import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import re
import functools
import numpy as np
import jax
import jax.numpy as jp
import tensorflow as tf
from tensorflow.keras import layers
import tf2onnx
import onnxruntime as ort

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.training.checkpoint import load

from mujoco_playground.config import locomotion_params
from mujoco_playground import locomotion


# -------------------------
# Config
# -------------------------
ENV_NAME = "ZerothJoystickFlatTerrain"
CKPT_PATH = "/home/leo-benaharon/Desktop/mujoco_playground_custom/logs/ZerothJoystickFlatTerrain-20251002-211923/checkpoints/000074547200"
ONNX_PATH = "zeroth_policy.onnx"
DETERMINISTIC = True
EPS = 1e-8
CLIP = None  # set to a float (e.g., 5.0) if your training used clipping of normalized obs


# -------------------------
# Load env + networks
# -------------------------
ppo_params = locomotion_params.brax_ppo_config(ENV_NAME)

# build env
env_cfg = locomotion.get_default_config(ENV_NAME)
env = locomotion.load(ENV_NAME, config=env_cfg)

obs_size = env.observation_size  # dict-like: {'state': (D,), 'privileged_state': (P,), ...}
act_size = env.action_size

print("Observation sizes:", obs_size)
print("Action size:", act_size)

# Build JAX policy/inference fn
network_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    **ppo_params.network_factory,
    preprocess_observations_fn=running_statistics.normalize,  # match training
)
ppo_network = network_factory(obs_size, act_size)

# Load checkpoint: [0] running stats, [1] params/opt_state (structure depends on trainer)
ckpt = load(CKPT_PATH)
norm_state = ckpt[0]
params = ckpt[1]

make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
inference_fn = make_inference_fn((norm_state, params), deterministic=DETERMINISTIC)


# -------------------------
# Keras policy that mirrors JAX
# -------------------------
class MLP(tf.keras.Model):
    def __init__(self,
                 layer_sizes,
                 activation=tf.nn.swish,
                 kernel_init="lecun_uniform",
                 activate_final=False,
                 bias=True,
                 layer_norm=False,
                 mean_std=None,
                 eps=1e-8,
                 clip=None):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.kernel_init = kernel_init
        self.activate_final = activate_final
        self.bias = bias
        self.layer_norm = layer_norm
        self.eps = eps
        self.clip = clip

        if mean_std is not None:
            mean, std = mean_std
            self.mean = tf.Variable(mean, trainable=False, dtype=tf.float32, name="obs_mean")
            self.std  = tf.Variable(std,  trainable=False, dtype=tf.float32, name="obs_std")
        else:
            self.mean = None
            self.std  = None

        self.mlp_block = tf.keras.Sequential(name="MLP_0")
        for i, size in enumerate(self.layer_sizes):
            dense = layers.Dense(
                size,
                activation=self.activation,
                kernel_initializer=self.kernel_init,
                use_bias=self.bias,
                name=f"hidden_{i}"
            )
            self.mlp_block.add(dense)
            if self.layer_norm:
                self.mlp_block.add(layers.LayerNormalization(name=f"layer_norm_{i}"))
        # deactivate final layer if requested
        if not self.activate_final and self.mlp_block.layers:
            last = self.mlp_block.layers[-1]
            if hasattr(last, "activation") and last.activation is not None:
                last.activation = None

        self.submodules = [self.mlp_block]

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            x = inputs[0]
        else:
            x = inputs
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / tf.maximum(self.std, self.eps)
            if self.clip is not None:
                x = tf.clip_by_value(x, -self.clip, self.clip)
        logits = self.mlp_block(x)
        # Brax PPO policy typically parameterizes Tanh-Normal: split to loc/scale
        loc, _ = tf.split(logits, 2, axis=-1)
        return tf.tanh(loc)  # actions in [-1, 1]; rescale here if your env needs it


def make_policy_network(param_size, mean_std, hidden_layer_sizes, activation, layer_norm=False):
    return MLP(
        layer_sizes=list(hidden_layer_sizes) + [param_size],
        activation=activation,
        layer_norm=layer_norm,
        mean_std=mean_std,
        eps=EPS,
        clip=CLIP,
    )


# mean/std for the "state" part used by the policy
mean = norm_state.mean["state"]
std  = norm_state.std["state"]
mean_std = (tf.convert_to_tensor(np.array(mean, dtype=np.float32)),
            tf.convert_to_tensor(np.array(std, dtype=np.float32)))

tf_policy = make_policy_network(
    param_size=act_size * 2,
    mean_std=mean_std,
    hidden_layer_sizes=ppo_params.network_factory.policy_hidden_layer_sizes,
    activation=tf.nn.swish,
)

# Build variables
state_dim = int(np.prod(obs_size["state"]))
example_input = tf.zeros((1, state_dim), dtype=tf.float32)
_ = tf_policy(example_input)  # initialize variables
print("TF policy built. Output shape:", tf_policy(example_input).shape)


# -------------------------
# Weight transfer (Flax/Haiku robust)
# -------------------------
def find_dense_param_blocks(tree):
    """
    Recursively find dicts that look like Dense/Linear params.
    Returns ordered pairs (name, params_dict) where params_dict has kernel/bias or w/b.
    """
    found = {}

    def walk(d, path=()):
        if not isinstance(d, dict):
            return
        keys = set(d.keys())
        # Dense (Flax): kernel/bias; Haiku: w/b
        if ({"kernel", "bias"} <= keys) or ({"w", "b"} <= keys):
            name = path[-1] if path else "root"
            found[name] = d
            return
        for k, v in d.items():
            walk(v, path + (k,))

    walk(tree, ())
    return found


def extract_kernel_bias(layer_params):
    # Flax
    if "kernel" in layer_params and "bias" in layer_params:
        k = np.array(layer_params["kernel"])  # (in, out)
        b = np.array(layer_params["bias"])
        return k, b, "flax"
    # Haiku
    if "w" in layer_params and "b" in layer_params:
        k = np.array(layer_params["w"]).T     # (out, in) -> (in, out)
        b = np.array(layer_params["b"])
        return k, b, "haiku"
    raise KeyError(f"Unknown dense param format: {list(layer_params.keys())}")


def transfer_weights(jax_tree_params, keras_model: tf.keras.Model):
    """
    jax_tree_params: params['params'] subtree from checkpoint
    keras_model: built TF model with layers named hidden_0, hidden_1, ...
    """
    # Flatten JAX params to dense layers map
    dense_blocks = find_dense_param_blocks(jax_tree_params)

    # Prefer keys that look like 'hidden_0', 'hidden_1', ...
    def key_index(k):
        m = re.search(r"hidden_(\d+)", k)
        return int(m.group(1)) if m else None

    named_dense = [(k, v) for k, v in dense_blocks.items() if key_index(k) is not None]
    if not named_dense:
        # Fallback: try order by lexical path if names differ; match by index with TF layers
        # Build ordered list by trying to parse trailing integers; else sort by name
        named_dense = sorted(dense_blocks.items(), key=lambda kv: kv[0])

    # Build TF layer list in order
    tf_layers = [l for l in keras_model.get_layer("MLP_0").layers if isinstance(l, tf.keras.layers.Dense)]
    if len(tf_layers) != len(named_dense):
        # If LayerNorm present, tf_layers excludes them; still mismatch means JAX side has extra heads or different structure
        raise ValueError(f"Mismatched Dense count: TF={len(tf_layers)} vs JAX={len(named_dense)}. "
                         f"JAX Dense keys: {list(dense_blocks.keys())}")

    # Sort JAX layers by index if possible, else by name
    if all(key_index(k) is not None for k, _ in named_dense):
        named_dense.sort(key=lambda kv: key_index(kv[0]))
    else:
        named_dense.sort(key=lambda kv: kv[0])

    # Transfer one-by-one
    for (jax_name, lp), tf_layer in zip(named_dense, tf_layers):
        k, b, fmt = extract_kernel_bias(lp)
        # Sanity: shape match
        if k.shape != tuple(tf_layer.kernel.shape):
            raise ValueError(f"Kernel shape mismatch for {jax_name} ({fmt}): JAX {k.shape} vs TF {tuple(tf_layer.kernel.shape)}")
        if b.shape != tuple(tf_layer.bias.shape):
            raise ValueError(f"Bias shape mismatch for {jax_name} ({fmt}): JAX {b.shape} vs TF {tuple(tf_layer.bias.shape)}")
        tf_layer.set_weights([k, b])
        # Optional: print confirmations
        # print(f"Loaded {jax_name} -> {tf_layer.name} ({fmt}) {k.shape}")

    # Final check on first layer
    j0_k, _, _ = extract_kernel_bias(named_dense[0][1])
    tf0_k = tf_layers[0].get_weights()[0]
    print("max|ΔW(first dense)| =", np.max(np.abs(j0_k - tf0_k)))
    print("Weights transferred OK.")


# The network params are usually under params['params']
if "params" in params:
    jax_policy_params = params["params"]
else:
    # some trainers store directly at root
    jax_policy_params = params

# Build once more to ensure variables exist, then transfer
_ = tf_policy(example_input)
transfer_weights(jax_policy_params, tf_policy)


# -------------------------
# Export to ONNX
# -------------------------
tf_policy.output_names = ["continuous_actions"]
spec = [tf.TensorSpec(shape=(1, state_dim), dtype=tf.float32, name="obs")]
model_proto, _ = tf2onnx.convert.from_keras(
    tf_policy, input_signature=spec, opset=11, output_path=ONNX_PATH
)
print(f"Saved ONNX to {ONNX_PATH}")

# -------------------------
# Parity checks
# -------------------------
# Build identical test inputs
x_tf = np.ones((1, state_dim), dtype=np.float32)

# TF output
tf_out = tf_policy(x_tf)[0].numpy()

# ONNX output
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
onnx_out = sess.run(["continuous_actions"], {"obs": x_tf})[0][0]

# JAX output (construct full obs dict)
obs = {
    "state": jp.ones(obs_size["state"]),
    "privileged_state": jp.zeros(obs_size["privileged_state"]),
}
jax_out, _ = inference_fn(obs, jax.random.PRNGKey(0))
jax_out = np.array(jax_out)

print("Shapes -> TF:", tf_out.shape, "ONNX:", onnx_out.shape, "JAX:", jax_out.shape)
print("max|TF-ONNX| =", float(np.max(np.abs(tf_out - onnx_out))))
print("max|TF-JAX|  =", float(np.max(np.abs(tf_out - jax_out))))

# Optional: visualize
try:
    import matplotlib.pyplot as plt
    plt.plot(onnx_out, label="onnx")
    plt.plot(tf_out, label="tensorflow")
    plt.plot(jax_out, label="jax")
    plt.legend()
    plt.title("Action output parity check")
    plt.show()
except Exception as e:
    print("Plot skipped:", e)
