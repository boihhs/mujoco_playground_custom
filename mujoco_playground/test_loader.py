import mujoco
import mujoco.viewer
import jax.numpy as jp

# Load your XML model (replace with your file path)
model = mujoco.MjModel.from_xml_path("/home/leo-benaharon/Desktop/mujoco_playground_custom/mujoco_playground/_src/locomotion/zeroth/xmls/scene_mjx_feetonly_flat_terrain.xml")
data = mujoco.MjData(model)

home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
print(home_id)
mujoco.mj_resetDataKeyframe(model, data, home_id)
mujoco.mj_step(model, data)
print(data.xpos)

site_id_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
site_id_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

with mujoco.viewer.launch_passive(model, data) as viewer:
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    while viewer.is_running():
        mujoco.mj_step(model, data)
        print(data.site_xpos[site_id_L])
        print(data.site_xpos[site_id_R])
        print(jp.linalg.norm(data.site_xpos[site_id_L] - data.site_xpos[site_id_R]))

        viewer.sync()


