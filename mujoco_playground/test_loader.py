import mujoco
import mujoco.viewer

# Load your XML model (replace with your file path)
model = mujoco.MjModel.from_xml_path("/home/leo-benaharon/Desktop/mujoco_playground_custom/mujoco_playground/_src/locomotion/zeroth/xmls/scene_mjx_feetonly_flat_terrain.xml")
data = mujoco.MjData(model)

home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
print(home_id)
mujoco.mj_resetDataKeyframe(model, data, home_id)
mujoco.mj_step(model, data)
print(data.xpos)

with mujoco.viewer.launch_passive(model, data) as viewer:
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    while viewer.is_running():
        mujoco.mj_resetDataKeyframe(model, data, home_id)
        mujoco.mj_step(model, data)


        print(data.xpos)
        viewer.sync()


