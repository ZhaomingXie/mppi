import time
import mujoco
import mujoco.viewer
import os

def main():
    # Path to the Go2 XML file
    # Assuming the script is run from the root of the workspace or relative paths work as expected
    # Based on previous `find_by_name` result: unitree_mujoco/unitree_robots/go2/go2.xml
    model_path = os.path.join(os.path.dirname(__file__), "unitree_mujoco/unitree_robots/go2/scene.xml")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Load the model
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Create the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Close the viewer automatically after 30 wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < 30:
            step_start = time.time()

            # Step the simulation
            mujoco.mj_step(model, data)

            # Sync the viewer
            viewer.sync()

            # Time keeping
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
