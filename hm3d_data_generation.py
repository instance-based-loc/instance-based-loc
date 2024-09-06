import os
import numpy as np
import habitat
from habitat.datasets.image_nav.instance_image_nav_dataset import InstanceImageNavDatasetV1
from habitat.tasks.nav.instance_image_nav_task import InstanceImageNavigationTask
from pprint import pprint
from habitat.config.read_write import read_write
from tqdm import tqdm
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
import cv2 

# Define the stage
stage = "val"
split = "val"
num_sampled_episodes = 1000
num_saved_episodes = 20
max_timesteps = 150
min_timesteps = 30
save_path = "/scratch/vineeth.bhat/instance-loc/hm3d_trajectories/object_nav_trajectories"
save_path = "/scratch/vineeth.bhat/instance-loc/hm3d_trajectories/instance_nav_trajectories"

# Load the configuration from a YAML file
config_path = "dataloader/hm3d_config_instance_image_nav_mod.yaml"
# config_path = "dataloader/objnav_modded.yaml"

# Check if the configuration file exists
if not os.path.exists(config_path):
    raise RuntimeError(f"{config_path} does not exist!")

# Load the habitat configuration
habitat_config = habitat.get_config(config_path)

# Define data paths
data_path = f"/scratch/vineeth.bhat/instance-loc/data/datasets/instance_imagenav/hm3d/v3/{stage}/{stage}.json.gz"
# data_path = f"/scratch/vineeth.bhat/instance-loc/data/datasets/objectnav/hm3d/v2/{stage}/{stage}.json.gz"
scene_dataset = f"/scratch/vineeth.bhat/instance-loc/data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"

# Check if the data paths exist
if not os.path.exists(data_path):
    raise RuntimeError(f"{data_path} does not exist!")

if not os.path.exists(scene_dataset):
    raise RuntimeError(f"{scene_dataset} does not exist!")

# Update habitat configuration
robot_height=0.88
robot_radius=0.25
sensor_height=0.88
image_width=600
image_height=600
image_hfov=90
step_size=0.2
turn_angle=15
with read_write(habitat_config):
    habitat_config.habitat.dataset.split = split
    habitat_config.habitat.dataset.scenes_dir = "/scratch/vineeth.bhat/instance-loc/data/scene_datasets"
    habitat_config.habitat.dataset.data_path = data_path
    habitat_config.habitat.simulator.scene_dataset = scene_dataset
    habitat_config.habitat.environment.iterator_options.num_episode_sample = num_sampled_episodes
    habitat_config.habitat.simulator.agents.main_agent.height=robot_height
    habitat_config.habitat.simulator.agents.main_agent.radius=robot_radius
    habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = image_height
    habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = image_width
    habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = image_hfov
    habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position = [0,sensor_height,0]
    habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height = image_height
    habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width = image_width
    habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov = image_hfov
    habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.position = [0,sensor_height,0]
    habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = 50.0
    habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth = 0.02
    habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False

# Print the sensors of the main agent
pprint(habitat_config.habitat.simulator.agents.main_agent.sim_sensors)

# Initialize the Habitat environment
try:
    env = habitat.Env(habitat_config)
    print("Environment initialized successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to initialize the Habitat environment: {e}")

episode_counter = 0

follower = ShortestPathFollower(env.sim, goal_radius=1.5, return_one_hot=False)

while episode_counter < num_saved_episodes:
    obs = env.reset()
    rgb_data = []
    depth_data = []
    pose_data = []
    timesteps = 0
    goal_position = env.current_episode.goals[0].position
    
    # Iterate over every step of the episode
    while True:
        best_action = follower.get_next_action(goal_position)
        if best_action is None:
            print(f"Goal reached at timestep {timesteps}")
            break
        obs = env.step(best_action)
        
        rgb_data.append(obs['rgb'])
        depth_data.append(obs['depth'])

        q = env.sim.get_agent_state().sensor_states["depth"].rotation
        pose = np.concatenate(
            [np.array(env.sim.get_agent_state().sensor_states["depth"].position),
            np.array([q.w, q.x, q.y, q.z])]
        )

        pose_data.append(pose)

        timesteps += 1
        if env.episode_over or timesteps >= max_timesteps:
            if timesteps < min_timesteps:
                print(f"{timesteps} less than min. timesteps of {min_timesteps}")
                break
            
            episode_save_path = os.path.join(save_path, f"episode_{episode_counter}")
            os.makedirs(episode_save_path)

            pose_data_path = os.path.join(episode_save_path, "poses.npy")
            np.save(pose_data_path, pose_data)

            rgb_images_save_path = os.path.join(episode_save_path, "rgb")
            depth_images_save_path = os.path.join(episode_save_path, "depth")
            os.makedirs(rgb_images_save_path)
            os.makedirs(depth_images_save_path)

            for i in range(len(rgb_data)):
                rgb_image_path = os.path.join(rgb_images_save_path, f"{i}.png")
                depth_image_path = os.path.join(depth_images_save_path, f"{i}.npy")

                # Save RGB image
                rgb_image_bgr = cv2.cvtColor(rgb_data[i], cv2.COLOR_RGB2BGR)
                cv2.imwrite(rgb_image_path, rgb_image_bgr)

                # Save depth data
                np.save(depth_image_path, depth_data[i])

            episode_counter += 1
            print(f"Done with episode {episode_counter} out of {num_saved_episodes} total")
            break

# Close the environment after processing
env.close()
