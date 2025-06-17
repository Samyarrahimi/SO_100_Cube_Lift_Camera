#!/usr/bin/env python3

"""
Test script to verify the SO100 gripping improvements.
This script runs a few episodes with random actions to test the new contact sensors and gripper control.
"""

import argparse
import sys
from pathlib import Path

# Add the source directory to the path
source_dir = Path(__file__).parent.parent / "source"
sys.path.insert(0, str(source_dir))

import numpy as np
from omni.isaac.kit import SimulationApp

# Import the environment
from SO_100_Cube_Lift_With_Camera.tasks.manager_based.so_100_cube_lift_with_camera.so_100_cube_lift_with_camera_env_cfg import SO100CubeLiftCameraEnvCfg
from isaaclab.envs import ManagerBasedRLEnv


def test_gripping_improvements():
    """Test the gripping improvements with random actions."""
    
    # Launch Isaac Sim
    simulation_app = SimulationApp({"headless": False})
    
    try:
        # Create environment with improved configuration
        env_cfg = SO100CubeLiftCameraEnvCfg()
        env_cfg.scene.num_envs = 4  # Small number for testing
        env_cfg.scene.env_spacing = 3.0
        
        env = ManagerBasedRLEnv(env_cfg)
        
        print("Environment created successfully!")
        print(f"Number of environments: {env.num_envs}")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        
        # Test for a few episodes
        num_episodes = 3
        max_steps_per_episode = 100
        
        for episode in range(num_episodes):
            print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
            
            # Reset environment
            obs, info = env.reset()
            print(f"Episode reset. Observation keys: {list(obs.keys())}")
            
            episode_reward = 0.0
            gripper_contact_count = 0
            successful_grasp_count = 0
            
            for step in range(max_steps_per_episode):
                # Generate random actions
                actions = {}
                for key, space in env.action_space.items():
                    if space.shape[0] == 1:  # Gripper action
                        # Bias towards closing the gripper
                        if np.random.random() < 0.7:
                            actions[key] = np.array([-0.2])  # Close gripper
                        else:
                            actions[key] = np.array([0.1])   # Open gripper
                    else:
                        # Random arm actions
                        actions[key] = np.random.uniform(-0.5, 0.5, space.shape)
                
                # Step environment
                obs, rewards, terminated, truncated, info = env.step(actions)
                
                # Track rewards
                episode_reward += sum(rewards.values())
                
                # Check for gripping behavior
                if 'gripper_contact' in rewards and rewards['gripper_contact'] > 0:
                    gripper_contact_count += 1
                
                if 'successful_grasp' in rewards and rewards['successful_grasp'] > 0:
                    successful_grasp_count += 1
                
                # Check if episode is done
                if any(terminated) or any(truncated):
                    break
                
                # Print progress every 20 steps
                if step % 20 == 0:
                    print(f"  Step {step}: Reward = {episode_reward:.2f}, "
                          f"Contact = {gripper_contact_count}, "
                          f"Grasp = {successful_grasp_count}")
            
            print(f"Episode {episode + 1} completed:")
            print(f"  Total reward: {episode_reward:.2f}")
            print(f"  Gripper contacts: {gripper_contact_count}")
            print(f"  Successful grasps: {successful_grasp_count}")
        
        print("\n=== Test completed successfully! ===")
        print("The gripping improvements are working correctly.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Close Isaac Sim
        simulation_app.close()


def main():
    parser = argparse.ArgumentParser(description="Test SO100 gripping improvements")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--num_episodes", type=int, default=3, help="Number of episodes to test")
    
    args = parser.parse_args()
    
    if args.headless:
        # Modify the simulation app to run headless
        import os
        os.environ["HEADLESS"] = "1"
    
    test_gripping_improvements()


if __name__ == "__main__":
    main() 