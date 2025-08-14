import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


class CustomCarRacingDiscreteWrapper(gym.Wrapper):
    def __init__(self, 
                 env, 
                 sample_actions):
                 #steering_actions=[0,0.33,0.67], 
                 #accel_actions=[0,0.33,0.67], 
                 #braking_actions=[0,0.5]):
        super().__init__(env)
        # Define your new discrete action space
        self.sample_actions = sample_actions
        #for steering in reversed(steering_actions):
        #    for accel in reversed(accel_actions):
        #        for braking in reversed(braking_actions):
        #            self.sample_actions.append((-steering,accel,braking))
        #            #self.sample_actions.append((steering,accel,braking))
        #for steering in steering_actions:
        #    for accel in accel_actions:
        #        for braking in braking_actions:
        #            self.sample_actions.append((steering,accel,braking))
        # self.action_space = spaces.Discrete(5) # 5 actions as defined above
        
        self.action_space = gym.spaces.Discrete(len(self.sample_actions))


    def step(self, action):
        # Map the discrete action to a continuous action for the base environment
        continuous_action = np.array(self.sample_actions[action])
        # Pass the continuous action to the original environment's step method
        observation, reward, terminated, truncated, info = self.env.step(continuous_action)
        return observation, reward, terminated, truncated, info
    

def make_env():
    # Create the original continuous CarRacing environment
    original_env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
    wrapped_env = CustomCarRacingDiscreteWrapper(original_env)
    # Apply the TimeLimit wrapper
    timelimit_env = TimeLimit(wrapped_env, max_episode_steps=1000)
    return timelimit_env

