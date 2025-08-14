import os
import argparse
import numpy as np
import platform
import matplotlib.pyplot as plt
import torch
from importlib.metadata import version
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.torch_layers import FlattenExtractor, NatureCNN

from wrappers.CustomNetwork import CustomCnnPolicy
from wrappers.CustomCarRacing import CustomCarRacingDiscreteWrapper
from wrappers.CustomObservation import FilterObservation

print(f"Python Version: {platform.python_version()}")
print(f"Torch Version: {version('torch')}")
print(f"Is Cuda Available: {torch.cuda.is_available()}")
# print(f"Cuda Version: {torch.version.cuda}")
print(f"Gymnasium Version: {version('gymnasium')}")
print(f"Numpy Version: {version('numpy')}")
print(f"Swig Version: {version('swig')}")
print(f"Stable Baselines3 Version: {version('stable_baselines3')}")
# print(f"IPython Version: {version('ipython')}")


def make_env(sample_actions):
    # Create the original continuous CarRacing environment
    original_env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
    wrapped_env = CustomCarRacingDiscreteWrapper(original_env,
                                                 sample_actions=sample_actions)
                                                 #steering_actions=steering_actions, 
                                                 #accel_actions=accel_actions, 
                                                 #braking_actions=braking_actions)
    # Apply the TimeLimit wrapper
    timelimit_env = TimeLimit(wrapped_env, max_episode_steps=1000)
    return timelimit_env


def learn(epochs, n_stack, log_dir, wrapper_kwargs, policy_kwargs, env_kwargs):
    # Create Training CarRacing environment
    env = make_vec_env(make_env,
                    n_envs=1,
                    env_kwargs=env_kwargs,
                    wrapper_class=FilterObservation,
                    wrapper_kwargs=wrapper_kwargs)
    env = VecFrameStack(env, n_stack=n_stack)
    env = VecTransposeImage(env)

    # Create Evaluation CarRacing environment
    env_val = make_vec_env(make_env,
                        n_envs=1,
                        env_kwargs=env_kwargs,
                        wrapper_class=FilterObservation,
                        wrapper_kwargs=wrapper_kwargs)
    env_val = VecFrameStack(env_val, n_stack=n_stack)
    env_val = VecTransposeImage(env_val)

    # Create Evaluation Callback
    # eval_freq - can cause learning instability if set to low
    eval_callback = EvalCallback(env_val,
                                best_model_save_path=log_dir,
                                log_path=log_dir,
                                eval_freq=int(epochs/50),#50,
                                render=False,
                                n_eval_episodes=20)

    # Initialize DQN
    # buffer_size - encourages exploration of other actions
    model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0, buffer_size=150_000)
    print(model.q_net)

    # Train the model
    model.learn(total_timesteps=epochs, #750_000,
                progress_bar=True,
                callback=eval_callback)

    # Save the model
    model.save(os.path.join(log_dir, "dqn_car_racing_script"))

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()
    env_val.close()


def evaluate(log_dir,wrapper_kwargs, env_kwargs):
    # Create Evaluation CarRacing environment
    env = make_vec_env(make_env,
                    n_envs=1,
                    seed=0,
                    env_kwargs=env_kwargs,
                    wrapper_class=FilterObservation,
                    wrapper_kwargs=wrapper_kwargs)
    env = VecFrameStack(env, n_stack=n_stack)
    env = VecTransposeImage(env)

    # Load the best model
    best_model_path = os.path.join(log_dir, "best_model.zip")
    best_model = DQN.load(best_model_path, env=env)

    mean_reward, std_reward = evaluate_policy(best_model, env, n_eval_episodes=20)
    print(f"Best Model - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Record video of the best model playing CarRacing
    env = VecVideoRecorder(env, f"{log_dir}/videos/",
                        video_length=10000,
                        record_video_trigger=lambda x: x == 0,
                        name_prefix="best_model_car_racing_mask_script")

    obs = env.reset()
    for _ in range(10000):
        action, _states = best_model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            break

    env.close()

    # Save Plot
    data = np.load(os.path.join(log_dir, "evaluations.npz"))
    plt.plot(data["timesteps"], np.mean(data["results"], axis=1) )
    plt.xlabel("Timesteps")
    plt.ylabel("Average Success")
    plt.title("Evaluation Results")
    plt.savefig(os.path.join(log_dir, "results.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Control Network with Stable Baseline3"
    )
    parser.add_argument("--n_steering_actions", default=4, type=int)
    parser.add_argument("--n_accel_actions", default=4, type=int)
    parser.add_argument("--n_braking_actions", default=1, type=int)
    parser.add_argument("--n_stack", default=1, type=int)
    parser.add_argument("--epochs", default=1_000_000, type=int)
    parser.add_argument("--widht", default=30, type=int)
    parser.add_argument("--height", default=30, type=int)
    parser.add_argument("--fov", default=72, type=int)
    parser.add_argument("--x_offset", default=1, type=int)
    parser.add_argument("--net_arch", default=[512,128], type=int, nargs='+')
    args = parser.parse_args()

    sample_actions = np.array([(-1.0, 1.0, 0), (-1.0, 0.5, 0), (-1.0, 0, 0.0), (-1.0, 0, 0.5), 
                             (-0.5, 1.0, 0), (-0.5, 0.5, 0), (-0.5, 0, 0.0), (-0.5, 0, 0.5),
                             (0.0, 1.0, 0),  (0.0, 0.5, 0),  (0.0, 0, 0.0),  (0.0, 0, 0.5), 
                             (0.5, 1.0, 0),  (0.5, 0.5, 0),  (0.5, 0, 0.0),  (0.5, 0, 0.5),
                             (1.0, 1.0, 0),  (1.0, 0.5, 0),  (1.0, 0, 0.0),  (1.0, 0, 0.5)])
    

    net_arch = args.net_arch
    epochs = args.epochs
    n_stack = args.n_stack
    steering_actions = np.linspace(0, 1.0, args.n_steering_actions)
    accel_actions = np.linspace(0, 1.0, args.n_accel_actions)
    braking_actions = np.linspace(0, 1.0, args.n_braking_actions)
    widht = args.widht
    height = args.height
    fov = args.fov
    x_offset = args.x_offset
    n_action_space = len(steering_actions) * 2 * len(accel_actions) * len(braking_actions)
    home_path = "."
    log_dir_str = f"CarRacing_{widht}_{height}_{n_stack}_net_{'_'.join(str(x) for x in net_arch)}_{len(sample_actions)}"
    network = ""
    log_dir = f"{home_path}/mask_logs/{log_dir_str}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    np.savetxt(f"{log_dir}/action_space.txt", sample_actions, fmt='%f')
    

    env_kwargs = dict(
        sample_actions=sample_actions
    )

    wrapper_kwargs = dict(
        width=widht, 
        height=height, 
        fov=fov,
        x_offset=x_offset
    )

    policy_kwargs = dict(
        net_arch=net_arch,
        features_extractor_class=FlattenExtractor,  #FlattenExtractor, NatureCNN
        features_extractor_kwargs=None,
        )
    
    learn(epochs=epochs, n_stack=n_stack, log_dir=log_dir, wrapper_kwargs=wrapper_kwargs, policy_kwargs=policy_kwargs, env_kwargs=env_kwargs)
    evaluate(log_dir=log_dir, wrapper_kwargs=wrapper_kwargs, env_kwargs=env_kwargs)