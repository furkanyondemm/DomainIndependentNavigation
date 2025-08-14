from stable_baselines3 import DQN
import numpy as np

class RL_DQN:
    def __init__(self, rl_model_path):
        self.sample_actions = np.loadtxt(f"{rl_model_path}/action_space.txt", dtype=float)
        print(f"Size of actionspace: {len(self.sample_actions)}")

        self.env = None
        self.best_model = DQN.load(f"{rl_model_path}/best_model.zip", env=self.env)

    def rlDQN(self, img):
        obs = np.expand_dims(img, axis=(0, 1))
        action, _states = self.best_model.predict(obs)
        select_action = self.sample_actions[action[0]]
        #braking
        if select_action[2] !=0:
            return -select_action[2], select_action[0]
        else:
            return select_action[1], select_action[0]  # accel, steering