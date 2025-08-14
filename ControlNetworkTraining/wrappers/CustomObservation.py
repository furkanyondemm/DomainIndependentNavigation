import torch
import gymnasium
import numpy as np
from gymnasium.spaces import Box
from torchvision import transforms as T
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage, VecFrameStack

class FilterObservation(gymnasium.ObservationWrapper[np.ndarray, int, np.ndarray]):
    # def __init__(self, env, width=84, height=84, threshold=120, fov=72):
    def __init__(self, env, width=30, height=30, threshold=120, fov=72, x_offset=0):
        super().__init__(env)
        self.lowest_y_pixel_ratio=70/96
        self.threshold = threshold
        self.fov = fov
        self.width = width
        self.height = height
        self.x_offset = x_offset
        self.fov_mask = self.init_fov_mask()
        self.observation_space = Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)
        
    def grayscale_observation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor and convert to grayscale
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation
    
    def resize_observation(self, observation):
        #transforms = T.Compose(
        #    [T.Resize((self.height, self.width), antialias=True), 
        #     T.Normalize(0, 255),
        #     T.ConvertImageDtype(dtype=torch.uint8)
        #    ]
        #)
        transforms = T.Compose(
            [T.ToTensor(),
             T.Grayscale(),
             T.Resize((self.height, self.width), antialias=False), 
             T.Normalize((0),(1)),
             T.ConvertImageDtype(dtype=torch.uint8)
            ]
        )
        observation = transforms(observation).squeeze(0)
        return observation
    
    def init_fov_mask(self):
        middle_x_pixel = int(self.width/2)
        fov_rad = np.deg2rad(self.fov)
        fov_mask = torch.zeros((self.height, self.width))
        lowest_y_pixel = int(self.lowest_y_pixel_ratio * self.height)
        for row in range(lowest_y_pixel):
            x_width = np.tan(fov_rad/2) * (lowest_y_pixel - row)
            x_width += self.x_offset
            if x_width > middle_x_pixel:
                x_width = middle_x_pixel
            fov_mask[row, int(middle_x_pixel - x_width):middle_x_pixel] = 1
            fov_mask[row, middle_x_pixel:int(middle_x_pixel + x_width)] = 1
        return fov_mask
    
    def filter_observation(self, observation):
        grey_mask = observation < self.threshold
        # mask = np.logical_and(fov_mask, grey_mask)
        mask = torch.logical_and(self.fov_mask, grey_mask)
        return mask
    
    def observation(self, observation):
        # height, width = observation.shape
        # Grayscale
        # observation = self.grayscale_observation(observation)
        # Resize
        observation = self.resize_observation(observation)
        # Filter FOV and Road
        observation = self.filter_observation(observation)
        return observation.unsqueeze(2)  #[:,:,None]


if __name__ == "__main__":
    env_str = "CarRacing-v3"
    log_dir = f"mask_logs/{env_str}"
    env_kwargs_dict={"continuous": False, "render_mode": "rgb_array"}
    n_stack = 4

    env = make_vec_env(env_str,
                    n_envs=1,
                    env_kwargs=env_kwargs_dict,
                    wrapper_class=FilterObservation)

    env = VecFrameStack(env, n_stack=n_stack)
    env = VecTransposeImage(env)

    print(type(env))
    print("Observation Space Size: ", env.observation_space)
    print("Action Space Size: ", env.action_space)