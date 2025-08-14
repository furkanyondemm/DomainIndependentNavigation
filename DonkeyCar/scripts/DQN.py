import gym
import gym_donkeycar  # noqa: F401
import cv2
import numpy as np
import time
from ultralytics import YOLO
from bird_view import BirdEyeTransformer
from stable_baselines3 import DQN
from rl_agent import RL_DQN


class DonkeyCarAgent:
    def __init__(self, seg_model_path="best.pt", rl_model_path="best_model_900_512_128_18.zip"):
         
        # --- ENV ---
        self.env_name = "donkey-waveshare-v0"
        self.conf = {
            "host": "127.0.0.1",
            "port": 9091,
            "cam_resolution": (480, 640, 3),
            "img_w": 640,
            "img_h": 480,
            "img_d": 3,
            "fov" : "72", 
            "offset_x" : "0",
            "offset_y" : "0",
            "offset_z" : "0",
            "rot_x" : "35"
        }
        self.env = gym.make(self.env_name, conf=self.conf)
        self.env.reset()
        # To subscribe to the first frame
        self.obs, _, self.done, _ = self.env.step([0.0, 0.0])
        # --- Perception ---
        self.model = YOLO(seg_model_path)
        self.road_class_id = 0  # YOLO-Klassen-ID für 'Straße'

        # Bird Eye View
        self.bev = BirdEyeTransformer()

        # --- Control ---
        self.rl = RL_DQN(rl_model_path=rl_model_path)


    def _get_frame_bgr(self):
        """Return latest frame from self.obs as BGR."""
        if isinstance(self.obs, np.ndarray):
            return cv2.cvtColor(self.obs, cv2.COLOR_RGB2BGR), self.done
        return None, self.done

    

    def _segment_road_mask(self, frame_bgr):
        """Generate a binary road mask using YOLO segmentation."""
        res = self.model.predict(frame_bgr, task='segment', verbose=False)[0]
        mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        if res.masks is not None:
            for seg, cls_id in zip(res.masks.xy, res.boxes.cls):
                if int(cls_id) == self.road_class_id:
                    poly = np.array([seg], dtype=np.int32)
                    cv2.fillPoly(mask, poly, 255)
        return mask
        
    def _bev_crop_for_rl(self, road_mask):
        """Apply Bird’s-Eye View transformation and crop the ROI for RL."""
        return self.bev.transform(road_mask)
    
    def _bev_filter_mask(self, bev_crop):
        return np.array(bev_crop > 127, dtype=np.uint8)
    
    def _decide_action(self, bev_crop):
        """Use the RL model to predict action and scale it for the environment."""
        accel, steering, *_ = self.rl.rlDQN(bev_crop)
        return accel, steering

    def run_step(self):
        """Run a single perception → decision → control cycle."""
        start_time= time.time() 
        frame_bgr, done = self._get_frame_bgr()
        if frame_bgr is None:
            return done

        road_mask = self._segment_road_mask(frame_bgr)
        bev_crop = self._bev_crop_for_rl(road_mask)
        bev_mask = self._bev_filter_mask(bev_crop=bev_crop)
        accel, steering = self._decide_action(bev_mask)

        # Send command to simulator
        self.obs, reward, self.done, info = self.env.step([steering, accel]) 
        # Calc Time
        stop_time=time.time()
        inference_frequency = 1 / (stop_time - start_time)
        # Optional logging
        print(f"accel:{accel:.2f} steering:{steering:.2f} in {inference_frequency:4.2f} Hz")

        return done


    def run(self):
        """Continuous main loop."""
        try:
            while True:
                done = self.run_step()
                #if done:
                    #self.env.reset()
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            self.env.close()


if __name__ == "__main__":
    seg_model_path = "weights/best_donkey.pt"
    rl_model_path = "mask_logs/CarRacing_30_30_1_net_512_256_128_20"
    
    agent = DonkeyCarAgent(seg_model_path=seg_model_path, rl_model_path=rl_model_path)
    agent.run()

