#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import cv2
import rclpy
import rclpy.logging
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from ultralytics import YOLO
import math
from stable_baselines3 import DQN
from torchvision import transforms as T
import torch

class RLDQNController:
    """Handles DQN-based reinforcement learning decision making for robot control."""

    def __init__(self):
        # Define discrete action space (steering, acceleration, braking)
        self.actions = []
        for steering in [0.0, 0.3, 0.6]:
            for accel in [0.0, 0.25, 0.5]:
                for brake in [0]:
                    self.actions.append((steering, accel, brake))
                    self.actions.append((-steering, accel, brake))

        self.env = None
        self.best_model = DQN.load("segModels/best_model_900_512_128_18.zip", env=self.env)

    def predict_action(self, observation):
        """Runs the DQN model to get steering, acceleration, and braking."""
        obs = np.expand_dims(observation, axis=(0, 1))

        action_index, _ = self.best_model.predict(obs)
        selected_action = self.actions[action_index[0]]

        steering, accel, brake = selected_action
        return accel, steering, brake


class BirdEyeTransformer:
    """Transforms an image into a bird's-eye view (top-down perspective)."""

    def __init__(self):
        # Rotation angles (in radians)
        self.alpha = (35 - 90) * math.pi / 180  # pitch
        self.beta = (90 - 90) * math.pi / 180   # yaw
        self.gamma = (90 - 90) * math.pi / 180  # roll
        self.focal_length = 400
        self.distance = 105
        self._matrix_cache = {}  # Cache for transformation matrices

    def _get_transformation_matrix(self, w, h):
        """Returns a cached or newly computed transformation matrix for the given image size."""
        key = (w, h)
        if key in self._matrix_cache:
            return self._matrix_cache[key]

        A1 = np.array([[1, 0, -w / 2],
                       [0, 1, -h / 2],
                       [0, 0, 0],
                       [0, 0, 1]], dtype=np.float32)

        RX = np.array([[1, 0, 0, 0],
                       [0, math.cos(self.alpha), -math.sin(self.alpha), 0],
                       [0, math.sin(self.alpha), math.cos(self.alpha), 0],
                       [0, 0, 0, 1]], dtype=np.float32)

        RY = np.array([[math.cos(self.beta), 0, -math.sin(self.beta), 0],
                       [0, 1, 0, 0],
                       [math.sin(self.beta), 0, math.cos(self.beta), 0],
                       [0, 0, 0, 1]], dtype=np.float32)

        RZ = np.array([[math.cos(self.gamma), -math.sin(self.gamma), 0, 0],
                       [math.sin(self.gamma), math.cos(self.gamma), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], dtype=np.float32)

        R = np.dot(np.dot(RX, RY), RZ)

        T_bird = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, self.distance],
                           [0, 0, 0, 1]], dtype=np.float32)

        K = np.array([[self.focal_length, 0, w / 2, 0],
                      [0, self.focal_length, h / 2, 0],
                      [0, 0, 1, 0]], dtype=np.float32)

        M = np.dot(np.dot(np.dot(K, T_bird), R), A1)
        self._matrix_cache[key] = M
        return M

    def visualize(self, bev_image, cropped, original, show=False):
        """Optionally display images for debugging."""
        if show:
            cv2.imshow("YOLOv8 Road Segmentation", bev_image)
            cv2.imshow("Bird's-Eye View", cropped)
            cv2.imshow("Original Image", original)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rclpy.shutdown()

    def crop_and_resize(self, img, x1=145, x2=495, y1=0, y2=350, size=(30, 30)):
        """Crop the image to region of interest and resize."""
        cropped = img[y1:y2, x1:x2]
        if len(cropped.shape) == 3:
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        cropped = torch.from_numpy(cropped).float().unsqueeze(0)
        transforms = T.Compose([T.Resize(size, antialias=False)])
        cropped = transforms(cropped)
        cropped = cropped.squeeze(0).numpy().astype(np.uint8)
        return cropped
    
    def transform(self, frame):
        """Convert an input image into a bird's-eye view and crop for the RL model."""
        h, w = frame.shape[:2]
        M = self._get_transformation_matrix(w, h)

        bev_image = cv2.warpPerspective(frame, M, (w, h), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)
        cropped = self.crop_and_resize(bev_image)
        # True to visualize
        self.visualize(bev_image, cropped, frame, show=False)
        return cropped


class RoadSegmentationNode(Node):
    """ROS2 Node: Segments road using YOLOv8, applies bird's-eye view, and gets control commands from DQN."""

    def __init__(self, model_path):
        super().__init__('yolov8_segmentation_node')
        self.get_logger().info("YOLOv8 Road Segmentation Node started.")

        self.bridge = CvBridge()
        self.model = YOLO(model_path)

        self.road_class_id = 0
        self.rl_controller = RLDQNController()
        self.bev_transformer = BirdEyeTransformer()

        self.subscription = self.create_subscription(
            Image,
            '/camera_sensor/image_raw',
            self.image_callback,
            10
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.cmd_msg = Twist()

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            results = self.model.predict(frame, task='segment', verbose=False)[0]
            road_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

            if results.masks is not None:
                for seg, cls_id in zip(results.masks.xy, results.boxes.cls):
                    if int(cls_id) == self.road_class_id:
                        mask = np.zeros_like(road_mask)
                        polygon = np.array([seg], dtype=np.int32)
                        cv2.fillPoly(mask, polygon, 255)
                        road_mask = np.maximum(road_mask, mask)
            
            cropped_mask = self.bev_transformer.transform(road_mask)
            accel, steering, brake = self.rl_controller.predict_action(cropped_mask)
            self.get_logger().info(f"Action: steer={steering}, accel={accel}, brake={brake}")

            self.cmd_msg.linear.x = accel
            self.cmd_msg.angular.z = -steering
            self.cmd_pub.publish(self.cmd_msg)

        except Exception as e:
            self.get_logger().error(f"Error in image processing: {e}")


def main(args=None):
    rclpy.init(args=args)
    model_path = 'segModels/best.pt'
    node = RoadSegmentationNode(model_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
