import numpy as np
import math
import cv2

class BirdEyeTransformer:
    """Transforms an image into a bird's-eye view (top-down perspective)."""
    def __init__(self):
        # Rotation angles (in radians)
        self.alpha = (35 - 90) * math.pi / 180 # pitch
        self.beta = (90 - 90) * math.pi / 180  # yaw
        self.gamma = (90 - 90) * math.pi / 180 # roll
        self.focal_length = 380
        self.dist = 120 
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
                           [0, 0, 1, self.dist],
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
                print("Programm endet...")
                cv2.destroyAllWindows()
                

    def crop_and_resize(self, img, x1=145, x2=495, y1=0, y2=350, size=(30,30)):
        """Crop the image to region of interest, apply treshold and resize."""
        cropped = img[y1:y2, x1:x2]
        _, cropped = cv2.threshold(cropped, 254, 255, cv2.THRESH_BINARY)
        return cv2.resize(cropped, size, interpolation=cv2.INTER_AREA)

    def transform(self, frame):
        """Convert an input image into a bird's-eye view and crop for the RL model."""
        h, w = frame.shape[:2]
        M = self._get_transformation_matrix(w, h)

        bev_image = cv2.warpPerspective(frame, M, (w, h), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)
        cropped = self.crop_and_resize(bev_image)
        # to visualize TRUE
        self.visualize(bev_image, cropped, frame, show=False)
        return cropped