import cv2 as cv
import yaml

CONFIG_PATH = "../config/config.yaml"
with open(CONFIG_PATH, "r" , encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

class BackgroundSubtractor:
    def __init__(self, method="mog2"):
        if method == "mog2":
            self.model = cv.createBackgroundSubtractorMOG2(history=cfg['background']['history'],
                                                           varThreshold=cfg['background']['var_threshold'],
                                                           detectShadows=cfg['background']['detect_shadows'])
        elif method == "knn":
            self.model = cv.createBackgroundSubtractorKNN(detectShadows=cfg['background']['detect_shadows'])
        else:
            raise ValueError("Unknown method")

    def apply(self, frame,lr=0.01):
        fg_mask = self.model.apply(frame,lr)
        return fg_mask
