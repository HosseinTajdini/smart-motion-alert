import cv2 as cv
class BackgroundSubtractor:
    def __init__(self, method="mog2"):
        if method == "mog2":
            self.model = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        elif method == "knn":
            self.model = cv.createBackgroundSubtractorKNN()
        else:
            raise ValueError("Unknown method")

    def apply(self, frame):
        fg_mask = self.model.apply(frame)
        return fg_mask
