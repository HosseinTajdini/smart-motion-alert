import cv2 as cv
from background_subtractor import BackgroundSubtractor
import yaml

CONFIG_PATH = "../config/config.yaml"
with open(CONFIG_PATH, "r" , encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

class MotionDetector:

    def __init__(self):
        self.fgbg = BackgroundSubtractor(method=cfg['background']['method'])
        self.clean = None
        self.fg_mask = None
        self.contours = []



    def update(self,frame,min_area:int=300,learningRate:float=.01):
        fg_mask = self.fgbg.apply(frame,learningRate)
        self.fg_mask = fg_mask

        blur = cv.GaussianBlur(fg_mask,(cfg['preprocess']['blur_kernel'],cfg['preprocess']['blur_kernel']),0)

        _, thresh = cv.threshold( blur, cfg['preprocess']['threshold'], 255, cv.THRESH_BINARY) 

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (cfg['preprocess']['morph_kernel'], cfg['preprocess']['morph_kernel']))

        clean = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=cfg['preprocess']['morph_iterations'])

        clean = cv.dilate(clean, kernel, iterations=cfg['preprocess']['dilate_iterations'])
        self.clean = clean

        contours, _ = cv.findContours(clean, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.contours = contours

        motions = []
        for cnt in self.contours:
            area = cv.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv.boundingRect(cnt)
            motions.append((x, y, w, h))
        return motions   


    def show(self):
        cv.imshow("Clean",self.clean)
        cv.imshow("FG_Mask",self.fg_mask)
