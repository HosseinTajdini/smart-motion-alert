
import cv2 as cv
import numpy as np
import yaml
from Human_Alert import Alert
from motion_detector import MotionDetector

def rescaleFrame(frame,scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimentions = (width,height)
    return cv.resize(frame,dimentions,interpolation = cv.INTER_AREA)


CONFIG_PATH = "../config/config.yaml"
with open(CONFIG_PATH, "r" , encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

video = cv.VideoCapture(cfg["video"]["source"])

motionDetector = MotionDetector()

alert = Alert(5)
while True:
    _,frame = video.read()
    if frame is None :
        break

    resized_frame = rescaleFrame(frame)

    motions = motionDetector.update(resized_frame,cfg['contours']['min_area'])

    for motion in motions:
        x,y,w,h = motion  
        if cfg['debug']['show_contours']:
            cv.rectangle(resized_frame,(x,y),(x+w,y+h),(0,250,0),thickness=2)

    if cfg['alert']['enabled']:
        alert.printAlert(motions)

    cv.imshow("Display",resized_frame)

    if cfg['debug']['show_mask']:
        motionDetector.show()

    if cv.waitKey(20) & 0xFF==ord('0'):
            break

video.release()
cv.destroyAllWindows()    
