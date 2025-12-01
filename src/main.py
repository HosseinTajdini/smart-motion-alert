##YOLO + MotionDetector
import cv2 as cv
import numpy as np
import yaml
import utils
from human_alert import Alert
from classifier import ObjectClassifier
import time

# Config
CONFIG_PATH = "../config/config.yaml"
with open(CONFIG_PATH, "r" , encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

#Video 
video = cv.VideoCapture(cfg["video"]["source"])

#Alert System 
alert = Alert(5)

#Classifier
if cfg['detector']['enabled']:
    classifier = ObjectClassifier(weights_path=cfg['detector']['weights_path'],
                                  conf_threshold=cfg['detector']['conf_threshold'],
                                  allowed_classes=cfg['detector']['allowed_classes'])
else:
    classifier = None

frame_idx = 0
DETECT_EVERY = 3 # skip frame for optimization

last_detection = None
stale_frames = 0
STALE_LIMIT = 5 

prev_time = time.time()
fps = 0
#Base Loop
while True:
    _,frame = video.read()
    if frame is None :
        break
    frame_idx += 1  
    run_detection = (frame_idx % DETECT_EVERY == 0)

    resized_frame = utils.rescaleFrame(frame,1)

    if classifier is not None and run_detection:
        chosen_detection = classifier.detect(resized_frame)
        if chosen_detection is not None:
            last_detection = chosen_detection
            stale_frames = 0   
        else:
            stale_frames += 1

        if last_detection is not None and stale_frames < STALE_LIMIT:
            x1,y1,x2,y2 = last_detection['bbox']
            if cfg['debug']['show_contours']:

                cv.rectangle(resized_frame,(x1,y1),(x2,y2),(0,0,255),thickness=2)
                label_text = last_detection["label"]
                cv.putText(resized_frame, label_text, (x1, y1-10),cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    #Show FPS
    current_time = time.time()
    delta = current_time - prev_time
    if delta > 0:
        fps = 1 / delta
    prev_time = current_time
    cv.putText(resized_frame,f"FPS: {fps:.1f}",(10, 30),cv.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 0),2)

    cv.imshow("Display",resized_frame)

    if cfg['alert']['enabled']:
        alert.printAlert(last_detection)

    if cv.waitKey(20) & 0xFF==ord('0'):
            break

video.release()
cv.destroyAllWindows()    
