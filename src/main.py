##YOLO + MotionDetector
import cv2 as cv
import numpy as np
import yaml
import utils
from Human_Alert import Alert
from motion_detector import MotionDetector
from classifier import ObjectClassifier
# Config
CONFIG_PATH = "../config/config.yaml"
with open(CONFIG_PATH, "r" , encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
#Video 
video = cv.VideoCapture(cfg["video"]["source"])
#MotionDetector
motionDetector = MotionDetector()
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

#Base Loop
while True:
    _,frame = video.read()
    if frame is None :
        break
    frame_idx += 1  
    run_detection = (frame_idx % DETECT_EVERY == 0)

    resized_frame = utils.rescaleFrame(frame,1)

    motions = motionDetector.update(resized_frame,cfg['contours']['min_area'])

    valid_motions = [] # for alert
    for motion in motions:
        x,y,w,h = motion 

        roi = resized_frame[y:y+h, x:x+w]

        is_valid = False # motion is valid

        if classifier is not None and run_detection:
            chosen_detection = classifier.detect(roi)

            if chosen_detection is not None:
                is_valid = True    

        if  is_valid == False :
            continue

        valid_motions.append(motion)  

        if cfg['debug']['show_contours']:

            cv.rectangle(resized_frame,(x,y),(x+w,y+h),(0,250,0),thickness=2)

            if classifier is not None and chosen_detection is not None:
                label_text = chosen_detection["label"]
                cv.putText(resized_frame, label_text, (x, y-10),cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv.imshow("Display",resized_frame)

    if cfg['alert']['enabled']:
        alert.printAlert(valid_motions)



    if cfg['debug']['show_mask']:
        motionDetector.show()
    if cv.waitKey(20) & 0xFF==ord('0'):
            break

video.release()
cv.destroyAllWindows()    
