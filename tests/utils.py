import cv2 as cv
def rescaleFrame(frame,scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimentions = (width,height)
    return cv.resize(frame,dimentions,interpolation = cv.INTER_AREA)
