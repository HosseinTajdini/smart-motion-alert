import cv2 

def rescaleFrame(frame,scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimentions = (width,height)
    return cv.resize(frame,dimentions,interpolation = cv.INTER_AREA)
cam = cv2.VideoCapture(0)

while True:
    _,frame = cam.read()
    resized_frame = rescaleFrame(frame)
    gray = cv.cvtColor(resized_frame,cv.COLOR_BGR2GRAY)
    cv.imshow("Camera",gray)

    if cv.waitKey(20) & 0xFF==ord('d'):
            break

cam.release()
cv.destroyAllWindows()    
