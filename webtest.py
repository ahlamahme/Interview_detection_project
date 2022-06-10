from webcam import *
import cv2
def tt(cap):
    f=cap.getNextFrame()
    cv2.imwrite("frame.jpg", f)
if __name__== "__main__":
    cap = VideoCamera(0)
    tt(cap)
    tt(cap)
    