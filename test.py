import cv2
import time
import pymsgbox
from FER import lock

slouch = False
count = 0
session_end = False
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

time.sleep(0.1)
t_last = time.time()
class Posture():
    def __init__(self,cap):
        self.video_capture = cap
        self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        

    def startPost(self,uid):
        global session_end
        print("enteringStartPost")
        while not(session_end):
           
            frame=self.video_capture.getNextFrame()[0]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # use appropriate flag based on version of OpenCV
            if int(cv2.__version__.split('.')[0]) >= 3:
                cv_flag = cv2.CASCADE_SCALE_IMAGE
            else:
                cv_flag = cv2.cv.CV_HAAR_SCALE_IMAGE

            faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv_flag
            )
            # for each face, draw a green rectangle around it and append to the image
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                if (y+h) > 400:
                    print('slouch')
                    global slouch
                    slouch = True
                else:
                    slouch = False
                    print("No slouching detected")
            global t_last
            if slouch == True and time.time() - t_last > 10:
                global count
                cv2.imwrite(uid+"/frame%d.jpg" % slouch, frame)
                count  +=  1
                #pymsgbox.alert('Stop Slouching!', 'PostureFix')
                t_last = time.time()

            time.sleep(0.5)
        print("outofpostwhile")    

    def stopPost(self):
        global session_end
        session_end = True
        self.video_capture.close()


if __name__ == '__main__':
    s= Posture(cv2.VideoCapture(0))
    s.startPost("0")
