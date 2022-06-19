import cv2
import threading

class VideoCamera(object):
  # filename can be 0 to access the webcam
  def __init__(self, filename):
    self.lock = threading.Lock()
    self.openVideo(filename)
    self.img = None

  def openVideo(self, filename):
    self.lock.acquire()
    self.videoCap = cv2.VideoCapture(filename)
    self.lock.release()

  def getNextFrame(self):
         
      # if no video opened return None
        self.lock.acquire()
        #global img
        if self.videoCap.isOpened():
            while(1):
              ret, self.img = self.videoCap.read()
              if ret :
                    break
        self.lock.release()
        return self.img ; 

  def close(self):
    self.videoCap.release()