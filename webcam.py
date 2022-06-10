import cv2
import threading

class VideoCamera(object):
  # filename can be 0 to access the webcam
  def __init__(self, filename):
    self.lock = threading.Lock()
    self.openVideo(filename)

  def openVideo(self, filename):
    self.lock.acquire()
    self.videoCap = cv2.VideoCapture(filename)
    self.lock.release()

  def getNextFrame(self):
        
      self.lock.acquire()
      img = None
      # if no video opened return None
      if self.videoCap.isOpened():
        ret, img = self.videoCap.read()
      self.lock.release()
      return img,ret

  def close(self):
    self.videoCap.release()