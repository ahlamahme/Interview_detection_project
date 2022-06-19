import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from multiprocessing import Lock, Process
import keyboard
import os
import threading
from threading import RLock
os.environ['KMP_DUPLICATE_LIB_OK']='True'

end_session=True

class FER():
    def __init__(self,cap):
        self.emotion_labels = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
        self.face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.total_predictions = []
        self.cap=cap
        self.lock = RLock()

    def load_model(self):
        json_file = open('Saved-Models-facial/model8258.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        self.loaded_model.load_weights('Saved-Models-facial/model8258.h5')

    def predict_emotion(self):
        #lock.acquire()
        #with self.lock:
        frame= self.cap.getNextFrame()
        #lock.release()  
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)
                predictions = self.loaded_model.predict(roi)[0]
                pred_list = list(predictions)
                self.total_predictions.append(pred_list)
                #pred_np = np.squeeze(np.array(pred_list).tolist(), axis=1) # Get rid of 'array' & 'dtype' statments.

    def stop_facial(self):
        global end_session
        end_session=False


    def reset(self):
        global end_session
        end_session=True

    def analyse_face(self,q,uid):
        self.reset()
        while True:
             self.predict_emotion();
             if (end_session==False):
                 break;
             if keyboard.is_pressed ("esc")  :
                 break
        print("break works")        
        total_predictions_np =  np.mean(np.array(self.total_predictions).tolist(), axis=0)
        print("hh")
        fig = plt.figure()
        plt.bar(self.emotion_labels, total_predictions_np, color = 'blue')
        plt.ylabel("Mean probabilty (%)")
        plt.title("Face Emotions recoqnized for q"+q)
        print("before save")
        fig.savefig(uid+'/Session Summary_face'+q+'.png')

if __name__== "__main__":
    f1 = FER(cv2.VideoCapture(0))
    f1.load_model()
    f1.analyse_face("4","0")


    
        