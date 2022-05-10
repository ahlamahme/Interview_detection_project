#!/usr/bin/env python
# coding: utf-8


import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import os
import datetime
import time
import matplotlib.pyplot as plt
#import docx
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from time import sleep





#record:
filename='records/video_demo.avi'
frames_per_seconds=6.0
my_res='720p'

# Standard Video Dimensions Sizes
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}
 # Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

# Set resolution for the video capture
# Function adapted from https://kirr.co/0l6qmh
def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

# grab resolution dimensions and set video capture to it.
def get_dims(cap, res='1080p'):
  width, height = STD_DIMENSIONS["480p"]
  if res in STD_DIMENSIONS:
    width,height = STD_DIMENSIONS[res]
    ## change the current caputre device
    ## to the resulting resolution
    change_res(cap, width, height)
    return width, height    

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']  


def facial_function():
    #counts
    Angry_count=0     
    Disgust_count=0
    Fear_count=0
    Happy_count=0
    Neutral_count=0
    Sad_count=0
    Surprise_count=0
    figure, axis = plt.subplots(2, 2)
    emotion_labels = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Load model from JSON file
    json_file = open('Saved-Models-facial/model8258.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    classifier = model_from_json(loaded_model_json)

    # Load weights and them to model
    classifier.load_weights('Saved-Models-facial/model8258.h5')

    cap = cv2.VideoCapture(0)
    dims=get_dims(cap,res=my_res)
    video_type_cv2=get_video_type(filename)
    out = cv2.VideoWriter(filename, get_video_type(filename), frames_per_seconds, get_dims(cap, my_res))
    neutral_face=[]
    neutral_time=[]
    happy_time=[]
    happy_face=[]
    sad_time=[]
    sad_face=[]
    surprise_time=[]
    surprise_face=[]
    fear_time=[]
    fear_face=[]
    disgust_time=[]
    disgust_face=[]
    angry_time=[]
    angry_face=[]
   
    

    while True:
     _, frame = cap.read()
     datet=str(datetime.datetime.now())
     t = time.localtime()
     current_time = time.strftime("%H:%M:%S", t)
    
     labels = []
     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
     faces = face_classifier.detectMultiScale(gray)

     for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
  
        cv2.putText(frame,datet,(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        out.write(frame)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)
            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            if (prediction.argmax()==0):                    
                    Angry_count=Angry_count+1
                    angry_face.append(Angry_count)
                    angry_time.append(current_time)   
                    plot1 = plt.figure(1)
                    plt.plot(angry_time,angry_face)                    
                    plt.xticks(rotation=45)
                    plt.title('Angry')
                    plt.xlabel('Time')
                    plt.ylabel('count')
                    plot1.savefig('imags/angry_report.png')      
                    

                      
            elif (prediction.argmax()==1):                
                    Disgust_count=Disgust_count+1
                    disgust_face.append(Disgust_count)
                    disgust_time.append(current_time)   
                    plot2 = plt.figure(2)
                    plt.plot(disgust_time,disgust_face)                    
                    plt.xticks(rotation=45)
                    plt.title('Disgust')
                    plt.xlabel('Time')
                    plt.ylabel('count')
                    plot2.savefig('imags/disgust_report.png')
                 
            elif (prediction.argmax()==2):
                    Fear_count=Fear_count+1
                    fear_face.append(Fear_count)
                    fear_time.append(current_time)   
                    plot3 = plt.figure(3)
                    plt.plot(fear_time,fear_face)                    
                    plt.xticks(rotation=45)
                    plt.title('Fear')
                    plt.xlabel('Time')
                    plt.ylabel('count')
                    plot3.savefig('imags/fear_report.png')
              
            elif (prediction.argmax()==3):                    
                    Happy_count=Happy_count+1                
                    happy_face.append(Happy_count)
                    happy_time.append(current_time)   
                    plot4 = plt.figure(4)
                    plt.plot(happy_time,happy_face)                    
                    plt.xticks(rotation=45)
                    plt.title('Happy')
                    plt.xlabel('Time')
                    plt.ylabel('count')
                    plot4.savefig('imags/happy_report.png')
                    
                    
                    
            elif (prediction.argmax()==4):                                       
                    Neutral_count=Neutral_count+1                
                    neutral_face.append(Neutral_count)
                    neutral_time.append(current_time)     
                    plot5 = plt.figure(5)
                    plt.plot(neutral_time,neutral_face)                     
                    plt.xticks(rotation=45)
                    plt.title('Neutral')
                    plt.xlabel('Time')
                    plt.ylabel('count')
                    plot5.savefig('imags/neutral_report.png')
                  
                   
            elif (prediction.argmax()==5):                 
                    Sad_count=Sad_count+1
                    sad_face.append(Sad_count)
                    sad_time.append(current_time)     
                    plot6 = plt.figure(6)
                    plt.plot(sad_time,sad_face)                     
                    plt.xticks(rotation=45)
                    plt.title('Sad')
                    plt.xlabel('Time')
                    plt.ylabel('count')
                    plot6.savefig('imags/sad_report.png')
                    
            elif (prediction.argmax()==6):
                    Surprise_count=Surprise_count+1 
                    surprise_face.append(Surprise_count)
                    surprise_time.append(current_time)     
                    plot7 = plt.figure(7)
                    plt.plot(surprise_time,surprise_face)                     
                    plt.xticks(rotation=45)
                    plt.title('Surprise')
                    plt.xlabel('Time')
                    plt.ylabel('count')
                    plot7.savefig('imags/surprise_report.png')
           
            label_position = (x,y-10)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)   
        

            
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
     cv2.imshow('Emotion Detector',frame)
     
   

     if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    # defining labels
    activities_label =  ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
    activities = [1,2, 3,4,5,6,7]
    slices= [Angry_count,Surprise_count, Happy_count, Neutral_count,Disgust_count,Sad_count, Fear_count]
 
    # color for each label
    colors = ['r', 'y', 'g', 'b','brown','black','orange']
 
    
    fig = plt.figure(8)
    plt.bar(activities, slices, tick_label = activities_label,
        width = 0.8, color = colors)
    plt.title('The faces count!')
    plt.savefig('imags/count_report.png')

    plt.show()
       
facial_function()







