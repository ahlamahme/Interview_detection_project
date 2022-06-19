from asyncio.windows_events import NULL
from urllib import response
from flask import Flask,render_template,request,jsonify, session
from chat import *
import cv2
from AVrecordeR import *
from SER import *
from test import *
from EYE import *
from FER import *
from volume_meter import *
from finalReport import *


import os

app = Flask(__name__, static_folder='static')
app.secret_key = "27eduCBA09"


id = 0
#session["uid"] = id
parent_dir = "D:/hana/asu/spring2022/gp2/integrated/Interview_detection_project-master/"
@app.route("/")
def login():
    global id,parent_dir
    session["uid"] = id
    id+=1
    directory = str(session["uid"])
    path = os.path.join(parent_dir, directory)
    os.mkdir(path)
    return '''<h1>The session value is: {}</h1>'''.format(session["uid"])

if __name__ == "__main__":
    app.run(debug=True)