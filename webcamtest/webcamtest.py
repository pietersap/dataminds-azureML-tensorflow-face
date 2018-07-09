
import myImageLibrary
import cv2
import os
import time
from keras.models import load_model
import numpy as np
import base64

CASCADE_FOLDER = "C:/Users/pisa/AppData/Local/Programs/Python/Python35/Lib/site-packages/cv2/data"
FACE_CASCADE = cv2.CascadeClassifier(os.path.join(CASCADE_FOLDER,'haarcascade_frontalface_default.xml'))

def get_model():
    return load_model("my_model.h5")

index_to_name = {
    "0":"buscemi",
    "1":"clooney",
    "2":"dicaprio",
    "3":"jennifer",
    "4":"pieter",
    "5":"unknown"
}

# global model
# model = get_model()

# this method for quick testing differs from run() in score.py
# in that it takes a second parameter, the model. In score.py, this is a global
# variable. 

def run(input_bytes,model):

    input_bytes = base64.b64decode(input_bytes)   
    img = np.loads(input_bytes)
    prediction = model.predict(x=img)
    index = np.argmax(prediction)
    return prediction

def monitor_faces(model):
    cam = cv2.VideoCapture(0)
    
    while True:
        time.sleep(500/1000)
        ret_val, img = cam.read()
        # cv2.imwrite("current_frame.jpg",img)
        # img = cv2.imread("current_frame.jpg")
        cv2.imshow('images',img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
        faces = myImageLibrary.extract_face(img,FACE_CASCADE)
        if len(faces) == 0:
            print("nobody detected")
        else:
            people = []
            for face in faces:
                face = np.around(myImageLibrary.resize_crop(face,96)/255.0,decimals=12)
                face_base64 = myImageLibrary.preprocess(face)
                index = run(face_base64,model)
                people.append(index)
                # people.append(index_to_name[str(index)])
            print("Detected: ",people)
    cv2.destroyAllWindows()
    cam.release()

model = load_model("my_model.h5")
monitor_faces(model)