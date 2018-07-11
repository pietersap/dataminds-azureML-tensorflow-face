
import cv2 
import pickle
from keras.models import load_model
import numpy as np
import myImageLibrary
#import matplotlib.pyplot as plt
import os
import base64
import time


#QUICK LOCAL TESTING OUTSIDE WORKBENCH ENVIRONMENT

FACE_CASCADE = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")



print("loading model...")
model = load_model("my_model.h5")
print("model loaded")


def direct_run(image,model):
    prediction = model.predict(x=image)
    print(prediction)
    return prediction

def run(input_bytes,model):

    time_start = time.time()
    print("Decoding and loading to numpy...")  
    input_bytes = base64.b64decode(input_bytes) 
    img = np.loads(input_bytes)
    print("predicting...")
    prediction = model.predict(x=img)
    print(prediction)
    total_time = time.time() - start_time
    print("prediction took {0} seconds".format(total_time))
    return str(prediction.tolist()), total_time

print("reading images...")
images = myImageLibrary.get_images(os.path.join("images","test_images"))
faces = [np.around(myImageLibrary.resize_crop(image,96).transpose(2,0,1)/255.0,decimals=12) for image in myImageLibrary.extract_faces_bulk(images,FACE_CASCADE)]

for face in faces:
    # try:
        start_time = time.time()
        print("preprocessing image")
        img_base64 = myImageLibrary.preprocess(face,transpose=False)
        total_time = time.time() - start_time
        print("preprocessing took {0} seconds".format(total_time))        
        r = run(img_base64,model)
        # r = direct_run(np.expand_dims(face,axis=0),model)
        # print(r)
    # except:
    #     print("this one failed")


