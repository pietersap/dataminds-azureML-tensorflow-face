import sys
import os
import argparse
from azureml.logging import get_azureml_logger

import tensorflow as tf
import cv2
import numpy as np
import glob
import json

import myImageLibrary

import keras
from keras.models import load_model
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, Lambda, Dense
from keras.models import Model
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# initialize the logger
logger = get_azureml_logger()
# add experiment arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', help='Number of epochs', default=5,type=int)
args = parser.parse_args()

SQUARE_SIZE = 96 # size to which input images will be reshaped
EPOCHS = args.epochs
print("EPOCHS: {0}".format(EPOCHS))
# This is how you log scalar metrics
# logger.log("MyMetric", value)
# Create the outputs folder - save any outputs you want managed by AzureML here
#os.makedirs('./outputs', exist_ok=True)

SHARED_FOLDER = os.environ["AZUREML_NATIVE_SHARE_DIRECTORY"]
print("SHARED FOLDER ",SHARED_FOLDER)


#subfolder = "azureml-dataminds-share" #pisa-dsvm (linux dsvm)
subfolder = 'dataminds' # local

MODEL_PATH =  os.path.join(SHARED_FOLDER,subfolder,'facenet_nn4_small2_v7.h5') #path to facenet keras model
PRETRAINED_WEIGHTS_PATH = os.path.join(SHARED_FOLDER,subfolder,'pretrained_weights.h5')
FACE_CASCADE = cv2.CascadeClassifier(os.path.join(SHARED_FOLDER,subfolder,'haarcascade_frontalface_default.xml'))

print("model path: ",MODEL_PATH)
print("weights path: ",PRETRAINED_WEIGHTS_PATH)


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def face_list_to_array(face_list):
    shape = list(face_list[0].shape)
    shape[:0] = [len(face_list)]
    faces_array = np.concatenate(face_list).reshape(shape)
    return faces_array

# (1) READING AND PREPARING THE DATA

#find all image folders in /SHARED_FOLDER/subfolder/images/
image_folder = os.path.join(SHARED_FOLDER,subfolder,'images')
list_dir = os.listdir(image_folder)
folders = []
for name in list_dir: #do not include files in folder list
    if os.path.isdir(os.path.join(os.path.abspath(image_folder), name)):
        folders.append(name)

print("{0} names: ".format(len(folders)),folders)


number_to_text_label = {}
face_list = []
number_labels = []
print("Reading and processing images...")
for i, folder in enumerate(folders):
    print("{0}..".format(folder))   
    image_list = myImageLibrary.get_images(os.path.join(image_folder,folder))
    processed_image_list =  [np.around(myImageLibrary.resize_crop(image,SQUARE_SIZE).transpose(2,0,1)/255.0,decimals=12) for image in myImageLibrary.extract_faces_bulk(image_list,FACE_CASCADE)]
    # processed = face extracted, normalized, resized and cropped, transposed to "channels first"
    number_labels = number_labels+([i]*len(processed_image_list))
    face_list = face_list+processed_image_list
    number_to_text_label[str(i)] = folder

face_array = face_list_to_array(face_list)
label_array = np.array(number_labels)
output_onehot = convert_to_one_hot(label_array,len(folders)).T 

X_train, X_test, Y_train, Y_test = train_test_split(face_array,output_onehot)

print("Loading facenet model...(ignore warning, we compile it later)")
facemodel = load_model(MODEL_PATH)
# this will also log a warning: no training configuration found in file. This file only contains the model, no weights and no training config. Weights are loaded
# in the line below, training config will be set after the extra layer is added to the facenet network (see the SoftmaxModel() function; model.compile(..))
facemodel.load_weights(PRETRAINED_WEIGHTS_PATH)

def SoftmaxModel(facemodel,classes=5,input_shape=(3,96,96)):
     
    X_input = Input(input_shape)
    encoding = facemodel(X_input)
    X = Activation('relu')(encoding)
    X = Dense(classes,activation='softmax')(X)    
    model = Model(inputs=X_input,outputs=X)
    model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
    return model

print("Building classification model...")
softmaxmodel = SoftmaxModel(facemodel,classes=len(folders),input_shape=(3,SQUARE_SIZE,SQUARE_SIZE))

print("Fitting model...")
softmaxmodel.fit(x=X_train,y=Y_train,epochs=EPOCHS)

#saving to a single hdf5 file. The file will contain
#the architecture, weights, training config (loss, optimizer) and the state
#of the optimizer, allowing to resume training where you left off. 
print("Saving model to outputs folder")
softmaxmodel.save(os.path.join("outputs","my_model.h5"))
with open(os.path.join("outputs","number_to_text_label.json"),"w") as jsonfile:
    jsonfile.write(json.dumps(number_to_text_label))
print("To enable name resolution when testing, please retrieve number_to_text_label.json and save it in project directory before deploying with CLI.")

print("Saving model to shared folder")
softmaxmodel.save(os.path.join(SHARED_FOLDER,"my_model.h5"))
with open(os.path.join(SHARED_FOLDER,"number_to_text_label.json"),"w") as jsonfile:
    jsonfile.write(json.dumps(number_to_text_label))

print("Done")

print("Evaluating fitted model")
#softmaxmodel = load_model(os.path.join(SHARED_FOLDER,"my_model.h5"))
accuracy = softmaxmodel.evaluate(x=X_test,y=Y_test)[1]
print(accuracy)
logger.log("accuracy",accuracy)
pred = softmaxmodel.predict(X_test)
cm = confusion_matrix(np.argmax(Y_test,axis=1),np.argmax(pred,axis=1))
print(cm)
