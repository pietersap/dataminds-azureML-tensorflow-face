import sys
import os
import argparse
from azureml.logging import get_azureml_logger

import tensorflow as tf
import cv2
import numpy as np
import glob

import myImageLibrary

import keras
from keras.models import load_model
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, Lambda, Dense
from keras.models import Model

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

unknown = True
EPOCHS = 5

# initialize the logger
logger = get_azureml_logger()
# add experiment arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', help='Number of epochs', default=5,type=int)
args = parser.parse_args()

EPOCHS = args.epochs
print("EPOCHS: {0}".format(EPOCHS))
# This is how you log scalar metrics
# logger.log("MyMetric", value)
# Create the outputs folder - save any outputs you want managed by AzureML here
#os.makedirs('./outputs', exist_ok=True)

SHARED_FOLDER = os.environ["AZUREML_NATIVE_SHARE_DIRECTORY"]
#SHARED FOLDER: C:\Users\gbaeke\.azureml\share\MLexperimentation-pisa\PieterWorkspace\dataminds-session

MODEL_PATH =  os.path.join(SHARED_FOLDER,'dataminds','facenet_nn4_small2_v7.h5') #path to facenet keras model
PRETRAINED_WEIGHTS_PATH = os.path.join(SHARED_FOLDER,'dataminds','pretrained_weights.h5')
FACE_CASCADE = cv2.CascadeClassifier(os.path.join(SHARED_FOLDER,'dataminds','haarcascade_frontalface_default.xml'))

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def face_list_array(face_list):
    shape = list(face_list[0].shape)
    shape[:0] = [len(face_list)]
    faces_array = np.concatenate(face_list).reshape(shape)
    return faces_array

# (1) READING AND PREPARING THE DATA
print("Reading images...")
images_buscemi = myImageLibrary.get_images(os.path.join(SHARED_FOLDER,'dataminds','images','buscemi'))
images_jennifer = myImageLibrary.get_images(os.path.join(SHARED_FOLDER,'dataminds','images','jennifer'))
images_dicaprio = myImageLibrary.get_images(os.path.join(SHARED_FOLDER,'dataminds','images','dicaprio'))
images_clooney = myImageLibrary.get_images(os.path.join(SHARED_FOLDER,'dataminds','images','clooney'))
if unknown:
    images_unknown = myImageLibrary.get_images(os.path.join(SHARED_FOLDER,'dataminds','images','unknown'))
images_pieter = myImageLibrary.get_images(os.path.join(SHARED_FOLDER,'dataminds','images','pieter'))
images_pieter = images_pieter + myImageLibrary.get_images(os.path.join(SHARED_FOLDER,'dataminds','images','pieter2'))
images_pieter3 = myImageLibrary.get_images(os.path.join(SHARED_FOLDER,'dataminds','images','pieter3'))

print("Processing images...")
faces_buscemi = [np.around(myImageLibrary.resize_crop(image,96).transpose(2,0,1)/255.0,decimals=12) for image in myImageLibrary.extract_faces_bulk(images_buscemi,FACE_CASCADE)]
faces_jennifer = [np.around(myImageLibrary.resize_crop(image,96).transpose(2,0,1)/255.0,decimals=12) for image in myImageLibrary.extract_faces_bulk(images_jennifer,FACE_CASCADE)]
faces_dicaprio = [np.around(myImageLibrary.resize_crop(image,96).transpose(2,0,1)/255.0,decimals=12) for image in myImageLibrary.extract_faces_bulk(images_dicaprio,FACE_CASCADE)]
faces_clooney = [np.around(myImageLibrary.resize_crop(image,96).transpose(2,0,1)/255.0,decimals=12) for image in myImageLibrary.extract_faces_bulk(images_clooney,FACE_CASCADE)]
if unknown:
    faces_unknown = [np.around(myImageLibrary.resize_crop(image,96).transpose(2,0,1)/255.0,decimals=12) for image in myImageLibrary.extract_faces_bulk(images_unknown,FACE_CASCADE)]
faces_pieter = [np.around(myImageLibrary.resize_crop(image,96).transpose(2,0,1)/255.0,decimals=12) for image in myImageLibrary.extract_faces_bulk(images_pieter,FACE_CASCADE)]
faces_pieter3 = [np.around(myImageLibrary.resize_crop(image,96).transpose(2,0,1)/255.0,decimals=12) for image in myImageLibrary.extract_faces_bulk(images_pieter3,FACE_CASCADE)]
print(len(faces_buscemi))
print(len(faces_jennifer))
print(len(faces_dicaprio))
print(len(faces_clooney))
if unknown:
    print(len(faces_unknown))
print(len(faces_pieter))

faces_buscemi_array = face_list_array(faces_buscemi)
faces_jennifer_array = face_list_array(faces_jennifer)
faces_dicaprio_array = face_list_array(faces_dicaprio)
faces_clooney_array = face_list_array(faces_clooney)
if unknown:
    faces_unknown_array = face_list_array(faces_unknown)
faces_pieter_array = face_list_array(faces_pieter)
faces_pieter3_array = face_list_array(faces_pieter3)

faces_softmax = np.concatenate((faces_buscemi_array,faces_clooney_array,faces_dicaprio_array,faces_jennifer_array,faces_pieter_array ))
if unknown:
    faces_softmax_all = np.concatenate((faces_softmax, faces_unknown_array))
# array with "_all" in name contain also the unknown faces. In previous tries, these were excluded. 
y_softmax = np.array([0]*faces_buscemi_array.shape[0]+[1]*faces_clooney_array.shape[0]+[2]*faces_dicaprio_array.shape[0]+[3]*faces_jennifer_array.shape[0]+[4]*faces_pieter_array.shape[0])
y_softmax_pieter = np.array([4]*faces_pieter3_array.shape[0])



if unknown:
    y_softmax_all = np.array(list(y_softmax) + [5] * faces_unknown_array.shape[0])
    y_softmax_all_oh = convert_to_one_hot(y_softmax_all,6).T 
    y_softmax_pieter_oh = convert_to_one_hot(y_softmax_pieter,6).T

else:
    y_softmax_oh = convert_to_one_hot(y_softmax,5).T
    y_softmax_pieter_oh = convert_to_one_hot(y_softmax_pieter,5).T



if unknown:
    X_train, X_test, Y_train, Y_test = train_test_split(faces_softmax_all,y_softmax_all_oh)
else:
    X_train, X_test, Y_train, Y_test = train_test_split(faces_softmax,y_softmax_oh)

print("SHAPE: ",Y_test.shape)

print("Loading facenet model...")
facemodel = load_model(MODEL_PATH)
# this will also log a warning: no training configuration found in file. This file only contains the model, no weights and no training config. Weights are loaded
# in the line below, training config will be set after the extra layer is added to the facenet network (see the SoftmaxModel() function; model.compile(..))
facemodel.load_weights(PRETRAINED_WEIGHTS_PATH)


# (2) BUILDING AND FITTING THE MODEL 

# Classification is done by adding a dense layer to the pretrained facenet
# model (an example of transfer learning) with a softmax activation.
# The dense layer has 6 layers for the 6 categories.
#       (0) buscemi (Steve Buscemi)
#       (1) clooney (George Clooney)
#       (2) dicaprio (Leonardo Dicaprio)
#       (3) jennifer (Jennifer Aniston)
#       (4) pieter (myself)
#       (5) unknown
# The weights of facenet itself are also updated during training
# An alternative was to only train the new weights of the extra dense layer,
# (by using facenet.trainable = False) but this performs worse.

def SoftmaxModel(facemodel,classes=5,input_shape=(3,96,96)):
     
    X_input = Input(input_shape)
    encoding = facemodel(X_input)
    X = Activation('relu')(encoding)
    X = Dense(classes,activation='softmax')(X)    
    model = Model(inputs=X_input,outputs=X)
    model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
    return model

print("Building classification model...")
if unknown:
    softmaxmodel = SoftmaxModel(facemodel,classes=6)
else:
    softmaxmodel = SoftmaxModel(facemodel,classes=5)
print("Fitting model...")
softmaxmodel.fit(x=X_train,y=Y_train,epochs=EPOCHS)



#saving to a single hdf5 file. The file will contain
#the architecture, weights, training config (loss, optimizer) and the state
#of the optimizer, allowing to resume training where you left off. 
print("Saving model to outputs folder")
softmaxmodel.save(os.path.join("outputs","my_model.h5"))
print("Saving model to shared folder")
softmaxmodel.save(os.path.join(SHARED_FOLDER,"my_model.h5"))
print("Done")

print("Evaluating fitted model")
#softmaxmodel = load_model(os.path.join(SHARED_FOLDER,"my_model.h5"))
accuracy = softmaxmodel.evaluate(x=X_test,y=Y_test)[1]
print(accuracy)
logger.log("accuracy",accuracy)
pred = softmaxmodel.predict(X_test)
cm = confusion_matrix(np.argmax(Y_test,axis=1),np.argmax(pred,axis=1))
print(cm)

accuracy2 = softmaxmodel.evaluate(x=faces_pieter3_array,y=y_softmax_pieter_oh)[1]
print(accuracy2)
logger.log("accuracy2",accuracy)
pred2 = softmaxmodel.predict(faces_pieter3_array)
cm2 = confusion_matrix(np.argmax(y_softmax_pieter_oh,axis=1),np.argmax(pred2,axis=1))
print(cm2)
print(pred2)