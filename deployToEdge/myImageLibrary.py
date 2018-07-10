import numpy as np
import pydot
from IPython.display import SVG
import scipy.misc
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import cv2
import glob
import os
import math
import pickle
import base64


def showimage(title,img):
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def extract_face(img,FACE_CASCADE):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    rectangles = FACE_CASCADE.detectMultiScale(img_gray, 1.3, 5)
    faces = []
    for (x,y,w,h) in rectangles:
        cropped_face = img[y:y+h,x:x+w]
        faces.append(cropped_face)
    return faces

def extract_faces_bulk(img_list,FACE_CASCADE):
    return_list = []
    for img in img_list:
        try:
            faces = extract_face(img,FACE_CASCADE)
            if len(faces)>0:
                return_list.append(faces[0])
        except:
            pass
    return return_list

def crop_center(image,cropx,cropy):
    w = image.shape[1]
    h = image.shape[0]
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)
    return image[starty:starty+cropy, startx:startx+cropx]

def resize_crop(image,square_size):
    w = image.shape[1]
    h = image.shape[0]
    min_dim = min(w,h)  
    max_square_image = crop_center(image, min_dim, min_dim)
    result = cv2.resize(max_square_image,(square_size,square_size),0,0)
    return result



def get_images(parent_directory):
    results = glob.glob(parent_directory+'/*.jpg')+glob.glob(parent_directory+'/*.jpeg')+glob.glob(parent_directory+'/*.png')
    images = []
    for result in results:
        images.append(cv2.imread(result))
    return images

def preprocess(img,transpose=True):
    #takes an image (face) of shape (x,x,3), transforms it to shape (1,3,x,x), pickles it and encodes to base64.
    if transpose:
        img = img.transpose(2,0,1)
    img = np.expand_dims(img,axis=0)
    img_pkl = pickle.dumps(img)
    img_base64 = base64.b64encode(img_pkl)
    return img_base64







