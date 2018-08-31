import argparse
import sys
import cv2
import time
import json
import pickle
import requests
import myImageLibrary
import base64
import numpy as np
import os

#FACE_CASCADE = cv2.CascadeClassifier("C:/Users/pisa/AppData/Local/Programs/Python/Python36-32/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
FACE_CASCADE = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def callFaceService(img_base64,url,key=None):

        #call the webservice here
        headers = {'Content-Type': 'application/json'}
        if key is not None and key is not []:
        #key is only needed when deploying to AKS cluster. Local service does not require key. Default = None.
            headers['Authorization'] = 'Bearer ' + key
        # data = '{"input_bytes": [{"image base64 string": "' + str(img_base64) + '"}]}'
        data = '{"input_bytes": "'+str(img_base64,'utf-8')+'"}'

        res = requests.post(url,headers=headers,data=data)
        try:
            resDict = json.loads(res.json())
            return resDict['index']
        except:
            print("ERROR: webservice returned message " + res.text)
            return 'error'  
        

def main(args):

    if os.path.exists("number_to_text_label.json"):
        with open("number_to_text_label.json","r") as file:
                index_to_name = json.loads(file.read())
    else:
        print("[WARNING] number_to_text_label.json not found.")
        index_to_name = None

    parser = argparse.ArgumentParser(description='Call the face recognition service')
    parser.add_argument('-u', '--url', default='http://localhost/score', type=str, help='Service URL')
    parser.add_argument('-p', '--path', default='test_image.jpg', type=str, help='Path to test image')
    parser.add_argument('-k','--key',type=str,help='service key',default=None)
    #key is only needed when deploying to AKS cluster. Local service does not require key. Default = None.
    args = parser.parse_args()
    path = args.path
    key = args.key
    url = args.url
    img_raw = cv2.imread(path)
    faces = myImageLibrary.extract_face(img_raw,FACE_CASCADE)
    if len(faces) == 0:
        print("No face found. Try a different image")
    else:
        face = myImageLibrary.resize_crop(faces[0],96)
        face = np.around(face/255.0, decimals = 12)    
        img_base64 = myImageLibrary.preprocess(face)
        person = callFaceService(img_base64=img_base64,url=url,key=key)
        if index_to_name is None:
            print(person) #printing index number
        else:
            print(index_to_name[person]) #printing name

if __name__ == '__main__':
    from sys import argv
    try:
        main(argv)
    except KeyboardInterrupt:
        pass
    sys.exit()