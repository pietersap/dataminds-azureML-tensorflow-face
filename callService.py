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

#FACE_CASCADE = cv2.CascadeClassifier("C:/Users/pisa/AppData/Local/Programs/Python/Python36-32/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
FACE_CASCADE = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def main(args):
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

        #call the webservice here (see also the removeThis sample project in workbench)
        headers = {'Content-Type': 'application/json'}
        if key is not None and key is not []:
        #key is only needed when deploying to AKS cluster. Local service does not require key. Default = None.
            headers['Authorization'] = 'Bearer ' + key
        # data = '{"input_bytes": [{"image base64 string": "' + str(img_base64) + '"}]}'
        data = '{"input_bytes": "'+str(img_base64,'utf-8')+'"}'

        startTime = time.time()
        res = requests.post(url,headers=headers,data=data)
        try:
            print("RESULT: ",type(res))
            resDict = json.loads(res.json())
            # apiDuration   = int(float(resDict['executionTimeMs']))
            # localDuration = int(float(1000.0*(time.time() - startTime)))
            print(resDict)
        except:
            print("ERROR: webservice returned message " + res.text)
    

def call_face_service_data(data,url):  
    return None

if __name__ == '__main__':
    from sys import argv
    try:
        main(argv)
    except KeyboardInterrupt:
        pass
    sys.exit()