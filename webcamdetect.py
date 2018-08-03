# reading webcam input and recognizing faces using the service.

import myImageLibrary
import callService

import time
import cv2
import base64
import argparse
import sys

FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
SQUARE_SIZE = 96

def main(args):
    parser = argparse.ArgumentParser(description='Call the face recognition service')
    parser.add_argument('-u', '--url', default='http://localhost:32773/score', type=str, help='Service URL')
    parser.add_argument('-k','--key',type=str,help='service key',default=None)
    #key is only needed when deploying to AKS cluster. Local service does not require key. Default = None.
    args = parser.parse_args()
    key = args.key
    url = args.url

    monitor_faces(url,key=key,plot=True)

def monitor_faces(url,key=None,plot=True):
    cam = cv2.VideoCapture(0)    
    while True:
        time.sleep(500/1000)
        ret_val, img = cam.read()
        faces, rectangles = myImageLibrary.extract_face(img,FACE_CASCADE, return_rectangles = True)
        if plot:
            for (x,y,w,h) in rectangles:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.imshow('Hit escape to quit.',img)
            if cv2.waitKey(1) == 27: 
                break  # esc to quit
        if len(faces) == 0:
            print("No faces detected")
        else:
            people = []
            for face in faces:
                face = myImageLibrary.resize_crop(face,SQUARE_SIZE)
                face_base64 = myImageLibrary.preprocess(face,normalize=True)
                person = callService.callFaceService(face_base64,url=url,key=key)
                people.append(person)
            print("Detected: ",people)
    if plot:
        cv2.destroyAllWindows()
    cam.release()

if __name__ == '__main__':
    from sys import argv
    try:
        main(argv)
    except KeyboardInterrupt:
        print("Keyboardinterrupt")
    sys.exit()