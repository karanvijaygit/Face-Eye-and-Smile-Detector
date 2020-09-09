# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 17:21:14 2020

@author: VIJAYK1
"""
# Smile_Detection

import cv2
import os
os.chdir('C:/Users/vijayk1/Desktop/CV_az/smile_det')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
def detect(gray,frame): # Frame - Actual color image, gray - grayscale of the image
    faces = face_cascade.detectMultiScale(gray,1.3,5) # gray is the grayscale image, 1.3 is the scale factor for image shrinking and 5 is number of neighbours
    # faces contain the boxes of x,y,w,h parameters. x,y is coordinate of top left corner, w is width and h is height
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        # x,y is coordinnate of top left corner of rectangle and x+w,y+h is coordinate of bottom right corner
        # 255,0,0 is color of rectangle box and 2 is thickness of box
        # Once we are inside the face level, we only need to search for regions which is in region of interest
        roi_gray = gray[y:y+h,x:x+w] 
        roi_color = frame[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray,1.1,22)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        smile = smile_cascade.detectMultiScale(roi_gray,1.7,44)
        # Number of neighbours can be increased to have more refinement in detection
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
    return frame

# Face Recognition with Webcam
video_capture = cv2.VideoCapture(0)
# Video capture - 0 means webcam internal, 1 means external webcam
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas = detect(gray,frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()



