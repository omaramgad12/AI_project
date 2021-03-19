
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 01:00:02 2020

@author: meshm
"""

import cv2
import os

#Preparing Realtime Video Camera Window
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

#Using 'haarcascade_frontalface_default.xml' 
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Entering The Person ID Starting From Number 1 ex: 1 = Mohamed , 2 = Yehya ,, etc
face_id = input('\nEnter User id ==> ')

print("\n [INFO] Initializing face capture. Look at the camera and wait ...")
# Initialize individual sampling face count
count = 0

#start detect your face and take 100 pictures
while(True):

    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder with the id and pic number
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)

    #You Can Press 'ESC' for exiting video
    k = cv2.waitKey(100) & 0xff 
    if k == 27:
        break
    elif count >= 100: #The Number of Pictures That the application takes for one user
         break

print("\n Exiting Program >> Collecting Dataset is done now we will train it")
cam.release()
cv2.destroyAllWindows()


