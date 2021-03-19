# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 01:00:02 2020

@author: meshm
"""    

import numpy as np
from PIL import Image #pillow package
import os
import cv2
# Path for database
path = 'dataset'

#using haarcascade_frontalface_default.xml
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml"); 
recognizer = cv2.face.LBPHFaceRecognizer_create()



# function to get the images and label data
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []


    #seperate the name and id >> and save the faces x,y,w,h in the faceSamples array
    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') 
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml')

# Print the numer of faces trained and end program
print("\n {0} faces trained. now we can recognize the trained faces".format(len(np.unique(ids))))
