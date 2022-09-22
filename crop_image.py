import re
import numpy as np
import cv2
import json
import time
import os

if __name__ == "__main__":
    #path: location where cropped faces will be stored(Can be empty)
    path = "cropped/Martin_Wainaina"
    #list of files in original folder (current location of photos)
    files = os.listdir("original/Martin_Wainaina")
    print(files)

    #Loop through each folder
    count = 1
    #load frontal face haar cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
    for i in range(len(files)):
        #read image
        img = cv2.imread("original/Martin_Wainaina/"+files[i])
        #convert image from BGR to Gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #pass it through classifier
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        #output if faces consist of all faces in an image
        #loop through each face in an image
        for (x,y,w, h) in faces:
            #x, y -> startin point
            #w, h -> width and height of faces
            #roi = region of interest of cropped image
            roi_color = img[y:y+h, x:x+w]
            #write image in jpg fornat
            #count is used to name image files
            cv2.imwrite(path+"/"+str(count)+".jpg", roi_color)
            count = count + 1
