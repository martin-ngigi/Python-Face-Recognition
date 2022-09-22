import re
import numpy as np
import cv2
import json
import os

if __name__ == "__main__":
    #Path here the cropped images will be stored
    path = "cropped/martin"
    count = 1
    #Load haar cascade classifier to crop faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
    #load video
    cap = cv2.VideoCapture('martinVideo.mp4')
    #loop through video
    while(cap.isOpened()):
        #read faces
        ret, frame = cap.read()
        count = count +1
        #After every 5 frames, save the face [can be chnaged]
        if (count %5==0):
            print(count)
            #If frame is not null
            if ret == True:
                #Convert image from BGR to GRAY
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #pass it throough face cascade classifier
                faces = face_cascade.detectMultiScale(gray, 1.3, 4)
                #Loop throuhg each face in frame
                for(x,y,w,h) in faces:
                    #crop face from image
                    roi_color = frame[y:y+h, x:x+w]
                    #save face in jpg formart
                    cv2.imwrite(path+"/"+str(count)+".jpg", roi_color)
            #if frame is null
            else:
                #release loaded video
                cap.release()