# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 12:41:27 2022

@author: HP
"""
import cv2
import numpy as np
import sys

#cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(r"C:\Users\ramp1\Desktop\ML\haarcascades\haarcascade_frontalface_alt.xml")
kernel_27x27 = np.ones((27,27),np.float32)/729
video_capture = cv2.VideoCapture(0)
print(cv2.COLOR_BGR2GRAY)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=6,
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()