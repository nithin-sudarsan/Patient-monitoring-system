import os
import shutil
import cv2
import numpy as np


img = cv2.imread('D:\\Pictures\\Camera Roll\\New2.jpg')
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

# Detect faces in the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.25, 6)

# Loop through all detected faces and crop them
for (x, y, w, h) in faces:
    # Draw a rectangle around the face
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)

    # Crop the face region from the image
    face_crop = img[y:y+h, x:x+w]

    # Display the cropped face
    cv2.imwrite('New_cropped.png', face_crop)

crp = cv2.imread('New_cropped.png')
hsv = cv2.cvtColor(crp, cv2.COLOR_BGR2HSV)
cv2.imwrite('New_cropped_hsv.png', hsv)
exec(open("UI.py").read())
