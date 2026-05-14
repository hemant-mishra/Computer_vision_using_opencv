import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("/home/labuser/Downloads/CV_OPENCV/Object_Detection/hemant_photo.jpg")
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

face_cascade = cv.CascadeClassifier("/home/labuser/Downloads/CV_OPENCV/Object_Detection/haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(img_gray)

img1 = img.copy()   # ✅ FIXED HERE

for (x, y, w, h) in faces:
    cv.rectangle(img1, (x, y), (x+w, y+h), (255, 0, 0), 8)

# detect_eye
eye_cascade = cv.CascadeClassifier("/home/labuser/Downloads/CV_OPENCV/Object_Detection/haarcascade_eye.xml")
eyes = eye_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)  # ✅ grayscale

for (x, y, w, h) in eyes:
    cv.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 4)

plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))  # ✅ proper display
plt.axis("off")
plt.show()

# detect_smile
smile_cascade = cv.CascadeClassifier("/home/labuser/Downloads/CV_OPENCV/Object_Detection/haarcascade_smile.xml")
smile = smile_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)  # ✅ grayscale

for (x, y, w, h) in smile:
    cv.rectangle(img1, (x, y), (x+w, y+h), (255, 255, 255), 6)

plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))  # ✅ proper display
plt.axis("off")
plt.show()
