import cv2
import numpy as np

# 1) Load image
img = cv2.imread("/home/labuser/Downloads/CV_OPENCV/Object_Detection/hemant_photo.jpg")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2) HOG + SVM detector (dlib)
import dlib
detector = dlib.get_frontal_face_detector()  # HOG + linear SVM (pretrained)

# 3) Detect faces (upsample=1 helps detect smaller faces)
dets = detector(rgb,1)

# 4) Draw bounding boxes and count
for d in dets:
    x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

count_faces = len(dets)
print("Number of faces detected:", count_faces)

cv2.imshow("HOG+SVM Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()