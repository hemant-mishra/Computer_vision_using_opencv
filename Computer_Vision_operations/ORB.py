import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("/home/labuser/Downloads/GuidedProject1602257891588/house.PNG")
if img is None:
    raise FileNotFoundError("Image not found. Check the path!")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

orb = cv.ORB_create(nfeatures=1000)  # you can tune this
kp = orb.detect(gray, None)
kp, des = orb.compute(gray, kp)

print("Keypoints:", len(kp))
print("Descriptors shape:", des.shape if des is not None else None)

img_kp = cv.drawKeypoints(gray, kp, None, color=(0,255,0),
                          flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(10,6))
plt.imshow(img_kp, cmap="gray")
plt.title("ORB Keypoints")
plt.axis("off")
plt.show()