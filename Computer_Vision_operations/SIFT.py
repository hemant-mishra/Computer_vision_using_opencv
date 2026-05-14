import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img1=cv.imread("/home/labuser/Downloads/GuidedProject1602257891588/taj.jpg")
img=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)

sift=cv.SIFT_create()

keypoints, descryptor=sift.detectAndCompute(img,None)
print(descryptor)
img_with_keypoints=cv.drawKeypoints(img1,keypoints,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

img_with_kp=cv.cvtColor(img_with_keypoints,cv.COLOR_BGR2RGB)
plt.imshow(img_with_kp)
plt.show()