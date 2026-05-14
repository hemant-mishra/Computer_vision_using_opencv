import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img=cv.imread("/home/labuser/Downloads/GuidedProject1602257891588/house.PNG")
img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
sobelx=cv.Sobel(img,cv.CV_64F,1,0,ksize=3)
sobely=cv.Sobel(img,cv.CV_64F,0,1,ksize=3)

plt.imshow(img)
plt.show()

plt.imshow(sobelx)
plt.show()

plt.imshow(sobely)
plt.show()

can=cv.Canny(img,100,200)
plt.imshow(can)
plt.show()
