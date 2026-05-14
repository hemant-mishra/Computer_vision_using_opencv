import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# EROSION
img1=cv.imread("/home/labuser/Downloads/GuidedProject1602257891588/taj.jpg")
img=cv.cvtColor(img1,cv.COLOR_BGR2RGB)
# rows,cols,ch=img.shape
kernal=np.ones((10,10),np.uint8)
erosion=cv.erode(img,kernal,iterations=1)
plt.imshow(erosion)
plt.show()
# print(kernal)

#Dilation
kernal=np.ones((5,5),np.uint8)
dilation=cv.dilate(img,kernal,iterations=1)
plt.imshow(dilation)
plt.show()
print(kernal)

#opening
kernal=np.ones((5,5),np.uint8)
open=cv.morphologyEx(img,cv.MORPH_OPEN,kernel=kernal)
plt.imshow(open)
plt.show()

#close
close=cv.morphologyEx(img,cv.MORPH_CLOSE,kernel=kernal)
plt.imshow(close)
plt.show()

# Gradient
gradient=cv.morphologyEx(img,cv.MORPH_GRADIENT,kernel=kernal)
plt.imshow(gradient)
plt.show()

# Morph_Hat
morph=cv.morphologyEx(img,cv.MORPH_TOPHAT,kernel=kernal)
plt.imshow(morph)
plt.show()

# morph_hat
hat=cv.morphologyEx(img,cv.MORPH_BLACKHAT,kernel=kernal)
plt.imshow(morph)
plt.show()