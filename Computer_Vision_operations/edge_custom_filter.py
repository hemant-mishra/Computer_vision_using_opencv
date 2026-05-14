import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img=cv.imread("/home/labuser/Downloads/GuidedProject1602257891588/balloons.jpg")
img_resized=cv.resize(img,(300,300))

#bileteral filter
ib=cv.bilateralFilter(img_resized,-1,50,50)
ib_c=cv.cvtColor(ib,cv.COLOR_BGR2RGB)
plt.imshow(ib_c)
plt.show()

#dilation, erosion

im_dilate=cv.dilate(img_resized,(3,3),iterations=1)
im_erode=cv.erode(im_dilate,(3,3),iterations=1)
im_de=cv.cvtColor(im_erode,cv.COLOR_BGR2RGB)
plt.imshow(im_de)
plt.show()

#gaussianblur

img_gb=cv.GaussianBlur(img_resized,(25,25),0)
edges=img_resized-img_gb
sharp_leaves=img_resized+edges

plt.subplot(221)
plt.imshow(cv.cvtColor(img_resized,cv.COLOR_BGR2RGB))
plt.subplot(222)
plt.imshow(cv.cvtColor(img_gb,cv.COLOR_BGR2RGB))
plt.subplot(223)
plt.imshow(cv.cvtColor(edges,cv.COLOR_BGR2RGB))
plt.subplot(224)
plt.imshow(cv.cvtColor(sharp_leaves,cv.COLOR_BGR2RGB))
plt.show()