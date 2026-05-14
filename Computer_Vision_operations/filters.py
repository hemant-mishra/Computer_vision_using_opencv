import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread("/home/labuser/Downloads/GuidedProject1602257891588/balloon.jpg")
# img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

#mean filter
m=cv.blur(img,(3,3))
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(m)
plt.show()

#gaussain filter

g=cv.GaussianBlur(img,(3,3),0)
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(g)
plt.show()

#medain filter

M=cv.medianBlur(img,9)
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(M)
plt.show()

# DIFFERENT BORDER TYPE
#median for different border type

kernal=np.ones((3,3),np.float32)/9

borders = [
    ("CONSTANT", cv.BORDER_CONSTANT),
    ("REPLICATE", cv.BORDER_REPLICATE),
    ("REFLECT", cv.BORDER_REFLECT),
    ("REFLECT_101", cv.BORDER_REFLECT_101),
    ("WRAP", cv.BORDER_WRAP),
]

for i in range(len(borders)):
    m=cv.filter2D(img,kernel=kernal,borderType=borders[i])
    plt.subplot(2,3,i)
    plt.imshow(m,cmap="gray")
    plt.show()