import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread("/home/labuser/Downloads/GuidedProject1602257891588/house.PNG")
img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

imgGB=cv.GaussianBlur(img,(3,3),0)

circles=cv.HoughCircles(imgGB,cv.HOUGH_GRADIENT,dp=1.2,minDist=30,param1=120,param2=20,minRadius=5,maxRadius=20)
output=img.copy()
print(circles)

if circles is not None:
    circles=np.uint8(np.around(circles))
    print(circles)
    for (x,y, r) in circles[0,:]:
         cv.circle(output,(x,y),r,(0,255,0),2)
         cv.circle(output,(x,y),2,(0,0,255),3)
else:
     print("no circle detected")

plt.subplot(131)
plt.imshow(img)
plt.axis("off")
plt.subplot(132)
plt.imshow(imgGB)
plt.axis("off")
plt.subplot(133)
plt.imshow(output)
plt.axis("off")
plt.show()