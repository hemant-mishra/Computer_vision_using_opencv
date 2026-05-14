import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread("/home/labuser/Downloads/GuidedProject1602257891588/house.PNG")
img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
plt.imshow(img,cmap="gray")
plt.show()
edges=cv.Canny(img,50,150,apertureSize=3)
plt.imshow(edges)
plt.show()

lines=cv.HoughLinesP(edges,rho=1,theta=np.pi/180,threshold=50,minLineLength=20,maxLineGap=30)
# print(lines)
if lines is not None:
    for line in lines:
        # print(line)
        x1,y1,x2,y2=line[0]
        cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)
plt.imshow(img)
plt.show()
