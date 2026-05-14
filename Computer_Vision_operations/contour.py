import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img=cv.imread("/home/labuser/Downloads/GuidedProject1602257891588/balloons.jpg")
img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
thresh,baloon_thresh=cv.threshold(img,100,255,cv.THRESH_BINARY)
baloon_contour,contour_hirarchy=cv.findContours(baloon_thresh, cv.RETR_TREE ,cv.CHAIN_APPROX_SIMPLE)

canvas=255*np.ones((baloon_thresh.shape[0],baloon_thresh.shape[1],3),dtype=np.uint8)

for i in range(len(baloon_contour)):
    color=(np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256))
    cv.drawContours(canvas,baloon_contour,i,color,2,cv.LINE_8)
plt.imshow(canvas)
plt.show()

canvas1=255*np.ones((baloon_thresh.shape[0],baloon_thresh.shape[1],3),dtype=np.uint8)
for i in range(len(baloon_contour)):
    color=(np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256))
    baloon_hull=[cv.convexHull(baloon_contour[i])]
    cv.drawContours(canvas1,baloon_hull,0,color,cv.LINE_8)
plt.imshow(canvas1)
plt.show()

canvas2=255*np.ones((baloon_thresh.shape[0],baloon_thresh.shape[1],3),dtype=np.uint8)
baloon_rect=cv.resize(baloon_thresh,(400,400))
for i in range(len(baloon_contour)):
    color=(np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256))
    x,y,w,h=cv.boundingRect(baloon_contour[i])
    cv.rectangle(baloon_rect,(x,y),(x+w,y+h),color,2)
plt.imshow(baloon_rect)
plt.show()

baloon_circle=cv.resize(baloon_thresh,(400,400))
for i in range(len(baloon_contour)):
    color=(np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256))
    (x,y),radius=cv.minEnclosingCircle(baloon_contour[i])
    center=(int(x),int(y))
    radius=int(radius)
    cv.circle(baloon_circle,center,radius,color,2)
plt.imshow(baloon_circle)
plt.show()

baloon_rot_rect=cv.resize(baloon_thresh,(400,400))
for i in range(len(baloon_contour)):
    color=(np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256))
    rot_rect=cv.minAreaRect(baloon_contour[i])
    box=cv.boxPoints(rot_rect)
    box=np.intp(box)
    cv.drawContours(baloon_rot_rect,[box],0,color,2)
plt.imshow(baloon_rot_rect)
plt.show()

baloon_ellips=cv.resize(baloon_thresh,(400,400))
for i in range(len(baloon_contour)):
    if(baloon_contour[i].shape[0]>=5):
        color=(np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256))
        ellips=cv.fitEllipse(baloon_contour[i])
        cv.ellipse(baloon_ellips,ellips,color,2)
plt.imshow(baloon_ellips)
plt.show()