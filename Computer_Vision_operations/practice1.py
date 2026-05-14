import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


img=cv.imread("/home/labuser/Downloads/GuidedProject1602257891588/taj.jpg",0)
# new=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# plt.imshow(new)
#Transalation
rols, cols=img.shape
M=np.float32([[1,0,100],[0,1,50]])
dst=cv.warpAffine(img,M,(cols,rols))
plt.imshow(dst)
plt.show()
# trans=cv.resize(img,None, fx=2,fy=2, interpolation=cv.INTER_CUBIC)
# plt.imshow(trans)
# plt.show()

#Rotation
R=cv.getRotationMatrix2D((cols/2,rols/2),90,1)
Rotated=cv.warpAffine(img,R,(cols,rols))
plt.imshow(Rotated)
plt.show()

#rotation
img=cv.imread("/home/labuser/Downloads/GuidedProject1602257891588/taj.jpg")
rols, cols, ch=img.shape
pts1=np.float32([[50,50],[200,50],[50,200]])
pts2=np.float32([[10,100],[200,50],[100,250]])
r=cv.getAffineTransform(pts1,pts2)
rtd_img=cv.warpAffine(img,r,(cols,rols))
plt.imshow(rtd_img)
plt.show()


# perspective rotation
img=cv.imread("/home/labuser/Downloads/GuidedProject1602257891588/taj.jpg")
rols,cols,ch=img.shape
pts1=np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2=np.float32([[0,0],[300,0],[0,300],[300,300]])
rperM=cv.getPerspectiveTransform(pts1,pts2)
rper=cv.warpPerspective(img,rperM,(300,300))
rpe=cv.cvtColor(rper,cv.COLOR_BGR2RGB)
plt.imshow(rpe)
plt.show()
