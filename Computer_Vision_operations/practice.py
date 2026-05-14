import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

mandrill_color=cv.imread("/home/labuser/Downloads/GuidedProject1602257891588/taj.jpg",1)#default - IMREAD_COLOR or 1 
mandrill_unchanged=cv.imread("/home/labuser/Downloads/GuidedProject1602257891588/taj.jpg",cv.IMREAD_UNCHANGED) # -1 
mandrill_grayscale=cv.imread("/home/labuser/Downloads/GuidedProject1602257891588/taj.jpg",cv.IMREAD_GRAYSCALE) # 0 
# print(type(mandrill_color))
# print(mandrill_unchanged)
# print(mandrill_color)

resized_img=cv.resize(mandrill_color,(150,150))
print(mandrill_color.shape)
print(resized_img.shape)

cv.namedWindow("taj-img",cv.WINDOW_NORMAL)

cv.imshow("taj-img",resized_img)

cv.waitKey(0)

cv.destroyAllWindows()

cv.imwrite("taj_copy.jpg",resized_img)

flag=[]
for i in dir(cv):
    if i.startswith("COLOR_"):
        flag.append(i)
print(len(flag))

new_img=cv.cvtColor(mandrill_color,cv.COLOR_BGR2GRAY)
cv.imshow("new-img",new_img)
cv.waitKey(0)
cv.destroyAllWindows()
