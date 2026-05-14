import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

cap=cv.VideoCapture("/home/labuser/Downloads/CV_OPENCV/Video_processing/video_proc/video_sample.mp4")

while(cap.isOpened()):
    ret, frame=cap.read()
    if(ret==True):
        gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        cv.imshow("frame",gray)
        cv.waitKey(50)
    else:
        break
cap.release()
cv.destroyAllWindows()
print("done")

