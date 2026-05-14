import numpy as np
import cv2 as cv

captured_video=cv.VideoCapture()
# valid_read, image_frame=captured_video.read()

while(captured_video.isOpened()):
    valid_read, image_frame=captured_video.read()

    if(valid_read):
        gray=cv.cvtColor(image_frame,cv.COLOR_BGR2GRAY)
        cv.imshow("img",gray)
        cv.waitKey(50)
        if (cv.getWindowProperty("frame",0)==-1):
            break

captured_video.release()
cv.destroyAllWindows()
print("done")