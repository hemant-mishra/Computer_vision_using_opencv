import numpy as np
import cv2 as cv

video=cv.VideoCapture("/home/labuser/Downloads/CV_OPENCV/Video_processing/video_proc/video_sample.avi")

codec=cv.VideoWriter_fourcc(*"DIVX")
out=cv.VideoWriter("output_written.avi",codec,10,(1280,720))

while (video.isOpened()):
    rat, frame=video.read()
    if rat==True:
        frame=cv.flip(frame,0)
        out.write(frame)
    else:
        break
video.release()
out.release()
cv.destroyAllWindows()
print("done")