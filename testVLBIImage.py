import cv
import cv2
import numpy as np
import time
import sys, optparse


CAMERA_DEVICE_INDEX=0   #check /dev/, ID is attached to video device (0 is in the internal)

cam0= cv2.VideoCapture(0)

ret,frame = cam0.read()
cv2.imshow('frame',frame)


cv2.imwrite('test.jpg',frame)
cv2.waitKey(1)

cam0.release()
cv2.destroyAllWindows()


