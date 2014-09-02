import cv
import cv2
import numpy as np
import time
import sys, optparse
import matplotlib.pyplot as plt


#rame=cv2.imread('test.jpg',0)
cam0t=cv2.VideoCapture(1)
ret,frame = cam0t.read()
frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
frame=~frame
cv2.imshow('image',frame)
cv2.waitKey(0)


edges=cv2.Canny(frame,70,255)


#ret,thresh1 = cv2.threshold(hsv,90,255,cv2.THRESH_BINARY)
#thresh1=~thresh1
cv2.imshow('image',edges)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#filledI = np.zeros(thresh1.shape[0:2]).astype('uint8')
#convexI = np.zeros(thresh1.shape[0:2]).astype('uint8')
cv2.drawContours(edges, contours, 0, (0,255,0), 10)
cv2.imshow('image',edges)
cv2.waitKey(0)

contours= sorted(contours, key = cv2.contourArea, reverse = False)
pos=[]
for cnt in range(len(contours)):
	peri = cv2.arcLength(contours[cnt], True)
	approx = cv2.approxPolyDP(contours[cnt], 0.02 * peri, True)
#	cv2.drawContours(thresh1, [approx], -1, (255, 255, 0), 6)
	M=cv2.moments(contours[cnt])
	try:
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])
		cv2.circle(edges,(cx,cy),10,(255,255,0),5)
		pos.append([cx,cy])
	except:
		print 'error'
	try:
		print cnt,cx,cy
	except:
		print 'error'

cv2.imshow('image',edges)
cv2.waitKey(30)



cv2.imshow('image',frame)
cv2.waitKey(30)

#cv2.destroyAllWindows()


