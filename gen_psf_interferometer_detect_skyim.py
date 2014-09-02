#! /usr/bin/python
"""
Interferometer/PSF simulator

Created by: Jack Hickish
Minor Modifications by: Griffin Foster

TODO: add color
TODO: adjust detection paramters
TODO: add freeze/unfreeze command
TODO: add rotation command
"""

import cv2 #for ubuntu 12.04 install see: http://karytech.blogspot.com/2012/05/opencv-24-on-ubuntu-1204.html
import cv
import numpy as np
import time
import sys, optparse

def cvfast_conv(image,psf):
    max_size = np.array([np.max([image.shape[0],psf.shape[0]]),np.max([image.shape[1],psf.shape[1]])])
    n = cv.GetOptimalDFTSize(max_size[0]*2)
    m = cv.GetOptimalDFTSize(max_size[1]*2)
    imagePad=np.zeros((n,m))
    imagePad[:image.shape[0],:image.shape[1]]=image

    imageDirty=np.fft.irfft2(np.fft.rfft2(imagePad) * np.fft.rfft2(psf, imagePad.shape))
    print image.shape, psf.shape, n,m
    return imageDirty[psf.shape[0]/2:image.shape[0]+psf.shape[0]/2,psf.shape[1]/2:image.shape[1]+psf.shape[1]/2]
    #return imagePad

def cv2array(im): 
    depth2dtype = { 
          cv.IPL_DEPTH_8U: 'uint8', 
          cv.IPL_DEPTH_8S: 'int8', 
          cv.IPL_DEPTH_16U: 'uint16', 
          cv.IPL_DEPTH_16S: 'int16', 
          cv.IPL_DEPTH_32S: 'int32', 
          cv.IPL_DEPTH_32F: 'float32', 
          cv.IPL_DEPTH_64F: 'float64', 
    } 

    arrdtype=im.depth 
    a = np.fromstring( 
           im.tostring(), 
           dtype=depth2dtype[im.depth], 
           count=im.width*im.height*im.nChannels) 
    a.shape = (im.height,im.width,im.nChannels) 
    return a 

def array2cv(a): 
    dtype2depth = { 
        'uint8':   cv.IPL_DEPTH_8U, 
        'int8':    cv.IPL_DEPTH_8S, 
        'uint16':  cv.IPL_DEPTH_16U, 
        'int16':   cv.IPL_DEPTH_16S, 
        'int32':   cv.IPL_DEPTH_32S, 
        'float32': cv.IPL_DEPTH_32F, 
        'float64': cv.IPL_DEPTH_64F, 
   } 
    try: 
        nChannels = a.shape[2] 
    except: 
        nChannels = 1 
    cv_im = cv.CreateImageHeader((a.shape[1],a.shape[0]), 
    dtype2depth[str(a.dtype)], nChannels) 
    cv.SetData(cv_im, a.tostring(),a.dtype.itemsize*nChannels*a.shape[1]) 
    return cv_im

o = optparse.OptionParser()
o.set_usage('%prog [options]')
o.set_description(__doc__)
o.add_option('-i','--input',dest='input', default=None,
    help='Input \'sky\' image, Default: HARDCODED')
o.add_option('-c','--camera',dest='camera', default=1, type='int',
    help='Camera device ID in /dev/video*, Default: 1')
o.add_option('-r','--res',dest='res', default=4, type='int',
    help='Resolution factor, increase this value to decrease the resolution, Default: 4')
opts, args = o.parse_args(sys.argv[1:])

CAMERA_DEVICE_INDEX=opts.camera   #check /dev/, ID is attached to video device (0 is in the internal)

cv.NamedWindow("Antenna Layout", cv.CV_WINDOW_AUTOSIZE)
cv.NamedWindow("Target Image", cv.CV_WINDOW_AUTOSIZE)
cv.NamedWindow("Point Spread", cv.CV_WINDOW_AUTOSIZE)
cv.NamedWindow("Observed Image", cv.CV_WINDOW_AUTOSIZE)
cam0 = cv.CaptureFromCAM(CAMERA_DEVICE_INDEX)

if opts.input is None:
    target_image = cv2.imread('/home/griffin/Downloads/interactiveInterferometer/astro_test_image.jpg')
else:
    target_image=cv2.imread(opts.input)
target_img_grey = cv2.cvtColor(target_image,cv2.cv.CV_BGR2GRAY)
target_img_lying = target_img_grey.copy()
#saturated image
target_img_grey[target_img_grey>100] = 255
target_img_grey[target_img_grey<255] = 0

RESCALE_FACTOR=opts.res #decrease to change the effective resolution
ysize=480
xsize=640

#make a 2D Gaussian to modulate the PSF with
def gauss2d(x0,y0,amp,stdx,stdy):
    return lambda x,y: amp*np.exp(-1.*( (((x-x0)**2.)/(2*stdx**2.)) + (((y-y0)**2.)/(2*stdy**2.)) ))
gaussFunc=gauss2d(0.,0.,1.,30.,30.)
xx = np.arange(xsize)-(xsize/2)
yy = np.arange(ysize)-(ysize/2)
xv, yv = np.meshgrid(xx, yy)
gaussGrid=gaussFunc(xv,yv)

while(True):
    layout_img = cv.QueryFrame(cam0)

    layout_img_grey = cv.CreateImage((layout_img.width,layout_img.height),layout_img.depth,1)
    cv.CvtColor(layout_img,layout_img_grey,cv.CV_BGR2GRAY)
    layout_img_grey_arr = cv2array(layout_img_grey)

    station_locs = np.zeros([ysize/RESCALE_FACTOR,xsize/RESCALE_FACTOR])
    #cv2.HoughCircles(image, method, dp, minDist, circles, param1, param2, minRadius, maxRadius)
    #   image: input webcam image size
    #   method: only cv.CV_HOUGH_GRADIENT exists
    #   dp: Inverse ratio of the accumulator resolution to the image resolution. this basically affects the min/max radius
    #   minDist: Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
    #   circles: set to None
    #   param1: threshold parameter
    #   param2: The smaller it is, the more false circles may be detected.
    #   minRadius: Minimum circle radius
    #   maxRadius: Maximum circle radius
    circles = cv2.HoughCircles(layout_img_grey_arr, cv.CV_HOUGH_GRADIENT,2,10,None,100,35,5,30)
    if circles is not None:
        for cn,circle in enumerate(circles[0]):
            x,y = circle[1],circle[0]
            print "we have circle at %d,%d"%(x,y)
            try:
                layout_img_grey_arr[x-5:x+5,y-5:y+5]=255
            except:
                pass
            station_locs[x/RESCALE_FACTOR,y/RESCALE_FACTOR]=1


    psf = np.fft.fftshift(np.abs(np.fft.fft2(station_locs,s=[ysize,xsize]))**2)
    #psf=psf[(ysize/2)-64:(ysize/2)+64,(xsize/2)-64:(xsize/2)+64] #only select the central region of the PSF
    psf /= psf.max()
    psf=psf*gaussGrid #apply a Gaussian taper to the PSF

    psf_img = array2cv(psf)

    #target_arr = target_img_grey[:,:]
    #dirty_arr = cvfast_conv(target_arr,psf)
    dirty_arr = cvfast_conv(target_img_lying,psf)
    
    dirty_arr /= dirty_arr.max()
    dirty_img = array2cv(dirty_arr)

    cv.ShowImage("Antenna Layout",array2cv(layout_img_grey_arr))
    cv.ShowImage("Target Image",array2cv(target_img_lying))
    cv.ShowImage("Point Spread",psf_img)
    cv.ShowImage("Observed Image",dirty_img)

    if cv.WaitKey(50)!=-1:
        break

cv.DestroyAllWindows()
