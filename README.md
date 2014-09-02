liveInterferometer
==================

#README Document for the webcam interactive interferometer
-----------------------------------------------------------------
Created: 26.2.14
Updated: 26.2.14

Required Equipment:
1. extrenal USB webcam

In the circle detection script mode you require
a. coins or small round objects
b. flat, light colored area (A3 or A2 paper)

In the edge detection mode you can draw the stations on a piece of white paper using a black marker



Software:
python script: gen_psf_interferometer_detect_skyim.py
test sky image: such as astro_test_image.jpg

The code is in the directory: /home/griffin/Downloads/interactiveInterferometer

##To Run:
-----------------------------------------------------------------
1. before plugging in the webcam, run:

    ls /dev/video*

2. plug in the webcam and again run:

    ls /dev/video*

A new video device should appear, remember the ID number as this will be used in the script. For this example the external webcam is /dev/video1, so it has ID:1

3. run the script: gen_psf_interferometer_detect_skyim.py

For script help run: python gen_psf_interferometer_detect_skyim.py -h

To run the script with the webcam ID and test sky image: python gen_psf_interferometer_detect_skyim.py -c 1 -i /home/griffin/Downloads/interactiveInterferometer/astro_test_image.jpg


##Changing Settings:
-----------------------------------------------------------------
A few parameters can be altered.

A Gaussian funstion is applied to the PSF, this can be disabled by commenting out the line:

    psf=psf*gaussGrid

To change the size of the Gaussian function change the last two number in the line (they are in pixels):

    gaussFunc=gauss2d(0.,0.,1.,30.,30.)

The circles are detected using the cv2.HoughCircles() function, it takes a number of parameters which are documented in the code and on the opencv webpage.
Interferometer Live Demo Using Open CV image detection

#Edge Detection

The code can also be run in edge detection mode.

When run in the edge detection mode the code does not use HoughCircles but rather the cv2.Canny algorithm for edge detection. Countours are then used to find the centroid of the contour and hence the location of the points. This algorithm allows you to do things like draw stations on a piece of paper.

To run the script with the webcam ID and test sky image: python gen_psf_interferometer_detect_skyimCJC.py -c 1 -i /home/griffin/Downloads/interactiveInterferometer/astro_test_image.jpg

