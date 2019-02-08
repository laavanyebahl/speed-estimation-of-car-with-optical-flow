# Background

The aim of  this project is to predict the speed of a moving car with just a front facing dashboard camera.
The video is challenging because of the lighting conditions. 

First one can think of a general approach of training a CNN with the images and the labeled data. But, then if we train with the original frame from the video for regression and predicting the speed, the model will simply overfit, because it will try to memorize evry image with a speed value. 

In order to make find a useful input feature to train on, we look into optical flow.

# Optical Flow
Optical flow takes in two consecutive frames of a video in grayscale and gives out a matrix with the same dimesnion as the input image, with each pixel denoting the change in its position compared to the previous frame. 

The direction/ orientation and magnitude at every pielof the optical flow can be stored by using the HSV color channel. 

The file **frames_to_opticalFlow.py** is used to generate an optical flow image from two consecutive frames.

The input and output looks like this:

#input gif
#output gif

# Data preprocessing
I wrote the following 2 programs :
* **video_to_frames.py:** Takes in video and converts into frames
* **video_to_frames_and_optical_flow.py:** Takes in video, converts into frames and also to optical flow frames, so that this redundant thing does not happen at every epoch during training in bacthes.



