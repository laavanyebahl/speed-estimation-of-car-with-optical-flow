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

# Training

__Model__

Following model is used in **model.py:**
``` 
def CNNModel():
    model = Sequential()
    # model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape = (160, 320, 3)))
    # model.add(Cropping2D(cropping=((20, 20), (0, 0)), input_shape = (240, 320, 3)))
    # normalize data
    # model.add(Lambda(lambda x: (x / 255) - 0.5, input_shape = (240, 320, 3)))
    # model.add(BatchNormalization(input_shape = (240, 320, 3)))
    model.add(Convolution2D(24,5,5, subsample=(2,2), init = 'he_normal' , input_shape = (240, 320, 3)))
    # model.add(Convolution2D(24,5,5, subsample=(2,2), init = 'he_normal' ))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), init = 'he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), init = 'he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Convolution2D(64,3,3))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Convolution2D(64,3,3))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Activation('elu'))
    model.add(Dense(100))
    model.add(Activation('elu'))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Activation('elu'))
    model.add(Dense(10))
    model.add(Activation('elu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model
```


__Data Preparation and generation__   

We take 4 consecutive images and calculate the average optical flow to remove any outliers and smooth out predictions.

In **train_model.py:** we have the following two functions:
* prepareData() :  Stores the paths of 4 consecutive optical flow frames + the path of the original frame
* generateData() :  Generator funciton which loads the images from the path, makes a single train image from 4 consecutives frames and orginal image and feeds it into the network in batches by using yield

__Data Augmentation__  
In the generateData() function we double the data for training by simply flipping the combined ttrain images and adding it in the batch along with the same label.

__Training Graph__   

 
# Testing


# Early stopping

I used the kears eraly stopping feature to stop the network when it starts overfirring.
I define the patience as 3. That means it looks for 3 more epochs for improvement in validationn loss other wise it stops the trainign if it does not improvve and saves the mode with the best validatoin loss.

# Output visualization
We can see the optical glow overlayed on th prigin al video image  below.





