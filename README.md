# Background

The aim of  this project is to predict the speed of a moving car with just a front facing dashboard camera.
The video is challenging because of the lighting conditions. 

The input looks like this:    
![in](/output/input.gif)


First one can think of a general approach of training a CNN with the images and the labeled data. But, then if we train with the original frame from the video for regression and predicting the speed, the model will simply overfit, because it will try to memorize every image with a speed value since there aren't any common general features to associate speed with.

In order to make a useful input feature to train on, we look into optical flow.

# Optical Flow
Optical flow takes in two consecutive frames of a video in grayscale and gives out a matrix with the same dimensions as the input image, with each pixel denoting the change in its position compared to the previous frame. 

The direction/ orientation and magnitude at every pixel of the optical flow can be stored by using the HSV color channel. 

By training the network with optical flow, the network learns which pixels of the optical flow are important and weigh them accordingly.

The file **frames_to_opticalFlow.py** is used to generate a dense optical flow image from two consecutive frames using the 
openCV function ```cv2.calcOpticalFlowFarneback()```.   
The optical flow looks like this:

![flow](/output/flow.gif)

# Data preprocessing
I wrote the following 2 programs :
* **video_to_frames.py:** Takes in video and converts into frames
* **video_to_frames_and_optical_flow.py:** Takes in video, converts into frames and also to optical flow frames, so that this redundant thing does not happen at every epoch during training in bacthes.

# Training

3 efforts were made to train the network:   
* **Train using only original video frames:** Network gave good loss, but there was a lot of difference between loss an validation loss. the network seems to overfit and memorize the images.
* **Train using only optical flow frame:** Network gave very good loss, but required more EPOCHS
* **Train using both original video + optical flow frame (0.1 x original_video  + optical_flow):** The newtork gave the least loss and least EPOCHS.

2nd approach gives the most realistic resuts on the test video.

__Model__

Following model is used in **model.py:**
``` 
def CNNModel():
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, input_shape = (240, 320, 3), subsample=(2,2), init = 'he_normal'))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), init = 'he_normal'))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), init = 'he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, subsample = (1,1), init = 'he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample= (1,1), border_mode = 'valid', init = 'he_normal'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Dense(100, init = 'he_normal'))
    model.add(ELU())
    model.add(Dense(50, init = 'he_normal'))
    model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Dense(10, init = 'he_normal'))
    model.add(ELU())
    model.add(Dense(1, init = 'he_normal'))

    adam = Adam(lr=1e-4)
    model.compile(optimizer = adam, loss = 'mse')

    return model
```

__Data Preparation and generation__   

We take 4 consecutive images and calculate the average optical flow to remove any outliers and smooth out predictions.

In **train_model.py:** we have the following two functions:
* prepareData() :  Stores the paths of 4 consecutive optical flow frames + the path of the original frame
* generateData() :  Generator funciton which loads the images from the path, makes a single train image from 4 consecutive frames and orginal image and feeds it into the network in batches by using yield.

__Data Augmentation__  
In the generateData() function we double the data for training by simply flipping the train images and adding it in the batch along with the same label.

__Training Log and Graph__   

```
127/127 [==============================] - 312s 2s/step - loss: 30.8945 - val_loss: 11.2830
Epoch 2/50
127/127 [==============================] - 280s 2s/step - loss: 7.8224 - val_loss: 5.7935
Epoch 3/50
127/127 [==============================] - 282s 2s/step - loss: 3.5847 - val_loss: 2.6088
Epoch 4/50
127/127 [==============================] - 281s 2s/step - loss: 2.1205 - val_loss: 1.8640
Epoch 5/50
127/127 [==============================] - 281s 2s/step - loss: 1.5385 - val_loss: 1.0753
Epoch 6/50
127/127 [==============================] - 281s 2s/step - loss: 1.0618 - val_loss: 1.7238
Epoch 7/50
127/127 [==============================] - 283s 2s/step - loss: 0.8148 - val_loss: 0.8372
Epoch 8/50
127/127 [==============================] - 283s 2s/step - loss: 0.6916 - val_loss: 0.8000
Epoch 9/50
127/127 [==============================] - 281s 2s/step - loss: 0.5722 - val_loss: 0.8313
Epoch 10/50
127/127 [==============================] - 282s 2s/step - loss: 0.5420 - val_loss: 0.6292
Epoch 11/50
127/127 [==============================] - 282s 2s/step - loss: 0.4552 - val_loss: 0.4124
Epoch 12/50
127/127 [==============================] - 283s 2s/step - loss: 0.4100 - val_loss: 0.5331
Epoch 13/50
127/127 [==============================] - 280s 2s/step - loss: 0.3925 - val_loss: 0.6852
Training model complete...

```

![graph](/output/graph.png)


# Testing 
For testing we convert to Optical flow on the go and take the average of the previous four frames to remove outliers and predict the speed for each such frame.

# Early stopping   

I used the keras eraly stopping feature to stop the network when it starts overfitting.
I define the patience as 2. That means it looks for 2 more epochs for improvement in validationn loss otherwise it stops the training if it does not improve and saves the model with the best validation loss.
Here, the network is stopped at 13th epoch.

# Output visualization
We can see the output video and the optical flow overlayed on the original video image  below.

![out](/output/out.gif)

![out_opt](/output/out_flow.gif)



