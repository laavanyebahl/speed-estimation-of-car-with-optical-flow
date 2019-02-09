# Background

The aim of  this project is to predict the speed of a moving car with just a front facing dashboard camera.
The video is challenging because of the lighting conditions. 

The input looks like this:    
![in](/output/input.gif)


First one can think of a general approach of training a CNN with the images and the labeled data. But, then if we train with the original frame from the video for regression and predicting the speed, the model will simply overfit, because it will try to memorize every image with a speed value since there aren't any common general features to associate speed with.

In order to make a useful input feature to train on, we look into optical flow.

# Optical Flow
Optical flow takes in two consecutive frames of a video in grayscale and gives out a matrix with the same dimensions as the input image, with each pixel denoting the change in its position and speed compared to the previous frame. 

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
* **Train using only original video frames:** Network gave good loss, but there was a lot of difference between loss and validation loss. the network seems to overfit and memorize the images.
* **Train using only optical flow frame:** Network gave very good loss, but required more EPOCHS
* **Train using both original video + optical flow frame (0.1 x original_video  + optical_flow):** The newtork gave the least loss with the least EPOCHS.

2nd approach gives the most realistic resuts on the test video and its output is attached.

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
    model.add(ELU())
    model.add(Dense(100, init = 'he_normal'))
    model.add(ELU())
    model.add(Dense(50, init = 'he_normal'))
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
127/127 [==============================] - 330s 3s/step - loss: 41.9508 - val_loss: 19.4800
Epoch 2/50
127/127 [==============================] - 292s 2s/step - loss: 16.0890 - val_loss: 13.3413
Epoch 3/50
127/127 [==============================] - 285s 2s/step - loss: 10.6050 - val_loss: 8.8628
Epoch 4/50
127/127 [==============================] - 293s 2s/step - loss: 7.5929 - val_loss: 6.4190
Epoch 5/50
127/127 [==============================] - 316s 2s/step - loss: 5.7750 - val_loss: 6.1499
Epoch 6/50
127/127 [==============================] - 322s 3s/step - loss: 4.4170 - val_loss: 3.9023
Epoch 7/50
127/127 [==============================] - 321s 3s/step - loss: 3.4074 - val_loss: 3.3072
Epoch 8/50
127/127 [==============================] - 311s 2s/step - loss: 2.7530 - val_loss: 3.9980
Epoch 9/50
127/127 [==============================] - 312s 2s/step - loss: 2.3145 - val_loss: 2.7451
Epoch 10/50
127/127 [==============================] - 307s 2s/step - loss: 1.8891 - val_loss: 1.8045
Epoch 11/50
127/127 [==============================] - 310s 2s/step - loss: 1.6338 - val_loss: 1.7173
Epoch 12/50
127/127 [==============================] - 307s 2s/step - loss: 1.4434 - val_loss: 1.9637
Epoch 13/50
127/127 [==============================] - 308s 2s/step - loss: 1.2507 - val_loss: 2.0172
Epoch 14/50
127/127 [==============================] - 302s 2s/step - loss: 1.1852 - val_loss: 1.6852
Epoch 15/50
127/127 [==============================] - 291s 2s/step - loss: 0.9678 - val_loss: 1.6554
Epoch 16/50
127/127 [==============================] - 286s 2s/step - loss: 0.9465 - val_loss: 0.9293
Epoch 17/50
127/127 [==============================] - 287s 2s/step - loss: 0.8984 - val_loss: 0.9464
Epoch 18/50
127/127 [==============================] - 303s 2s/step - loss: 0.7462 - val_loss: 1.3606
Epoch 19/50
127/127 [==============================] - 309s 2s/step - loss: 0.7392 - val_loss: 1.4743

Training model complete...

```

![graph](/output/graph.png)

__Early stopping__

I used the keras eraly stopping feature to stop the network when it starts overfitting.
I define the patience as 2. That means it looks for 2 more epochs for improvement in validationn loss otherwise it stops the training if it does not improve and saves the model with the best validation loss.
Here, the network is stopped at 19th epoch.



# Testing 
For testing we convert to Optical flow on the go and take the average of the previous four frames to remove outliers and predict the speed for each such frame.

The program saves the predicted labels in test.py as well as outputs two videos : test video with predicted label on it and a combined optical flow + test video as shown at the end.


# Output visualization
We can see the output video and the optical flow overlayed on the original video image  below.

![out](/output/out.gif)

![out_opt](/output/out_flow.gif)



