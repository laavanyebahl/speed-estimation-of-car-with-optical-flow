
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout, Reshape, BatchNormalization, Activation


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
    model.add(Activation('elu'))
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Activation('elu'))
    model.add(Dense(50))
    # model.add(Dropout(0.5))
    model.add(Activation('elu'))
    model.add(Dense(10))
    model.add(Activation('elu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model
