from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout, Reshape, BatchNormalization, ELU
from keras.optimizers import Adam


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
    # model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Dense(10, init = 'he_normal'))
    model.add(ELU())
    model.add(Dense(1, init = 'he_normal'))

    adam = Adam(lr=1e-4)
    model.compile(optimizer = adam, loss = 'mse')

    return model
