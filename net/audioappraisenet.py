# import the necessary packages
from keras.models import Sequential
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras import backend as K

class AudioAppraiseNet:
    @staticmethod
    def build(width, height, depth, classes, reg, init="he_normal"):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model.add(Conv2D(16, (3, 3), padding="same",
            input_shape=inputShape, kernel_initializer=init, kernel_regularizer=reg))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(Conv2D(16, (3, 3), padding="same", kernel_initializer=init, kernel_regularizer=reg))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer=init, kernel_regularizer=reg))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init, kernel_regularizer=reg))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer=init, kernel_regularizer=reg))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer=init, kernel_regularizer=reg))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128, kernel_initializer=init, kernel_regularizer=reg))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model

    @staticmethod
    def build_inceptionv3(width, height, depth, classes, reg, init="he_normal"):
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        input_tensor = Input(shape=inputShape)
        base_model = InceptionV3(include_top=True, input_tensor=input_tensor, classes=classes)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        x = Dense(128, kernel_initializer=init, kernel_regularizer=reg)(x)
        x = Activation('relu')(x)
        x = Dense(classes)(x)
        predictions = Activation('softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
