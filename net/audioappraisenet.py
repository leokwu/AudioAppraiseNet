# import the necessary packages
from keras.models import Sequential
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet import MobileNet
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

        model.add(Conv2D(32, (3, 3), padding="same",
            input_shape=inputShape, kernel_initializer=init, kernel_regularizer=reg))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer=init, kernel_regularizer=reg))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer=init, kernel_regularizer=reg))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init, kernel_regularizer=reg))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # for i in range(25):
        model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer=init, kernel_regularizer=reg))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer=init, kernel_regularizer=reg))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
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
    def build_mobilenetv1(width, height, depth, classes, reg, init="he_normal"):
        inputShape = (height, width, depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        input_tensor = Input(shape=inputShape)
        base_model = MobileNet(include_top=False, weights='imagenet', input_tensor=input_tensor,
                                 input_shape=inputShape, pooling='avg')
        for i, layer in enumerate(base_model.layers):
            print(i, layer.name)
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model.output
        x = Dense(1024, kernel_initializer=init, kernel_regularizer=reg, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        # for i, layer in enumerate(base_model.layers):
        #     print(i, layer.name)
        # for layer in model.layers[:154]:
        #     layer.trainable = False
        # for layer in model.layers[154:]:
        #     layer.trainable = True
        return model


    @staticmethod
    def build_mobilenetv2(width, height, depth, classes, reg, init="he_normal"):
        inputShape = (height, width, depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        input_tensor = Input(shape=inputShape)
        base_model = MobileNetV2(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=inputShape, pooling='avg')
        for i, layer in enumerate(base_model.layers):
            print(i, layer.name)
        for layer in base_model.layers:
            layer.trainable = False
        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(128, kernel_initializer=init, kernel_regularizer=reg))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation("softmax"))
        # x = base_model.output
        # x = Dense(1024, kernel_initializer=init, kernel_regularizer=reg, activation='relu')(x)
        # x = BatchNormalization()(x)
        # x = Dense(1024, activation='relu')(x)
        # x = BatchNormalization()(x)
        # x = Dropout(0.25)(x)
        # x = Dense(1024, activation='relu')(x)
        # x = BatchNormalization()(x)
        # x = Dense(512, activation='relu')(x)
        # x = BatchNormalization()(x)
        # x = Dropout(0.5)(x)
        # predictions = Dense(classes, activation='softmax')(x)
        # model = Model(inputs=base_model.input, outputs=predictions)
        model.summary()
        # for i, layer in enumerate(base_model.layers):
        #     print(i, layer.name)
        # for layer in model.layers[:154]:
        #     layer.trainable = False
        # for layer in model.layers[154:]:
        #     layer.trainable = True
        return model

    @staticmethod
    def build_inceptionv3(width, height, depth, classes, reg, init="he_normal"):
        inputShape = (height, width, depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        input_tensor = Input(shape=inputShape)
        base_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=inputShape, pooling='avg')
        for i, layer in enumerate(base_model.layers):
            print(i, layer.name)
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model.output
        x = Dense(512, kernel_initializer=init, kernel_regularizer=reg, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Dense(1024, kernel_initializer=init, kernel_regularizer=reg, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Dense(1024, kernel_initializer=init, kernel_regularizer=reg, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Dense(1024, kernel_initializer=init, kernel_regularizer=reg, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(512, kernel_initializer=init, kernel_regularizer=reg, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        # for i, layer in enumerate(base_model.layers):
        #     print(i, layer.name)
        # for layer in model.layers[:249]:
        #     layer.trainable = False
        # for layer in model.layers[249:]:
        #     layer.trainable = True
        return model
