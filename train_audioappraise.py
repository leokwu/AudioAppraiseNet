# USAGE
# python3.6 train_audioappraise.py --dataset /datasets/f0_classify --model audioappraise.model --le le.pickle

import matplotlib
matplotlib.use("Agg")

from net.audioappraisenet import AudioAppraiseNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os
import tensorflow as tf
from keras import backend as K
import keras
from keras.regularizers import l2
from keras.models import Model
# from keras.utils import multi_gpu_model

def train_process(config):
    INIT_LR = 1e-4
    BS = 32
    EPOCHS = 128
    # EPOCHS = 50

    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(config.dataset))
    # print("imagePaths: ", imagePaths)
    data = []
    labels = []

    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        # print(": ", imagePath.split(os.path.sep)[1])
        # print("os.path.sep: ", os.path.sep)
        # print("label: ", label)
        image = cv2.imread(imagePath)
        # image = cv2.resize(image, (224, 224)) # mobilenetv2 (96, 96) (128, 128) (160, 160) (192, 192) (224, 224)
        image = cv2.resize(image, (128, 128))
        data.append(image)
        labels.append(label)

    data = np.array(data, dtype="float") / 255.0

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = np_utils.to_categorical(labels, 2)

    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

    # aug = ImageDataGenerator()
    # aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
    aug = ImageDataGenerator(zoom_range=0.15,
                             width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                             horizontal_flip=True, fill_mode="nearest")

    print("[INFO] compiling model...")
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    # mobilenetv2
    model = AudioAppraiseNet.build_mobilenetv2(width=128, height=128, depth=3,
    # model = AudioAppraiseNet.build(width=300, height=300, depth=3,
                               classes=len(le.classes_), reg=l2(0.0004))
    # inceptionv3
    # model = AudioAppraiseNet.build_inceptionv3(width=32, height=32, depth=3,
    #                                           classes=len(le.classes_), reg=l2(0.0004))
    # model = multi_gpu_model(model, gpus=4)
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    callbacks = [keras.callbacks.TensorBoard(log_dir='./logs', update_freq='epoch'),
                 keras.callbacks.ModelCheckpoint(filepath='./checkpoints/audioappraisenet.{epoch:02d}.hdf5',
                                                 verbose=0, save_best_only=False, save_weights_only=False, mode='auto',
                                                 period=1)
                 ]
    print("len(trainX) // BS: ", len(trainX) // BS)
    print("[INFO] training network for {} epochs...".format(EPOCHS))
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                            validation_data=(testX, testY),
                            steps_per_epoch=len(trainX) // BS,
                            epochs=EPOCHS,
                            # workers=10,
                            # use_multiprocessing=True,
                            callbacks=callbacks)

    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=BS)
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1), target_names=le.classes_))

    print("[INFO] serializing network to '{}'...".format(config.model))
    model.save(config.model)
    model.save_weights('./weights/audioappraisenet_weights.h5')
    # model.load_weights('./weights/audioappraisenet_weights.h5')
    # print model info as json/yaml
    # json_strig = model.to_json()
    # print("json_strig: ", json_strig)
    # yaml_string = model.to_yaml()
    # print("yaml_string: ", yaml_string)

    # output intermediate layer model result
    # layer_name = 'dense_2'
    # intermediate_layer_model = Model(inputs=model.input,
    #                                     outputs=model.get_layer(layer_name).output)
    # intermediate_output = intermediate_layer_model.predict(testX, batch_size=BS)
    # print("intermediate_output: --------> ", intermediate_output)

    f = open(config.le, "wb")
    f.write(pickle.dumps(le, True))
    f.close()

    print("input is: ", model.input.op.name)
    print("output is: ", model.output.op.name)
    # save pb model
    # sess = K.get_session()
    # frozen_graph_def = tf.graph_util.convert_variables_to_constants(
    #    sess,
    #    sess.graph_def,
        # output_node_names=["dense_6/Softmax"])
    #    output_node_names=["dense_5/Softmax"])
    #with tf.gfile.GFile('./model/audioappraisenet_model.pb', "wb") as f:
    #    f.write(frozen_graph_def.SerializeToString())
    # tf.train.write_graph(frozen_graph_def, 'model', 'audioappraisenet_model.pb', as_text=True)
    # tf.train.write_graph(frozen_graph_def, 'model', 'audioappraisenet_model.pb', as_text=False)

    # Training set accuracy--------------------------------
    result = model.evaluate(trainX, trainY, batch_size=10)
    print('\nTrain Acc:', result[1])
    # print('\nTrain Los:', result[0])

    # Testing set accuracy---------------------------------
    result = model.evaluate(testX, testY, batch_size=10)
    print('\nTest Acc:', result[1])
    # print('\nTest Los:', result[0])

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, EPOCHS), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, EPOCHS), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(config.plot)

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset")
    parser.add_argument("-m", "--model", type=str, required=True,
                    help="path to trained model")
    parser.add_argument("-l", "--le", type=str, required=True,
                    help="path to label encoder")
    parser.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output loss/accuracy plot")

    config = parser.parse_args()
    train_process(config)
