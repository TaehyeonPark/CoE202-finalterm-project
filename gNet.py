import tensorflow as tf
import keras
from keras import Sequential, Input
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from utils import *


class GestureNet():
    def __init__(self, args) -> None:
        super(GestureNet, self).__init__()
        self.input_shape = args.get("inputShape")
        self.output_shape = args.get("outputShape")

    def model(self) -> None:
        model = Sequential()
        model.add(Input((21*2,)))
        model.add(Dense(48))
        model.add(BatchNormalization())
        model.add(Activation(activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(24))
        model.add(BatchNormalization())
        model.add(Activation(activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(6, activation='softmax'))
        model.compile(
            optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model._name = "GestureNet"
        model.summary()
        return model

    def loadModel(self, model_location) -> None:
        model = self.model()
        return model.load_weights(model_location)


args = getGNetArgs()
gNet = GestureNet(args=args)
model = gNet.model()
