import keras
from keras import Sequential, Input
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from utils import *


class GestureNet():
    def model(self):
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
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        model._name = "GestureNet"
        return model


if __name__ == '__main__':
    gNet = GestureNet()
    model = gNet.model()
