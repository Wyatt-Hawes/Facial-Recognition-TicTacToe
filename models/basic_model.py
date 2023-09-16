from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.src.optimizers import SGD, RMSprop,Adam

from .model import Model
from keras.utils import plot_model
from keras.models import Sequential

import tensorflow as tf


import keras
class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here

        basic_model = Sequential()
        #                                           5,5
        basic_model.add(Conv2D(8, (3, 3), activation='relu', input_shape=input_shape))
        basic_model.add(MaxPooling2D((2, 2)))

        basic_model.add(Conv2D(16, (3, 3), activation='relu'))
        basic_model.add(MaxPooling2D((2, 2)))

        basic_model.add(Conv2D(16, (3, 3), activation='relu'))
        basic_model.add(MaxPooling2D((2, 2)))

        basic_model.add(Flatten())
        basic_model.add(Dense(32, activation='relu'))

        basic_model.add(Dense(categories_count, activation='softmax'))



        self.model = basic_model
        self.print_summary()

    def _compile_model(self):
        # Your code goes here
        self.model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )


        #self.model.compile(optimizer=RMSprop(learning_rate=.001), loss='categorical_crossentropy',metrics="accuracy")
