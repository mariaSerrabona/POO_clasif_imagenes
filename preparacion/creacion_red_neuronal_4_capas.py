

import pandas as pnd
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import BatchNormalization
from preparacion.preparacion_redes_conv import preparacion_capa_concolucion
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras

class red_neuronal_4_capas():

    def __init__(self,ancho_imagen, largo_imagen):
        self.ancho_imagen = ancho_imagen
        self.largo_imagen = largo_imagen

    def creacion(self):
        dimensionImagen = (self.ancho_imagen, self.largo_imagen, 1)

        #Se crea la red neuronal capa por capa

        redNeuronas4Convolucion = Sequential()
        redNeuronas4Convolucion.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=dimensionImagen))
        redNeuronas4Convolucion.add(BatchNormalization())

        redNeuronas4Convolucion.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        redNeuronas4Convolucion.add(BatchNormalization())
        redNeuronas4Convolucion.add(MaxPooling2D(pool_size=(2, 2)))
        redNeuronas4Convolucion.add(Dropout(0.25))

        redNeuronas4Convolucion.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        redNeuronas4Convolucion.add(BatchNormalization())
        redNeuronas4Convolucion.add(Dropout(0.25))

        redNeuronas4Convolucion.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        redNeuronas4Convolucion.add(BatchNormalization())
        redNeuronas4Convolucion.add(MaxPooling2D(pool_size=(2, 2)))
        redNeuronas4Convolucion.add(Dropout(0.25))

        redNeuronas4Convolucion.add(Flatten())

        redNeuronas4Convolucion.add(Dense(512, activation='relu'))
        redNeuronas4Convolucion.add(BatchNormalization())
        redNeuronas4Convolucion.add(Dropout(0.5))

        redNeuronas4Convolucion.add(Dense(128, activation='relu'))
        redNeuronas4Convolucion.add(BatchNormalization())
        redNeuronas4Convolucion.add(Dropout(0.5))

        redNeuronas4Convolucion.add(Dense(10, activation='softmax'))

        #8 - Compilación del modelo
        redNeuronas4Convolucion.compile(loss=keras.losses.categorical_crossentropy,
                                        optimizer=keras.optimizers.Adam(),
                                        metrics=['accuracy'])
        return redNeuronas4Convolucion