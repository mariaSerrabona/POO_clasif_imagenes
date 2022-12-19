#-----------------------------------------------------------------------------------------
# @Autor: Aurélien Vannieuwenhuyze
# @Empresa: Junior Makers Place
# @Libro:
# @Capítulo: 12 - Clasificación de imágenes
#
# Módulos necesarios:
#   PANDAS 0.24.2
#   KERAS 2.2.4
#   PILOW 6.0.0
#   SCIKIT-LEARN 0.20.3
#   NUMPY 1.16.3
#   MATPLOTLIB : 3.0.3
#
# Para instalar un módulo:
#   Haga clic en el menú File > Settings > Project:nombre_del_proyecto > Project interpreter > botón +
#   Introduzca el nombre del módulo en la zona de búsqueda situada en la parte superior izquierda
#   Elegir la versión en la parte inferior derecha
#   Haga clic en el botón install situado en la parte inferior izquierda
#-----------------------------------------------------------------------------------------



#************************************************************************************
#
# REDES NEURONALES CONVOLUCIONALES CON 1 CAPA DE CONVOLUCIONES
#
#************************************************************************************


import pandas as pnd
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from preparacion_redes_conv import preparacion_capa_concolucion
import keras



#Definición de largo y ancho de la imagen
LARGO_IMAGEN = 28
ANCHO_IMAGEN = 28

#Carga de los datos de entrenamiento
observaciones_entrenamiento = pnd.read_csv('datas/zalando/fashion-mnist_train.csv')
observaciones_test = pnd.read_csv('datas/zalando/fashion-mnist_test.csv')


class capa_concolucion():
    def __init__(self,observaciones_entrenamiento,observaciones_test, ancho_imagen, largo_imagen ):
        self.observaciones_entrenamiento=observaciones_entrenamiento
        self.observaciones_test=observaciones_test
        self.ancho_imagen=ancho_imagen
        self.largo_imagen=largo_imagen

    preparacion=preparacion_capa_concolucion(self.observaciones_entrenamiento, self.ancho_imagen, self.largo_imagen)
#----------------------- CNN 1 ------------------------


    def creacion_red(self):

        #Se especifican las dimensiones de la imagen de entrada
        dimensionImagen = (self.ancho_imagen, self.largo_imagen, 1)

        #Se crea la red neuronal capa por capa
        redNeurona1Convolucion = Sequential()

        #1- Adición de la capa de convolución que contiene
        #  Capa coculta de 32 neuronas
        #  Un filtro de 3x3 (Kernel) recorriendo la imagen
        #  Una función de activación de tipo ReLU (Rectified Linear Activation)
        #  Una imagen de entrada de 28px * 28 px
        redNeurona1Convolucion.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=dimensionImagen))

        #2- Definición de la función de pooling con un filtro de 2px por 2 px
        redNeurona1Convolucion.add(MaxPooling2D(pool_size=(2, 2)))

        #3- Adición de una función de ignorancia
        redNeurona1Convolucion.add(Dropout(0.2))

        #5 - Se transforma en una sola línea
        redNeurona1Convolucion.add(Flatten())

        #6 - Adición de una red neuronal compuesta por 128 neuronas con una función de activación de tipo Relu
        redNeurona1Convolucion.add(Dense(128, activation='relu'))

        #7 - Adición de una red neuronal compuesta por 10 neuronas con una función de activación de tipo softmax
        redNeurona1Convolucion.add(Dense(10, activation='softmax'))

        #8 - Compilación del modelo

        redNeurona1Convolucion.compile(loss=keras.losses.categorical_crossentropy,
                                        optimizer=keras.optimizers.Adam(),
                                        metrics=['accuracy'])

    def aprendizaje(self):
        #9 - Aprendizaje
        historico_aprendizaje  = redNeurona1Convolucion.fit(preparacion.separacion_datos(self.observaciones_test, 2), preparacion.separacion_datos(self.observaciones_test, 4))
                                batch_size=256,
                                epochs=10,
                                verbose=1,
                                validation_data=(preparacion.separacion_datos(self.observaciones_test, 3), preparacion.separacion_datos(self.observaciones_test, 5))

        return historico_aprendizaje

    def evaluacion_modelo(self):
        #10 - Evaluación del modelo
        evaluacion = self.aprendizaje().evaluate(preparacion.separacion_datos(self.observaciones_test, 0), preparacion.separacion_datos(self.observaciones_test, 1), verbose=0)
        print('Error:', evaluacion[0])
        print('Precisión:', evaluacion[1])
        return evaluacion



#11 - Visualización de la fase de aprendizaje
import matplotlib.pyplot as plt

#Datos de precisión (accurary)
plt.plot(historico_aprendizaje.history['accuracy'])
plt.plot(historico_aprendizaje.history['val_accuracy'])
plt.title('Precisión del modelo')
plt.ylabel('Precisión')
plt.xlabel('Epoch')
plt.legend(['Aprendizaje', 'Test'], loc='upper left')
plt.show()

#Datos de validación y error
plt.plot(historico_aprendizaje.history['loss'])
plt.plot(historico_aprendizaje.history['val_loss'])
plt.title('Error')
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.legend(['Aprendizaje', 'Test'], loc='upper left')
plt.show()


