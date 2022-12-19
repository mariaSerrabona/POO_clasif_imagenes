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
from preparacion.preparacion_redes_conv import preparacion_capa_concolucion
from preparacion.creacion_red_neuronal import RedNeuronal
import keras
import matplotlib.pyplot as plt


class capa_convolucion():
    def __init__(self,observaciones_entrenamiento,observaciones_test, ancho_imagen, largo_imagen ):
        self.observaciones_entrenamiento=observaciones_entrenamiento
        self.observaciones_test=observaciones_test
        self.ancho_imagen=ancho_imagen
        self.largo_imagen=largo_imagen

    def preparacion_red(self):
        preparacion=preparacion_capa_concolucion(self.observaciones_entrenamiento, self.ancho_imagen, self.largo_imagen)
        return preparacion


    def creacion_red(self):

        preparacion=preparacion_capa_concolucion(self.observaciones_entrenamiento, self.ancho_imagen, self.largo_imagen)
        creacion= RedNeuronal(self.ancho_imagen, self.largo_imagen)

        #9 - Aprendizaje
        historico_aprendizaje  = creacion.creacion_red().fit(preparacion.separacion_datos(2), preparacion.separacion_datos(4)),batch_size=256,epochs=10,verbose=1,validation_data=(preparacion.separacion_datos(3), preparacion.separacion_datos(5))


        #10 - Evaluación del modelo
        evaluacion = self.aprendizaje().evaluate(preparacion.separacion_datos(0), preparacion.separacion_datos(1), verbose=0)
        print('Error:', evaluacion[0])
        print('Precisión:', evaluacion[1])
        return historico_aprendizaje




def main():

    largo_imagen = 28
    ancho_imagen = 28

    observaciones_entrenamiento = pnd.read_csv('datas/zalando/fashion-mnist_train.csv')
    observaciones_test = pnd.read_csv('datas/zalando/fashion-mnist_test.csv')

    #11 - Visualización de la fase de aprendizaje
    capa_convolucion=(observaciones_entrenamiento, observaciones_test, ancho_imagen, largo_imagen)

    #Datos de precisión (accurary)
    plt.plot(capa_convolucion.creacion_red.history['accuracy'])
    plt.plot(capa_convolucion.creacion_red.history['val_accuracy'])
    plt.title('Precisión del modelo')
    plt.ylabel('Precisión')
    plt.xlabel('Epoch')
    plt.legend(['Aprendizaje', 'Test'], loc='upper left')
    plt.show()

    #Datos de validación y error
    plt.plot(capa_convolucion.creacion_red.history['loss'])
    plt.plot(capa_convolucion.creacion_red.history['val_loss'])
    plt.title('Error')
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.legend(['Aprendizaje', 'Test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()

