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
# RED NEURONAL DE 4 CAPAS DE CONVOLUCIONES CON UNA CANTIDAD DE IMAGENES EN AUMENTO
#
#************************************************************************************

import pandas as pnd
from preparacion.preparacion_redes_conv import preparacion_capa_concolucion
from preparacion.creacion_red_neuronal_4_capas import red_neuronal_4_capas
from keras.preprocessing.image import ImageDataGenerator
import time
import matplotlib.pyplot as plt


class capas_convolucion4():

    def __init__(self,observaciones_entrenamiento,observaciones_test, ancho_imagen, largo_imagen ):
        self.observaciones_entrenamiento=observaciones_entrenamiento
        self.observaciones_test=observaciones_test
        self.ancho_imagen=ancho_imagen
        self.largo_imagen=largo_imagen


    def preparacion_red(self):
        preparacion=preparacion_capa_concolucion(self.observaciones_entrenamiento,self.observaciones_test, self.ancho_imagen, self.largo_imagen)
        return preparacion


    def generador_imagenes(self, indice):
        red_neuronal_4_capas=red_neuronal_4_capas.creacion()
        preparacion=preparacion_capa_concolucion(self.observaciones_entrenamiento,self.observaciones_test, self.ancho_imagen, self.largo_imagen)
        #9 - Aumento de la cantidad de imágenes
        generador_imagenes = ImageDataGenerator(rotation_range=8,
                                width_shift_range=0.08,
                                shear_range=0.3,
                                height_shift_range=0.08,
                                zoom_range=0.08)
        if indice ==0:
            nuevas_imagenes_aprendizaje = generador_imagenes.flow(preparacion.separacion_datos(2), preparacion.separacion_datos(4), batch_size=256)
            return nuevas_imagenes_aprendizaje
        if indice ==1:
            nuevas_imagenes_validacion = generador_imagenes.flow(preparacion.separacion_datos(3), preparacion.separacion_datos(5), batch_size=256)
            return nuevas_imagenes_validacion


#10 - Aprendizaje
    def aprendizaje(self):
        #start = time.clock()
        historico_aprendizaje = red_neuronal_4_capas.creacion().fit_generator(self.generador_imagenes(0),
                                                        steps_per_epoch=48000//256,
                                                        epochs=50,
                                                        validation_data=self.generador_imagenes(1),
                                                        validation_steps=12000//256,
                                                        use_multiprocessing=False,
                                                        verbose=1 )

        #stop = time.clock()
        #print("Tiempo de aprendizaje = "+str(stop-start))
        return historico_aprendizaje

    def evaluacion(self):
        preparacion=preparacion_capa_concolucion(self.observaciones_entrenamiento,self.observaciones_test, self.ancho_imagen, self.largo_imagen)

        #11 - Evaluación del modelo
        evaluacion = red_neuronal_4_capas.creacion().evaluate(preparacion.separacion_datos(0), preparacion.separacion_datos(1), verbose=0)
        print('Error:', evaluacion[0])
        print('Precisión:', evaluacion[1])


#12 - Visualización de la fase de aprendizaje
def main():
    largo_imagen = 28
    ancho_imagen = 28

    #Carga de los datos de entrenamiento
    observaciones_entrenamiento = pnd.read_csv('datas/zalando/fashion-mnist_train.csv')
    observaciones_test = pnd.read_csv('datas/zalando/fashion-mnist_test.csv')
    capas_conv_aumentada4=capas_convolucion4(observaciones_entrenamiento, observaciones_test, ancho_imagen, largo_imagen)

    #Datos de precisión (accuracy)
    plt.plot(capas_conv_aumentada4.aprendizaje().history['accuracy'])
    plt.plot(capas_conv_aumentada4.aprendizaje().history['val_accuracy'])
    plt.title('Precisión del modelo')
    plt.ylabel('Precisión')
    plt.xlabel('Epoch')
    plt.legend(['Aprendizaje', 'Test'], loc='upper left')
    plt.show()

    #Datos de validación y error
    plt.plot(capas_conv_aumentada4.aprendizaje().history['loss'])
    plt.plot(capas_conv_aumentada4.aprendizaje().history['val_loss'])
    plt.title('Error')
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.legend(['Aprendizaje', 'Test'], loc='upper left')
    plt.show()


    #Guardado del modelo
    # serializar modelo a JSON
    modelo_json = red_neuronal_4_capas.creacion().to_json()
    with open("modelo/modelo_4convoluciones.json", "w") as json_file:
        json_file.write(modelo_json)

    # serializar pesos a HDF5
    red_neuronal_4_capas.creacion().save_weights("modelo/modelo_4convoluciones.h5")
    print("¡Modelo guardado!")
