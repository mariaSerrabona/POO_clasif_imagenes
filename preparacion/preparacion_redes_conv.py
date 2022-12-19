import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class preparacion_capa_concolucion():

    def __init__(self,observaciones_entrenamiento,observaciones_test, ancho_imagen, largo_imagen ):
        self.observaciones_entrenamiento=observaciones_entrenamiento
        self.observaciones_test=observaciones_test
        self.ancho_imagen=ancho_imagen
        self.largo_imagen=largo_imagen

    def pixel(self):
        #Solo se guardan las características "píxeles"
        X = np.array(self.observaciones_entrenamiento.iloc[:, 1:])
        return X

    def tabla_categorias(self):
        #Se crea una tabla de categorías con la ayuda del módulo Keras
        y = to_categorical(np.array(self.observaciones_entrenamiento.iloc[:, 0]))

        return y

    def separacion_datos(self, indice):

        #Distribución de los datos de entrenamiento en datos de aprendizaje y datos de validación
        #80 % de datos de aprendizaje y 20 % de datos de validación
        X_aprendizaje, X_validacion, y_aprendizaje, y_validacion = train_test_split(self.pixel(), self.tabla_categorias(), test_size=0.2, random_state=13)


        # Se redimensionan las imágenes al formato 28*28 y se realiza una adaptación de escala en los datos de los píxeles
        X_aprendizaje = X_aprendizaje.reshape(X_aprendizaje.shape[0], self.ancho_imagen, self.largo_imagen, 1)
        X_aprendizaje = X_aprendizaje.astype('float32')
        X_aprendizaje /= 255

        # Se hace lo mismo con los datos de validación
        X_validacion = X_validacion.reshape(X_validacion.shape[0], self.ancho_imagen, self.largo_imagen, 1)
        X_validacion = X_validacion.astype('float32')
        X_validacion /= 255


        X_test = np.array(self.observaciones_test.iloc[:, 1:])
        y_test = to_categorical(np.array(self.observaciones_test.iloc[:, 0]))

        X_test = X_test.reshape(X_test.shape[0], self.ancho_imagen, self.largo_imagen, 1)
        X_test = X_test.astype('float32')
        X_test /= 255


        if indice==0:
            return X_test
        if indice==1:
            return y_test

        if indice==2:
            return X_aprendizaje
        if indice==3:
            return X_validacion

        if indice==4:
            return y_aprendizaje
        if indice==5:
            return y_validacion
