from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras


class RedNeuronal():

    def __init__(self, ancho_imagen, largo_imagen):
        self.ancho_imagen = ancho_imagen
        self.largo_imagen=largo_imagen

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

        return redNeurona1Convolucion