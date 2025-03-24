import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore

# Definir el modelo secuencial
entrada = keras.Input(shape=(3,))
x = layers.Dense(5, activation='relu')(entrada)  # Capa oculta
salida = layers.Dense(3, activation='sigmoid')(x)  # Capa de salida

modelo = keras.Model(inputs=entrada, outputs=salida)

# Mostrar resumen de la red
modelo.summary()

modelo.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

