import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore

# Datos de entrada (X) - 4 muestras con 3 características cada una
X = np.array([
    [2, 4, -3],
    [1, -2, 3],
    [0, 1, -1],
    [5, -3, 2]])
# Salidas esperadas (Y) - 4 muestras con 3 valores de salida cada una
Y = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [0, 1, 1]])

# Definir el modelo secuencial (funcional)
entrada = keras.Input(shape=(3,))
x = layers.Dense(5, activation='relu')(entrada)  # Capa oculta
salida = layers.Dense(3, activation='sigmoid')(x)  # Capa de salida
modelo = keras.Model(inputs=entrada, outputs=salida)

modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar la red neuronal
modelo.fit(X, Y, epochs=500, verbose=1)

# Evaluar modelo
loss, accuracy = modelo.evaluate(X, Y)
print(f"Pérdida: {loss:.4f}, Precisión: {accuracy:.4f}")

# Hacer predicciones con nuevos datos
nueva_entrada = np.array([[3, -1, 2]])
prediccion = modelo.predict(nueva_entrada)
print("Predicción:", prediccion)

# Simular predicciones binarias
print("Predicción binaria:", np.round(prediccion))

# Graficar el error
plt.plot(modelo.history.history['loss'], label='Pérdida (Loss)')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.title('Evolución de la pérdida en el entrenamiento')
plt.legend()
plt.show()

