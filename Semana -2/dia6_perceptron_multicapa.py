import numpy as np

# Definir datos de entrada
x = np.array([2, 4, -3])
w1 = np.random.rand(3, 5)  # Pesos capa 1 (3 entradas, 5 neuronas)
b1 = np.random.rand(5)     # Bias capa 1

w2 = np.random.rand(5, 3)  # Pesos capa 2 (5 entradas, 3 neuronas)
b2 = np.random.rand(3)     # Bias capa 2

# Función de activación
def relu(z):
    return np.maximum(0, z)

# Paso hacia adelante (forward pass)
z1 = np.dot(x, w1) + b1
a1 = relu(z1)

z2 = np.dot(a1, w2) + b2
salida = relu(z2)

print("Salida final de la red multicapa:", salida)
