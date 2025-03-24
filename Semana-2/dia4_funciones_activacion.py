import numpy as np

# Valores de entrada
x = np.array([2, 3, 5])
w = np.array([0.4, 0.2, 0.9])
b = 3

# Cálculo de z
z = np.dot(x, w) + b

# Funciones de activación
relu = np.maximum(0, z)
tanh = np.tanh(z)
sigmoide = 1 / (1 + np.exp(-z))

print("Valor de z:", z)
print("ReLU:", relu)
print("Tanh:", tanh)
print("Sigmoide:", sigmoide)
