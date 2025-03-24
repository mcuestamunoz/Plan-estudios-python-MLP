import numpy as np

# Crear dos matrices aleatorias de 3x3 y calcular su producto
A = np.random.randint(0, 10, (3, 3))
B = np.random.randint(0, 10, (3, 3))
producto = np.dot(A, B)

print("Matriz A:", A)
print("Matriz B:", B)
print("Producto de A y B:", producto)
