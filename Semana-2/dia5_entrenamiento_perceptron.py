import numpy as np
import matplotlib.pyplot as plt

# Datos de entrada
x = np.array([2, 4, -3])
w = np.array([[0.1, 0.3, 0.1],
              [0.2, 0.5, 0.2],
              [0.3, 0.4, 0.3]])
b = np.array([1, 2, 1])
y_real = np.array([1, 0, 1])
alpha = 0.01
epochs = 1000

errores = []

def mlp(x, w, b):
    z = np.dot(x, w) + b
    return 1 / (1 + np.exp(-z))

for epoch in range(epochs):
    salida = mlp(x, w, b)
    error = np.mean((y_real - salida) ** 2)
    errores.append(error)

    gradiente_w = -2 * np.outer(x, (y_real - salida))
    gradiente_b = -2 * (y_real - salida)

    w = w - alpha * gradiente_w
    b = b - alpha * gradiente_b

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Error = {error:.4f}")

# Graficar evolución del error
plt.plot(range(epochs), errores, label="Error de MSE")
plt.xlabel("Épocas")
plt.ylabel("Error")
plt.title("Evolución del error en entrenamiento")
plt.legend()
plt.show()
