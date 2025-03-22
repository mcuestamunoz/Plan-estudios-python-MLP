import numpy as np
import matplotlib.pyplot as plt

# Datos de entrada y salida esperada
x = np.array([2, 4, -3])
y_real = np.array([1, 0, 1])

# Inicialización de pesos y bias para una red multicapa
w1 = np.random.rand(3, 5)  # 3 entradas, 5 neuronas en capa oculta
b1 = np.random.rand(5)

w2 = np.random.rand(5, 3)  # 5 neuronas en capa oculta, 3 en salida
b2 = np.random.rand(3)

alpha = 0.01
epochs = 1000
errores = []

# Función de activación ReLU y derivada
def relu(z):
    return np.maximum(0, z)

def relu_derivada(z):
    return (z > 0).astype(float)

# Entrenamiento de la red
for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(x, w1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, w2) + b2
    salida = relu(z2)

    # Cálculo del error
    error = np.mean((y_real - salida) ** 2)
    errores.append(error)

    # Backpropagation (gradientes)
    delta2 = -2 * (y_real - salida) * relu_derivada(z2)
    gradiente_w2 = np.outer(a1, delta2)
    gradiente_b2 = delta2

    delta1 = np.dot(delta2, w2.T) * relu_derivada(z1)
    gradiente_w1 = np.outer(x, delta1)
    gradiente_b1 = delta1

    # Actualización de pesos y bias
    w2 -= alpha * gradiente_w2
    b2 -= alpha * gradiente_b2
    w1 -= alpha * gradiente_w1
    b1 -= alpha * gradiente_b1

    # Mostrar error cada 100 épocas
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Error = {error:.4f}")

# Gráfico de evolución del error
plt.plot(range(epochs), errores, label="Error MSE")
plt.xlabel("Épocas")
plt.ylabel("Error")
plt.title("Evolución del error en el entrenamiento MLP")
plt.legend()
plt.show()
