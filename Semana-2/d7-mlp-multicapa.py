import numpy as np
import matplotlib.pyplot as plt

# ✅ Datos de entrada (5 ejemplos con 3 características cada uno)
X = np.array([
    [9, 4, -3],
    [3, 1, 7],
    [5, 8, -2],
    [6, 2, 0],
    [1, 7, 3]])

# ✅ Inicializamos pesos y sesgos aleatorios
input_size = 3    # Número de características en X
hidden_size = 4   # Número de neuronas en la capa oculta
output_size = 3   # Número de neuronas en la capa de salida

W1 = np.random.rand(input_size, hidden_size)  # Pesos de entrada → capa oculta
b1 = np.random.rand(1, hidden_size)  # Bias de la capa oculta

W2 = np.random.rand(hidden_size, output_size)  # Pesos de capa oculta → salida
b2 = np.random.rand(1, output_size)  # Bias de la capa de salida

# ✅ Definimos la salida esperada (y_real)
y_real = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 0]])

alpha = 0.001
epochs = 1000
errores = []

# ✅ Función MLP con una capa oculta
def mlp(X, W1, b1, W2, b2, activacion):
    # Capa oculta
    z1 = np.dot(X, W1) + b1

    if activacion == "sigmoide":
        a1 = 1 / (1 + np.exp(-z1)) # asignamos a1 a activación 1
    elif activacion == "relu":
        a1 = np.maximum(0, z1)
    elif activacion == "tanh":
        a1 = np.tanh(z1)
    elif activacion == "leaky relu":
        a1 = np.where(z1 > 0, z1, 0.01 * z1)
    else:
        print("Error: Opción no válida")
        return None

    # Capa de salida (equivalente a a2 o activación 2)
    z2 = np.dot(a1, W2) + b2
    salida = 1 / (1 + np.exp(-z2))  # Usamos sigmoide en la salida

    return a1, salida

# ✅ Elegir función de activación
activacion = input("Elige una función de activación (sigmoide/relu/tanh/leaky relu): ").strip().lower()

# ✅ Bucle de entrenamiento
for epoch in range(epochs):
    a1, salida = mlp(X, W1, b1, W2, b2, activacion)
    error = np.mean((y_real - salida) ** 2)  # MSE

    errores.append(error)

    # ✅ Gradiente para capa de salida
    gradiente_w2 = -2 * np.dot(a1.T, (y_real - salida)) / len(X)
    gradiente_b2 = -2 * np.mean((y_real - salida), axis=0)

    # ✅ Gradiente para capa oculta
    delta_hidden = np.dot((y_real - salida), W2.T) * (a1 * (1 - a1))  # Derivada de sigmoide
    gradiente_w1 = -2 * np.dot(X.T, delta_hidden) / len(X)
    gradiente_b1 = -2 * np.mean(delta_hidden, axis=0)

    # ✅ Actualización de pesos y sesgos
    W2 = W2 - alpha * gradiente_w2
    b2 = b2 - alpha * gradiente_b2
    W1 = W1 - alpha * gradiente_w1
    b1 = b1 - alpha * gradiente_b1

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Error = {error:.4f}")

# ✅ Graficar evolución del error
plt.plot(range(epochs), errores, label="Error de MSE")
plt.xlabel("Épocas")
plt.ylabel("Error")
plt.title("Evolución del Error en Entrenamiento")
plt.legend()
plt.grid()
plt.show()
