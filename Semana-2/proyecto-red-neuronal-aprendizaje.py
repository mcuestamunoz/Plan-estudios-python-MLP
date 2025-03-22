import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [9, 4, -3],
    [3, 1, 7],
    [5, 8, -2],
    [6, 2, 0],
    [1, 7, 3]])
w = np.random.rand(3, 3)
b = np.random.rand(1, 3) 
y_real = np.array([
    [1, 0, 1],  # Salida esperada para el primer ejemplo
    [0, 1, 0],  # Salida esperada para el segundo ejemplo
    [1, 1, 0],  # Salida esperada para el tercer ejemplo
    [0, 0, 1],  # Salida esperada para el cuarto ejemplo
    [1, 0, 0]])   # Salida esperada para el quinto ejemplo
alpha = 0.0005
epochs = 1000
errores = []

def mlp(x, w, b, activacion):
    z = np.dot(x, w) + b

    if activacion == "sigmoide":
        salida = 1 / (1 + np.exp(-z))
    elif activacion == "relu":
        salida = np.maximum(0, z)
    elif activacion == "tanh":
        salida = np.tanh(z)
    elif activacion == "leakly relu":
        salida = np.where(z > 0, z, 0.01 * z)
    else:
        print("Error: Opción no válida")
        salida = None

    return salida

activacion = input("Elige una función de activación (sigmoide/relu/tanh/leakly relu): ").strip().lower()

for epoch in range(epochs):
    salida = mlp(X, w, b, activacion)
    error = np.mean((y_real - salida) ** 2) 

    errores.append(error)

    gradiente_w = -2 * np.dot(X.T, (y_real - salida)) / len(X)
    gradiente_b = -2 * np.mean((y_real - salida), axis=0)


    w = w - alpha * gradiente_w
    b = b - alpha * gradiente_b

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Error = {error:.4f}")


plt.plot(range(epochs), errores, label="Error de MSE")
plt.xlabel("Bucles")
plt.ylabel("Error")
plt.title("Evolución del error en Entrenamiento")
plt.legend()
plt.show()
