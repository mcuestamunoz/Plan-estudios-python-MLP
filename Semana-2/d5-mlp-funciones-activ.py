# 3. Implementación en NumPy
#📌 Ejercicio:
#🔹 Modifica la función mlp() para usar Sigmoide o ReLU en lugar de la función escalón.
#🔹 Permitir que el usuario elija qué función de activación usar.
    #💡 Pistas:
        #Usa np.where(z > 0, z, 0) para ReLU.
        #Usa 1 / (1 + np.exp(-z)) para Sigmoide.

import numpy as np

x = np.array([2, 4, -3])
w = np.array([[0.5, 0.7, 0.2],
             [0.2, 7, 2], 
             [3, 0.6, 0.8]])
b = np.array([5, 5, 7])

activacion = input("Elige una función de activación (sigmoide/relu): ").strip().lower()

def mlp(x, w, b, activacion):
    z = np.dot(x, w) + b

    if activacion == "sigmoide":
        salida = 1 / (1 + np.exp(-z))
    elif activacion == "relu":
        salida = np.maximum(0, z)
    else:
        print("Error: Opción no válida")
        salida = None

    print("Valor de z: {:.4f}", z)
    return salida

salida = mlp(x, w, b, activacion)
print("Salida perceptrón: {:.4f}",salida)