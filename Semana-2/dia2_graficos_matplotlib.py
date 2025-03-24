import numpy as np
import matplotlib.pyplot as plt

# Graficar la función y = x^2 en un rango de -10 a 10
x = np.linspace(-10, 10, 100)
y = x**2

plt.plot(x, y, label="y = x^2")
plt.xlabel("Eje X")
plt.ylabel("Eje Y")
plt.title("Gráfico de y = x^2")
plt.legend()
plt.show()
