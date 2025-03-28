from sklearn.datasets import make_classification
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

# Generar dataset con 300 muestras, 5 características y 3 clases
X, y = make_classification(
    n_samples=300,
    n_features=5,
    n_classes=3,
    n_informative=4,
    n_redundant=0,
    random_state=42)

# Convertimos y a one-hot encoding para usar en redes neuronales
y_onehot = to_categorical(y)

print("Entradas (X):", X.shape)
print("Salidas (y):", y_onehot.shape)

# Definimos la arquitectura del modelo
entrada = keras.Input(shape=(5,))  # 5 características
x = layers.Dense(10, activation='relu')(entrada)
salida = layers.Dense(3, activation='softmax')(x)  # 3 clases

modelo = keras.Model(inputs=entrada, outputs=salida)

# Compilamos el modelo
modelo.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Resumen del modelo
modelo.summary()

modelo.fit(X, y_onehot, epochs=100, batch_size=16, verbose=1)

loss, accuracy = modelo.evaluate(X, y_onehot)
print(f"Pérdida final: {loss:.4f} | Precisión: {accuracy:.4f}")

# Nueva muestra
nueva_muestra = X[0].reshape(1, -1)  # Primera fila, reformateada
pred = modelo.predict(nueva_muestra)

print("Predicción (probabilidades):", pred)
print("Clase predicha:", np.argmax(pred))  # clase con mayor probabilidad
print("Clase real:", np.argmax(y_onehot[0]))  # para comparar


### Codigo para revisión 5 predicciones reales (entender funcionamiento de la red neuronal)

# Elegimos 5 muestras aleatorias del conjunto de entrenamiento
indices = np.random.choice(len(X), size=5, replace=False)

for i in indices:
    muestra = X[i].reshape(1, -1)           # Reformateamos para predicción
    pred = modelo.predict(muestra)          # Hacemos predicción
    clase_predicha = np.argmax(pred)        # Elegimos clase con mayor probabilidad
    clase_real = y[i]                       # Clase verdadera

    print(f"🧪 Muestra {i}:")
    print(f"  Probabilidades predichas: {np.round(pred[0], 3)}")
    print(f"  Clase predicha: {clase_predicha}")
    print(f"  Clase real:     {clase_real}")
    print("  ✅ Correcto ✅" if clase_predicha == clase_real else "  ❌ Incorrecto ❌")
    print("-" * 40)
