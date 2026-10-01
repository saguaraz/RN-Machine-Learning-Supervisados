import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# 1. DATASET: Tabla de verdad de la puerta AND
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

# 2. CONSTRUCCIÓN Y ENTRENAMIENTO DE LA NEURONA
neurona = Perceptron(max_iter=100, random_state=42)
neurona.fit(X, y)

# 3. EVALUACIÓN Y PREDICCIONES
predicciones = neurona.predict(X)

print("--- TABLA DE VERDAD PROBADA POR LA NEURONA ---")
for entrada, objetivo, pred in zip(X, y, predicciones):
    print(f"Entrada: {entrada} -> Esperado: {objetivo} | Predicho: {pred}")

print(f"\nPrecisión del modelo: {accuracy_score(y, predicciones) * 100}%")

# 4. INSPECCIÓN DE PESOS APRENDIDOS
pesos = neurona.coef_[0]
sesgo = neurona.intercept_[0]

print("\n--- PARÁMETROS APRENDIDOS POR LA NEURONA ---")
print(f"Peso W1 (para Entrada 1): {pesos[0]}")
print(f"Peso W2 (para Entrada 2): {pesos[1]}")
print(f"Sesgo (Bias b): {sesgo}")