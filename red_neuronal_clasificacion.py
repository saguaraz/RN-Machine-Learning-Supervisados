import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. CARGA Y EXPLORACIÓN DEL DATASET
# ==========================================
print("--- 1. Carga de datos etiquetados ---")
iris = load_iris()

# Convertimos a DataFrame para mostrarlo como tabla estructurada
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['especie'] = df['target'].map({0: iris.target_names[0], 1: iris.target_names[1], 2: iris.target_names[2]})

print("Primeras filas del conjunto de datos:")
print(df.head(), "\n")

# Separar Variables de Entrada (X) y Etiqueta/Target (y)
X = iris.data
y = iris.target

# ==========================================
# 2. PREPARACIÓN DE DATOS (Train/Test & Escalado)
# ==========================================
# División: 80% entrenamiento, 20% evaluación
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Las Redes Neuronales son sensibles a la escala de las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. CONSTRUCCIÓN Y ENTRENAMIENTO DE LA RED NEURONAL
# ==========================================
print("--- 2. Entrenando el Perceptrón Multicapa (MLP) ---")
# Red Neuronal con 1 capa oculta de 10 neuronas
mlp = MLPClassifier(
    hidden_layer_sizes=(10,), 
    activation='relu', 
    solver='adam', 
    max_iter=500, 
    random_state=42
)

# Entrenar la red
mlp.fit(X_train_scaled, y_train)
print("¡Modelo entrenado exitosamente!\n")

# ==========================================
# 4. EVALUACIÓN DEL MODELO
# ==========================================
y_pred = mlp.predict(X_test_scaled)

print("--- 3. Reporte de Clasificación ---")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ==========================================
# 5. VISUALIZACIÓN DE RESULTADOS
# ==========================================
plt.figure(figsize=(12, 5))

# Gráfico 1: Curva de Pérdida (Pérdida/Loss durante las épocas)
plt.subplot(1, 2, 1)
plt.plot(mlp.loss_curve_, color='blue', lw=2)
plt.title("Curva de Aprendizaje (Pérdida)")
plt.xlabel("Épocas / Iteraciones")
plt.ylabel("Pérdida (Loss)")
plt.grid(True)

# Gráfico 2: Matriz de Confusión
plt.subplot(1, 2, 2)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Realidad")

plt.tight_layout()
plt.show()

# ==========================================
# 6. PREDICCIÓN CON NUEVOS DATOS
# ==========================================
print("--- 4. Prueba con una nueva muestra desconocida ---")
# Muestra: [sepal length, sepal width, petal length, petal width]
nueva_flor = np.array([[5.1, 3.5, 1.4, 0.2]]) 
nueva_flor_scaled = scaler.transform(nueva_flor)

prediccion = mlp.predict(nueva_flor_scaled)
probabilidades = mlp.predict_proba(nueva_flor_scaled)

print(f"Medidas de la flor: {nueva_flor[0]}")
print(f"Especie predicha: {iris.target_names[prediccion[0]]}")
print(f"Probabilidades por clase: {probabilidades[0]}")