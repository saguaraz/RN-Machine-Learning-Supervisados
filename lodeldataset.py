from sklearn.datasets import load_iris

# Carga el dataset
iris = load_iris()

# Accede a las características y etiquetas
print(iris.feature_names) # Muestra los nombres de las columnas
print(iris.target_names)  # Muestra: ['setosa', 'versicolor', 'virginica']
print(iris.data[0])       # Muestra las medidas de la primera flor: [5.1, 3.5, 1.4, 0.2]
print(iris.target[0])