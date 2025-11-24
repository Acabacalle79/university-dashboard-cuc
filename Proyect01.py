# -*- coding: utf-8 -*-
"""
Proyecto Final - Data Mining
Iris Species Classification
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'


# Importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
import pickle

# Carga del dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print("Primeras filas del dataset:")
print(X.head())

# Análisis exploratorio de datos (EDA)
print("\nGenerando visualizaciones del análisis exploratorio...")

# Pairplot
sns.pairplot(
    pd.concat([X, pd.DataFrame({"species": y})], axis=1),
    hue="species"
)
plt.show()

# Heatmap de correlación
plt.figure(figsize=(10, 6))
sns.heatmap(X.corr(), annot=True, cmap="viridis")
plt.title("Matriz de correlación")
plt.show()

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrenamiento del modelo
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Cálculo de métricas
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("\n======= Métricas del Modelo =======")
print(f"Accuracy: {acc:.4f}")
print(f"Precision (weighted): {prec:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1-score (weighted): {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Guardar modelo
with open("modelo_iris.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModelo guardado como: modelo_iris.pkl")
