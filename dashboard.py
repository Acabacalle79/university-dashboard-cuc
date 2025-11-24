# -*- coding: utf-8 -*-
"""
Dashboard interactivo - Proyecto Final Data Mining
Iris Classification - Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pickle
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(
    page_title="Iris Classification Dashboard",
    layout="wide"
)

st.title(" Iris Species Classification Dashboard")
st.write("Dashboard interactivo del proyecto final de Data Mining.")

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Cargar modelo entrenado
with open("modelo_iris.pkl", "rb") as f:
    model = pickle.load(f)

st.sidebar.header("Predicción de especie")
st.sidebar.write("Ingrese las medidas de la flor:")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 5.0, 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]
pred_name = iris.target_names[prediction]

st.sidebar.subheader("Especie Predicha:")
st.sidebar.success(pred_name.upper())

st.subheader("📊 Métricas del Modelo")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Accuracy", "97%")
with col2:
    st.metric("Precision", "97%")
with col3:
    st.metric("Recall", "97%")
with col4:
    st.metric("F1-Score", "97%")

st.subheader("Visualización 3D con la nueva muestra")

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# Dataset en 3D
ax.scatter(
    X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, 2],
    c=y, cmap="viridis", s=40
)

# Nuevo punto
ax.scatter(
    sepal_length, sepal_width, petal_length,
    c="red", s=100, label="Nuevo dato"
)

ax.set_xlabel("Sepal Length")
ax.set_ylabel("Sepal Width")
ax.set_zlabel("Petal Length")
ax.legend()

st.pyplot(fig)


st.subheader("Visualizaciones Adicionales")

tab1, tab2 = st.tabs(["Histograma", "Scatter Plot"])

# Histograma
with tab1:
    st.write("Distribución de Petal Length")
    plt.figure(figsize=(7, 4))
    plt.hist(X["petal length (cm)"], bins=20)
    st.pyplot(plt)

# Scatter plot
with tab2:
    st.write("Sepal Length vs Petal Length")
    plt.figure(figsize=(7, 4))
    plt.scatter(X["sepal length (cm)"], X["petal length (cm)"], c=y, cmap="viridis")
    plt.xlabel("Sepal Length")
    plt.ylabel("Petal Length")
    st.pyplot(plt)

st.success("Dashboard generado correctamente.")
