import streamlit as st
import joblib
import numpy as np

pipeline = joblib.load('pipeline_transformer.sav')
lr_model = joblib.load('lr_model.sav')
svm_model = joblib.load('svm_model.sav')
dt_model = joblib.load('dt_model.sav')


st.title("Bienvenid@ a tu app para clasificación de Flores Iris ")

st.text('A continuación, ingresa los datos de tu flor. Toma en cuenta que las medidas son en cm.')

sp_length = float(st.number_input('Largo del sépalo', min_value=0.0, max_value=20.0))
sp_width = float(st.number_input('Ancho del sépalo', min_value=0.0, max_value=20.0))
pet_length = float(st.number_input('Largo del petalo',min_value=0.0, max_value=20.0))
pet_width = float(st.number_input('Ancho del petalo',min_value=0.0, max_value=20.0))

model_option = st.selectbox("Seleccione el modelo", ["Regresión Logística", "SVM", "Árbol de Decisión"])

def get_prediction(model, features):
    features = pipeline.transform(features)
    prediction = model.predict(features)
    return prediction

btn = st.button('Classify')

st.title("Predicción de tipos de flores Iris")

# Botón para realizar la predicción
if btn:
    features = ([[sp_length, sp_width, pet_length, pet_width]])
    if model_option== "SVM":
        prediction = get_prediction(svm_model, features)
    elif model_option == "Regresión Logística":
        prediction = get_prediction(lr_model, features)
    elif model_option == "Árbol de Decisión":
        prediction = get_prediction(dt_model, features)
    st.write(f"La flor es de tipo: {prediction[0]}")