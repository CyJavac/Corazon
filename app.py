import joblib
import streamlit as st
import requests
from PIL import Image
import numpy as np
import pandas as pd
from io import BytesIO

# Cargar el modelo y el escalador
modelo = joblib.load("modelo_knn.bin")
escalador = joblib.load("escalador.bin")

def predecir(edad, colesterol):
    datos = pd.DataFrame([[edad, colesterol]], columns=["edad", "colesterol"])
    datos_escalados = escalador.transform(datos)
    resultado = modelo.predict(datos_escalados)[0]
    return resultado

# Interfaz de usuario con Streamlit
st.title("Asistente IA para cardiólogos")
st.write("### Realizado por Alfredo Diaz")

# Pestaña de entrada de datos
tab1, tab2 = st.tabs(["Entrada de Datos", "Predicción"])

with tab1:
    st.write("Esta aplicación permite predecir si una persona tiene problemas cardíacos con base en su edad y nivel de colesterol.")
    
    edad = st.number_input("Edad (18-80 años):", min_value=18, max_value=80, step=1)
    colesterol = st.number_input("Colesterol (50-600):", min_value=50, max_value=600, step=1)
    
    if st.button("Predecir"):
        resultado = predecir(edad, colesterol)
        st.session_state.resultado = resultado
        tab2.select()

with tab2:
    if "resultado" in st.session_state:
        resultado = st.session_state.resultado
        if resultado == 1:
            st.write("## Tiene problema cardiaco", unsafe_allow_html=True)
            imagen_url = "https://www.clinicadeloccidente.com/wp-content/uploads/sintomas-cardio-linkedin-1080x627.jpg"
        else:
            st.write("## No tiene problema cardiaco", unsafe_allow_html=True)
            imagen_url = "https://s28461.pcdn.co/wp-content/uploads/2017/07/Tu-corazo%CC%81n-consejos-para-mantenerlo-sano-y-fuerte.jpg"
        
        response = requests.get(imagen_url)
        imagen = Image.open(BytesIO(response.content))
        st.image(imagen, use_column_width=True)
