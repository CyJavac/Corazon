import joblib
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import requests
from io import BytesIO
import numpy as np
import pandas as pd

# Cargar el modelo y el escalador
modelo = joblib.load("modelo_knn.bin")
escalador = joblib.load("escalador.bin")

def predecir():
    edad = float(entry_edad.get())
    colesterol = float(entry_colesterol.get())
    datos = pd.DataFrame([[edad, colesterol]], columns=["edad", "colesterol"])
    datos_escalados = escalador.transform(datos)
    resultado = modelo.predict(datos_escalados)[0]
    
    notebook.select(frame_resultado)
    for widget in frame_resultado.winfo_children():
        widget.destroy()
    
    if resultado == 1:
        label_resultado = tk.Label(frame_resultado, text="Tiene problema cardiaco", font=("Arial", 14), fg="red")
        imagen_url = "https://www.clinicadeloccidente.com/wp-content/uploads/sintomas-cardio-linkedin-1080x627.jpg"
    else:
        label_resultado = tk.Label(frame_resultado, text="No tiene problema cardiaco", font=("Arial", 14), fg="green")
        imagen_url = "https://s28461.pcdn.co/wp-content/uploads/2017/07/Tu-corazo%CC%81n-consejos-para-mantenerlo-sano-y-fuerte.jpg"
    
    label_resultado.pack()
    
    response = requests.get(imagen_url)
    imagen = Image.open(BytesIO(response.content))
    imagen = imagen.resize((400, 250), Image.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(imagen)
    label_imagen = tk.Label(frame_resultado, image=img_tk)
    label_imagen.image = img_tk
    label_imagen.pack()

# Crear la ventana principal
root = tk.Tk()
root.title("Asistente IA para cardiólogos")

notebook = ttk.Notebook(root)
frame_entrada = ttk.Frame(notebook)
frame_resultado = ttk.Frame(notebook)
notebook.add(frame_entrada, text="Entrada de Datos")
notebook.add(frame_resultado, text="Predicción")
notebook.pack(expand=True, fill="both")

# Pestaña de entrada de datos
titulo = tk.Label(frame_entrada, text="Asistente IA para cardiólogos", font=("Arial", 18, "bold"))
titulo.pack(pady=10)

autor = tk.Label(frame_entrada, text="Realizado por Alfredo Diaz", font=("Arial", 12))
autor.pack()

introduccion = tk.Label(frame_entrada, text="Esta aplicación permite predecir si una persona tiene problemas cardíacos con base en su edad y nivel de colesterol.", wraplength=500, justify="center")
introduccion.pack(pady=10)

label_edad = tk.Label(frame_entrada, text="Edad (18-80 años):")
label_edad.pack()
entry_edad = tk.Entry(frame_entrada)
entry_edad.pack()

label_colesterol = tk.Label(frame_entrada, text="Colesterol (50-600):")
label_colesterol.pack()
entry_colesterol = tk.Entry(frame_entrada)
entry_colesterol.pack()

btn_predecir = tk.Button(frame_entrada, text="Predecir", command=predecir)
btn_predecir.pack(pady=10)

root.mainloop()
