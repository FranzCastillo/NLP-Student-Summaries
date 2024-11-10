"""
 * Nombre: dashboard.py
 * Autores:
    - Francisco Castillo, 21562
    - Andrés Montoya, 21552
    - Fernanda Esquivel, 21542
    - Diego Lemus, 21469
 * Descripción: 
 * Lenguaje: Python
 * Recursos: Streamlit, Numpy, BERT, ROUGE
 * Historial: 
    - Creado el 10/11/2024
    - Modificado el 10/11/2024
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer

# Configuración de la paleta de colores y estilos
st.set_page_config(page_title="Evaluación Automática de Resúmenes")
st.markdown(
    """
    <style>
    .main {
        background-color: #CAD6E1;
    }
    .reportview-container {
        color: #08519C;
    }
    .title {
        text-align: center;
        font-size: 32px;
        color: #238B45;
        font-weight: bold;
    }
    .container {
        background-color: #DEEBF7;
        padding: 15px;
        border-radius: 10px;
        color: #08519C;
    }
    .button {
        background-color: #379656;
        color: white;
        border-radius: 5px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

#Título del dashboard
st.markdown('<p class="title">Evaluación Automática de Resúmenes</p>', unsafe_allow_html=True)

#Cargar el modelo BERT y tokenizer
@st.cache_resource
def load_model():
    model = TFBertForSequenceClassification.from_pretrained("../src/bert_summary_model")  # Ruta a la carpeta del modelo
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

model, tokenizer = load_model()

#Función para evaluar con BERT
def evaluate_with_bert(model, tokenizer, text, summary):
    #Tokenizar y preparar los datos de entrada para BERT
    inputs = tokenizer(text + " [SEP] " + summary, return_tensors="tf", max_length=512, truncation=True, padding="max_length")
    outputs = model(inputs)
    
    #Extraer puntuaciones de content y wording
    scores = outputs.logits.numpy().flatten()
    return scores[0], scores[1]

#Configuración de layout (tamaño de columnas)
col1, col2, col3 = st.columns([4, 4, 2])

with col1:
    st.subheader("Texto a resumir")
    original_file = st.file_uploader("Cargar archivo:", key="original")
    if original_file is not None:
        original_text = original_file.read().decode("utf-8")
    else:
        original_text = st.text_area("Escriba el texto completo aquí")

with col2:
    st.subheader("Resumen Realizado")
    summary_file = st.file_uploader("Cargar archivo:", key="summary")
    if summary_file is not None:
        summary_text = summary_file.read().decode("utf-8")
    else:
        summary_text = st.text_area("Escriba el resumen aquí")

with col3:
    st.subheader("Selección del modelo")
    model_choice = st.selectbox("Evaluar con:", ["BERT", "ROUGE", "Ambos"])

#Botón para evaluar
if st.button("Evaluar", key="evaluate", help="Haz clic para evaluar el resumen"):
    if original_text and summary_text:
        if model_choice in ["BERT", "Ambos"]: #Evaluar con BERT o ambos
            bert_content_score, bert_wording_score = evaluate_with_bert(model, tokenizer, original_text, summary_text)
            st.subheader("Puntuación BERT")
            st.write(f"Content Score: {bert_content_score:.2f}")
            st.write(f"Wording Score: {bert_wording_score:.2f}")
    else:
        st.warning("Por favor, cargue o ingrese ambos textos para evaluar.")
