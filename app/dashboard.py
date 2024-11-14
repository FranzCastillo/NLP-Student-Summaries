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

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from rouge_score import rouge_scorer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import TFBertForSequenceClassification, BertTokenizer

# Constants
MODEL_PATH = "../src/bert_summary_model"
PREDICTIONS_FILE = "../src/BERT_predictions.csv"


# Configure Streamlit page
def configure_page():
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


configure_page()


# Load BERT model and tokenizer
@st.cache_resource
def load_bert_model() -> Tuple[TFBertForSequenceClassification, BertTokenizer]:
    model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer


model, tokenizer = load_bert_model()


# Evaluate text and summary with BERT
def evaluate_with_bert(model: TFBertForSequenceClassification, tokenizer: BertTokenizer, text: str, summary: str) -> \
        Tuple[float, float]:
    inputs = tokenizer(text + " [SEP] " + summary, return_tensors="tf", max_length=512, truncation=True,
                       padding="max_length")
    outputs = model(inputs)
    scores = outputs.logits.numpy().flatten()
    return scores[0], scores[1]


# Load predictions from CSV
@st.cache_data
def load_predictions() -> Tuple[np.ndarray, np.ndarray]:
    df_predictions = pd.read_csv(PREDICTIONS_FILE)
    test_targets = df_predictions[['Real_Content', 'Real_Wording']].values
    pred_labels = df_predictions[['Predicted_Content', 'Predicted_Wording']].values
    return test_targets, pred_labels


scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


# Get ROUGE scores
def calculate_rouge_scores(text: str, summary: str) -> pd.DataFrame:
    scores = scorer.score(text, summary)
    scores_dict = {
        'rouge1_recall': scores['rouge1'].recall,
        'rouge1_precision': scores['rouge1'].precision,
        'rouge1_f1': scores['rouge1'].fmeasure,
        'rouge2_recall': scores['rouge2'].recall,
        'rouge2_precision': scores['rouge2'].precision,
        'rouge2_f1': scores['rouge2'].fmeasure,
        'rougeL_recall': scores['rougeL'].recall,
        'rougeL_precision': scores['rougeL'].precision,
        'rougeL_f1': scores['rougeL'].fmeasure
    }
    return pd.DataFrame(scores_dict, index=[0])


# Navigation menu
menu_option = st.sidebar.radio("Selecciona una sección:",
                               ["Evaluación de Resúmenes", "Desempeño de los Modelos", "Métricas de los modelos", "Análisis de Resúmenes"])

if menu_option == "Evaluación de Resúmenes":
    st.markdown('<p class="title">Evaluación Automática de Resúmenes</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([4, 4, 2])

    with col1:
        st.subheader("Texto a resumir")
        original_file = st.file_uploader("Cargar archivo:", key="original")
        original_text = original_file.read().decode("utf-8") if original_file else st.text_area(
            "Escriba el texto completo aquí")

    with col2:
        st.subheader("Resumen Realizado")
        summary_file = st.file_uploader("Cargar archivo:", key="summary")
        summary_text = summary_file.read().decode("utf-8") if summary_file else st.text_area("Escriba el resumen aquí")

    with col3:
        st.subheader("Selección del modelo")
        model_choice = st.selectbox("Evaluar con:", ["BERT", "ROUGE", "Ambos"])

    if st.button("Evaluar", key="evaluate", help="Haz clic para evaluar el resumen"):
        if original_text and summary_text:
            if model_choice == "BERT":
                bert_content_score, bert_wording_score = evaluate_with_bert(model, tokenizer, original_text,
                                                                            summary_text)
                st.subheader("Puntuación BERT")
                st.write(f"Content Score: {bert_content_score:.2f}")
                st.write(f"Wording Score: {bert_wording_score:.2f}")
            elif model_choice == "ROUGE":
                rouge_scores = calculate_rouge_scores(original_text, summary_text)
                st.subheader("Puntuación ROUGE")
                st.write(rouge_scores)
            elif model_choice == "Ambos":
                bert_content_score, bert_wording_score = evaluate_with_bert(model, tokenizer, original_text,
                                                                            summary_text)
                rouge_scores = calculate_rouge_scores(original_text, summary_text)
                st.subheader("Puntuación BERT")
                st.write(f"Content Score: {bert_content_score:.2f}")
                st.write(f"Wording Score: {bert_wording_score:.2f}")
                st.subheader("Puntuación ROUGE")
                st.write(rouge_scores)
        else:
            st.warning("Por favor, cargue o ingrese ambos textos para evaluar.")

elif menu_option == "Desempeño de los Modelos":
    st.markdown('<p class="title">Desempeño de los Modelos</p>', unsafe_allow_html=True)

    df_predictions = pd.read_csv(PREDICTIONS_FILE)

    content_scores = df_predictions['Predicted_Content']
    wording_scores = df_predictions['Predicted_Wording']

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].violinplot(content_scores, showmeans=True)
    ax[0].set_title("Distribución de Predicted_Content")
    ax[0].set_ylabel("Puntaje Predicho (Content)")

    ax[1].violinplot(wording_scores, showmeans=True)
    ax[1].set_title("Distribución de Predicted_Wording")
    ax[1].set_ylabel("Puntaje Predicho (Wording)")

    fig.suptitle("Comparación de Distribuciones de Puntajes: Content y Wording")
    st.pyplot(fig)
    

elif menu_option == "Métricas de los modelos":
    st.markdown('<p class="title">Métricas de los modelos</p>', unsafe_allow_html=True)
    st.sidebar.subheader("Métricas")
    metric_choice = st.sidebar.radio("Selecciona la métrica:", ["MSE", "MAE", "R²"])
    test_targets, pred_labels = load_predictions()

    if metric_choice == "MSE":
        mse_content = mean_squared_error(test_targets[:, 0], pred_labels[:, 0])
        mse_wording = mean_squared_error(test_targets[:, 1], pred_labels[:, 1])
        metric_description = "Esta métrica muestra el Error Cuadrático Medio, lo que significa que mide el promedio de los errores al cuadrado entre los valores reales y las predicciones."
        st.write(f"**MSE (Content):** {mse_content:.4f}")
        st.write(f"**MSE (Wording):** {mse_wording:.4f}")
        squared_errors_content = np.square(test_targets[:, 0] - pred_labels[:, 0])
        squared_errors_wording = np.square(test_targets[:, 1] - pred_labels[:, 1])
        st.subheader("Histograma de Errores Cuadrados (MSE)")
        fig_mse, (ax_mse_content, ax_mse_wording) = plt.subplots(1, 2, figsize=(12, 5))
        ax_mse_content.hist(squared_errors_content, bins=20, color='#08519C', edgecolor='white')
        ax_mse_content.set_title("Errores Cuadrados - Content")
        ax_mse_content.set_xlabel("Error Cuadrado")
        ax_mse_content.set_ylabel("Frecuencia")
        ax_mse_wording.hist(squared_errors_wording, bins=20, color='#238B45', edgecolor='white')
        ax_mse_wording.set_title("Errores Cuadrados - Wording")
        ax_mse_wording.set_xlabel("Error Cuadrado")
        ax_mse_wording.set_ylabel("Frecuencia")
        st.pyplot(fig_mse)

    elif metric_choice == "MAE":
        mae_content = mean_absolute_error(test_targets[:, 0], pred_labels[:, 0])
        mae_wording = mean_absolute_error(test_targets[:, 1], pred_labels[:, 1])
        metric_description = "Esta métrica muestra el Error Absoluto Medio, que representa el promedio de los errores absolutos entre los valores reales y las predicciones."
        st.write(f"**MAE (Content):** {mae_content:.4f}")
        st.write(f"**MAE (Wording):** {mae_wording:.4f}")
        abs_errors_content = np.abs(test_targets[:, 0] - pred_labels[:, 0])
        abs_errors_wording = np.abs(test_targets[:, 1] - pred_labels[:, 1])
        st.subheader("Histograma de Errores Absolutos (MAE)")
        fig_mae, (ax_mae_content, ax_mae_wording) = plt.subplots(1, 2, figsize=(12, 5))
        ax_mae_content.hist(abs_errors_content, bins=20, color='#08519C', edgecolor='white')
        ax_mae_content.set_title("Errores Absolutos - Content")
        ax_mae_content.set_xlabel("Error Absoluto")
        ax_mae_content.set_ylabel("Frecuencia")
        ax_mae_wording.hist(abs_errors_wording, bins=20, color='#238B45', edgecolor='white')
        ax_mae_wording.set_title("Errores Absolutos - Wording")
        ax_mae_wording.set_xlabel("Error Absoluto")
        ax_mae_wording.set_ylabel("Frecuencia")
        st.pyplot(fig_mae)

    else:  # R^2
        r2_content = r2_score(test_targets[:, 0], pred_labels[:, 0])
        r2_wording = r2_score(test_targets[:, 1], pred_labels[:, 1])
        metric_description = "Esta métrica muestra el Coeficiente de Determinación (R²), que indica qué tan bien se ajustan las predicciones a los valores reales."
        st.write(f"**R² (Content):** {r2_content:.4f}")
        st.write(f"**R² (Wording):** {r2_wording:.4f}")
        residuals_content = test_targets[:, 0] - pred_labels[:, 0]
        residuals_wording = test_targets[:, 1] - pred_labels[:, 1]
        st.subheader("Histograma de Residuos (R²)")
        fig_r2, (ax_r2_content, ax_r2_wording) = plt.subplots(1, 2, figsize=(12, 5))
        ax_r2_content.hist(residuals_content, bins=20, color='#08519C', edgecolor='white')
        ax_r2_content.set_title("Residuos - Content")
        ax_r2_content.set_xlabel("Residuo")
        ax_r2_content.set_ylabel("Frecuencia")
        ax_r2_wording.hist(residuals_wording, bins=20, color='#238B45', edgecolor='white')
        ax_r2_wording.set_title("Residuos - Wording")
        ax_r2_wording.set_xlabel("Residuo")
        ax_r2_wording.set_ylabel("Frecuencia")
        st.pyplot(fig_r2)

    st.sidebar.write(metric_description)

elif menu_option == "Análisis de Resúmenes":
    st.markdown('<p class="title">Análisis de Resúmenes</p>', unsafe_allow_html=True)
    st.sidebar.subheader("Filtros de Resúmenes")
    wording_range = st.sidebar.slider(
        "Rango de Wording Score",
        0.0,  # Min value
        1.0,  # Max value
        (0.0, 1.0),  # Default value
        0.01  # Step
    )
    content_range = st.sidebar.slider("Rango de Content Score", 0.0, 1.0, (0.0, 1.0), 0.01)
    length_range = st.sidebar.slider(
        "Rango de Longitud del Resumen",
        114,  # Valores obtenidos desde results.ipnyb
        3940,
        (114, 3940), 10
    )

    # Load data
    df_merged = pd.read_csv('../results/rouge_scores.csv')

    if 'summary_length' not in df_merged.columns:
        df_merged['summary_length'] = df_merged['text'].apply(len)

    # Filter data
    df_filtered = df_merged[
        (df_merged['wording'] >= wording_range[0]) & (df_merged['wording'] <= wording_range[1]) &
        (df_merged['content'] >= content_range[0]) & (df_merged['content'] <= content_range[1]) &
        (df_merged['summary_length'] >= length_range[0]) & (df_merged['summary_length'] <= length_range[1])
    ]

    st.subheader("Resúmenes Filtrados")
    st.write(df_filtered[['text']])
