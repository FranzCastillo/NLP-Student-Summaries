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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
            background-color: #CAD6E1;  /* Columbia Blue */
        }
        .reportview-container {
            color: #08519C;  /* Polynesian Blue */
        }
        .title {
            text-align: center;
            font-size: 32px;
            color: #238B45;  /* Sea Green */
            font-weight: bold;
        }
        .container {
            background-color: #DEEBF7;  /* Alice Blue */
            padding: 15px;
            border-radius: 10px;
            color: #08519C;  /* Polynesian Blue */
        }
        .button {
            background-color: #379656;  /* Shamrock Green */
            color: white;
            border-radius: 5px;
            font-weight: bold;
        }
        .stRadio > label {
            color: #379656;  /* Shamrock Green */
        }
        div[role="radiogroup"] label {
            color: #379656 !important;  /* Shamrock Green */
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
    metric_choice = st.sidebar.radio("Selecciona la gráfica:", ["Distribución", "Dispersión"])
    test_targets, pred_labels = load_predictions()

    if metric_choice == "Distribución":
        metric_description = "Este gráfico de violín muestra de manera comparativa la distribución de puntajes predichos, donde se observa la concentración de valores en diferentes puntos del rango de puntajes."
        df_predictions = pd.read_csv(PREDICTIONS_FILE)

        content_scores = df_predictions['Predicted_Content']
        wording_scores = df_predictions['Predicted_Wording']

        fig = go.Figure()

        fig.add_trace(go.Violin(y=content_scores, name="Predicted_Content", box_visible=True, meanline_visible=True))
        fig.add_trace(go.Violin(y=wording_scores, name="Predicted_Wording", box_visible=True, meanline_visible=True))

        fig.update_layout(
            title="Distribuciones de Puntajes: Content y Wording",
            yaxis_title="Puntaje Predicho",
            xaxis_title="Tipo de Puntaje",
        )

        st.plotly_chart(fig)
    
    elif metric_choice == "Dispersión":
        metric_description = "Este gráfico muestra cómo se comportan las predicciones del modelo en relación con los valores reales, permitiendo visualizar patrones y sesgos en las predicciones."
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Content", "Wording"))

        fig.add_trace(go.Scatter(x=test_targets[:, 0], y=pred_labels[:, 0],
                                mode='markers',
                                name="Content",
                                marker=dict(color="blue", opacity=0.6)),
                    row=1, col=1)
        fig.update_xaxes(title_text="Valor Real (Content)", row=1, col=1)
        fig.update_yaxes(title_text="Predicción (Content)", row=1, col=1)

        fig.add_trace(go.Scatter(x=test_targets[:, 1], y=pred_labels[:, 1],
                                mode='markers',
                                name="Wording",
                                marker=dict(color="green", opacity=0.6)),
                    row=1, col=2)
        fig.update_xaxes(title_text="Valor Real (Wording)", row=1, col=2)
        fig.update_yaxes(title_text="Predicción (Wording)", row=1, col=2)

        fig.update_layout(title="Distribución de Predicciones vs Valores Reales", showlegend=False)
        st.plotly_chart(fig)
        
    st.sidebar.write(metric_description)

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
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Errores Cuadrados - Content", "Errores Cuadrados - Wording"))
        fig.add_trace(go.Histogram(x=squared_errors_content, name="Content", nbinsx=20), row=1, col=1)
        fig.add_trace(go.Histogram(x=squared_errors_wording, name="Wording", nbinsx=20), row=1, col=2)
        fig.update_layout(title="Histograma de Errores Cuadrados (MSE)", xaxis_title="Error Cuadrado", yaxis_title="Frecuencia")
        st.plotly_chart(fig)

    elif metric_choice == "MAE":
        mae_content = mean_absolute_error(test_targets[:, 0], pred_labels[:, 0])
        mae_wording = mean_absolute_error(test_targets[:, 1], pred_labels[:, 1])
        metric_description = "Esta métrica muestra el Error Absoluto Medio, que representa el promedio de los errores absolutos entre los valores reales y las predicciones."
        st.write(f"**MAE (Content):** {mae_content:.4f}")
        st.write(f"**MAE (Wording):** {mae_wording:.4f}")
        abs_errors_content = np.abs(test_targets[:, 0] - pred_labels[:, 0])
        abs_errors_wording = np.abs(test_targets[:, 1] - pred_labels[:, 1])
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Errores Absolutos - Content", "Errores Absolutos - Wording"))
        fig.add_trace(go.Histogram(x=abs_errors_content, name="Content", nbinsx=20), row=1, col=1)
        fig.add_trace(go.Histogram(x=abs_errors_wording, name="Wording", nbinsx=20), row=1, col=2)
        fig.update_layout(title="Histograma de Errores Absolutos (MAE)", xaxis_title="Error Absoluto", yaxis_title="Frecuencia")
        st.plotly_chart(fig)

    else:  # R^2
        r2_content = r2_score(test_targets[:, 0], pred_labels[:, 0])
        r2_wording = r2_score(test_targets[:, 1], pred_labels[:, 1])
        metric_description = "Esta métrica muestra el Coeficiente de Determinación (R²), que indica qué tan bien se ajustan las predicciones a los valores reales."
        st.write(f"**R² (Content):** {r2_content:.4f}")
        st.write(f"**R² (Wording):** {r2_wording:.4f}")
        residuals_content = test_targets[:, 0] - pred_labels[:, 0]
        residuals_wording = test_targets[:, 1] - pred_labels[:, 1]
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Residuos - Content", "Residuos - Wording"))
        fig.add_trace(go.Histogram(x=residuals_content, name="Content", nbinsx=20), row=1, col=1)
        fig.add_trace(go.Histogram(x=residuals_wording, name="Wording", nbinsx=20), row=1, col=2)
        fig.update_layout(title="Histograma de Residuos (R²)", xaxis_title="Residuo", yaxis_title="Frecuencia")
        st.plotly_chart(fig)

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
