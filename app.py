import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification



def inference(text, prob=False):

    tokenizer = AutoTokenizer.from_pretrained("1Arhat/SYAS1-PTBR")
    model = AutoModelForSequenceClassification.from_pretrained("1Arhat/SYAS1-PTBR")

    # Processar seu texto
    inputs = tokenizer(text, return_tensors="pt")

    # Previsões do modelo

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    if prob:
        probs = F.softmax(logits, dim=1)  # Calcula as probabilidades
        labels = ["Negativo", "Neutro", "Positivo"]

        resultado = {label: probs[0][idx].item() for idx, label in enumerate(labels)}
        return resultado  # Retorna um dicionário
 
    else:
        return torch.argmax(logits, dim=1).item() # Retorna a classe

st.header("Análise de Sentimentos com SYAS1-PTBR")

if "placeholder" not in st.session_state:
    st.session_state.placeholder = None  # Ou um valor padrão

# Agora pode usar sem erro
placeholder = st.session_state.placeholder


if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

text_input = st.text_input(
        "Digite algo",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
        placeholder=st.session_state.placeholder,
    )

if not text_input:
    st.stop()

    
prediction = inference(text_input, prob=True)


col1, col2, col3 = st.columns(3)
col1.metric('Negativo', f"{(prediction['Negativo'] * 100):.2f}%")
col2.metric('Neutro', f"{(prediction['Neutro'] * 100):.2f}%")
col3.metric('Positivo', f"{(prediction['Positivo'] * 100):.2f}%")


progress_text = "Pensando..."
neutro_bar = st.progress(0, text=progress_text)
neutro_bar.progress(prediction['Neutro'], text='Neutro')

negativo_bar = st.progress(0, text=progress_text)
negativo_bar.progress(prediction['Negativo'], text='Negativo')

positivo_bar = st.progress(0, text=progress_text)
positivo_bar.progress(prediction['Positivo'], text='Positivo')

