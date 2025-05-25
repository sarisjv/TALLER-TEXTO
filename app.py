import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from transformers import pipeline

# Configuración inicial
st.set_page_config(page_title="Análisis de Opiniones", layout="wide")
st.title("📊 Analizador de Opiniones")

# Descarga recursos NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Carga modelos optimizados
@st.cache_resource
def load_models():
    return (
        pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"),
        pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
    )

sentiment_analyzer, summarizer = load_models()

# Interfaz
with st.sidebar:
    st.header("⚙️ Configuración")
    max_words = st.slider("Palabras en nube", 50, 200, 100)

# Funciones clave
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud)
    plt.axis("off")
    st.pyplot(plt)

def analyze_sentiment(text):
    result = sentiment_analyzer(text[:512])[0]
    return result['label'], result['score']

# Carga de datos
uploaded_file = st.file_uploader("Sube tus opiniones (CSV)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file).head(20)
    st.success(f"✅ {len(df)} opiniones cargadas")
    
    # Análisis
    tab1, tab2 = st.tabs(["📈 Visualización", "🧠 Análisis"])
    
    with tab1:
        st.subheader("Nube de palabras")
        generate_wordcloud(" ".join(df.iloc[:, 0].astype(str)))
    
    with tab2:
        st.subheader("Sentimientos")
        df['Análisis'] = df.iloc[:, 0].apply(lambda x: analyze_sentiment(x))
        st.bar_chart(df['Análisis'].apply(lambda x: x[0]).value_counts())