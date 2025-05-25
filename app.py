import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk

# Configuración inicial ligera
st.set_page_config(page_title="Análisis de Opiniones", layout="wide")
st.title("📊 Analizador de Opiniones")

# Verificación de recursos NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Carga diferida de modelos pesados
MODELS_LOADED = False

def load_models():
    global MODELS_LOADED
    if not MODELS_LOADED:
        from transformers import pipeline
        sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # Forzar CPU
        )
        summarizer = pipeline(
            "summarization", 
            model="sshleifer/distilbart-cnn-6-6",
            device=-1  # Forzar CPU
        )
        MODELS_LOADED = True
        return sentiment_analyzer, summarizer
    return None, None

# Interfaz simplificada
with st.sidebar:
    st.header("⚙️ Configuración")
    max_words = st.slider("Palabras en nube", 50, 200, 100)
    analyze_option = st.checkbox("Realizar análisis de sentimientos", True)

# Función optimizada para wordcloud
def generate_wordcloud(text):
    try:
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=max_words
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud)
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Error generando nube de palabras: {str(e)}")

# Función de análisis con manejo de errores
def analyze_sentiment(text, analyzer):
    try:
        if len(text) > 500:
            text = text[:500] + "..."
        result = analyzer(text)[0]
        return result['label'], result['score']
    except Exception as e:
        st.error(f"Error en análisis: {str(e)}")
        return "NEUTRAL", 0.5

# Procesamiento del archivo
uploaded_file = st.file_uploader("Sube tus opiniones (CSV o Excel)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Carga de datos optimizada
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        if len(df) > 50:
            df = df.head(50)
            st.warning("Mostrando solo las primeras 50 filas para mejor rendimiento")
        
        text_column = df.columns[0]
        st.success(f"Datos cargados correctamente. {len(df)} registros encontrados.")
        
        # Pestañas de análisis
        tab1, tab2 = st.tabs(["Visualización", "Análisis"])
        
        with tab1:
            st.subheader("Nube de palabras")
            all_text = " ".join(df[text_column].astype(str))
            generate_wordcloud(all_text)
        
        with tab2:
            if analyze_option:
                with st.spinner("Cargando modelo de análisis (esto puede tomar unos segundos)..."):
                    sentiment_analyzer, _ = load_models()
                
                st.subheader("Análisis de sentimientos")
                
                # Procesamiento por lotes
                batch_size = 5
                results = []
                
                progress_bar = st.progress(0)
                for i in range(0, len(df), batch_size):
                    batch = df[text_column].iloc[i:i+batch_size].tolist()
                    batch_results = [analyze_sentiment(str(text), sentiment_analyzer) for text in batch]
                    results.extend(batch_results)
                    progress_bar.progress(min((i+batch_size)/len(df), 1.0))
                
                df['Sentimiento'] = [r[0] for r in results]
                df['Confianza'] = [r[1] for r in results]
                
                st.bar_chart(df['Sentimiento'].value_counts())
                st.dataframe(df[[text_column, 'Sentimiento', 'Confianza']].head(10))
            else:
                st.info("El análisis de sentimientos está desactivado en la configuración")
    
    except Exception as e:
        st.error(f"Error procesando archivo: {str(e)}")

  
