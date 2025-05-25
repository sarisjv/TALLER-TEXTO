import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from collections import Counter

# Configuración ligera
st.set_page_config(page_title="Análisis de Opiniones", layout="centered")
st.title("📊 Analizador de Opiniones")

# Verificación de recursos NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Carga diferida y optimizada del modelo
@st.cache_resource(ttl=3600, show_spinner=False)
def load_sentiment_model():
    from transformers import pipeline
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,  # Forzar CPU
        truncation=True,
        max_length=512
    )

# Interfaz minimalista
with st.sidebar:
    st.header("⚙️ Configuración")
    max_records = st.slider("Máx. registros a analizar", 10, 50, 20)
    show_wordcloud = st.checkbox("Mostrar nube de palabras", True)

# Función optimizada para wordcloud
def generate_wordcloud(text):
    try:
        wordcloud = WordCloud(
            width=600, 
            height=300,
            background_color='white',
            max_words=100
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wordcloud)
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.warning(f"No se pudo generar la nube de palabras: {str(e)}")

# Función de análisis optimizada
def analyze_sentiment_batch(texts, analyzer):
    try:
        results = analyzer(texts)
        return [(r['label'], r['score']) for r in results]
    except Exception as e:
        st.error(f"Error en análisis: {str(e)}")
        return [("ERROR", 0.0)] * len(texts)

# Procesamiento del archivo
uploaded_file = st.file_uploader("Sube tus opiniones (CSV)", type=["csv"])

if uploaded_file:
    try:
        # Carga optimizada de datos
        df = pd.read_csv(uploaded_file).head(max_records)
        text_column = df.columns[0]
        
        with st.expander("📂 Vista previa de datos"):
            st.dataframe(df.head(3), use_container_width=True)
        
        # Análisis principal
        if st.button("Analizar opiniones"):
            with st.spinner("Cargando modelo de análisis..."):
                analyzer = load_sentiment_model()
            
            # Procesamiento por lotes optimizado
            texts = df[text_column].astype(str).tolist()
            batch_size = 4  # Tamaño reducido para Streamlit.app
            results = []
            
            progress_bar = st.progress(0)
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_results = analyze_sentiment_batch(batch, analyzer)
                results.extend(batch_results)
                progress_bar.progress(min((i+batch_size)/len(texts), 1.0))
            
            # Asignar resultados
            df['Sentimiento'] = [r[0] for r in results]
            df['Confianza'] = [r[1] for r in results]
            
            # Mostrar resultados
            st.subheader("Resultados del análisis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total opiniones", len(df))
                st.bar_chart(df['Sentimiento'].value_counts())
            
            with col2:
                avg_confidence = df['Confianza'].mean()
                st.metric("Confianza promedio", f"{avg_confidence:.0%}")
                st.write(df['Sentimiento'].value_counts(normalize=True))
            
            if show_wordcloud:
                st.subheader("Nube de palabras")
                generate_wordcloud(" ".join(texts))
            
            st.success("Análisis completado!")
            
            # Opción para descargar resultados
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Descargar resultados",
                csv,
                "resultados_analisis.csv",
                "text/csv"
            )
    
    except Exception as e:
        st.error(f"Error procesando archivo: {str(e)}")
else:
    st.info("Por favor, sube un archivo CSV para comenzar el análisis")

st.caption("Nota: Esta aplicación usa modelos optimizados para funcionar en Streamlit.app")
