import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

# Configuración inicial optimizada
st.set_page_config(page_title="Análisis de Opiniones", layout="wide")
st.title("📊 Analizador de Opiniones")

# Descarga recursos NLTK (solo si no están descargados)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Carga modelos solo cuando sea necesario y con caché
@st.cache_resource(show_spinner=False, ttl=3600)
def load_sentiment_model():
    from transformers import pipeline
    return pipeline("sentiment-analysis", 
                   model="distilbert-base-uncased-finetuned-sst-2-english",
                   truncation=True)

@st.cache_resource(show_spinner=False, ttl=3600)
def load_summarizer():
    from transformers import pipeline
    return pipeline("summarization", 
                   model="sshleifer/distilbart-cnn-6-6",
                   truncation=True)

# Interfaz
with st.sidebar:
    st.header("⚙️ Configuración")
    max_words = st.slider("Palabras en nube", 50, 200, 100)
    use_summarization = st.checkbox("Usar resumen automático", False)

# Funciones optimizadas
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         max_words=max_words).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud)
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)  # Liberar memoria

def analyze_sentiment(text):
    try:
        analyzer = load_sentiment_model()
        result = analyzer(text[:512])[0]
        return result['label'], result['score']
    except Exception as e:
        st.error(f"Error en análisis: {str(e)}")
        return "ERROR", 0.0

# Carga de datos con validación
uploaded_file = st.file_uploader("Sube tus opiniones (CSV)", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file).head(20)
        text_column = df.columns[0]
        st.success(f"✅ {len(df)} opiniones cargadas")
        
        # Mostrar vista previa
        with st.expander("🔍 Vista previa de datos"):
            st.dataframe(df.head(3))
        
        # Análisis en pestañas
        tab1, tab2 = st.tabs(["📈 Visualización", "🧠 Análisis"])
        
        with tab1:
            st.subheader("Nube de palabras")
            all_text = " ".join(df[text_column].astype(str))
            generate_wordcloud(all_text)
        
        with tab2:
            st.subheader("Sentimientos")
            with st.spinner("Analizando sentimientos..."):
                # Procesamiento por lotes para mejor rendimiento
                sample_texts = df[text_column].astype(str).tolist()
                results = [analyze_sentiment(t) for t in sample_texts]
                df['Análisis'] = results
                
                # Mostrar resultados
                col1, col2 = st.columns(2)
                with col1:
                    st.bar_chart(df['Análisis'].apply(lambda x: x[0]).value_counts())
                with col2:
                    st.write("Distribución de sentimientos:")
                    st.dataframe(df['Análisis'].apply(lambda x: x[0]).value_counts())
                
                if use_summarization:
                    with st.spinner("Generando resumenes..."):
                        try:
                            summarizer = load_summarizer()
                            sample_text = df[text_column].iloc[0][:1024]
                            summary = summarizer(sample_text, max_length=130, min_length=30, do_sample=False)
                            st.subheader("Resumen automático")
                            st.write(summary[0]['summary_text'])
                        except Exception as e:
                            st.warning(f"No se pudo generar el resumen: {str(e)}")
    
    except Exception as e:
        st.error(f"Error procesando archivo: {str(e)}")
