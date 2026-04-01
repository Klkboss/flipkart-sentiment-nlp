import streamlit as st
import pandas as pd
import joblib
import re
import nltk
import subprocess
import sys
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Flipkart Review Analyzer", layout="wide")

# ---------------- DOWNLOAD NLTK ----------------
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()

# ---------------- LOAD SPACY (FIXED) ----------------
@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

nlp = load_spacy()

# ---------------- LOAD ML MODELS ----------------
@st.cache_resource
def load_models():
    model = joblib.load('sentiment_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    le = joblib.load('label_encoder.pkl')
    return model, tfidf, le

model, tfidf, le = load_models()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# ---------------- ASPECT EXTRACTION ----------------
def extract_aspects(text):
    if not nlp:
        return []

    doc = nlp(text)
    aspects = []

    for token in doc:
        if token.pos_ == "NOUN":
            adjectives = []

            # Direct adjectives
            for child in token.children:
                if child.pos_ == "ADJ":
                    adjectives.append(child.text)

            # Adjectives via verb
            if token.dep_ in ["nsubj", "nsubjpass"] and token.head.pos_ in ["AUX", "VERB"]:
                for child in token.head.children:
                    if child.pos_ == "ADJ":
                        adjectives.append(child.text)

            if adjectives:
                compound_parts = [child.text for child in token.children if child.dep_ == "compound"]
                aspect_name = " ".join(compound_parts + [token.text]).title() if compound_parts else token.text.title()

                adj_text = " ".join(adjectives)

                # VADER
                score = sia.polarity_scores(adj_text)['compound']

                if score >= 0.05:
                    sentiment = "Positive"
                elif score <= -0.05:
                    sentiment = "Negative"
                else:
                    # ML fallback
                    vectorized_adj = tfidf.transform([adj_text])
                    prediction = model.predict(vectorized_adj)
                    sentiment = le.inverse_transform(prediction)[0].capitalize()

                emoji = "🟢" if sentiment == "Positive" else "🔴" if sentiment == "Negative" else "🟡"
                aspects.append(f"{aspect_name}: {sentiment} {emoji}")

    return aspects

# ---------------- UI ----------------
st.title("🛍️ Flipkart Business Intelligence & Sentiment Analyzer")
st.write("Analyze individual product feedback or upload bulk data for business insights.")

tab1, tab2 = st.tabs(["🔍 Aspect-Based Analysis (Single)", "📊 Business Dashboard (Batch)"])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("Deep Dive: Single Review Analysis")

    user_input = st.text_area(
        "Enter a product review:",
        "The screen is beautiful but the battery life is terrible."
    )

    if st.button("Analyze Review"):
        if user_input:
            cleaned = clean_text(user_input)
            vec = tfidf.transform([cleaned])
            prediction = model.predict(vec)
            sentiment = le.inverse_transform(prediction)[0]

            st.markdown("### 1. Overall Sentiment")
            if sentiment.lower() == 'positive':
                st.success(f"{sentiment} 🟢")
            elif sentiment.lower() == 'negative':
                st.error(f"{sentiment} 🔴")
            else:
                st.warning(f"{sentiment} 🟡")

            st.markdown("### 2. Aspect Extraction")
            aspects = extract_aspects(user_input)

            if aspects:
                for a in aspects:
                    st.markdown(f"- {a}")
            else:
                st.write("No aspects detected.")
        else:
            st.warning("Please enter a review.")

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("Batch Upload & Business Insights")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        text_col = None
        if 'Summary' in df.columns:
            text_col = 'Summary'
        elif 'Review' in df.columns:
            text_col = 'Review'

        if text_col:
            st.info(f"Analyzing {len(df)} reviews...")

            with st.spinner("Processing..."):
                df['Cleaned'] = df[text_col].astype(str).apply(clean_text)
                vec = tfidf.transform(df['Cleaned'])
                preds = model.predict(vec)
                df['Predicted_Sentiment'] = le.inverse_transform(preds)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Sentiment Distribution")
                counts = df['Predicted_Sentiment'].value_counts().reset_index()
                counts.columns = ['Sentiment', 'Count']
                fig = px.pie(counts, values='Count', names='Sentiment')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### Negative Word Cloud")
                neg_text = " ".join(df[df['Predicted_Sentiment'] == 'negative']['Cleaned'])

                if neg_text:
                    wc = WordCloud(width=800, height=400).generate(neg_text)
                    fig_wc, ax = plt.subplots()
                    ax.imshow(wc)
                    ax.axis("off")
                    st.pyplot(fig_wc)
                else:
                    st.write("No negative reviews found.")

            st.markdown("### Data Preview")
            st.dataframe(df[[text_col, 'Predicted_Sentiment']])
        else:
            st.error("CSV must contain 'Summary' or 'Review'")