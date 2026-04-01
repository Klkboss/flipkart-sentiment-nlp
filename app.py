import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy

# Page Config
st.set_page_config(page_title="Flipkart Review Analyzer", layout="wide")

# Load NLTK and Spacy securely
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()

@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("SpaCy model not found. Please run: python -m spacy download en_core_web_sm")
        return None

nlp = load_spacy()

# Load saved ML models
@st.cache_resource
def load_models():
    model = joblib.load('sentiment_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    le = joblib.load('label_encoder.pkl')
    return model, tfidf, le

model, tfidf, le = load_models()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Extract Aspects (Nouns) and Descriptors (Adjectives)
def extract_aspects(text):
    if not nlp: return []
    doc = nlp(text)
    aspects = []
    
    for token in doc:
        if token.pos_ == "NOUN":
            adjectives = []
            
            # 1. Adjectives that are direct children (amod)
            for child in token.children:
                if child.pos_ == "ADJ":
                    adjectives.append(child.text)
            
            # 2. Adjectives connected via a verb
            if token.dep_ in ["nsubj", "nsubjpass"] and token.head.pos_ in ["AUX", "VERB"]:
                for child in token.head.children:
                    if child.pos_ == "ADJ":
                        adjectives.append(child.text)
            
            if adjectives:
                compound_parts = [child.text for child in token.children if child.dep_ == "compound"]
                if compound_parts:
                    aspect_name = " ".join(compound_parts + [token.text]).title()
                else:
                    aspect_name = token.text.title()

                adj_text = " ".join(adjectives)
                
                # 1. Try VADER first
                score = sia.polarity_scores(adj_text)['compound']
                
                if score >= 0.05:
                    sentiment = "Positive"
                elif score <= -0.05:
                    sentiment = "Negative"
                else:
                    # 2. Fallback to ML Model
                    vectorized_adj = tfidf.transform([adj_text])
                    prediction = model.predict(vectorized_adj)
                    sentiment = le.inverse_transform(prediction)[0].capitalize()
                
                if sentiment == 'Positive':
                    sentiment_emoji = '🟢'
                elif sentiment == 'Negative':
                    sentiment_emoji = '🔴'
                else:
                    sentiment_emoji = '🟡'
                
                aspects.append(f"{aspect_name}: {sentiment} {sentiment_emoji}")
                
    return aspects

# --- UI LAYOUT ---
st.title("🛍️ Flipkart Business Intelligence & Sentiment Analyzer")
st.write("Analyze individual product feedback or upload bulk data for business insights.")

# Create Tabs
tab1, tab2 = st.tabs(["🔍 Aspect-Based Analysis (Single)", "📊 Business Dashboard (Batch)"])

# --- TAB 1: SINGLE REVIEW & ASPECT EXTRACTION ---
with tab1:
    st.subheader("Deep Dive: Single Review Analysis")
    user_input = st.text_area("Enter a product review:", "The screen is beautiful but the battery life is terrible.")

    if st.button("Analyze Review"):
        if user_input:
            # 1. Predict Overall Sentiment
            cleaned_input = clean_text(user_input)
            vectorized_input = tfidf.transform([cleaned_input])
            prediction = model.predict(vectorized_input)
            sentiment = le.inverse_transform(prediction)[0]
            
            # Display Sentiment
            st.markdown("### 1. Overall Sentiment")
            if sentiment.lower() == 'positive':
                st.success(f"**{sentiment.capitalize()}** 🟢")
            elif sentiment.lower() == 'negative':
                st.error(f"**{sentiment.capitalize()}** 🔴")
            else:
                st.warning(f"**{sentiment.capitalize()}** 🟡")

            # 2. Extract Aspects (Level 2 Feature)
            st.markdown("### 2. Aspect Extraction (What is the customer talking about?)")
            aspects = extract_aspects(user_input)
            if aspects:
                for aspect in aspects:
                    st.markdown(f"- {aspect}")
            else:
                st.write("No specific product features/adjectives detected in this review.")
        else:
            st.warning("Please enter a review.")


# --- TAB 2: BUSINESS DASHBOARD ---
with tab2:
    st.subheader("Batch Upload & Business Insights")
    st.write("Upload a CSV file containing a column named `Summary` or `Review` to analyze hundreds of reviews instantly.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        
        # Check if right column exists
        text_col = None
        if 'Summary' in df_batch.columns: text_col = 'Summary'
        elif 'Review' in df_batch.columns: text_col = 'Review'
        
        if text_col:
            st.info(f"Analyzing {len(df_batch)} reviews...")
            
            with st.spinner("Processing text and predicting sentiment... (This may take a few minutes for large datasets)"):
                # Clean and Predict
                df_batch['Cleaned'] = df_batch[text_col].astype(str).apply(clean_text)
                vec_batch = tfidf.transform(df_batch['Cleaned'])
                preds = model.predict(vec_batch)
                df_batch['Predicted_Sentiment'] = le.inverse_transform(preds)
            
            # Layout for charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Sentiment Distribution")
                sentiment_counts = df_batch['Predicted_Sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                fig = px.pie(sentiment_counts, values='Count', names='Sentiment', 
                             color='Sentiment',
                             color_discrete_map={'positive':'#00cc96', 'negative':'#EF553B', 'neutral':'#FFA15A'})
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.markdown("### Common Words in Negative Reviews")
                neg_reviews = " ".join(df_batch[df_batch['Predicted_Sentiment'] == 'negative']['Cleaned'])
                if neg_reviews:
                    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(neg_reviews)
                    fig_wc, ax = plt.subplots()
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig_wc)
                else:
                    st.write("No negative reviews found to generate word cloud.")
            
            # Show the raw data table
            st.markdown("### Raw Data & Predictions")
            st.dataframe(df_batch[[text_col, 'Predicted_Sentiment']])
            
        else:
            st.error("Your CSV must contain a column named 'Summary' or 'Review'.")