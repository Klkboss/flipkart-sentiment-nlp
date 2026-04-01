# Flipkart E-Commerce Sentiment Analysis

An end-to-end Natural Language Processing (NLP) pipeline that classifies customer product reviews into **Positive**, **Neutral**, or **Negative** sentiments. 

This project was trained on a massive dataset of **205,052 real product reviews** scraped from Flipkart.

## 🚀 Features
* **Custom Text Preprocessing:** Regex cleaning, stop-word removal, and WordNet Lemmatization.
* **Class Imbalance Handling:** Implemented `class_weight='balanced'` in the Logistic Regression model to accurately predict minority classes (Negative/Neutral) against a heavily skewed Positive dataset (81%).
* **Feature Extraction:** TF-IDF Vectorization (capped at 5000 max features).
* **Interactive UI:** Built a front-end web application using Streamlit for real-time user inference.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **NLP & ML:** NLTK, Scikit-learn (Logistic Regression, TF-IDF)
* **Frontend UI:** Streamlit

## ⚙️ How to Run Locally

1. Clone this repository:
   ```bash
   git clone [https://github.com/YourUsername/flipkart-sentiment-nlp.git](https://github.com/YourUsername/flipkart-sentiment-nlp.git)
   cd flipkart-sentiment-nlp