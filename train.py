import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

print("Loading dataset...")
# Using your exact uploaded filename
df = pd.read_csv('Dataset-SA.csv')

# Drop rows with missing values in 'Summary' or 'Sentiment'
df.dropna(subset=['Summary', 'Sentiment'], inplace=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

print("Cleaning text (this will take a minute for 205k rows)...")
df['Clean_Summary'] = df['Summary'].apply(clean_text)

print("Encoding labels and splitting data...")
le = LabelEncoder()
df['Sentiment_Encoded'] = le.fit_transform(df['Sentiment'])

X_train, X_test, y_train, y_test = train_test_split(
    df['Clean_Summary'], 
    df['Sentiment_Encoded'], 
    test_size=0.2, 
    random_state=42
)

print("Vectorizing text (TF-IDF)...")
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)

print("Training model (with balanced class weights)...")
# class_weight='balanced' handles the massive difference between Positive (166k) and Negative (28k) reviews
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

print("Saving model files...")
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("Done! Model saved successfully. You can now run your Streamlit app.")