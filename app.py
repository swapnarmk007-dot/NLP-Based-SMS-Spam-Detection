import streamlit as st
import joblib
import re
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Page Configuration
st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📩",
    layout="centered"
)

# Load Models
model = joblib.load("models/svm_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# NLP Preprocessing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):

    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = word_tokenize(text)

    words = [
        word for word in words
        if word not in stop_words
    ]

    words = [
        stemmer.stem(word)
        for word in words
    ]

    return ' '.join(words)

# Application Title
st.title("📩 SMS Spam Detection System")

st.write(
    "Enter an SMS message to identify whether "
    "it is Spam or Normal."
)

# User Input
message = st.text_area("Enter your SMS message:")

# Prediction
if st.button("Predict"):

    if message.strip():

        cleaned_message = clean_text(message)

        message_vector = tfidf.transform(
            [cleaned_message]
        )

        prediction = model.predict(message_vector)

        if prediction[0] == 1:

            st.error("🚨 SPAM MESSAGE")

        else:

            st.success("✅ NORMAL MESSAGE")

    else:

        st.warning("Please enter a message.")

st.markdown("---")

st.caption("Built using Python, NLP and Machine Learning.")
st.caption("Algorithm: Support Vector Machine (SVM)")