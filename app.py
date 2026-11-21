import os
import re
import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Page Configuration
st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📩",
    layout="centered",
)


@st.cache_resource
def setup_nltk():
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("tokenizers/punkt", "punkt"),
    ]
    for resource_path, pkg_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(pkg_name, quiet=True)


@st.cache_resource
def load_artifacts():
    model_path = os.path.join("models", "svm_model.pkl")
    tfidf_path = os.path.join("models", "tfidf_vectorizer.pkl")
    return joblib.load(model_path), joblib.load(tfidf_path)


setup_nltk()
model, tfidf = load_artifacts()

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]

    return " ".join(words)


# Application Title
st.title("📩 SMS Spam Detection System")
st.write("Enter an SMS message to identify whether it is Spam or Normal.")

# User Input
message = st.text_area("Enter your SMS message:")

# Prediction
if st.button("Predict"):
    if message.strip():
        cleaned_message = clean_text(message)
        message_vector = tfidf.transform([cleaned_message])
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
