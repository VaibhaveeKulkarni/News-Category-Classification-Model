import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Download necessary NLTK data (only if not already available in the environment)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- Text Preprocessing Functions (must match training pipeline) ---
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_punctuation_and_special_chars(text):
    # Keep alphanumeric characters and spaces, remove everything else.
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

def simple_tokenize_text(text):
    # Using the same simple regex tokenizer as in the notebook
    return re.findall(r'\b\w+\b', text)

def remove_stopwords_func(tokens):
    return [word for word in tokens if word not in stop_words]

def lemmatize_tokens_func(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess_text(text):
    # Apply the cleaning steps in order
    text = text.lower()
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_punctuation_and_special_chars(text)
    text = remove_extra_spaces(text)

    # Apply NLP preprocessing steps
    tokens = simple_tokenize_text(text)
    tokens = remove_stopwords_func(tokens)
    tokens = lemmatize_tokens_func(tokens)

    return ' '.join(tokens)

# --- Load the saved model and vectorizer ---
model = None
tfidf_vectorizer = None

model_path = 'news_model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'

if not os.path.exists(model_path):
    st.error(f"Error: Model file '{model_path}' not found. Please ensure it's in the same directory.")
    st.stop()

if not os.path.exists(vectorizer_path):
    st.error(f"Error: TF-IDF vectorizer file '{vectorizer_path}' not found. Please ensure it's in the same directory.")
    st.stop()

try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    with open(vectorizer_path, 'rb') as file:
        tfidf_vectorizer = pickle.load(file)
    st.success("Model and TF-IDF vectorizer loaded successfully!")
except Exception as e:
    st.error(f"An error occurred while loading the model or vectorizer: {e}")
    st.stop()

# --- Streamlit UI ---
st.title("News Article Classifier")
st.write("Enter a news article below to classify its category.")

news_article_input = st.text_area("News Article Text", height=200)

# Define class labels (assuming AG News dataset categories based on common usage)
# If actual labels are different, this mapping should be updated.
class_labels = {
    1: 'World',
    2: 'Sports',
    3: 'Business',
    4: 'Sci/Tech'
}

if st.button("Classify News Article"):
    if news_article_input:
        with st.spinner("Processing and classifying..."):
            # Preprocess the input text
            processed_text = preprocess_text(news_article_input)

            # Transform the processed text using the loaded TF-IDF vectorizer
            text_tfidf = tfidf_vectorizer.transform([processed_text])

            # Predict the news category
            prediction = model.predict(text_tfidf)[0]
            predicted_category = class_labels.get(prediction, f'Unknown Category ({prediction})')

            st.subheader("Prediction Results:")
            st.write(f"**Predicted Category:** {predicted_category}")

            # Display confidence score if the model supports it
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(text_tfidf)[0]
                confidence = probabilities[model.classes_.tolist().index(prediction)] * 100
                st.write(f"**Confidence Score:** {confidence:.2f}%")
            else:
                st.info("This model does not provide probability scores directly.")
    else:
        st.warning("Please enter some text to classify.")
