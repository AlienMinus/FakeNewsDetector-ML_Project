import streamlit as st
import re
import os
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

st.title("📰 Fake News Classifier")

# Use the new Keras format if possible
model_path = "Models/my_model.keras"
tokenizer_path = "Models/tokenizer.pickle"

if os.path.exists(model_path) and os.path.exists(tokenizer_path):
    st.success("Loading pretrained model and tokenizer...")
    model = load_model(model_path, compile=False)
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
else:
    st.error("Model or tokenizer not found. Ensure 'fakenews_model.keras' and 'tokenizer.pickle' exist.")
    st.stop()

# ...rest of your code...
# Text Preprocessing
tokenizer_num_words = 5000
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# Text Prediction
st.subheader("Predict Fake/Real News")
input_text = st.text_area("Enter News Text")
if st.button("Predict") and input_text:
    cleaned = clean_text(input_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded_seq = pad_sequences(seq, maxlen=100)
    prediction = model.predict(padded_seq)[0][0]
    label = "FAKE" if prediction < 0.5 else "REAL"
    st.write(f"Prediction: **{label}** ({prediction:.2f})")