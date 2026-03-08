import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense


word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

model = load_model('simple_rnn_imdb.h5')

def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

# function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded = [word_index.get(word, 2) + 3 for word in words]  # 2 is for unknown words
    padded_encoded = sequence.pad_sequences([encoded], maxlen=500)
    return padded_encoded


### step 3 create predict function
def predict_sentiment(text):
    preprocessed = preprocess_text(text)
    prediction = model.predict(preprocessed)[0][0]
    return 'Positive' if prediction >= 0.5 else 'Negative', prediction

import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
user_input = st.text_area("Enter a movie review:")
if st.button("Predict Sentiment"):
    sentiment, confidence = predict_sentiment(user_input)
    st.write(f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.4f})")
else:
    st.write("Enter a movie review and click 'Predict Sentiment' to see the result.")

