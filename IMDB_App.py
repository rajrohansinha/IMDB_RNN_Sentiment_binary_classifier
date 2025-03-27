# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('Simple_RNN_IMDB_Sentiment_Classification.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = pad_sequences([encoded_review], maxlen=500)  # Fixed NameError
    return padded_review

# Streamlit app
st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎥", layout="centered")

# App title and description
st.title('🎥 IMDB Movie Review Sentiment Analysis')
st.markdown("""
This app uses a pre-trained Simple RNN model to classify movie reviews as **Positive** or **Negative**.  
Simply enter a movie review in the text box below and click **Classify** to see the sentiment prediction.
""")

# User input
st.subheader("Enter a Movie Review:")
user_input = st.text_area('Type your review here...', height=150)

if st.button('Classify'):
    if user_input.strip():
        # Preprocess the input
        preprocessed_input = preprocess_text(user_input)

        # Make prediction
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

        # Display the result
        st.subheader("Prediction Result:")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Prediction Score:** {prediction[0][0]:.4f}")
    else:
        st.warning("⚠️ Please enter a valid movie review before clicking Classify.")
else:
    st.info("Enter a movie review and click **Classify** to get started.")