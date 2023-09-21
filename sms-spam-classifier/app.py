import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score
import numpy as np
import diffprivlib.models as dp_models
from diffprivlib.mechanisms import Laplace

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Define your privacy budget (ε)
epsilon = 1.0  # Set your desired privacy budget here

# Function to add Laplace noise for Differential Privacy
def add_laplace_noise(data, epsilon):
    sensitivity = 1.0  # Sensitivity of the function being privatized
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, len(data))
    return data + noise

# Load the TF-IDF vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app title
st.title("Email/SMS Spam Classifier")

# Text input for user to enter a message
input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # Preprocess the user input
    def transform_text(text):
        text = text.lower()
        text = nltk.word_tokenize(text)

        y = []
        for i in text:
            if i.isalnum():
                y.append(i)

        text = y[:]
        y.clear()

        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)

        text = y[:]
        y.clear()

        for i in text:
            y.append(ps.stem(i))

        return " ".join(y)

    transformed_sms = transform_text(input_sms)

    # Vectorize the preprocessed message
    vector_input = tfidf.transform([transformed_sms])

    # Perform Differential Privacy on the vectorized input
    vector_input_dp = add_laplace_noise(vector_input.toarray(), epsilon)

    # Make predictions using the privacy-preserving model
    result = model.predict(vector_input_dp)[0]

    # Display the prediction result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
