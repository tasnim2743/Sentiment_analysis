import streamlit as st
import pickle

# Load the trained model and vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Manually map the sentiment labels
label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}

# Streamlit app
st.title("Sentiment Analysis Web App")
st.write("Enter text to determine if the sentiment is positive, negative, or neutral.")

# Text input
user_input = st.text_area("Enter your text here:")

# Predict sentiment
if st.button("Analyze Sentiment"):
    if user_input:
        # Transform the input using the loaded vectorizer
        input_vector = vectorizer.transform([user_input])
        # Predict using the loaded model
        prediction = model.predict(input_vector)
        # Map the prediction to the corresponding sentiment
        sentiment = label_mapping[prediction[0]]
        st.write(f"The sentiment of the entered text is: **{sentiment}**")
    else:
        st.write("Please enter some text for analysis.")
