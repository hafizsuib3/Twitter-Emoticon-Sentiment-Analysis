%%writefile app.py
import streamlit as st
import joblib
import re

model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf.pkl")

def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    return text

st.title("📊 Sentiment Analysis App")
st.write("Type a sentence or tweet to predict sentiment:")

user_input = st.text_area("Your text:")

if st.button("Predict"):
    if user_input.strip():
        cleaned = clean_tweet(user_input)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        sentiment = "😊 Positive" if prediction == 1 else "☹️ Negative"
        st.success(f"Predicted Sentiment: {sentiment}")
    else:
        st.warning("Please enter some text")
