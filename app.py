import streamlit as st
import joblib
import fitz  # PyMuPDF for PDF extraction
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import time
from req import *

# Load the trained models
model_paths = {
    "Logistic Regression": "D:/ML IA3 PROJECT/saved_models/Logistic_Regression_Tuned.joblib",
    "Random Forest": "D:/ML IA3 PROJECT/saved_models/Random_Forest_Tuned.joblib",
    "XGBoost": "D:/ML IA3 PROJECT/saved_models/XGBoost_Tuned.joblib"
}
models = {name: joblib.load(path) for name, path in model_paths.items()}

# Load the vectorizer & label encoder
vectorizer = joblib.load("D:/ML IA3 PROJECT/saved_models/vectorizer.joblib")
label_encoder = joblib.load("D:/ML IA3 PROJECT/saved_models/label_encoder.joblib")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# Streamlit UI
st.set_page_config(page_title="AI Resume Screener", layout="centered")

# UI Enhancements
st.markdown("<h1 style='text-align: center; font-size: 32px; color: #4CAF50; font-family: Arial;'>AI Resume Screening System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a resume to predict the best job category!</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload Resume (PDF)", type=["pdf"])

if uploaded_file:
    st.success("✅ File uploaded successfully!")

    # Progress Bar
    with st.spinner("🔍 Analyzing Resume... Please wait!"):
        time.sleep(2)  # Simulate processing time

        # Extract and preprocess text
        resume_text = extract_text_from_pdf(uploaded_file)
        cleaned_text = preprocess_text(resume_text)

        # Convert text into numerical features
        vectorized_text = vectorizer.transform([cleaned_text])

        # Predict using all models
        predictions = {name: model.predict(vectorized_text)[0] for name, model in models.items()}
        predictions = {name: label_encoder.inverse_transform([pred])[0] for name, pred in predictions.items()}

        # Majority Voting
        prediction_values = list(predictions.values())  # Convert dict_values to a list
        final_prediction = max(set(prediction_values), key=prediction_values.count)

        # Job suitability levels
        suitability_levels = {
            "Data Scientist": "Highly Suitable",
            "Data Analyst" : "Highly Suitable",
            "Machine Learning Engineer": "Highly Suitable",
            "Full Stack Developer": "Highly Suitable",
            "Cloud Engineer": "Highly Suitable",
            "Backend Developer": "Moderate",
            "Frontend Developer": "Moderate",
            "Python Developer": "Moderate",
            "Mobile App Developer (iOS/Android)": "Moderate"
        }

        # Get suitability level
        job_suitability = suitability_levels.get(final_prediction, "Moderate")

    st.success("✅ Resume analysis completed!")

    # Display results
    st.markdown(f"<h2 style='text-align: center; font-size: 24px; color: #E91E63;'>Final Prediction: {final_prediction}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center; font-size: 20px; color: #009688;'>Suitability Level: {job_suitability}</h3>", unsafe_allow_html=True)

    st.balloons()  # 🎈 Fun UI effect

# Footer
st.markdown("<p style='text-align: center; font-size: 12px;'> AI can make mistakes. Check important info.</p>", unsafe_allow_html=True)
