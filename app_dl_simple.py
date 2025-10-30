"""
Simple Deep Learning Resume Screening App - TensorFlow Only
Multiple DL Models: DNN, CNN, LSTM + Traditional ML Ensemble
"""

import streamlit as st
import joblib
import fitz
import re
import numpy as np
import pandas as pd
import time
import os
import tensorflow as tf
from tensorflow import keras

# Load models
@st.cache_resource
def load_dl_models():
    models = {}
    model_files = {
        'Simple DNN': 'saved_models/Simple_DNN.keras',
        'Advanced DNN': 'saved_models/Advanced_DNN.keras',
        'CNN': 'saved_models/CNN_Text.keras',
        'LSTM': 'saved_models/LSTM_Network.keras'
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = keras.models.load_model(path)
    
    return models

@st.cache_resource
def load_vectorizer_and_encoder():
    vectorizer_dl = None
    vectorizer_ml = None
    label_encoder = None
    
    # Try to load DL vectorizer (3000 features)
    if os.path.exists('saved_models/vectorizer_dl.joblib'):
        vectorizer_dl = joblib.load('saved_models/vectorizer_dl.joblib')
    
    # Try to load ML vectorizer (5000 features)
    if os.path.exists('saved_models/vectorizer_enhanced.joblib'):
        vectorizer_ml = joblib.load('saved_models/vectorizer_enhanced.joblib')
    
    if os.path.exists('saved_models/label_encoder_dl.joblib'):
        label_encoder = joblib.load('saved_models/label_encoder_dl.joblib')
    elif os.path.exists('saved_models/label_encoder_enhanced.joblib'):
        label_encoder = joblib.load('saved_models/label_encoder_enhanced.joblib')
    
    return vectorizer_dl, vectorizer_ml, label_encoder

# Load traditional ML models
@st.cache_resource
def load_traditional_models():
    models = {}
    model_paths = {
        "Logistic Regression": "saved_models/Logistic_Regression_Enhanced.joblib",
        "Random Forest": "saved_models/Random_Forest_Enhanced.joblib",
        "XGBoost": "saved_models/XGBoost_Enhanced.joblib",
    }
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    
    return models

dl_models = load_dl_models()
vectorizer_dl, vectorizer_ml, label_encoder = load_vectorizer_and_encoder()
ml_models = load_traditional_models()

# PDF extraction
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Enhanced preprocessing
def enhanced_preprocessing(text):
    if pd.isna(text) or not text:
        return ""
    text = text.lower()
    
    # Extract experience
    exp_pattern = r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)'
    exp_match = re.search(exp_pattern, text)
    if exp_match:
        text += f" experience_years_{exp_match.group(1)}"
    
    # Extract education
    edu_keywords = {
        'phd': 'doctorate', 'ph.d': 'doctorate',
        'm.s': 'masters', 'm.tech': 'masters', 'mba': 'masters',
        'b.tech': 'bachelors', 'b.s': 'bachelors'
    }
    for edu, level in edu_keywords.items():
        if edu in text:
            text += f" {level}_level"
    
    # Companies
    faang = ['google', 'apple', 'facebook', 'amazon', 'netflix', 'microsoft', 'meta']
    for company in faang:
        if company in text:
            text += " tier1_company"
            break
    
    text = re.sub(r'[^\w\s\+\#\.]', ' ', text)
    text = ' '.join(text.split())
    return text

# Streamlit UI
st.set_page_config(page_title="Deep Learning Resume AI", layout="wide")

st.markdown("""
    <style>
    .big-title {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #555;
        margin-bottom: 30px;
    }
    .model-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🤖 Deep Learning Resume AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Powered by TensorFlow: DNN • CNN • LSTM • Ensemble Learning</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🧠 Deep Learning Models")
    st.markdown("---")
    
    for model_name in ['Simple DNN', 'Advanced DNN', 'CNN', 'LSTM']:
        if model_name in dl_models:
            st.success(f"✅ {model_name}")
        else:
            st.warning(f"⏳ {model_name}")
    
    st.markdown("---")
    st.markdown("## 📊 Traditional ML")
    st.info(f"✅ {len(ml_models)} models loaded")
    
    st.markdown("---")
    st.markdown("### 🎯 Job Categories")
    if label_encoder:
        for cat in label_encoder.classes_:
            st.markdown(f"• {cat}")

# Main upload area
st.markdown("### 📂 Upload Resume for Analysis")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    st.success("✅ Resume uploaded successfully!")
    
    with st.spinner("🔍 Analyzing with Deep Learning models..."):
        time.sleep(0.5)
        
        # Extract text
        resume_text = extract_text_from_pdf(uploaded_file)
        st.info(f"📄 Extracted {len(resume_text)} characters from resume")
        
        if vectorizer_dl and label_encoder:
            # Preprocess
            cleaned_text = enhanced_preprocessing(resume_text)
            text_tfidf = vectorizer_dl.transform([cleaned_text]).toarray()
            text_cnn = text_tfidf.reshape(1, text_tfidf.shape[1], 1)
            
            st.markdown("---")
            st.markdown("## 🎯 Deep Learning Model Predictions")
            
            # Show DL predictions
            cols = st.columns(len(dl_models))
            dl_predictions = []
            dl_confidences = []
            
            for idx, (model_name, model) in enumerate(dl_models.items()):
                with cols[idx]:
                    if 'CNN' in model_name or 'LSTM' in model_name:
                        probs = model.predict(text_cnn, verbose=0)[0]
                    else:
                        probs = model.predict(text_tfidf, verbose=0)[0]
                    
                    pred_idx = np.argmax(probs)
                    category = label_encoder.inverse_transform([pred_idx])[0]
                    confidence = float(probs[pred_idx] * 100)  # Convert to Python float
                    
                    dl_predictions.append(category)
                    dl_confidences.append(confidence)
                    
                    st.markdown(f'<div class="model-box">', unsafe_allow_html=True)
                    st.markdown(f"**{model_name}**")
                    st.markdown(f"### {category}")
                    st.progress(float(confidence/100))  # Ensure Python float
                    st.caption(f"Confidence: {confidence:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Traditional ML predictions
            if len(ml_models) > 0 and vectorizer_ml:
                st.markdown("---")
                st.markdown("## 📊 Traditional ML Predictions")
                
                ml_cols = st.columns(len(ml_models))
                ml_predictions = []
                
                for idx, (name, model) in enumerate(ml_models.items()):
                    with ml_cols[idx]:
                        pred = model.predict(vectorizer_ml.transform([cleaned_text]))[0]
                        category = label_encoder.inverse_transform([pred])[0]
                        ml_predictions.append(category)
                        
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(vectorizer_ml.transform([cleaned_text]))[0]
                            conf = float(max(proba) * 100)
                        else:
                            conf = 85.0
                        
                        st.markdown(f'<div class="model-box">', unsafe_allow_html=True)
                        st.markdown(f"**{name}**")
                        st.markdown(f"### {category}")
                        st.progress(float(conf/100))  # Ensure Python float
                        st.caption(f"Confidence: {conf:.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                ml_predictions = []
            
            # Final Ensemble
            st.markdown("---")
            st.markdown("## 🏆 Final Ensemble Decision")
            
            all_predictions = dl_predictions + ml_predictions
            
            if len(all_predictions) > 0:
                from collections import Counter
                vote_counts = Counter(all_predictions)
                final_prediction = vote_counts.most_common(1)[0][0]
                agreement = (vote_counts[final_prediction] / len(all_predictions)) * 100
                
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown("### 🎯 Recommended Category")
                    st.markdown(f"# **{final_prediction}**")
                
                with col2:
                    st.markdown("### 📈 Model Agreement")
                    st.markdown(f"# **{agreement:.0f}%**")
                    st.caption(f"{vote_counts[final_prediction]}/{len(all_predictions)} models agree")
                
                with col3:
                    avg_confidence = np.mean(dl_confidences)
                    st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                
                # Detailed breakdown
                st.markdown("---")
                st.markdown("### 📋 All Model Votes")
                
                vote_data = []
                for idx, model_name in enumerate(list(dl_models.keys()) + list(ml_models.keys())):
                    if idx < len(all_predictions):
                        vote_data.append({
                            'Model': model_name,
                            'Prediction': all_predictions[idx],
                            'Confidence': f"{dl_confidences[idx]:.1f}%" if idx < len(dl_confidences) else "N/A"
                        })
                
                st.dataframe(vote_data, use_container_width=True)
                
            else:
                st.error("⚠️ No models available. Please train models first!")
        else:
            st.error("⚠️ Vectorizer or Label Encoder not found. Please train models first!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p><b>🚀 Deep Learning Resume Screening System</b></p>
        <p>Technologies: TensorFlow • Keras • Deep Neural Networks • CNN • LSTM • Ensemble Learning</p>
        <p style='font-size: 12px;'>Built with ❤️ using State-of-the-art Deep Learning</p>
    </div>
""", unsafe_allow_html=True)
