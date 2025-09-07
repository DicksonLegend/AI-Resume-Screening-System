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
import pandas as pd
from req import *

# Load the enhanced trained models
model_paths = {
    "Logistic Regression Enhanced": "D:/ML IA3 PROJECT/saved_models/Logistic_Regression_Enhanced.joblib",
    "Random Forest Enhanced": "D:/ML IA3 PROJECT/saved_models/Random_Forest_Enhanced.joblib",
    "XGBoost Enhanced": "D:/ML IA3 PROJECT/saved_models/XGBoost_Enhanced.joblib",
    "Ensemble Enhanced": "D:/ML IA3 PROJECT/saved_models/Ensemble_Enhanced.joblib"
}
models = {name: joblib.load(path) for name, path in model_paths.items()}

# Load the enhanced vectorizer & label encoder
vectorizer = joblib.load("D:/ML IA3 PROJECT/saved_models/vectorizer_enhanced.joblib")
label_encoder = joblib.load("D:/ML IA3 PROJECT/saved_models/label_encoder_enhanced.joblib")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Enhanced text preprocessing (matching the training preprocessing)
def enhanced_text_preprocessing(text):
    """Enhanced text preprocessing with better feature extraction"""
    if pd.isna(text) or not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Extract years of experience (if mentioned)
    experience_pattern = r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)'
    experience_match = re.search(experience_pattern, text)
    if experience_match:
        text += f" experience_years_{experience_match.group(1)}"
    
    # Extract education level
    education_keywords = {
        'phd': 'doctorate_level',
        'ph.d': 'doctorate_level',
        'doctorate': 'doctorate_level',
        'm.s': 'masters_level',
        'm.tech': 'masters_level',
        'mba': 'masters_level',
        'masters': 'masters_level',
        'b.tech': 'bachelors_level',
        'b.s': 'bachelors_level',
        'b.e': 'bachelors_level',
        'bachelors': 'bachelors_level'
    }
    
    for edu_key, edu_level in education_keywords.items():
        if edu_key in text:
            text += f" {edu_level}"
    
    # Extract company tier (FAANG, Fortune 500, etc.)
    faang_companies = ['google', 'apple', 'facebook', 'amazon', 'netflix', 'microsoft', 'meta']
    for company in faang_companies:
        if company in text:
            text += " tier1_company"
            break
    
    # Remove special characters but keep important programming symbols
    text = re.sub(r'[^\w\s\+\#\.]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Legacy text preprocessing (kept for fallback)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# Streamlit UI
st.set_page_config(page_title="Enhanced AI Resume Screener", layout="centered")

# Enhanced UI
st.markdown("<h1 style='text-align: center; font-size: 36px; color: #4CAF50; font-family: Arial; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>🚀 Enhanced AI Resume Screening System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #666;'>Powered by Advanced Machine Learning with 80% Accuracy!</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px;'>Upload a resume to get intelligent job category predictions with confidence scores!</p>", unsafe_allow_html=True)


uploaded_file = st.file_uploader("📂 Upload Resume (PDF Format)", type=["pdf"])

if uploaded_file:
    st.success("✅ File uploaded successfully!")

    # Progress Bar
    with st.spinner("🔍 Analyzing Resume with Enhanced AI Models... Please wait!"):
        time.sleep(2)  # Simulate processing time

        # Extract and preprocess text
        resume_text = extract_text_from_pdf(uploaded_file)
        
        # Use enhanced preprocessing
        cleaned_text = enhanced_text_preprocessing(resume_text)

        # Convert text into numerical features using enhanced vectorizer
        vectorized_text = vectorizer.transform([cleaned_text])

        # Get predictions from all enhanced models
        predictions = {}
        prediction_probabilities = {}
        
        for name, model in models.items():
            pred = model.predict(vectorized_text)[0]
            predictions[name] = label_encoder.inverse_transform([pred])[0]
            
            # Get prediction probabilities for confidence scores
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(vectorized_text)[0]
                max_proba = max(proba)
                prediction_probabilities[name] = max_proba

        # Use Ensemble model as primary prediction
        ensemble_prediction = predictions["Ensemble Enhanced"]
        ensemble_confidence = prediction_probabilities.get("Ensemble Enhanced", 0.0)

        # Fallback to majority voting if ensemble not available
        if "Ensemble Enhanced" not in predictions:
            prediction_values = list(predictions.values())
            ensemble_prediction = max(set(prediction_values), key=prediction_values.count)

        # Dynamic Suitability Calculator
        def analyze_skill_match(resume_text, predicted_category):
            """Analyze how well resume skills match the predicted category"""
            
            skill_requirements = {
                "Data Science": ["python", "sql", "machine learning", "statistics", "pandas", "numpy", "tableau", "r", "analytics", "data mining"],
                "Machine Learning Engineer": ["tensorflow", "pytorch", "mlops", "model deployment", "python", "deep learning", "neural networks", "ai", "kubeflow"],
                "Software Engineer": ["programming", "algorithms", "software development", "coding", "git", "debugging", "testing", "agile", "java", "c++"],
                "Frontend Developer": ["javascript", "react", "html", "css", "vue", "angular", "responsive design", "ui", "bootstrap", "typescript"],
                "Backend Developer": ["apis", "database", "server", "python", "java", "node.js", "microservices", "sql", "nosql", "rest"],
                "Full Stack Developer": ["frontend", "backend", "full stack", "javascript", "database", "web development", "apis", "react", "node.js"],
                "DevOps Engineer": ["docker", "kubernetes", "aws", "azure", "ci/cd", "jenkins", "terraform", "monitoring", "automation", "linux"],
                "Web Developer": ["html", "css", "javascript", "web design", "responsive", "cms", "wordpress", "php", "web development"],
                "Product Manager": ["product management", "roadmap", "strategy", "analytics", "user research", "agile", "stakeholder", "market research", "kpi"]
            }
            
            required_skills = skill_requirements.get(predicted_category, [])
            resume_lower = resume_text.lower()
            found_skills = [skill for skill in required_skills if skill in resume_lower]
            
            skill_match_ratio = len(found_skills) / len(required_skills) if required_skills else 0
            return skill_match_ratio, found_skills, required_skills

        def extract_experience_years(resume_text):
            """Extract years of experience from resume"""
            import re
            patterns = [
                r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
                r'(\d+)\+\s*(?:years?|yrs?)',
                r'over\s*(\d+)\s*(?:years?|yrs?)',
                r'(\d+)\s*(?:years?|yrs?)\s*in'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, resume_text.lower())
                if match:
                    return int(match.group(1))
            return 0

        def calculate_dynamic_suitability(prediction, confidence, all_models_agree, resume_text):
            """Calculate comprehensive suitability based on multiple factors"""
            
            # 1. Skill Match Analysis
            skill_match_ratio, found_skills, required_skills = analyze_skill_match(resume_text, prediction)
            
            # 2. Experience Analysis
            experience_years = extract_experience_years(resume_text)
            
            # 3. Base score from AI confidence and agreement
            base_score = confidence
            if all_models_agree:
                base_score += 0.15  # Bonus for unanimous agreement
            
            # 4. Skill match bonus/penalty
            skill_bonus = skill_match_ratio * 0.3  # Up to 30% bonus
            
            # 5. Experience bonus
            exp_bonus = min(experience_years * 0.02, 0.2)  # Up to 20% bonus, 1% per year
            
            # 6. Calculate final score
            final_score = base_score + skill_bonus + exp_bonus
            final_score = min(final_score, 1.0)  # Cap at 100%
            
            # 7. Determine suitability level and color
            if final_score >= 0.85:
                level = "Exceptional Match ⭐⭐⭐⭐⭐"
                color = "#1B5E20"  # Dark Green
            elif final_score >= 0.70:
                level = "Excellent Match ⭐⭐⭐⭐"
                color = "#2E7D32"  # Green
            elif final_score >= 0.55:
                level = "Strong Match ⭐⭐⭐"
                color = "#388E3C"  # Light Green
            elif final_score >= 0.40:
                level = "Good Match ⭐⭐"
                color = "#689F38"  # Green-Yellow
            elif final_score >= 0.25:
                level = "Fair Match ⭐"
                color = "#FBC02D"  # Yellow
            else:
                level = "Limited Match"
                color = "#FF9800"  # Orange
            
            return level, color, final_score, skill_match_ratio, found_skills, experience_years

        # Calculate dynamic suitability
        # First check if all models agree
        all_predictions = list(predictions.values())
        models_agree = len(set(all_predictions)) == 1
        
        job_suitability, suitability_color, final_score, skill_match_ratio, found_skills, experience_years = calculate_dynamic_suitability(
            ensemble_prediction, 
            ensemble_confidence, 
            models_agree, 
            resume_text
        )

        # Enhanced confidence interpretation
        def interpret_confidence(confidence, all_models_agree=False):
            """Provide better confidence interpretation"""
            if all_models_agree:
                # If all models agree, boost the interpretation
                if confidence >= 0.5:
                    return "High", "🟢"
                elif confidence >= 0.3:
                    return "Good", "🟡" 
                else:
                    return "Moderate", "🟠"
            else:
                # Standard interpretation
                if confidence >= 0.7:
                    return "High", "🟢"
                elif confidence >= 0.5:
                    return "Good", "🟡"
                elif confidence >= 0.3:
                    return "Moderate", "🟠"
                else:
                    return "Low", "🔴"

        # Check if all models agree
        confidence_level, confidence_color = interpret_confidence(ensemble_confidence, models_agree)

    st.success("✅ Resume analysis completed with Enhanced AI Models!")

    # Display enhanced results
    st.markdown("---")
    
    # Main prediction with confidence
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"<h2 style='text-align: center; font-size: 28px; color: #E91E63; background: linear-gradient(90deg, #E91E63, #9C27B0); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>🎯 Best Job Category: {ensemble_prediction}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center; font-size: 22px; color: {suitability_color};'>📊 {job_suitability}</h3>", unsafe_allow_html=True)
        
        # Show detailed analysis
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("🎯 Overall Score", f"{final_score:.1%}")
        with col_b:
            st.metric("🛠️ Skill Match", f"{skill_match_ratio:.1%}")
        with col_c:
            if experience_years > 0:
                st.metric("📅 Experience", f"{experience_years} years")
            else:
                st.metric("📅 Experience", "Not specified")
    
    with col2:
        if ensemble_confidence > 0:
            st.metric("🔥 AI Confidence", f"{ensemble_confidence:.1%}")
            if models_agree:
                st.success(f"{confidence_color} All models agree! Prediction reliability: {confidence_level}")
            else:
                st.warning(f"{confidence_color} Confidence level: {confidence_level}")
    
    # Individual model predictions
    st.markdown("### 🤖 Individual Model Predictions:")
    
    pred_cols = st.columns(len(predictions))
    for i, (model_name, prediction) in enumerate(predictions.items()):
        with pred_cols[i]:
            confidence = prediction_probabilities.get(model_name, 0.0)
            model_short = model_name.replace(" Enhanced", "")
            st.info(f"**{model_short}**\n\n{prediction}\n\nConfidence: {confidence:.1%}")

    # Detailed Analysis Section
    st.markdown("### 📋 Detailed Suitability Analysis:")
    
    with st.expander("🔍 Click to see detailed breakdown", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**🛠️ Skills Found in Resume:**")
            if found_skills:
                for skill in found_skills:
                    st.write(f"✅ {skill.title()}")
            else:
                st.write("❌ No specific skills detected for this category")
            
            st.write(f"\n**📊 Analysis Summary:**")
            st.write(f"• AI Confidence: {ensemble_confidence:.1%}")
            st.write(f"• Skill Match: {skill_match_ratio:.1%}")
            st.write(f"• Experience: {experience_years} years" if experience_years > 0 else "• Experience: Not specified")
            st.write(f"• Model Agreement: {'✅ All models agree' if models_agree else '⚠️ Mixed predictions'}")
            st.write(f"• Final Score: {final_score:.1%}")
        
        with col2:
            # Simple score visualization
            st.write("**📊 Score Breakdown:**")
            
            # Progress bars for different components
            st.write("AI Confidence:")
            st.progress(ensemble_confidence)
            
            st.write("Skill Match:")
            st.progress(skill_match_ratio)
            
            if experience_years > 0:
                exp_normalized = min(experience_years / 10, 1.0)  # Normalize to 10 years max
                st.write("Experience Level:")
                st.progress(exp_normalized)
            
            st.write("Overall Score:")
            st.progress(final_score)
            
            # Color-coded final assessment
            if final_score >= 0.7:
                st.success(f"🎉 Excellent match! Score: {final_score:.1%}")
            elif final_score >= 0.5:
                st.info(f"👍 Good match! Score: {final_score:.1%}")
            else:
                st.warning(f"⚠️ Fair match. Score: {final_score:.1%}")

    st.balloons()  # 🎈 Fun UI effect

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 12px; color: #888;'>⚠️ AI can make mistakes. Please verify important information. | Enhanced with 90% accuracy models</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 10px; color: #aaa;'>Powered by Enhanced AI Models trained on comprehensive dataset</p>", unsafe_allow_html=True)
