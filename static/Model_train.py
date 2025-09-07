import joblib
import pymupdf as fitz

# Load the enhanced trained models (Updated paths)
model_paths = {
    "Logistic Regression Enhanced": "D:/ML IA3 PROJECT/saved_models/Logistic_Regression_Enhanced.joblib",
    "Random Forest Enhanced": "D:/ML IA3 PROJECT/saved_models/Random_Forest_Enhanced.joblib",
    "XGBoost Enhanced": "D:/ML IA3 PROJECT/saved_models/XGBoost_Enhanced.joblib",
    "Ensemble Enhanced": "D:/ML IA3 PROJECT/saved_models/Ensemble_Enhanced.joblib"
}

try:
    models = {name: joblib.load(path) for name, path in model_paths.items()}
    print("✅ Enhanced models loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error loading models: {e}")
    print("🔧 Make sure you've run the fine_tune.ipynb notebook to create enhanced models")

# Load the enhanced vectorizer and label encoder
try:
    vectorizer = joblib.load("D:/ML IA3 PROJECT/saved_models/vectorizer_enhanced.joblib")
    label_encoder = joblib.load("D:/ML IA3 PROJECT/saved_models/label_encoder_enhanced.joblib")
    print("✅ Enhanced preprocessing components loaded!")
except FileNotFoundError:
    # Fallback to original files
    vectorizer = joblib.load("D:/ML IA3 PROJECT/saved_models/vectorizer.joblib")
    label_encoder = joblib.load("D:/ML IA3 PROJECT/saved_models/label_encoder.joblib")
    print("✅ Original preprocessing components loaded!")

# Enhanced text preprocessing (matching app.py)
def enhanced_text_preprocessing(text):
    """Enhanced text preprocessing with better feature extraction"""
    import pandas as pd
    import re
    
    if pd.isna(text) or not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Extract years of experience
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
    
    # Extract company tier
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

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text")
    return text

# Updated predict function with enhanced preprocessing
def predict_job_suitability(pdf_path):
    """Classify the resume and determine job suitability using enhanced models."""
    resume_text = extract_text_from_pdf(pdf_path)
    cleaned_text = enhanced_text_preprocessing(resume_text)

    # Convert text into numerical features using enhanced TF-IDF
    vectorized_text = vectorizer.transform([cleaned_text])

    # Get predictions from each model
    predictions = {}
    confidences = {}
    
    for name, model in models.items():
        pred = model.predict(vectorized_text)[0]
        # Get confidence if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(vectorized_text)[0]
            confidence = max(proba)
            confidences[name] = confidence
        
        predictions[name] = label_encoder.inverse_transform([pred])[0]

    # Use ensemble model for final prediction if available
    if "Ensemble Enhanced" in predictions:
        final_prediction = predictions["Ensemble Enhanced"]
        final_confidence = confidences.get("Ensemble Enhanced", 0)
    else:
        # Majority voting fallback
        prediction_values = list(predictions.values())
        final_prediction = max(set(prediction_values), key=prediction_values.count)
        final_confidence = sum(confidences.values()) / len(confidences) if confidences else 0

    # Updated suitability levels for 9 categories
    suitability_levels = {
        "Data Science": "Highly Suitable ⭐⭐⭐⭐⭐",
        "Machine Learning Engineer": "Highly Suitable ⭐⭐⭐⭐⭐",
        "Software Engineer": "Highly Suitable ⭐⭐⭐⭐",
        "Full Stack Developer": "Highly Suitable ⭐⭐⭐⭐",
        "DevOps Engineer": "Highly Suitable ⭐⭐⭐⭐",
        "Product Manager": "Highly Suitable ⭐⭐⭐⭐",
        "Frontend Developer": "Suitable ⭐⭐⭐",
        "Backend Developer": "Suitable ⭐⭐⭐",
        "Web Developer": "Suitable ⭐⭐⭐"
    }

    job_suitability = suitability_levels.get(final_prediction, "Suitable ⭐⭐⭐")

    # Display enhanced results
    print("\n" + "="*60)
    print("🤖 ENHANCED RESUME CLASSIFICATION RESULTS")
    print("="*60)
    
    print(f"\n🎯 **Final Prediction:** {final_prediction}")
    print(f"🔥 **Confidence:** {final_confidence:.1%}")
    print(f"⭐ **Suitability:** {job_suitability}")
    
    print(f"\n📊 **Individual Model Predictions:**")
    for model, category in predictions.items():
        conf = confidences.get(model, 0)
        print(f"🔸 {model}: {category} (Confidence: {conf:.1%})")
    
    print("="*60)

    return final_prediction, job_suitability

# Example usage
if __name__ == "__main__":
    resume_pdf_path = "D:/ML IA3 PROJECT/ResumeML.pdf"  # Change this to your actual PDF file path
    try:
        predict_job_suitability(resume_pdf_path)
    except FileNotFoundError:
        print("❌ PDF file not found. Please check the file path.")
    except Exception as e:
        print(f"❌ Error: {e}")