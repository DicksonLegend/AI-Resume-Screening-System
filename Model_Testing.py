import joblib
import pymupdf as fitz


# Load the saved models
model_paths = {
    "Logistic Regression": "D:\ML IA3 PROJECT\saved_models\Logistic_Regression_Tuned.joblib",
    "Random Forest": "D:\ML IA3 PROJECT\saved_models\Random_Forest_Tuned.joblib",
    "XGBoost": "D:\ML IA3 PROJECT\saved_models\XGBoost_Tuned.joblib"
}
models = {name: joblib.load(path) for name, path in model_paths.items()}

# Load the saved vectorizer and label encoder
vectorizer = joblib.load("D:\\ML IA3 PROJECT\\saved_models\\vectorizer.joblib")
label_encoder = joblib.load("D:\\ML IA3 PROJECT\\saved_models\\label_encoder.joblib")

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    import fitz
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text")
    return text

# Function to preprocess text
def preprocess_text(text):
    import re, string
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# Predict suitability function
def predict_job_suitability(pdf_path):
    """Classify the resume and determine job suitability."""
    resume_text = extract_text_from_pdf(pdf_path)
    cleaned_text = preprocess_text(resume_text)

    # Convert text into numerical features using TF-IDF
    vectorized_text = vectorizer.transform([cleaned_text])

    # Get predictions from each model
    predictions = {name: model.predict(vectorized_text)[0] for name, model in models.items()}

    # Convert predictions back to category names
    predictions = {name: label_encoder.inverse_transform([pred])[0] for name, pred in predictions.items()}

    # Majority voting (If at least 2 models agree, use that label)
    prediction_values = list(predictions.values())
    final_prediction = max(set(prediction_values), key=prediction_values.count)

    # ✅ Define suitability levels based on new dataset
    suitability_levels = {
        "Data Scientist": "Highly Suitable",
        "Machine Learning Engineer": "Highly Suitable",
        "Full Stack Developer": "Highly Suitable",
        "Cloud Engineer": "Highly Suitable",
        "Backend Developer": "Moderate",
        "Frontend Developer": "Moderate",
        "Python Developer": "Moderate",
        "Mobile App Developer (iOS/Android)": "Moderate"
    }

    # Get job suitability level
    job_suitability = suitability_levels.get(final_prediction, "Moderate")

    # Display results
    print("\n🔹 **Resume Classification Results:**")
    for model, category in predictions.items():
        print(f"🔸 {model}: {category}")
    print(f"\n✅ **Final Decision:** {final_prediction} - **{job_suitability}**")

    return final_prediction, job_suitability

# Example usage
resume_pdf_path = "D:\ML IA3 PROJECT\ResumeML.pdf"  # Change this to your actual PDF file path
predict_job_suitability(resume_pdf_path)
