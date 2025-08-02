import os
import re
import string
import fitz  # PyMuPDF for PDF processing
import numpy as np
import pandas as pd
import nltk
import joblib

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Create necessary folders
os.makedirs("static", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

# ----------- Step 1: Load and preprocess dataset -----------
df = pd.read_csv("resume_data.csv")  # Make sure your dataset has 'Resume' and 'Category' columns

# Preprocess function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# Apply preprocessing
df["cleaned_text"] = df["Resume"].apply(preprocess_text)

# ----------- Step 2: Feature extraction using TF-IDF -----------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["cleaned_text"])

# ----------- Step 3: Label Encoding -----------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Category"])

# ----------- Step 4: Train-test split -----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------- Step 5: Define and train models -----------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)  # use encoded y_train
    y_pred = model.predict(X_test)
    print(f"\n📊 {name} Performance:\n")
    print(classification_report(y_test, y_pred))
    print("--------------------------------------------------")
    
    # Save model
    joblib.dump(model, f"saved_models/{name.replace(' ', '_')}.joblib")
    print(f"✅ {name} model saved as saved_models/{name.replace(' ', '_')}.joblib")

# ----------- Step 6: Save vectorizer and label encoder -----------
joblib.dump(vectorizer, "saved_models/vectorizer.joblib")
print("✅ Vectorizer saved as saved_models/vectorizer.joblib")

joblib.dump(label_encoder, "saved_models/label_encoder.joblib")
print("✅ Label Encoder saved as saved_models/label_encoder.joblib")
