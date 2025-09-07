import os
import re
import string
import fitz  # PyMuPDF for PDF processing
import numpy as np
import pandas as pd
import nltk
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Create necessary folders
os.makedirs("static", exist_ok=True)
os.makedirs("models", exist_ok=True)  # Use models folder instead of saved_models
os.makedirs("saved_models", exist_ok=True)  # Keep saved_models for backward compatibility

print("🚀 Starting Resume Classification Model Training...")
print(f"📅 Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

# ----------- Step 1: Load and preprocess dataset -----------
print("📂 Loading dataset...")
df = pd.read_csv("comprehensive_resume_dataset.csv")  # Large multi-category dataset with 50 realistic resumes

print(f"✅ Dataset loaded successfully!")
print(f"📊 Dataset shape: {df.shape}")
print(f"📋 Categories: {df['Category'].unique()}")
print(f"📈 Category distribution:\n{df['Category'].value_counts()}")
print("-" * 40)

# Preprocess function
def preprocess_text(text):
    """Enhanced text preprocessing with better cleaning"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    # Remove special characters but keep important ones
    text = re.sub(r'[^\w\s\+\#\.]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words("english") and len(word) > 2]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

print("🔄 Preprocessing text data...")
# Apply preprocessing
df["cleaned_text"] = df["Resume"].apply(preprocess_text)
print("✅ Text preprocessing completed!")
print("-" * 40)

# ----------- Step 2: Feature extraction using TF-IDF -----------
print("🔤 Extracting features using TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),  # Include unigrams and bigrams
    min_df=2,  # Ignore terms that appear in less than 2 documents
    max_df=0.8,  # Ignore terms that appear in more than 80% of documents
    stop_words='english'
)
X = vectorizer.fit_transform(df["cleaned_text"])
print(f"✅ Feature extraction completed! Shape: {X.shape}")
print("-" * 40)

# ----------- Step 3: Label Encoding -----------
print("🏷️ Encoding labels...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Category"])
print(f"✅ Label encoding completed! Classes: {label_encoder.classes_}")
print("-" * 40)

# ----------- Step 4: Train-test split -----------
print("🔀 Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Ensure balanced split across all categories
)
print(f"✅ Data split completed!")
print(f"📊 Training set shape: {X_train.shape}")
print(f"📊 Test set shape: {X_test.shape}")
print("-" * 40)

# ----------- Step 5: Define and train models (basic configurations) -----------
print("🤖 Training machine learning models...")

# Define models with basic configurations
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
}

# Store results for comparison
results = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Cross-validation score (reduce CV folds for small dataset)
    cv_folds = min(3, len(y_train) // len(label_encoder.classes_))  # Adaptive CV based on dataset size
    if cv_folds < 2:
        cv_folds = 2
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'f1_score': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"✅ {name} Training completed!")
    print(f"📊 Test Accuracy: {accuracy:.4f}")
    print(f"📊 F1 Score: {f1:.4f}")
    print(f"📊 CV Score ({cv_folds}-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Save model to models folder
    model_filename = f"models/{name.replace(' ', '_')}.joblib"
    joblib.dump(model, model_filename)
    print(f"💾 {name} model saved as {model_filename}")
    
    # Also save to saved_models for backward compatibility
    legacy_filename = f"saved_models/{name.replace(' ', '_')}.joblib"
    joblib.dump(model, legacy_filename)
    print(f"💾 {name} model also saved as {legacy_filename}")
    
    print("=" * 60)

# ----------- Step 6: Save vectorizer and label encoder -----------
print("💾 Saving preprocessing components...")

# Save to models folder
vectorizer_path = "models/vectorizer.joblib"
label_encoder_path = "models/label_encoder.joblib"

joblib.dump(vectorizer, vectorizer_path)
joblib.dump(label_encoder, label_encoder_path)

print(f"✅ Vectorizer saved as {vectorizer_path}")
print(f"✅ Label Encoder saved as {label_encoder_path}")

# Also save to saved_models for backward compatibility
joblib.dump(vectorizer, "saved_models/vectorizer.joblib")
joblib.dump(label_encoder, "saved_models/label_encoder.joblib")
print("✅ Backup copies saved to saved_models folder")

# ----------- Step 7: Model comparison and summary -----------
print("\n" + "=" * 60)
print("📊 MODEL COMPARISON SUMMARY")
print("=" * 60)

best_model = max(results, key=lambda x: results[x]['accuracy'])
print(f"🏆 Best Model: {best_model}")
print(f"🎯 Best Accuracy: {results[best_model]['accuracy']:.4f}")

print(f"\n📈 All Model Results:")
for model_name, metrics in results.items():
    print(f"• {model_name}:")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - F1 Score: {metrics['f1_score']:.4f}")
    print(f"  - CV Score: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")

print("\n" + "=" * 60)
print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
print(f"📅 Training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📂 Models saved in: ./models/ folder")
print(f"📂 Backup models in: ./saved_models/ folder")
print("=" * 60)
