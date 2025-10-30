"""
Simple Deep Learning Resume Classification - TensorFlow Only
Fast training and deployment for project demo
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🚀 DEEP LEARNING RESUME CLASSIFICATION - TENSORFLOW")
print("=" * 80)
print("\n📊 Loading Enhanced Resume Dataset...")

# Load dataset
df = pd.read_csv('comprehensive_resume_dataset.csv')
print(f"✅ Loaded {len(df)} resumes across {df['Category'].nunique()} categories")
print(f"\n📋 Categories: {df['Category'].unique().tolist()}")

# Prepare data
X = df['Resume'].values
y = df['Category'].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

print(f"\n🎯 Number of classes: {num_classes}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"📦 Training samples: {len(X_train)}")
print(f"📦 Testing samples: {len(X_test)}")

# ============================================================================
# TF-IDF VECTORIZATION
# ============================================================================
print("\n" + "=" * 80)
print("🔤 CREATING TF-IDF FEATURES")
print("=" * 80)

vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=1)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

feature_dim = X_train_tfidf.shape[1]
print(f"✅ Feature shape: {X_train_tfidf.shape}")
print(f"✅ Feature dimensions: {feature_dim}")

# ============================================================================
# MODEL 1: SIMPLE DEEP NEURAL NETWORK (DNN)
# ============================================================================
print("\n" + "=" * 80)
print("🧠 MODEL 1: SIMPLE DEEP NEURAL NETWORK")
print("=" * 80)

print("\n🏗️ Building Simple DNN...")
simple_dnn = keras.Sequential([
    layers.Input(shape=(feature_dim,)),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
], name='Simple_DNN')

simple_dnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🎓 Training Simple DNN (20 epochs)...")
history_simple = simple_dnn.fit(
    X_train_tfidf, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=8,
    verbose=1
)

# Evaluate
simple_pred = np.argmax(simple_dnn.predict(X_test_tfidf, verbose=0), axis=1)
simple_acc = accuracy_score(y_test, simple_pred)
print(f"\n🎯 Simple DNN Accuracy: {simple_acc * 100:.2f}%")

# ============================================================================
# MODEL 2: ADVANCED DEEP NEURAL NETWORK WITH BATCH NORMALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("🚀 MODEL 2: ADVANCED DEEP NEURAL NETWORK")
print("=" * 80)

print("\n🏗️ Building Advanced DNN with Batch Normalization...")
advanced_dnn = keras.Sequential([
    layers.Input(shape=(feature_dim,)),
    
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    
    layers.Dense(num_classes, activation='softmax')
], name='Advanced_DNN')

advanced_dnn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🎓 Training Advanced DNN (25 epochs with early stopping)...")
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001
)

history_advanced = advanced_dnn.fit(
    X_train_tfidf, y_train,
    validation_split=0.2,
    epochs=25,
    batch_size=8,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate
advanced_pred = np.argmax(advanced_dnn.predict(X_test_tfidf, verbose=0), axis=1)
advanced_acc = accuracy_score(y_test, advanced_pred)
print(f"\n🎯 Advanced DNN Accuracy: {advanced_acc * 100:.2f}%")

# ============================================================================
# MODEL 3: CNN FOR TEXT CLASSIFICATION
# ============================================================================
print("\n" + "=" * 80)
print("🌐 MODEL 3: CONVOLUTIONAL NEURAL NETWORK (CNN)")
print("=" * 80)

print("\n🏗️ Building CNN for Text Classification...")
# Reshape for CNN (add channel dimension)
X_train_cnn = X_train_tfidf.reshape(X_train_tfidf.shape[0], X_train_tfidf.shape[1], 1)
X_test_cnn = X_test_tfidf.reshape(X_test_tfidf.shape[0], X_test_tfidf.shape[1], 1)

cnn_model = keras.Sequential([
    layers.Input(shape=(feature_dim, 1)),
    
    layers.Conv1D(128, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Dropout(0.3),
    
    layers.Conv1D(64, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Dropout(0.3),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
], name='CNN_Text')

cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🎓 Training CNN (20 epochs)...")
history_cnn = cnn_model.fit(
    X_train_cnn, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=8,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate
cnn_pred = np.argmax(cnn_model.predict(X_test_cnn, verbose=0), axis=1)
cnn_acc = accuracy_score(y_test, cnn_pred)
print(f"\n🎯 CNN Accuracy: {cnn_acc * 100:.2f}%")

# ============================================================================
# MODEL 4: LSTM RECURRENT NEURAL NETWORK
# ============================================================================
print("\n" + "=" * 80)
print("🔄 MODEL 4: LSTM RECURRENT NEURAL NETWORK")
print("=" * 80)

print("\n🏗️ Building LSTM Network...")
lstm_model = keras.Sequential([
    layers.Input(shape=(feature_dim, 1)),
    
    layers.LSTM(128, return_sequences=True),
    layers.Dropout(0.3),
    
    layers.LSTM(64),
    layers.Dropout(0.3),
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
], name='LSTM_Network')

lstm_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🎓 Training LSTM (15 epochs - slower but powerful)...")
history_lstm = lstm_model.fit(
    X_train_cnn, y_train,
    validation_split=0.2,
    epochs=15,
    batch_size=8,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate
lstm_pred = np.argmax(lstm_model.predict(X_test_cnn, verbose=0), axis=1)
lstm_acc = accuracy_score(y_test, lstm_pred)
print(f"\n🎯 LSTM Accuracy: {lstm_acc * 100:.2f}%")

# ============================================================================
# ENSEMBLE OF ALL DEEP LEARNING MODELS
# ============================================================================
print("\n" + "=" * 80)
print("🏆 DEEP LEARNING ENSEMBLE")
print("=" * 80)

print("\n📊 Creating ensemble predictions...")
# Get probabilities from all models
simple_probs = simple_dnn.predict(X_test_tfidf, verbose=0)
advanced_probs = advanced_dnn.predict(X_test_tfidf, verbose=0)
cnn_probs = cnn_model.predict(X_test_cnn, verbose=0)
lstm_probs = lstm_model.predict(X_test_cnn, verbose=0)

# Average ensemble
ensemble_probs = (simple_probs + advanced_probs + cnn_probs + lstm_probs) / 4
ensemble_pred = np.argmax(ensemble_probs, axis=1)
ensemble_acc = accuracy_score(y_test, ensemble_pred)

print(f"\n🎯 Ensemble Accuracy: {ensemble_acc * 100:.2f}%")

# ============================================================================
# SAVE ALL MODELS
# ============================================================================
print("\n" + "=" * 80)
print("💾 SAVING MODELS")
print("=" * 80)

import os
os.makedirs('saved_models', exist_ok=True)

simple_dnn.save('saved_models/Simple_DNN.keras')
advanced_dnn.save('saved_models/Advanced_DNN.keras')
cnn_model.save('saved_models/CNN_Text.keras')
lstm_model.save('saved_models/LSTM_Network.keras')

joblib.dump(vectorizer, 'saved_models/vectorizer_dl.joblib')
joblib.dump(label_encoder, 'saved_models/label_encoder_dl.joblib')

print("✅ All models saved successfully!")

# ============================================================================
# FINAL RESULTS & CLASSIFICATION REPORTS
# ============================================================================
print("\n" + "=" * 80)
print("📊 FINAL RESULTS - ALL DEEP LEARNING MODELS")
print("=" * 80)

results = {
    'Simple DNN': simple_acc,
    'Advanced DNN': advanced_acc,
    'CNN': cnn_acc,
    'LSTM': lstm_acc,
    'Ensemble': ensemble_acc
}

print("\n🏆 Model Performance Comparison:")
for model_name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"   {model_name:15s}: {acc * 100:.2f}%")

# Best model classification report
best_model_name = max(results, key=results.get)
print(f"\n📈 Detailed Classification Report - {best_model_name}:")

if best_model_name == 'Simple DNN':
    best_pred = simple_pred
elif best_model_name == 'Advanced DNN':
    best_pred = advanced_pred
elif best_model_name == 'CNN':
    best_pred = cnn_pred
elif best_model_name == 'LSTM':
    best_pred = lstm_pred
else:
    best_pred = ensemble_pred

print(classification_report(y_test, best_pred, target_names=label_encoder.classes_))

print("\n" + "=" * 80)
print("✅ DEEP LEARNING TRAINING COMPLETE!")
print("=" * 80)
print("\n📁 Saved models:")
print("   - saved_models/Simple_DNN.keras")
print("   - saved_models/Advanced_DNN.keras")
print("   - saved_models/CNN_Text.keras")
print("   - saved_models/LSTM_Network.keras")
print("   - saved_models/vectorizer_dl.joblib")
print("   - saved_models/label_encoder_dl.joblib")
print("\n🚀 Ready to run: streamlit run app_dl_simple.py")
