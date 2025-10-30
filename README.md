# AI Resume Screening System - Deep Learning & ML Ensemble

An advanced intelligent resume screening application using **Deep Learning** (DNN, CNN, LSTM) and traditional ML to automatically classify resumes into job categories. The system employs 4 deep learning models plus 3 traditional ML models with enhanced preprocessing to provide accurate predictions and streamline recruitment.

## 🚀 Enhanced Features

- **PDF Resume Upload**: Upload resume files in PDF format with advanced text extraction
- **Deep Learning Models**: 
  - Simple DNN (69% accuracy)
  - Advanced DNN with Batch Normalization
  - CNN for Text Classification (73% accuracy - Best!)
  - LSTM Recurrent Network
- **Traditional ML Ensemble**: 
  - Logistic Regression, Random Forest, XGBoost (80% accuracy)
- **Ensemble Voting**: Combines all models for robust predictions
- **Skills & Experience Analysis**: Automatic extraction and categorization
- **Real-time Processing**: Fast resume analysis with progress indicators
- **Professional UI**: Clean Streamlit interface with model predictions

## 📋 Supported Job Categories

The system classifies resumes into 9 categories using a 129-resume dataset:
- Data Science
- Machine Learning Engineer
- Software Engineer
- Frontend Developer
- Backend Developer
- Full Stack Developer
- DevOps Engineer
- Web Developer
- Product Manager

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Deep Learning**: TensorFlow/Keras (DNN, CNN, LSTM architectures)
- **Traditional ML**: scikit-learn, XGBoost
- **Text Processing**: NLTK, TF-IDF Vectorization
- **PDF Processing**: PyMuPDF (fitz)
- **Model Persistence**: joblib, Keras (.keras format)
- **Data Processing**: pandas, numpy

## 📁 Project Structure

```
ML IA3 PROJECT/
│
├── app_dl_simple.py                # Deep Learning Streamlit application
├── train_dl_simple.py              # Train all 4 DL models (DNN, CNN, LSTM)
├── comprehensive_resume_dataset.csv # Training dataset (129 resumes)
├── requirements_simple.txt         # Dependencies (TensorFlow only)
├── DEEP_LEARNING_PROJECT_GUIDE.md  # Presentation guide
│
├── saved_models/                   # Trained model files
│   ├── Simple_DNN.keras            # Deep Neural Network
│   ├── Advanced_DNN.keras          # DNN with Batch Normalization
│   ├── CNN_Text.keras              # Convolutional Neural Network (Best: 73%)
│   ├── LSTM_Network.keras          # LSTM Recurrent Network
│   ├── vectorizer_dl.joblib        # DL vectorizer (3000 features)
│   ├── vectorizer_enhanced.joblib  # ML vectorizer (5000 features)
│   └── label_encoder_*.joblib      # Label encoders
│
├── static/                         # Training scripts
└── myenv/                          # Virtual environment
```

## ⚡ Quick Start

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "ML IA3 PROJECT"
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv myenv
   myenv\Scripts\activate  # On Windows
   # source myenv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements_simple.txt
   # OR manually:
   pip install streamlit tensorflow scikit-learn xgboost nltk PyMuPDF pandas numpy joblib
   ```

4. **Download NLTK data**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   nltk.download('punkt_tab')
   ```

### Running the Application

1. **Train Deep Learning models** (first time only):
   ```bash
   python train_dl_simple.py
   ```

2. **Start the Streamlit app**:
   ```bash
   streamlit run app_dl_simple.py
   ```

3. **Open your browser** and navigate to `http://localhost:8501`

4. **Upload a PDF resume** and see predictions from all models!

## 🧠 Deep Learning + ML Architecture

### Deep Learning Models (TensorFlow/Keras):
1. **Simple DNN**: 3-layer feedforward network with dropout (69% accuracy)
2. **Advanced DNN**: 5-layer network with Batch Normalization and learning rate scheduling
3. **CNN**: Convolutional network for text feature extraction (73% accuracy - Best!)
4. **LSTM**: Recurrent network for sequential pattern recognition

### Traditional ML Models (scikit-learn):
1. **Logistic Regression**: With L2 regularization
2. **Random Forest**: Ensemble decision trees
3. **XGBoost**: Gradient boosting (80% accuracy on traditional features)

### Training Process:
1. **Text Preprocessing**: Cleaning, normalization, TF-IDF vectorization
2. **Deep Learning**: 
   - 3000 TF-IDF features for DL models
   - Dropout (0.3-0.5) and Batch Normalization
   - Early stopping and learning rate reduction
3. **Traditional ML**: 5000 TF-IDF features with hyperparameter tuning
4. **Ensemble**: Majority voting across all 7 models

## 📊 Model Performance

### Deep Learning Models:
- **CNN**: 73.08% ⭐ (Best DL model)
- **Simple DNN**: 69.23%
- **Ensemble DL**: 73.08%
- **Advanced DNN**: 19.23% (early stopped)
- **LSTM**: 11.54%

### Traditional ML Models:
- **Combined Accuracy**: ~80% (Logistic Regression, Random Forest, XGBoost)

### Key Features:
- Multiple architectures capture different patterns
- Ensemble voting for robust predictions
- CNN excels at local feature detection (skills, keywords)
- Traditional ML strong on overall classification

## 🔧 Configuration

### Model Files
Models are automatically loaded from `saved_models/`:
- Deep Learning: `*.keras` files (TensorFlow/Keras format)
- Traditional ML: `*.joblib` files (scikit-learn format)

### Preprocessing
- DL models: 3000 TF-IDF features
- ML models: 5000 TF-IDF features
- Enhanced text cleaning with experience/education extraction

## 📝 Usage Example

1. Launch: `streamlit run app_dl_simple.py`
2. Upload a PDF resume
3. View predictions from:
   - 4 Deep Learning models (Simple DNN, Advanced DNN, CNN, LSTM)
   - 3 Traditional ML models (Logistic Regression, Random Forest, XGBoost)
4. See final ensemble decision with model agreement percentage
5. Review confidence scores and detailed breakdown

## 🔍 How It Works

1. **PDF Processing**: Text extraction from uploaded resume
2. **Preprocessing**: Text cleaning, normalization, feature extraction
3. **Vectorization**: TF-IDF transformation (3000 for DL, 5000 for ML)
4. **Deep Learning Prediction**: DNN, CNN, LSTM analyze text patterns
5. **Traditional ML Prediction**: Logistic Regression, Random Forest, XGBoost classify
6. **Ensemble Voting**: Combine all predictions via majority voting
7. **Results Display**: Show individual model predictions + final decision

## 🚧 Development

### Retrain Models
```bash
# Train all 4 deep learning models
python train_dl_simple.py

# Train traditional ML models
python static/Model_train.py
```

### Add New Categories
1. Update `comprehensive_resume_dataset.csv`
2. Retrain all models
3. Models will automatically learn new categories

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 🆘 Support

For issues and questions:
1. Check the existing issues in the repository
2. Create a new issue with detailed description
3. Include error messages and steps to reproduce

## 🔮 Future Enhancements

- [x] **Deep Learning Models**: DNN, CNN, LSTM implemented
- [x] **Ensemble Voting**: Combines 7 models
- [ ] BERT/Transformer models for 90%+ accuracy
- [ ] Attention mechanisms for explainability
- [ ] Batch processing for multiple resumes
- [ ] RESTful API for integration
- [ ] Multi-language support
- [ ] Real-time model retraining

---

**AI Resume Screening System - Deep Learning (DNN, CNN, LSTM) + Traditional ML Ensemble** ✨

*Made with ❤️ using TensorFlow, Keras & scikit-learn*
