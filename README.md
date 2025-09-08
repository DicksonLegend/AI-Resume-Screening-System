# AI Resume Screening System - Enhanced Multi-Model Architecture

An advanced intelligent resume screening application that leverages state-of-the-art machine learning to automatically classify resumes into job categories. The system employs a sophisticated 4-model ensemble architecture with enhanced preprocessing and dynamic suitability scoring to provide highly accurate predictions (80.77% accuracy) and streamline the recruitment process.

## 🚀 Enhanced Features

- **PDF Resume Upload**: Upload resume files in PDF format with advanced text extraction
- **4-Model Ensemble System**: 
  - Logistic Regression Enhanced
  - Random Forest Enhanced 
  - XGBoost Enhanced
  - Ensemble Enhanced (Voting Classifier)
- **Dynamic Suitability Scoring**: Multi-factor scoring combining AI confidence, skill matching, experience detection, and model agreement
- **Enhanced Confidence Interpretation**: Advanced confidence analysis with model consensus scoring
- **Skills & Experience Analysis**: Automatic extraction and analysis of technical skills and work experience
- **Company Tier Recognition**: Identifies top-tier companies in work history
- **Education Level Detection**: Automatically detects highest education qualification
- **Real-time Processing**: Fast resume analysis with detailed progress indicators
- **Professional UI**: Clean and intuitive Streamlit interface with comprehensive analytics

## 📋 Supported Job Categories

The system classifies resumes into 9 specialized categories using an enhanced 90-resume dataset:
- Data Science
- Machine Learning Engineer
- Software Engineer
- Frontend Developer
- Backend Developer
- Full Stack Developer
- DevOps Engineer
- Web Developer
- Product Manager

## 🛠️ Enhanced Tech Stack

- **Frontend**: Streamlit 1.49.1 with professional UI components
- **Machine Learning**: 
  - scikit-learn 1.6.1 (Enhanced ensemble methods)
  - XGBoost 2.1.4 (Advanced gradient boosting)
  - 4-model ensemble architecture with voting classifier
- **Text Processing**: 
  - NLTK 3.9.1 for advanced preprocessing
  - Enhanced TF-IDF vectorization (5000 features)
  - Custom preprocessing pipeline with experience extraction
- **PDF Processing**: PyMuPDF (fitz) for robust text extraction
- **Model Persistence**: joblib for enhanced model serialization
- **Data Processing**: pandas, numpy with advanced feature engineering
- **Development**: Jupyter notebooks for model fine-tuning and analysis

## 📁 Enhanced Project Structure

```
ML IA3 PROJECT/
│
├── app.py                           # Enhanced Streamlit application with 4-model integration
├── req.py                           # Requirements and dependencies
├── Model_Testing.py                 # Enhanced model testing with preprocessing pipeline
├── comprehensive_resume_dataset.csv # Enhanced 90-resume training dataset
├── resume_dataset.csv              # Original dataset (45 resumes)
├── db.sqlite3                      # Application database
├── README.md                       # This comprehensive documentation
│
├── saved_models/                   # Enhanced trained model files
│   ├── Logistic_Regression_Enhanced.joblib    # Enhanced Logistic Regression
│   ├── Random_Forest_Enhanced.joblib          # Enhanced Random Forest  
│   ├── XGBoost_Enhanced.joblib               # Enhanced XGBoost
│   ├── Ensemble_Enhanced.joblib              # Enhanced Ensemble Voting Classifier
│   ├── vectorizer_enhanced.joblib            # Enhanced TF-IDF vectorizer
│   ├── label_encoder_enhanced.joblib         # Enhanced label encoder
│   ├── Logistic_Regression.joblib            # Original models (legacy)
│   ├── Random_Forest.joblib
│   ├── XGBoost.joblib
│   ├── vectorizer.joblib
│   └── label_encoder.joblib
│
├── static/                         # Development and training files
│   ├── fine_tune.ipynb            # Advanced hyperparameter tuning notebook
│   └── Model_train.py             # Enhanced model training script
│
└── myenv/                         # Virtual environment (excluded from Git)
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

3. **Install enhanced dependencies**:
   ```bash
   pip install streamlit==1.49.1 joblib PyMuPDF nltk==3.9.1 scikit-learn==1.6.1 xgboost==2.1.4 pandas numpy tqdm
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

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Upload a PDF resume** and get instant job category predictions!

## 🧠 Enhanced Multi-Model Architecture

The system utilizes a sophisticated 4-model ensemble approach with advanced preprocessing for optimal performance:

### Enhanced Model Suite:
1. **Logistic Regression Enhanced**: Fine-tuned with advanced regularization and feature selection
2. **Random Forest Enhanced**: Optimized with 50-iteration hyperparameter tuning
3. **XGBoost Enhanced**: Gradient boosting with 30-iteration optimization
4. **Ensemble Enhanced**: Voting classifier combining all models for superior accuracy

### Advanced Training Process:
1. **Enhanced Data Preprocessing**: 
   - Advanced text cleaning and normalization
   - Experience extraction from work history
   - Education level detection
   - Company tier recognition
   - Skills analysis and categorization

2. **Feature Engineering**: 
   - Enhanced TF-IDF vectorization with 5000 optimized features
   - Custom preprocessing pipeline with domain-specific enhancements
   - Advanced text normalization techniques

3. **Model Training**: 
   - Individual model training with cross-validation
   - Enhanced ensemble learning with voting mechanisms
   - Advanced regularization and overfitting prevention

4. **Hyperparameter Optimization**: 
   - Comprehensive grid search with 50+ iterations for Random Forest
   - Advanced random search optimization for XGBoost
   - Ensemble weight optimization for voting classifier

5. **Model Validation**: 
   - Rigorous performance assessment using multiple metrics
   - Cross-validation with stratified sampling
   - Enhanced confidence scoring and model agreement analysis

## 📊 Enhanced Model Performance

The enhanced models achieve **80.77% accuracy** on the comprehensive test dataset:

### Dataset Enhancement:
- **Enhanced Dataset**: 90 high-quality resumes across 9 job categories
- **Data Augmentation**: Advanced techniques for balanced representation
- **Realistic Content**: Professional resumes with diverse skill sets and experience levels

### Performance Metrics:
- **Overall Accuracy**: 80.77% (significant improvement from baseline 66.67%)
- **Enhanced Preprocessing**: Custom pipeline with experience and education extraction
- **Model Agreement**: Advanced consensus scoring across all 4 models
- **Dynamic Suitability**: Multi-factor scoring system combining:
  - AI confidence levels
  - Skills matching analysis
  - Experience relevance scoring
  - Model agreement consensus

### Advanced Features:
- **Confidence Interpretation**: Sophisticated confidence analysis with model consensus
- **Skills Analysis**: Automatic detection and categorization of technical skills
- **Experience Extraction**: Years of experience calculation from work history
- **Education Detection**: Highest qualification level identification
- **Company Recognition**: Top-tier company identification in work history

## 🔧 Enhanced Configuration

### Enhanced Model Paths
The application automatically loads enhanced models from the `saved_models/` directory:
```python
enhanced_model_paths = {
    "Logistic Regression Enhanced": "saved_models/Logistic_Regression_Enhanced.joblib",
    "Random Forest Enhanced": "saved_models/Random_Forest_Enhanced.joblib", 
    "XGBoost Enhanced": "saved_models/XGBoost_Enhanced.joblib",
    "Ensemble Enhanced": "saved_models/Ensemble_Enhanced.joblib"
}
```

### Dynamic Suitability Configuration
The system includes a sophisticated multi-factor scoring algorithm:
```python
def calculate_dynamic_suitability(confidence, skills_score, experience_score, model_agreement):
    # Multi-factor scoring with weighted components
    # Confidence: 40%, Skills: 25%, Experience: 25%, Agreement: 10%
```

### Enhanced Preprocessing Pipeline
Advanced text processing with domain-specific enhancements:
- Experience extraction patterns
- Education level detection
- Company tier recognition
- Technical skills categorization

## 📝 Enhanced Usage Example

1. Launch the enhanced application: `streamlit run app.py`
2. Upload a PDF resume using the file uploader
3. Watch the advanced preprocessing pipeline analyze the resume
4. View comprehensive results including:
   - **Primary job category prediction**
   - **Dynamic suitability score** (0-100%)
   - **Confidence interpretation** with model agreement analysis
   - **Skills analysis** with technical skill extraction
   - **Experience analysis** with years calculation
   - **Education level** detection
   - **Company tier** recognition
5. Explore detailed analytics and model consensus scoring

## 🔍 How the Enhanced System Works

1. **Advanced PDF Processing**: Enhanced text extraction with improved formatting preservation
2. **Multi-Stage Preprocessing**: 
   - Text cleaning and normalization
   - Experience extraction using regex patterns
   - Education level detection
   - Skills categorization and analysis
   - Company tier recognition
3. **Enhanced Vectorization**: Advanced TF-IDF with 5000 optimized features
4. **4-Model Prediction**: All enhanced models make independent predictions
5. **Ensemble Analysis**: Sophisticated voting and consensus scoring
6. **Dynamic Suitability**: Multi-factor scoring combining confidence, skills, experience, and agreement
7. **Comprehensive Results**: Professional display with detailed analytics and insights

## 🚧 Enhanced Development

### Model Retraining with Enhanced Dataset
To retrain models with the enhanced 90-resume dataset:
1. Use the comprehensive dataset: `comprehensive_resume_dataset.csv`
2. Run the enhanced training script: `python static/Model_train.py`
3. Use the advanced fine-tuning notebook: `static/fine_tune.ipynb`
4. Enhanced models will be saved with `_Enhanced.joblib` suffix

### Adding New Categories
1. Update `comprehensive_resume_dataset.csv` with new job categories
2. Retrain all 4 enhanced models
3. Update the enhanced preprocessing pipeline
4. Modify dynamic suitability calculations in `app.py`

### Enhanced Development Features
- **Jupyter Notebook Integration**: Advanced model development with `fine_tune.ipynb`
- **Enhanced Preprocessing**: Custom pipeline with domain-specific features
- **Model Versioning**: Separate enhanced and legacy model versions
- **Comprehensive Testing**: Enhanced testing script with preprocessing validation

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

- [x] **Enhanced 4-Model Architecture**: Implemented with 80.77% accuracy
- [x] **Dynamic Suitability Scoring**: Multi-factor scoring system
- [x] **Advanced Confidence Analysis**: Model consensus and agreement scoring
- [x] **Skills & Experience Extraction**: Automatic analysis and categorization
- [ ] Support for multiple file formats (DOCX, TXT)
- [ ] Batch processing capabilities with enhanced analytics
- [ ] Advanced analytics dashboard with model insights
- [ ] RESTful API endpoints for enterprise integration
- [ ] Real-time model retraining with feedback loops
- [ ] Multi-language support with enhanced NLP
- [ ] Deep learning integration (BERT, transformers)
- [ ] Advanced visualization of model decisions
- [ ] Enhanced bias detection and fairness metrics

---

**Enhanced AI Resume Screening System - 4-Model Architecture with 80.77% Accuracy** ✨

*Made with ❤️ for efficient and intelligent resume screening*
