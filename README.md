# AI Resume Screening System

An intelligent resume screening application that uses machine learning to automatically classify resumes into job categories. The system employs multiple ML models with majority voting to provide accurate predictions and helps streamline the recruitment process.

## 🚀 Features

- **PDF Resume Upload**: Upload resume files in PDF format
- **Multi-Model Prediction**: Uses three fine-tuned models (Logistic Regression, Random Forest, XGBoost)
- **Majority Voting**: Combines predictions from all models for better accuracy
- **Job Suitability Assessment**: Provides suitability levels for different job categories
- **Real-time Processing**: Fast resume analysis with progress indicators
- **User-Friendly Interface**: Clean and intuitive Streamlit web interface

## 📋 Supported Job Categories

The system can classify resumes into the following categories:
- Data Scientist
- Data Analyst
- Machine Learning Engineer
- Full Stack Developer
- Cloud Engineer
- Backend Developer
- Frontend Developer
- Python Developer
- Mobile App Developer (iOS/Android)

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Machine Learning**: scikit-learn, XGBoost
- **Text Processing**: NLTK, TF-IDF Vectorization
- **PDF Processing**: PyMuPDF (fitz)
- **Model Persistence**: joblib
- **Data Processing**: pandas, numpy

## 📁 Project Structure

```
ML IA3 PROJECT/
│
├── app.py                    # Main Streamlit application
├── req.py                    # Requirements and dependencies
├── Model_Testing.py          # Model evaluation scripts
├── resume_dataset.csv        # Training dataset
├── db.sqlite3               # Database file
├── README.md                # This file
│
├── saved_models/            # Trained model files
│   ├── Logistic_Regression_Tuned.joblib
│   ├── Random_Forest_Tuned.joblib
│   ├── XGBoost_Tuned.joblib
│   ├── vectorizer.joblib
│   └── label_encoder.joblib
│
├── static/                  # Static files and notebooks
│   ├── fine_tune.ipynb     # Model fine-tuning notebook
│   └── Model_train.py      # Model training script
│
└── myenv/                  # Virtual environment
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
   pip install streamlit joblib PyMuPDF nltk scikit-learn xgboost pandas numpy
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

## 🧠 Model Training

The system uses three machine learning models that have been fine-tuned for optimal performance:

### Models Used:
- **Logistic Regression**: Fine-tuned with regularization parameters
- **Random Forest**: Optimized with tree parameters and feature selection
- **XGBoost**: Gradient boosting with hyperparameter optimization

### Training Process:
1. **Data Preprocessing**: Text cleaning, tokenization, stopword removal, lemmatization
2. **Feature Extraction**: TF-IDF vectorization with 5000 features
3. **Model Training**: Individual model training with cross-validation
4. **Hyperparameter Tuning**: Grid search and random search optimization
5. **Model Evaluation**: Performance assessment using classification metrics

## 📊 Model Performance

The models have been evaluated on a test dataset with the following components:
- **Text Preprocessing**: NLTK-based cleaning and normalization
- **Feature Engineering**: TF-IDF with optimized parameters
- **Ensemble Method**: Majority voting for final predictions
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score

## 🔧 Configuration

### Model Paths
Update the model paths in `app.py` if needed:
```python
model_paths = {
    "Logistic Regression": "path/to/Logistic_Regression_Tuned.joblib",
    "Random Forest": "path/to/Random_Forest_Tuned.joblib",
    "XGBoost": "path/to/XGBoost_Tuned.joblib"
}
```

### Suitability Levels
Customize job suitability levels in the `suitability_levels` dictionary in `app.py`.

## 📝 Usage Example

1. Launch the application using `streamlit run app.py`
2. Upload a PDF resume using the file uploader
3. Wait for the analysis to complete
4. View the predicted job category and suitability level
5. The system provides confidence through majority voting

## 🔍 How It Works

1. **PDF Text Extraction**: Uses PyMuPDF to extract text from uploaded PDF files
2. **Text Preprocessing**: Cleans and normalizes text using NLTK
3. **Vectorization**: Converts text to numerical features using TF-IDF
4. **Prediction**: Three models make independent predictions
5. **Ensemble**: Majority voting determines the final prediction
6. **Result Display**: Shows job category and suitability level

## 🚧 Development

### Model Retraining
To retrain models with new data:
1. Update `resume_dataset.csv` with new resume data
2. Run the training script: `python static/Model_train.py`
3. Use the fine-tuning notebook: `static/fine_tune.ipynb`

### Adding New Categories
1. Update the dataset with new job categories
2. Retrain all models
3. Update the suitability levels in `app.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This AI system is designed to assist in resume screening but should not be the sole basis for hiring decisions. Human review and judgment are essential in the recruitment process.

## 🆘 Support

For issues and questions:
1. Check the existing issues in the repository
2. Create a new issue with detailed description
3. Include error messages and steps to reproduce

## 🔮 Future Enhancements

- [ ] Support for multiple file formats (DOCX, TXT)
- [ ] Confidence scores for predictions
- [ ] Batch processing capabilities
- [ ] Advanced analytics dashboard
- [ ] API endpoints for integration
- [ ] Real-time model retraining
- [ ] Multi-language support

---

**Made with ❤️ for efficient resume screening**
