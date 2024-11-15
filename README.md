# Twitter Sentiment Analysis Project

## Overview
This project implements a multi-class sentiment analysis system that classifies tweets into four categories: positive, negative, neutral, and irrelevant. Unlike traditional binary sentiment classification, this approach provides more nuanced analysis of social media content using various machine learning algorithms.

## Tech Stack
### Programming Language
- Python 3.8+

### Machine Learning Libraries
- scikit-learn (for ML models and preprocessing)
- xgboost (for XGBoost implementation)
- nltk (for text processing)

### Data Processing
- pandas (for data manipulation)
- numpy (for numerical operations)

### Development Tools
- Jupyter Notebook (for interactive development)
- Git (for version control)
- pip (for package management)

### Visualization
- matplotlib
- seaborn
- plotly

## Environment Setup

### 1. Create Virtual Environment
```bash
# Using venv
python -m venv sentiment_env
# using anaconda 
conda create -p venv python==3.12 -y
# Activate virtual environment
# On Windows
sentiment_env\Scripts\activate
# On Unix or MacOS
source sentiment_env/bin/activate
```

### 2. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt

# Install NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 3. Configure Jupyter Notebook (Optional)
```bash
# Install Jupyter kernel for the virtual environment
pip install ipykernel
python -m ipykernel install --user --name=sentiment_env
```


## Requirements
All dependencies are listed in `requirements.txt`:
```
scikit-learn==1.0.2
pandas==1.4.2
numpy==1.22.3
xgboost==1.5.2
nltk==3.7
jupyter==1.0.0
matplotlib==3.5.2
seaborn==0.11.2
plotly==5.7.0
python-dotenv==0.20.0
```

## Dataset
The dataset used in this project is obtained from Kaggle and contains:
- Training data: 74,682 samples
- Validation data: 1,000 samples

Each sample contains:
- serial_number: Unique identifier
- source: Text source
- sentiment: Label (positive, negative, neutral, irrelevant)
- text: Tweet content

## Data Preprocessing
The preprocessing pipeline includes:
1. URL removal
2. @mentions removal
3. Lowercase conversion
4. Special character removal
5. Stop word removal
6. TF-IDF vectorization
7. Label encoding

## Models
The project implements and compares five machine learning models:
1. Linear SVC (Best performing - 98.6% accuracy)
2. K-Nearest Neighbors (97.1% accuracy)
3. XGBoost (96.8% accuracy)
4. Random Forest (96.7% accuracy)
5. Decision Tree (91.9% accuracy)

## Usage
1. Clone the repository:
```bash
git clone https://github.com/pushpendra-saini-pks/sentiment-analysis.git
cd sentiment-analysis
```

2. Setup environment and install dependencies (see Environment Setup section)



## Model Performance
Best performing model (Linear SVC):
```
              precision    recall  f1-score   support
  Irrelevant       0.99      0.98      0.99       172
    Negative       0.98      0.98      0.98       266
     Neutral       0.99      0.99      0.99       285
    Positive       0.98      0.98      0.98       277
```

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Future Work
- Implementation of deep learning models (BERT, RoBERTa)
- Ensemble modeling techniques
- Feature engineering improvements
- Dataset expansion

## References
1. Jemai, F., Hayouni, M., & Baccar, S. (2021). Twitter Sentiment Analysis Using Machine Learning
2. Jain, A. P., & Dandannavar, P. (2021). Sentiment Analysis for Different Industry Sectors Using Twitter Data
3. Sankar, H., & Subramaniyaswamy, V. (2021). Sentiment Analysis Using Machine Learning Approaches
4. El-Jawad, M. H. A., Hodhod, R., & Omar, Y. M. K. (2021). Twitter Sentiment Analysis Using Different Machine Learning Techniques
5. Zahoor, S., & Rohilla, R. (2021). Sentiment Analysis of Twitter Data Using Machine Learning Approaches

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Kaggle for providing the dataset
- Contributors to the scikit-learn and other open-source libraries used in this project

