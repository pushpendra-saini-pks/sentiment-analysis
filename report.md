# Sentiment Analysis Project Report

## 1. Problem Statement
The objective of this project is to build a sentiment analysis model that can accurately classify text data into four sentiment classes: positive, negative, neutral, and irrelevant. While previous work in this domain has primarily focused on binary classification (positive and negative) using algorithms like Naive Bayes, Logistic Regression, and SVC with good accuracy, this project extends the challenge to multi-class classification. We employed five machine learning algorithms - Linear SVC, Random Forest, Decision Tree, KNN, and XGBoost - achieving accuracy rates above 90% for the four-class classification task. This multi-class approach provides more nuanced sentiment analysis capabilities compared to traditional binary classification methods.

## 2. Literature Review
Based on the review of five relevant papers in sentiment analysis, the following key observations were made:

1. In [1], Fatma Jemai, Mohamed Hayouni, and Sahbi Baccar evaluated various algorithms including Naive Bayes (NB), Multinomial NB, Bernoulli NB, SVM, and Logistic Regression. Their study utilized NLTK's Twitter corpus containing 20,000 non-sentimental tweets and an additional 10,000 tweets evenly split between positive and negative sentiments (5,000 each).

2. In [2], Anuja P Jain and Padma Dandannavar implemented Naive Bayes, SVC, and Decision Tree algorithms on Twitter data from different sectors: IT Industry (Apple), Bank (ICICI), and Telecom (BSNL). Their analysis covered varying dataset sizes of 200, 2000, and 4000 tweets.

3. In [3], H. Sankar and V. Subramaniyaswamy explored both supervised and unsupervised machine learning approaches for sentiment analysis.

4. In [4], Mohammed H. Abd El-Jawad, Rania Hodhod, and Yasser M. K. Omar conducted experiments with Naive Bayes, Recurrent Neural Networks, Decision Trees, Neural Networks, and Random Forest. Their dataset comprised 1,048,588 tweets collected from GitHub, Kaggle repository, and UCI archive of English language Twitter 140 API.

5. In [5], Sheresh Zahoor and Rajesh Rohilla applied Naive Bayes, SVM, LSTM, and Random Forest to analyze approximately 16,000 tweets from Haryana assembly and 5,000 tweets from UNGA.

## 3. Dataset Overview
The dataset used for this project was obtained from Kaggle and contains the following attributes:
- `serial_number`: Unique identifier for each data sample
- `source`: The source of the text data (e.g., social media, customer reviews)
- `sentiment`: The sentiment label of the text, which can be one of four classes: positive, negative, neutral, or irrelevant
- `text`: The actual text data

The dataset consists of:
- Training data: 74,682 samples
- Validation data: 1,000 samples

## 4. Preprocessing Techniques
The following preprocessing techniques were applied to the dataset:
1. **Remove URLs**: Removed any URLs from tweets, including those starting with HTTP, HTTPS, and pic:\\, replacing them with empty strings.
2. **Remove @ mentions**: Removed Twitter usernames preceded by @ symbols as they don't provide relevant sentiment information.
3. **Convert to Lowercase**: Converted all text to lowercase to standardize the input.
4. **Regular Expression**: Used to remove HTML tags and other special characters from the text data.
5. **Stop Word Removal**: Removed common words (e.g., "the", "a", "and") that do not contribute significantly to the sentiment of the text.
6. **TF-IDF Vectorization**: Transformed the text data into numerical feature vectors using the Term Frequency-Inverse Document Frequency (TF-IDF) technique.
7. **Label Encoding**: Converted the sentiment labels (positive, negative, neutral, irrelevant) into numerical values for model training.

## 5. Modeling Approaches
The following machine learning models were trained and evaluated on the preprocessed dataset:
1. **Linear SVC**: A linear support vector classification model.
2. **Random Forest**: An ensemble learning method that combines multiple decision trees.
3. **Decision Tree**: A tree-based model that makes decisions based on feature importance.
4. **K-Nearest Neighbors (KNN)**: A non-parametric method that classifies based on the nearest neighbors.
5. **XGBoost**: A gradient boosting decision tree algorithm known for its high performance.

## 6. Results and Evaluation
The performance of the models was evaluated using the classification report, which includes precision, recall, F1-score, and overall accuracy.

**Linear SVC**:
```
              precision    recall  f1-score   support

  Irrelevant       0.99      0.98      0.99       172
    Negative       0.98      0.98      0.98       266
     Neutral       0.99      0.99      0.99       285
    Positive       0.98      0.98      0.98       277

    accuracy                           0.99      1000
   macro avg       0.99      0.99      0.99      1000
weighted avg       0.99      0.99      0.99      1000

Accuracy: 0.986
```

**Random Forest**:
```
              precision    recall  f1-score   support

  Irrelevant       0.98      0.96      0.97       172
    Negative       0.95      0.98      0.96       266
     Neutral       0.98      0.95      0.97       285
    Positive       0.97      0.97      0.97       277

    accuracy                           0.97      1000
   macro avg       0.97      0.97      0.97      1000
weighted avg       0.97      0.97      0.97      1000

Accuracy: 0.967
```

**Decision Tree**:
```
              precision    recall  f1-score   support

  Irrelevant       0.96      0.91      0.93       172
    Negative       0.88      0.95      0.92       266
     Neutral       0.93      0.92      0.92       285
    Positive       0.92      0.90      0.91       277

    accuracy                           0.92      1000
   macro avg       0.92      0.92      0.92      1000
weighted avg       0.92      0.92      0.92      1000

Accuracy: 0.919
```

**KNN**:
```
              precision    recall  f1-score   support

  Irrelevant       0.97      0.99      0.98       172
    Negative       0.94      0.98      0.96       266
     Neutral       0.99      0.95      0.97       285
    Positive       0.99      0.97      0.98       277

    accuracy                           0.97      1000
   macro avg       0.97      0.97      0.97      1000
weighted avg       0.97      0.97      0.97      1000

Accuracy: 0.971
```

**XGBoost**:
```
              precision    recall  f1-score   support

           0       0.98      0.97      0.97       172
           1       0.97      0.98      0.98       266
           2       0.97      0.95      0.96       285
           3       0.95      0.97      0.96       277

    accuracy                           0.97      1000
   macro avg       0.97      0.97      0.97      1000
weighted avg       0.97      0.97      0.97      1000

Accuracy: 0.968
```

Based on the results, the Linear SVC model achieved the highest accuracy of 0.986, followed by KNN (0.971), XGBoost (0.968), Random Forest (0.967), and Decision Tree (0.919).

## 7. Conclusion and Future Work
This sentiment analysis project demonstrates the ability to classify text data into four sentiment classes with high accuracy. The use of various preprocessing techniques and machine learning models has shown promising results.

For future work, the following improvements can be considered:
- Explore more advanced deep learning models, such as transformer-based architectures (e.g., BERT, RoBERTa), which have shown state-of-the-art performance in many NLP tasks.
- Experiment with ensemble techniques that combine multiple models to further improve the overall performance.
- Investigate the impact of different feature engineering techniques on the model's performance.
- Expand the dataset by incorporating more diverse sources of text data to improve the model's generalization.

## References
1. Jemai, F., Hayouni, M., & Baccar, S. (2021). Twitter Sentiment Analysis Using Machine Learning. [Journal details needed]
2. Jain, A. P., & Dandannavar, P. (2021). Sentiment Analysis for Different Industry Sectors Using Twitter Data. [Journal details needed]
3. Sankar, H., & Subramaniyaswamy, V. (2021). Sentiment Analysis Using Machine Learning Approaches. [Journal details needed]
4. El-Jawad, M. H. A., Hodhod, R., & Omar, Y. M. K. (2021). Twitter Sentiment Analysis Using Different Machine Learning Techniques. [Journal details needed]
5. Zahoor, S., & Rohilla, R. (2021). Sentiment Analysis of Twitter Data Using Machine Learning Approaches. [Journal details needed]