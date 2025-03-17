# Sentiment-Analysis-of-IMDB-Reviews# Sentiment Analysis of IMDB Reviews

### Overview
This project implements sentiment analysis on IMDB movie reviews using two algorithms: Naive Bayes (implemented from scratch) and Logistic Regression (using scikit-learn). The goal is to classify reviews as either positive or negative based on their textual content. The system includes text preprocessing, model training, and evaluation with detailed performance metrics.

### Dataset
- **Source**: IMDB dataset containing 50,000 movie reviews.
- **Labels**: Positive and Negative sentiment classes.
- **Split**: 80% training data and 20% testing data.

### Approach
#### Naive Bayes:
- **Architecture**: Implemented using a Non-Binary Bag of Words representation with add-1 smoothing.
- **Vocabulary**: All unique words extracted from the training data.
- **Probabilities**: Stored as Python dictionaries for each sentiment class, leveraging log-space calculations to handle underflow.

#### Logistic Regression:
- **Architecture**: Utilized scikit-learn's `LogisticRegression` model.
- **Features**: Transformed text documents into numerical representations using custom feature extraction.
- **Training**: Models trained on the Bag-of-Words features derived from the dataset.

### Evaluation Metrics
Both models were evaluated on the test set using standard metrics:
- True Positives, True Negatives, False Positives, False Negatives
- Sensitivity (Recall)
- Specificity
- Precision
- Negative Predictive Value
- Accuracy
- F-score

### Results
#### Naive Bayes:
- **Accuracy**: 84.8%
- **F-Score**: 0.8429
- **Precision**: 87.36%
- **Recall**: 81.3%
- **AUC**: 0.92

#### Logistic Regression:
- **Accuracy**: 83.0%
- **F-Score**: 0.827
- **Precision**: 87.1%
- **Recall**: 78.9%
- **AUC**: 0.50

Performance visualizations include ROC curves and confusion matrices.

### Model Performance
#### Features
- Text preprocessing (removal of unwanted characters, conversion to lowercase, tokenization, and label encoding).
- Implementation of Non-Binary Bag of Words for feature extraction.
- Add-1 smoothing for Naive Bayes.
- Comparison of both classifiers on the IMDB dataset.
- User input classification, allowing real-time analysis of user-generated text.

### Observations
- **Naive Bayes** outperformed Logistic Regression in terms of balanced metrics such as accuracy and recall, making it more effective for identifying positive cases.
- **Logistic Regression** exhibited higher negative predictive value, making it more suitable for scenarios prioritizing negative predictions.
- Both models demonstrated strong performance, with trade-offs in precision versus recall depending on the algorithm.

### Future Improvements
- Experimentation with advanced models, such as SVMs or neural networks.
- Incorporation of additional features, such as bigrams and sentiment lexicons.
- Hyperparameter tuning for optimized performance.
- Addressing class imbalance to improve recall further.

### Authors
**Harshith Deshalli Ravi**
