import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_fscore_support, 
    accuracy_score
)
import math
import re
import seaborn as sns
from collections import Counter
from sklearn.linear_model import LogisticRegression


class TextClassifier:
    def __init__(self, algorithm='naive_bayes'):
        self.algorithm = algorithm
        self.vocab = set()
        self.class_probs = {}
        self.word_probs = {}
        self.label_encoder = LabelEncoder()
        
        # Logistic regression
        self.lr_model = LogisticRegression(random_state=42, max_iter=1000)
        
    def preprocess_text(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        return text.split()
    
    def create_bow_vector(self, text, vocab=None):
        if vocab is None:
            vocab = self.vocab
        
        word_counts = Counter(word for word in text if word in vocab)
        return word_counts
    
    def train_naive_bayes(self, X_train, y_train):
        all_words = [word for doc in X_train for word in doc]
        self.vocab = set(all_words)
        
        total_docs = len(y_train)
        unique_classes = np.unique(y_train)
        self.class_probs = {c: np.sum(y_train == c) / total_docs for c in unique_classes}
        
        self.word_probs = {c: Counter() for c in unique_classes}
        class_word_counts = {c: Counter() for c in unique_classes}
        
        for doc, label in zip(X_train, y_train):
            class_word_counts[label].update(doc)
        
        for cls in unique_classes:
            total_words = sum(class_word_counts[cls].values())
            for word in self.vocab:
                word_count = class_word_counts[cls][word]
                self.word_probs[cls][word] = (word_count + 1) / (total_words + len(self.vocab))
    
    def predict_naive_bayes(self, document):
        bow = self.create_bow_vector(document)
        class_scores = {}
        
        for cls in self.class_probs.keys():
            log_prob = math.log(self.class_probs[cls] + 1e-10)
            for word, count in bow.items():
                log_prob += count * math.log(self.word_probs[cls].get(word, 1e-10))
            class_scores[cls] = log_prob
        
        max_score = max(class_scores.values())
        exp_scores = {cls: math.exp(score - max_score) for cls, score in class_scores.items()}
        total = sum(exp_scores.values())
        probs = {cls: score/total for cls, score in exp_scores.items()}
        
        predicted_class = max(class_scores, key=class_scores.get)
        return predicted_class, probs
    
    def train(self, X_train, y_train):
        processed_train = [self.preprocess_text(doc) for doc in X_train]
        
        if self.algorithm == 'naive_bayes':
            self.train_naive_bayes(processed_train, y_train)
        else:
            X_bow = np.array([
                sum(self.create_bow_vector(doc, self.vocab).values()) 
                for doc in processed_train
            ]).reshape(-1, 1)
            
            self.lr_model.fit(X_bow, y_train)
    
    def predict(self, document):
        processed_doc = self.preprocess_text(document)
        
        if self.algorithm == 'naive_bayes':
            return self.predict_naive_bayes(processed_doc)
        else:
            bow_vector = sum(self.create_bow_vector(processed_doc, self.vocab).values())
            prediction = self.lr_model.predict([[bow_vector]])[0]
            proba = self.lr_model.predict_proba([[bow_vector]])[0]
            return prediction, dict(zip(self.lr_model.classes_, proba))
        
def load_dataset(filepath):
    df = pd.read_csv(filepath)
    return df['review'].values, df['sentiment'].values

def evaluate_model(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    metrics = {
        'True Positives': tp,
        'True Negatives': tn,
        'False Positives': fp,
        'False Negatives': fn,
        'Sensitivity (Recall)': recall,
        'Specificity': tn / (tn + fp),
        'Precision': precision,
        'Negative Predictive Value': tn / (tn + fn),
        'Accuracy': accuracy_score(y_true, y_pred),
        'F-score': f1
    }
    
    return metrics, (tn, fp, fn, tp)

def plot_roc_and_confusion_matrix(y_true_nb, y_pred_nb, y_proba_nb, 
                                   y_true_lr, y_pred_lr, y_proba_lr):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    fpr_nb, tpr_nb, _ = roc_curve(y_true_nb, y_proba_nb)
    roc_auc_nb = auc(fpr_nb, tpr_nb)
    
    fpr_lr, tpr_lr, _ = roc_curve(y_true_lr, y_proba_lr)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    
    plt.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC = {roc_auc_nb:.2f})')
    plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    plt.subplot(132)
    cm_nb = confusion_matrix(y_true_nb, y_pred_nb)
    sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.title('Naive Bayes Confusion Matrix')
    
    plt.subplot(133)
    cm_lr = confusion_matrix(y_true_lr, y_pred_lr)
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.title('Logistic Regression Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('model_performance.png')
    plt.close()


def parse_arguments():
    algo = 0  
    train_size = 80  
    
    if len(sys.argv) == 3:
        try:
            algo = int(sys.argv[1])
            if algo not in [0, 1]:
                print("Invalid algorithm. Using default (0 - Naive Bayes).")
                algo = 0
            
            train_size = int(sys.argv[2])
            if train_size < 50 or train_size > 80:
                print("Train size must be between 50 and 80. Using default (80%).")
                train_size = 80
                
        except ValueError:
            print("Invalid arguments. Using defaults.")
            
    else:
        print("Usage: python cs585_P03_A20552830.py <algorithm> <train_size>")
        print("Using default: Naive Bayes with 80% training size")
    
    return algo, train_size/100

def main():
    algo, train_size = parse_arguments()
    file_path = 'IMDB Dataset_cleaned.csv'
    X, y = load_dataset('IMDB Dataset_cleaned.csv')
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    X_train, X_test = X[:int(len(X) * train_size)], X[int(len(X) * train_size):]
    y_train, y_test = y[:int(len(X) * train_size)], y[int(len(X) * train_size):]

    
    print(f"Deshalli Ravi, Harshith, A20552830 solution:")
    print(f"Training set size: {float(train_size)*100} %")
    
    results = {}
    
    if algo == 0:
        classifier = TextClassifier('naive_bayes')
        print(f"Classifier type: naive bayes")
    
        print("\nTraining classifier...")
        
        try:
            classifier.train(X_train, y_train)
        except Exception as e:
            print(f"Training failed: {e}")
            pass
        
        print("Testing classifier...")
        
        y_pred = [classifier.predict(doc)[0] for doc in X_test]
        
        y_proba = [classifier.predict(doc)[1][1] for doc in X_test]
        
        metrics, confusion = evaluate_model(y_test, y_pred, y_proba)
        
        results['naive_bayes'] = {
            'metrics': metrics,
            'confusion': confusion,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
    else:
        classifier = TextClassifier('logistic_regression')
        print(f"Classifier type: logistic regression")
    
        print("\nTraining classifier...")
        
        try:
            classifier.train(X_train, y_train)
        except Exception as e:
            print(f"Training failed: {e}")
            pass
        
        print("Testing classifier...")
        
        y_pred = [classifier.predict(doc)[0] for doc in X_test]
        
        y_proba = classifier.lr_model.predict_proba(
            [[sum(classifier.create_bow_vector(classifier.preprocess_text(doc), classifier.vocab).values())] for doc in X_test]
        )[:, 1]
        
        metrics, confusion = evaluate_model(y_test, y_pred, y_proba)
        
        results['logistic_regression'] = {
            'metrics': metrics,
            'confusion': confusion,
            'predictions': y_pred,
            'probabilities': y_proba
        }
    
    
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    
    while True:
        sentence = input("\nEnter your sentence/document: ")
        predicted_class, probs = classifier.predict(sentence)
        
        if predicted_class == 0:
            print(f"was classified as negative")
        else:
            print(f"was classified as positive")
        print(f"P(Class negative | S) = {probs[0]:.4f}")
        print(f"P(Class positive | S) = {probs[1]:.4f}")
        
        cont = input("\nDo you want to enter another sentence [Y/N]? ").upper()
        if cont != 'Y':
            break

if __name__ == "__main__":
    main()