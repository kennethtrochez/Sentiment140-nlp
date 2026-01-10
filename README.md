# Sentiment140 NLP Pipeline

This project implements an end-to-end sentiment analysis pipeline using the Sentiment140 Twitter dataset. The goal is to build and evaluate multiple models for text classification.

## Project Structure
- src/data: data ingestion and preprocessing
- src/features: feature engineering
- src/models: model training and evaluation

## Models
- TF-IDF + Logistic Regression (baseline)

## Baseline Model: (TF-IDF & Logistic Regression)
As a baseline, the text is vectorized using TF-IDF (Term Frequency-Inverse Document Frequency) and is classified using a logistic regression model.

TF-IDF emphasizes words that are more informative in relation to the tweets while decreasing the weight on terms that appear frequently across the dataset. This is a fast and interpretable baseline for sentiment classification.

The model was trained on 80% of the dataset and evaluated on the 20% that was left out as the test set.

**Results:**
- Accuracy: 0.7917
- F1 Score: 0.7952

Classification Report:
precision    recall  f1-score   support

0       0.80      0.77      0.79    160000
1       0.78      0.81      0.80    160000

accuracy                           0.79    320000
macro avg       0.79      0.79      0.79    320000
weighted avg    0.79      0.79      0.79    320000

Confusion Matrix:
[[123943  36057]
[ 30584 129416]]