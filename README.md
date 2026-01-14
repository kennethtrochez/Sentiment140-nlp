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

## DistilBERT
An initial run was performed on a 50k subset of the Sentiment140 dataset to verify that the transformer pipeline is working properly. This run achieved ~83% accuracy and ~82% F1 score on an unseen test set, demonstrating clear improvement over the Tf-IDF baseline. The model was then scaled to a 200k training subset with proper training/test split with a dedicated validation set. During early experimentation, the validation accuracy hit ~97%, while the test accuracy remained stable around 83%. This behavior is consistent with overfitting and validation bias, especially when training large transfromer models on short, repetitive texts such as tweets and evaluating on the same validation set multiple times during training.
To address this, early stopping and best-checkpoint selection were added, using validation F1 score as the model selection metric. F1 was chosen over loss because it better reflects classification performance and avoids cases where avlidation loss continues to decrease without improving generalization. With early stopping enables, training consistently stopped well before completing a full epoch, with validation and test performance aligned at around 83% F1. Overall, DistilBERT provides a clear improvement over the TF-IDF & Logistic Regression baseline, while demonstrating the importance of detailed evaluation when working on weakly supervised datasets.

