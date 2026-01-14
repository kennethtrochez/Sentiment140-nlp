# Sentiment140 NLP Pipeline

This project implements an end-to-end sentiment analysis pipeline using the Sentiment140 Twitter dataset. The goal is to build and evaluate multiple models for text classification.

## Project Structure
- src/data: data ingestion and preprocessing
- src/features: feature engineering
- src/models: model training and evaluation

## Models
- TF-IDF + Logistic Regression (baseline)
- DistilBERT

## Baseline Model: (TF-IDF & Logistic Regression)
As a baseline, the text is vectorized using TF-IDF (Term Frequency-Inverse Document Frequency) and is classified using a logistic regression model.

TF-IDF emphasizes words that are more informative in relation to the tweets while decreasing the weight on terms that appear frequently across the dataset. This is a fast and interpretable baseline for sentiment classification.

The model was trained on 80% of the dataset and evaluated on the 20% that was left out as the test set.

**TF-IDF + Logistic Regression 50k Results:**
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
An initial run was performed on a 50k subset of the Sentiment140 dataset to verify that the transformer pipeline is working properly. This run achieved ~83% accuracy and ~82% F1 score on an unseen test set, demonstrating clear improvement over the TF-IDF baseline. The parameters on this run were very basic and untuned as well as missing a validation set.

After validating the pipeline, the model was ran on a 200k training subset with a dedicated 10k validation set. To keep experiements consistent across runs, the evaluation was performed on a fixed 200k subset of the test set. Through testing it was found that changing the maximum length from 128 to 64 tokens helped the training speed without losing too much information. 

**200k Results:**
- Validation Accuracy: 0.8480
- Validation F1: 0.8513
- Test Accuracy: 0.8479
- Test F1: 0.8505

The validation and test scores being nearly identical shows a great sign that the model is generalizing well. This run on the 200k subset has seen a great jump in improvement compared to the basic 50k run. This 200k run will provide a solid benchmark to use when comparing to a 500k subset run and other transformer models.

**500k Results:**
- Validation Accuracy: 0.8494
- Validation F1: 0.8482
- Test Accuracy: 0.8530
- Test F1: 0.8505

Scaling the dataset fruther produced similar results to the 200k run. The improvement was not as substantial as the jump from the 50k prototype run to the fine tuned 200k run. However, the 500k subset run indicated that the model scales well by maintaining strong generalization, with validation and test metrics remaining closely aligned. This run represents the final benchmark for the DistilBERT model.
