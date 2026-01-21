# Sentiment140 NLP Pipeline

## Overview
This project implements an end-to-end sentiment analysis pipeline using the Sentiment140 Twitter dataset. The project progresses from a traditional TF-IDF + Logistic Regression baseline to a fine tuned DistilBERT transformer model. The pipeline covers data ingestion, preprocessing, feature engineering, model training, evaluation, and inference. 

The final DistilBERT model achieves strong generalization performance on a large subset of data as showcased on a 500k subset experiment. The model is deployed using FastAPI for real time use of any text a user can think of. The API is fully containerized with Docker to enable local deployment and personal use. 

This project demonstrates applied NLP modeling, performance benchmarking, and deploying a model as a usable API.

## Project Structure
- notebooks/: analysis and experiemenation

- reports/
    - figures/: evaluation plots and visuals

- src/
    - data/: data ingestion and preprocessing
    - features/: feature engineering
    - models/: model training and evaluation
    - api/: FastAPI inference service

- Dockerfile: container for API deployment
- requirements.txt: Python dependencies

## Models
- TF-IDF + Logistic Regression (baseline)
- DistilBERT

## Baseline Model: (TF-IDF & Logistic Regression)
As a baseline, the text is vectorized using TF-IDF (Term Frequency-Inverse Document Frequency) and is classified using a logistic regression model.

TF-IDF emphasizes words that are more informative in relation to the tweets while decreasing the weight on terms that appear frequently across the dataset. This is a fast and interpretable baseline for sentiment classification.

The model was trained on 80% of the dataset and evaluated on the 20% that was left out as the test set.

### **TF-IDF + Logistic Regression 50k Results:**
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

### **DistilBERT 200k Results:**
- Validation Accuracy: 0.8480
- Validation F1: 0.8513
- Test Accuracy: 0.8479
- Test F1: 0.8505

The validation and test scores being nearly identical shows a great sign that the model is generalizing well. This run on the 200k subset has seen a great jump in improvement compared to the basic 50k run. This 200k run will provide a solid benchmark to use when comparing to a 500k subset run and other transformer models.

### **DistilBERT 500k Results:**
- Validation Accuracy: 0.8494
- Validation F1: 0.8482
- Test Accuracy: 0.8530
- Test F1: 0.8505

Scaling the dataset fruther produced similar results to the 200k run. The improvement was not as substantial as the jump from the 50k prototype run to the fine tuned 200k run. However, the 500k subset run indicated that the model scales well by maintaining strong generalization, with validation and test metrics remaining closely aligned. This run represents the final benchmark for the DistilBERT model.

## **Docker Instructions:**

Prerequiste:
- Docker Desktop installed and running

### 1. Build the image:
From the project root: bash

docker build -t sentiment-api .

### 2. Run the container:
docker run -p 8000:8000 sentiment-api

If port 8000 is in use, use 8001:
docker run -p 8001:8000 sentiment-api

### 3. Open the API:
http://127.0.0.1:8000/docs (or 8001)

*To stop the container* : ctrl + C in the terminal