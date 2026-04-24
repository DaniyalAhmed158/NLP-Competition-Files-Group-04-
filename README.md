Task 1: Classidication using DistilBERT
Binary Classification Project: Distress vs. Normal Text. 
This repository contains three distinct approaches to solving a binary classification task, specifically designed to identify "Distress" (Class 1) vs. "Normal" (Class 0) patterns in text data. The project addresses a significant class imbalance (approximately 79:21 split) using various machine learning and deep learning strategies.
Project StructureThe project is divided into three primary experimental notebooks:1. Model 1: XGBoost BaselineArchitecture: Gradient Boosted Decision Trees using the XGBoost library.Key Features:Implements a robust baseline for binary classification.Monitors training performance using Log-Loss metrics.Utilizes GPU acceleration (T4) for efficient training iterations.2. Model 2: Feature-Engineered Ensemble (SVM + Random Forest)Architecture: A Soft-Voting Ensemble pipeline combining Support Vector Machines (SVM) and Random Forest Classifiers.Advanced Techniques:TF-IDF Vectorization: Uses n-grams (1-3) and sublinear scaling to capture rich linguistic context.Statistical Feature Selection: Employs Chi-Square ($\chi^2$) to select the top 2,000 most significant features, reducing noise.Imbalance Handling: Uses balanced class weights and Probability-Based Threshold Optimization to maximize the F1-score for the 21% minority class.3. Model 3: Transformer-Based Classification (DistilBERT)Architecture: A fine-tuned DistilBERT (Distilled BERT) model.Key Features:Leverages state-of-the-art Natural Language Processing (NLP) through the Hugging Face Transformers library.Designed for high-performance sequence classification while remaining computationally efficient.Includes a comprehensive evaluation suite with Precision-Recall curves and detailed classification reports.Dataset CharacteristicsTask: Binary Classification.Classes: 0 (Normal) and 1 (Distress).Distribution: Heavily imbalanced with a ~79% to 21% ratio.Preprocessing: Includes NLTK-based stopword removal, lemmatization, and custom regex-based cleaning.

Task 2: Text Classification using BERT Models
In this task, I implemented a text classification pipeline using transformer-based models in a Kaggle notebook.
Overview
The goal of this task was to perform classification using pre-trained BERT-based architectures. The implementation is designed to be flexible, allowing easy switching between different models.
Models Used
DistilBERT (lightweight and faster)
RoBERTa (more robust and higher performance)
Both models are integrated into a single codebase for ease of experimentation.
Implementation Details
Built using the Hugging Face Transformers library
Tokenization and preprocessing handled within the pipeline
Training and evaluation performed in a Kaggle Notebook environment
Modular design to support multiple model configurations
Model Switching
The code is designed to support multiple models without structural changes.
To switch between models, simply update the model name in the parameters section:
Python
model_name = "distilbert-base-uncased"
or
Python
model_name = "roberta-base"
This allows seamless experimentation between different architectures without modifying the rest of the code.
Key Features
Unified pipeline for multiple transformer models
Easy-to-modify parameter settings
Scalable and reusable notebook structure
Clean integration of training and evaluation steps
Notebook
The full implementation is available in the Kaggle notebook included in this repository.
