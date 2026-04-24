Task 1: Classidication using DistilBERT  
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
