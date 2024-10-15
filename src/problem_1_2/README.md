# Task 1 + 2: Tweet Sentiment Analysis Pipeline

This project implements a machine learning pipeline for processing, training, evaluating, and tuning models for sentiment analysis using Twitter data. The pipeline is modular and follows a structured approach, making it easy to extend and maintain.


### Files Overview

- **`preprocess.py`**:
    - Contains the `Preprocessor` class responsible for preprocessing the input data. This includes cleaning, tokenizing, and stemming the Twitter data for sentiment analysis.

- **`train.py`**:
    - Contains the `Trainer` class, which handles training multiple machine learning models such as Logistic Regression, Random Forest, Support Vector Machine (SVM), and Naive Bayes.

- **`model_evaluate.py`**:
    - Contains the `ModelEvaluator` class for evaluating the trained models and performing error analysis. It provides insights into where the models make incorrect predictions.

- **`tuning.py`**:
    - Contains the `Tuner` class, which performs hyperparameter tuning for machine learning models like Random Forest and LightGBM using grid search.

- **`pipeline_runner.py`**:
    - The main entry point for running the entire pipeline. This script orchestrates the preprocessing, training, evaluation, and tuning steps by calling the respective classes and methods.

- **`config.yaml`**:
    - A configuration file that defines parameters for data ingestion, preprocessing, and training. This makes the pipeline flexible and easy to configure without changing the core code.
