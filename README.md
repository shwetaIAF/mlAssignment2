# Machine Learning Assignment 2 – M.Tech AIML

## Problem Statement
To implement and compare multiple machine learning classification models and deploy them using Streamlit.

## Dataset Description
Human Activity Recognition Dataset from UCI repository.
- Instances: 10,299
- Features: 561
- Classes: 6

## Models Used & Evaluation Metrics

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|------|--------|-----|----------|--------|----|-----|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Random Forest | | | | | | |
| XGBoost | | | | | | |

## Model Observations

| Model | Observation |
|------|------------|
| Logistic Regression | Stable baseline performance |
| Decision Tree | Overfitting observed |
| KNN | Sensitive to feature scaling |
| Naive Bayes | Fast but lower accuracy |
| Random Forest | Strong ensemble performance |
| XGBoost | Best overall performance |

## Deployment
The application is deployed on Streamlit Community Cloud and provides:
- Dataset upload
- Model selection
- Evaluation metrics
- Confusion matrix & classification report
