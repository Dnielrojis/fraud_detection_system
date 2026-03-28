# Credit Card Fraud Detection System

## Project Overview
This project implements a machine learning solution to identify fraudulent credit card transactions. Due to the nature of fraud, the dataset is highly imbalanced, with fraudulent transactions representing only **0.167%** of the total data. The system utilizes a **Random Forest** model trained on **SMOTE-balanced** data to prioritize the detection of fraud (Recall) while maintaining high overall accuracy.

## Features
* **Data Cleaning & Preprocessing:** Automatic removal of duplicate records and feature scaling for "Time" and "Amount".
* **Handling Imbalance:** Implementation of **SMOTE** (Synthetic Minority Oversampling Technique) and Random Undersampling to improve model sensitivity.
* **Real-time API:** A **FastAPI** backend that provides a `/predict` endpoint for transaction inference.
* **Interactive Dashboard:** A **Streamlit** frontend allowing for manual transaction entry or batch processing via CSV upload.

## Dataset Summary
* **Total Transactions:** 283,726 (after removing 1,081 duplicates).
* **Fraudulent Cases:** 473 (0.167%).
* **Features:** 31 numerical features, including PCA-transformed variables (V1-V28), "Time", and "Amount".

## Model Performance
Several models were evaluated across different balancing techniques. The **Random Forest (SMOTE)** model was selected as the final production model.

| Metric | Score |
| :--- | :--- |
| **Recall** | 0.7579 |
| **ROC-AUC** | 0.9656 |
| **F1 Score** | 0.8276 |
| **Precision** | 0.9114 |

> **Note:** High Recall is prioritized to ensure a higher proportion of actual fraud cases are detected.

## Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Joblib, Pydantic
* **Backend:** FastAPI
* **Frontend:** Streamlit
* **Deployment:** Uvicorn

## Project Structure
* `fraud_model.pkl`: The trained Random Forest model.
* `scaler.pkl`: The saved StandardScaler for normalizing new input data.
* `main.py`: The FastAPI backend script.
* `app.py`: The Streamlit frontend script.

## Usage

### 1. Running the Backend (API)
The backend expects incoming JSON data with 30 features. Start the server using:
```bash
uvicorn main:app --port 8007 --reload