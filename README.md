# Credit Card Fraud Detection System

## Project Overview
This project implements a machine learning solution to identify fraudulent credit card transactions. Due to the nature of fraud, the dataset is highly imbalanced, with fraudulent transactions representing only **0.167%** of the total data. The system utilizes a **Random Forest** model trained on **SMOTE-balanced** data to prioritize the detection of fraud (Recall) while maintaining high overall accuracy.

## Objective
The primary objective of this project was to build an accurate, scalable, and deployable machine learning system capable of:

- Distinguishing fraudulent credit card transactions from legitimate ones in near-real-time.
- Minimising false negatives (missed fraud) to protect customers by prioritising Recall as the primary optimisation metric.
- Maintaining acceptable Precision to avoid excessive false alarms that burden fraud review teams.
- Providing both single-transaction and batch-processing capabilities via an accessible web interface.

### Dataset Summary
* **Total Transactions:** 283,726 (after removing 1,081 duplicates).
* **Fraudulent Cases:** 473 (0.167%).
* **Features:** 31 numerical features, including PCA-transformed variables (V1-V28), "Time", and "Amount".

## Key Findings
### Class Imbalance
Examination of the target variable distribution revealed a severe class imbalance: only 473 transactions (0.167%) were fraudulent, while the overwhelming majority were legitimate.
This imbalance is typical of real-world fraud datasets and, if unaddressed, would cause models to be biased toward predicting the majority class.

![Distribution of Transaction classes](images\Distribution%20of%20Transaction%20Classes.png)

### Feature Correlation
A Seaborn correlation matrix was used to identify relationships between each feature and the target Class variable. Several PCA-derived features (V1–V28) demonstrated meaningful positive and negative correlations with the target, indicating their discriminative utility. The raw Time and Amount fields showed weaker direct correlations but were retained and preprocessed.

![Distribution of Transaction classes](images\Seaborn%20Correlation%20Heatmap.png)

### Feature Distribution Analysis
Histograms were used to assess the skewness of the Amount and Time features:
- **Amount:** Highly right-skewed, with the vast majority of transactions clustered at lower values. Standard scaling was determined to be necessary before model training.

    ![Distribution of Tranasction Amounts](images\Distribution%20of%20Amount.png)
- **Time:** Moderately right-skewed. Scaling was similarly required, though the distribution was less extreme than Amount.

    ![Distribution of Tranasction Time](images\Distribution%20of%20Time.png)

### Temporal and Monetary Behaviour of Fraud
Density plots were used to contrast the temporal and monetary patterns of fraudulent versus legitimate transactions:
- **Time:** Visible and distinct peaks were observed for both classes, confirming that time of transaction carries predictive signal for fraud detection.

    ![Distribution of Transaction Time: Normal vs Fraud](images\Density%20Plot%20for%20Transaction%20time%20Against%20Class.png)
 
- **Amount:** The distributions for fraudulent and legitimate transactions were broadly similar, with only minor differences in peak density, suggesting that transaction amount alone is insufficient as a discriminator.

    ![Distribution of Transaction Time: Normal vs Fraud](images\Density%20Plot%20for%20Transaction%20Amount%20Against%20Class.png)

## Data Preprocessing
Prior to model training, the following preprocessing steps were applied:
- **Train-Test Split:** The dataset was split into training and test sets to ensure unbiased evaluation.
- **Feature Scaling:** The Amount and Time columns of the training data were scaled using Scikit-learn's StandardScaler. The fitted scaler was saved as scaler.pkl for consistent application during inference.
- **Class Imbalance Handling:** Three versions of the training data were created: the original imbalanced dataset, a SMOTE (Synthetic Minority Oversampling Technique) oversampled dataset, and a randomly undersampled dataset.


## Model Performance
Several models were evaluated across different balancing techniques. 
The table  presents the complete evaluation results for all nine models, sorted by PR-AUC (descending). The highlighted row (Random Forest - SMOTE) represents the selected final model.
The **Random Forest (SMOTE)** model was selected as the final production model.

| Model | Dataset | TP | FP | TN | FN | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | SMOTE | 72 | 7 | 56,644 | 23 | 0.9114 | 0.7579 | 0.8276 | 0.9656 | 0.8038 |
| **Random Forest** | Original | 69 | 2 | 56,649 | 26 | 0.9718 | 0.7263 | 0.8313 | 0.9239 | 0.7876 |
| **Gradient Boosting** | SMOTE | 80 | 616 | 56,035 | 15 | 0.1149 | 0.8421 | 0.2023 | 0.9771 | 0.7358 |
| **Random Forest** | Undersampled | 83 | 896 | 55,755 | 12 | 0.0848 | 0.8737 | 0.1546 | 0.9767 | 0.6983 |
| **Logistic Regression** | SMOTE | 83 | 1,482 | 55,169 | 12 | 0.0530 | 0.8737 | 0.1000 | 0.9619 | 0.6769 |
| **Logistic Regression** | Original | 83 | 1,393 | 55,258 | 12 | 0.0562 | 0.8737 | 0.1057 | 0.9658 | 0.6719 |
| **Gradient Boosting** | Original | 62 | 10 | 56,641 | 33 | 0.8611 | 0.6526 | 0.7425 | 0.8539 | 0.6234 |
| **Logistic Regression** | Undersampled | 83 | 1,516 | 55,135 | 12 | 0.0519 | 0.8737 | 0.0980 | 0.9559 | 0.6182 |
| **Gradient Boosting** | Undersampled | 82 | 1,772 | 54,879 | 13 | 0.0442 | 0.8632 | 0.0841 | 0.9725 | 0.5754 |

### Evaluation Visualization

![Precision-Recall Curve Comparison](images\Precision-Recall%20Curve%20Comparison.png)

The highlighted row below (Random Forest - SMOTE) represents the selected final model.
| Metric | Score |
| :--- | :--- |
| **Recall** | 0.7579 |
| **ROC-AUC** | 0.9656 |
| **F1 Score** | 0.8276 |
| **Precision** | 0.9114 |

> **Note:** The Random Forest classifier trained on SMOTE-balanced data was selected as the production model on the basis of the following considerations: 
- **Recall (0.7579):** The proportion of actual fraud cases detected  is the primary metric in fraud detection. Missing a fraudulent transaction has a higher cost than a false alarm. The SMOTE Random Forest achieved the highest Recall among all models that also maintained acceptable Precision.
- **Precision (0.9114):** The model maintained high Precision (91.14%), ensuring that the majority of flagged transactions were genuinely fraudulent. This reduces the operational burden on fraud review teams.
- **F1 Score (0.8276):** The harmonic mean of Precision and Recall was the second highest across all models, demonstrating a strong balance between the two metrics.
- **ROC-AUC (0.9656):** The model ranked among the top performers in area under the ROC curve, indicating excellent overall discriminatory ability.
- **PR-AUC (0.8038):** The highest PR-AUC across all models, a particularly meaningful metric for imbalanced datasets as it focuses on the minority class performance.

Alternative models were considered but ruled out for the following reasons:
- **Random Forest (Original):** Higher Precision (97.18%) but lower Recall (72.63%), meaning more fraudulent transactions would be missed.
- **Gradient Boosting (SMOTE):** Higher Recall (84.21%) but dramatically lower Precision (11.49%) and F1 Score (20.23%), indicating an excessive number of false positives that would render the system operationally impractical.
- **Logistic Regression models:** Consistently high Recall but very low Precision across all dataset variants, reflecting the limitations of a linear decision boundary for this task.

The saved model artefact is: fraud_model.pkl

## Deployment

The system has been deployed as a two-tier application:
- **Backend API:** A FastAPI service (main.py) that loads the saved model (fraud_model.pkl) and scaler (scaler.pkl), accepts 30-feature transaction payloads via a POST /predict endpoint, applies the same preprocessing pipeline used during training, and returns a prediction, fraud probability score, and human-readable message.
- **Frontend Application:** A Streamlit interface (app.py) supporting both single-transaction manual entry and batch CSV upload with per-row API prediction. Fraudulent transactions are highlighted in the batch output and a summary count is displayed.

It is recommended to deploy the FastAPI backend behind a production-grade ASGI server (e.g., Uvicorn with Gunicorn) and to secure the endpoint with API key authentication or OAuth2 before exposing it in a live financial environment.

## Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Joblib, Pydantic
* **Backend:** FastAPI
* **Frontend:** Streamlit
* **Deployment:** Uvicorn

## Project Structure
* `notebook\fraud_training.ipynb`: Analysis transformation and training of Model
* `backend\fraud_model.pkl`: The trained Random Forest model.
* `backend\scaler.pkl`: The saved StandardScaler for normalizing new input data.
* `backend\main.py`: The FastAPI backend script.
* `frontend\app.py`: The Streamlit frontend script.

## Usage

### 1. Running the Backend (API)
The backend expects incoming JSON data with 30 features. Start the server using:
```bash
uvicorn main:app --port 8007 --reload
