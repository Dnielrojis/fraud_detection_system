from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="Credit Card Fraud Detection API")

# 1. Load the model and scaler
try:
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print(f"Error loading model artifacts: {e}")

# 2. Define the Input Schema using Pydantic
# This ensures the API only accepts the correct 30 features
class Transaction(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float

@app.get("/")
def home():
    return {"message": "Fraud Detection API is Running"}

@app.post("/predict")
def predict(data: Transaction):
    try:
        # Convert input JSON to DataFrame
        df_input = pd.DataFrame([data.dict()])
        
        # 3. Preprocessing: Scale Time and Amount
        df_input[['Time', 'Amount']] = scaler.transform(df_input[['Time', 'Amount']])
        
        # 4. Inference
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]
        
        # 5. Return Response
        return {
            "prediction": int(prediction),
            "probability": round(float(probability), 4),
            "message": "Fraudulent transaction detected" if prediction == 1 else "Legitimate transaction"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))