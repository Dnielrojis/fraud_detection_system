import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Fraud Guard", layout="wide")

st.title("🛡️ Credit Card Fraud Detection System")
st.markdown("Enter transaction details manually or upload a batch file.")

API_URL = "http://localhost:8007/predict"

# Sidebar for Navigation
option = st.sidebar.selectbox("Choose Mode", ["Manual Entry", "CSV Upload"])

if option == "Manual Entry":
    st.subheader("Single Transaction Check")
    
    # Organize inputs into columns
    col1, col2, col3 = st.columns(3)
    
    inputs = {}
    with col1:
        inputs["Time"] = st.number_input("Time", value=0.0)
        inputs["Amount"] = st.number_input("Amount", value=10.0)
    
    # Loop to create V1-V28 inputs quickly
    for i in range(1, 29):
        col = [col1, col2, col3][i % 3]
        with col:
            inputs[f"V{i}"] = st.number_input(f"V{i}", value=0.0)

    if st.button("Analyze Transaction"):
        response = requests.post(API_URL, json=inputs)
        if response.status_code == 200:
            res = response.json()
            if res['prediction'] == 1:
                st.error(f"🚨 {res['message']} (Prob: {res['probability']})")
            else:
                st.success(f"✅ {res['message']} (Prob: {res['probability']})")
        else:
            st.error("API Error. Is the backend running?")

elif option == "CSV Upload":
    st.subheader("Batch Fraud Detection")
    uploaded_file = st.file_uploader("Upload Transaction CSV", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head())
        
        if st.button("Run Batch Prediction"):
            results = []
            # Progress bar for the i5's hard work
            progress = st.progress(0)
            
            for index, row in df.iterrows():
                # Send each row to the API
                resp = requests.post(API_URL, json=row.to_dict())
                if resp.status_code == 200:
                    results.append(resp.json()['prediction'])
                progress.progress((index + 1) / len(df))
            
            df['Is_Fraud'] = results
            
            # Show summary
            fraud_count = df['Is_Fraud'].sum()
            st.metric("Total Frauds Detected", fraud_count)
            
            # Highlight fraud rows
            st.dataframe(df.style.apply(lambda x: ['background-color: #ffcccc' if x.Is_Fraud == 1 else '' for i in x], axis=1))