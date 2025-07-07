import streamlit as st 
import joblib
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns 
import matplotlib.pyplot as plt


page = st.sidebar.selectbox("Select a page", ["Prediction", "Insights"])
model =joblib.load("loan_approval/model.pkl")
scaler =joblib.load("loan_approval/scaler.pkl")
numeric_features =joblib.load("loan_approval/numeric_features.pkl")
cibil_score=600
if page== "Prediction":
    st.title("Loan Approval Predictor")

    st.header("Enter your loan application details")

    cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900,value=400)
    no_of_dependents = st.slider("Number of Dependents", min_value=0,max_value=5,value=2)
    income_annum = st.number_input("Annual Income")
    loan_amount = st.number_input("Loan Amount")
    loan_term = st.number_input("Loan Term (months)")
    residential_assets_value = st.number_input("Residential Asset Value")
    commercial_assets_value = st.number_input("Commercial Asset Value")
    luxury_assets_value = st.number_input("Luxury Asset Value")
    bank_asset_value = st.number_input("Bank Asset Value")
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    
    if st.button("Predict"):
        if no_of_dependents is None:
            no_of_dependents=2.4987
        if income_annum is None:
            income_annum=5059123.9
        if loan_amount is None:
            loan_amount=15133450.456
        if loan_term is None:
            loan_term=10.9
        if cibil_score is None:
            cibil_score=599.93
        if residential_assets_value is None:
            residential_assets_value=7472616.53
        if commercial_assets_value is None:
            commercial_assets_value=4973155.30
        if luxury_assets_value is None:
            luxury_assets_value=15126305.92
        if bank_asset_value is None:
            bank_asset_value=4976692.43
        if education is None:
            education="Graduate"
        if self_employed is None:
            self_employed="Yes"
            
        input_df = pd.DataFrame({
            'no_of_dependents': [no_of_dependents],
            'education': [1 if education == "Graduate" else 0],
            'self_employed': [1 if self_employed == "Yes" else 0],
            'income_annum': [income_annum],
            'loan_amount': [loan_amount],
            'loan_term': [loan_term],
            'cibil_score': [cibil_score],
            'residential_assets_value': [residential_assets_value],
            'commercial_assets_value': [commercial_assets_value],
            'luxury_assets_value': [luxury_assets_value],
            'bank_asset_value': [bank_asset_value],
        })
        
        input_df[numeric_features]=scaler.transform(input_df[numeric_features])
        
        
        pred=model.predict(input_df)[0]
        prob=model.predict_proba(input_df)[0]
        if pred== 1:
            st.success(f"Loan has {prob[1]*100}% chance of approval ✅")
        else:
            st.error(f"Loan has {prob[0]*100}% chance of rejection ❌")
            
elif page=="Insights":
    st.title("📊 Data Insights")
    
    st.header(" Loan approval vs. CIBIL score ")
    
    cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900,value=400)
    df = pd.read_csv("loan_approval_dataset.csv")
    df.columns = df.columns.str.strip()
    df['education'] = df['education'].str.strip()
    df['self_employed'] = df['self_employed'].str.strip()
    df['loan_status'] = df['loan_status'].str.strip()
    df["education"]=df["education"].map({"Graduate":1,"Not Graduate":0})
    df["self_employed"]=df["self_employed"].map({"Yes":1,"No":0})
    df["loan_status"]=df["loan_status"].map({"Approved":1,"Rejected":0})
    fig, ax = plt.subplots()
    sns.kdeplot(df[df['loan_status']==1]['cibil_score'], label="Approved", shade=True)
    ax.axvline(cibil_score, color='red', linestyle='--', label='Your Score')
    plt.legend()
    st.pyplot(fig)

