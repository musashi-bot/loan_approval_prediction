# Loan Approval Prediction System

This project uses machine learning to predict whether a loan application will be approved or rejected based on applicant details like income, credit score, employment status, and more. It is deployed as an interactive web application using Streamlit.

---

## Demo

🔗 [Live Streamlit App](https://loanapprovalprediction-ly8ufdkyhdnth2hclgqdax.streamlit.app/)  

---

## Features

- Predict loan approval based on user inputs
- User-friendly Streamlit web interface
- Handles both numerical and categorical inputs
- Trained and optimized ML models (Random Forest, XGBoost)

---

## Machine Learning Pipeline

- **Data Preprocessing**
  - Mapping categorical features (`Yes/No`, `Graduate/Not Graduate`)
  - Feature scaling with `StandardScaler`
  - Class balancing with SMOTE
- **Model Training**
  - Explored Random Forest, XGBoost
  - Used GridSearchCV for hyperparameter tuning
- **Model Deployment**
  - Saved model and scaler using `joblib`
  - Deployed via Streamlit Cloud

---


