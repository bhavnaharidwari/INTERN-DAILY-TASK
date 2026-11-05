import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Task 5: Dynamic ML Model Loading, Error Handling & Real-Time Prediction
# SalaryInsight: Predictive Modeling for Informed Salary Predictions

MODEL_PATH = "saved_models/salary_prediction_model.joblib"
DATA_PATH = "data/statistics_practice_dataset_2000.csv"

def train_and_save_fallback_model():
    """Trains a simple baseline model if no saved model is found."""
    df = pd.read_csv(DATA_PATH)
    
    # Simple feature selection
    features = ['Age', 'ExperienceYears', 'PerformanceScore', 'WeeklyHours', 'ProjectsCompleted', 'SatisfactionScore', 'MonthlySalesK']
    target = 'AnnualSalary'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model

def load_machine_learning_model():
    """Dynamically loads the trained ML model with error handling."""
    model = None
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            st.success("✓ Trained Machine Learning model loaded successfully from disk!")
        else:
            st.warning(" Model file not found. Training a fresh baseline model dynamically...")
            model = train_and_save_fallback_model()
            st.success("✓ Fresh baseline model trained and saved successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}. Training backup model...")
        model = train_and_save_fallback_model()
        
    return model

def predict_salary_realtime(model, input_data):
    """Verifies compatibility and makes real-time prediction."""
    try:
        # Convert dictionary input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        return prediction
    except Exception as e:
        st.error(f"Compatibility/Prediction Error: {e}")
        return None

def run_task_5_demo():
    st.title("Task 5: Model Loading & Real-Time Prediction")
    st.write("This module tests dynamic model loading, error handling, and real-time inference.")
    
    # 1. Load model dynamically
    model = load_machine_learning_model()
    
    # 2. Input form for real-time predictions
    st.subheader("Test Real-Time Prediction Inputs")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=65, value=30)
        exp = st.number_input("Experience (Years)", min_value=0, max_value=40, value=6)
        perf = st.number_input("Performance Score (40-100)", min_value=40, max_value=100, value=75)
        hours = st.number_input("Weekly Hours", min_value=20.0, max_value=80.0, value=45.0)
        
    with col2:
        projects = st.number_input("Projects Completed", min_value=1, max_value=30, value=8)
        satisfaction = st.number_input("Satisfaction Score (1-10)", min_value=1.0, max_value=10.0, value=7.0)
        sales = st.number_input("Monthly Sales (₹ Thousands)", min_value=0.0, max_value=2000.0, value=150.0)
        
    if st.button("Calculate Predicted Salary"):
        input_dict = {
            'Age': age,
            'ExperienceYears': exp,
            'PerformanceScore': perf,
            'WeeklyHours': hours,
            'ProjectsCompleted': projects,
            'SatisfactionScore': satisfaction,
            'MonthlySalesK': sales
        }
        
        salary_pred = predict_salary_realtime(model, input_dict)
        if salary_pred is not None:
            st.success(f"Predicted Annual Salary: ₹{salary_pred:,.2f}")
            st.info(f"Monthly Salary Estimate: ₹{salary_pred/12:,.2f}")

if __name__ == "__main__":
    run_task_5_demo()
