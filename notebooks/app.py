import streamlit as st
import numpy as np
import pandas as pd
import joblib 


# Loads
model = joblib.load("FinalModel.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("📊 Employee Turnover Prediction")

# Inputs
over_time = st.selectbox("⏳ OverTime", ["Yes", "No"])
num_companies_worked = st.number_input("🏢 Number of Companies Worked For", 0, 10, step=1)
total_working_years = st.number_input("📅 Total Working Years", 0, 50, step=1)
job_level = st.selectbox("📊 Job Level", ["Assistant", "Junior", "Intermediate", "Senior", "Manager/Leader"])
marital_status = st.selectbox("💍 Marital Status", ["Single", "Married", "Divorced"])
years_in_current_role = st.number_input("🧑‍💼 Years in Current Role", 0, 20, step=1)
years_at_company = st.number_input("🏢 Years at Company", 0, 50, step=1)
age = st.number_input("🎂 Age", 18, 65, step=1)
stock_option_level = st.selectbox("📈 Stock Option Level", ["No Options", "Basic Options", "Moderate Options", "Advanced Options"])
job_satisfaction = st.selectbox("😊 Job Satisfaction", ["Dissatisfied", "Neutral", "Satisfied", "Extremely Satisfied"])
environment_satisfaction = st.selectbox("🏢 Environment Satisfaction", ["Dissatisfied", "Neutral", "Satisfied", "Extremely Satisfied"])
relationship_satisfaction = st.selectbox("🤝 Relationship Satisfaction", ["Dissatisfied", "Neutral", "Satisfied", "Extremely Satisfied"])
distance_from_home = st.number_input("🚗 Distance from Home (km)", 0, 50, step=1)
work_life_balance = st.selectbox("⚖️ Work-Life Balance", ["Poor", "Fair", "Good", "Excellent"])

# Encoding variables
satisfaction_mapping = {"Dissatisfied": 1, "Neutral": 2, "Satisfied": 3, "Extremely Satisfied": 4}
work_life_balance_mapping = {"Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4}
label_encoders = {
    "OverTime": {"Yes": 1, "No": 0},
    "MaritalStatus": {"Single": 0, "Married": 1, "Divorced": 2},
    "JobLevel": {"Assistant": 1, "Junior": 2, "Intermediate": 3, "Senior": 4, "Manager/Leader": 5},
    "StockOptionLevel": {"No Options": 0, "Basic Options": 1, "Moderate Options": 2, "Advanced Options": 3}
}

# Calculate new variables
job_change_frequency = num_companies_worked / total_working_years if total_working_years > 0 else 0
company_loyalty_ratio = years_at_company / age if age > 0 else 0
overall_satisfaction = (satisfaction_mapping[job_satisfaction] 
                        + satisfaction_mapping[environment_satisfaction] 
                        + satisfaction_mapping[relationship_satisfaction] ) / 3

# Create DataFrame
input_data = pd.DataFrame([[
    label_encoders["OverTime"][over_time],
    job_change_frequency,
    total_working_years,
    label_encoders["JobLevel"][job_level],
    label_encoders["MaritalStatus"][marital_status],
    years_in_current_role,
    company_loyalty_ratio,
    label_encoders["StockOptionLevel"][stock_option_level],
    overall_satisfaction,
    distance_from_home,
    work_life_balance_mapping[work_life_balance]
]], columns=[
    "OverTime", "JobChangeFrequency", "TotalWorkingYears", "JobLevel", "MaritalStatus",
    "YearsInCurrentRole", "CompanyLoyaltyRatio", "StockOptionLevel", "OverallSatisfaction",
    "DistanceFromHome", "WorkLifeBalance"
])

# Scale data
input_data_scaled = scaler.transform(input_data)

# Prediction
if st.button("🔍 Predict"):
    prediction = model.predict(input_data_scaled)[0]
    result = "🚨 **The employee WILL TURNOVER** 🚨" if prediction == 1 else "✅ **The employee will NOT turnover** ✅"
    st.success(result)
    
