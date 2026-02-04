
---

## ✅ `app.py`

```python
import streamlit as st
import numpy as np
import joblib

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Disease Risk Prediction System")
st.write("Enter patient details:")

age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0,1])
cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar > 120 (0/1)", [0,1])
restecg = st.selectbox("Rest ECG (0-2)", [0,1,2])
thalach = st.number_input("Max Heart Rate Achieved")
exang = st.selectbox("Exercise Induced Angina (0/1)", [0,1])
oldpeak = st.number_input("ST Depression")
slope = st.selectbox("Slope (0-2)", [0,1,2])
ca = st.selectbox("Number of Major Vessels (0-4)", [0,1,2,3,4])
thal = st.selectbox("Thalassemia (0-3)", [0,1,2,3])

input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                        restecg, thalach, exang, oldpeak,
                        slope, ca, thal]])

input_scaled = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")
