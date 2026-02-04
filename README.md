# Heart Disease Prediction Using Machine Learning Classification Models

## Problem Statement

Heart disease is one of the leading causes of death worldwide. Early and accurate prediction can significantly improve patient outcomes and reduce mortality. This project develops and compares multiple supervised machine learning models to classify whether a patient is likely to have heart disease based on medical attributes.

---

## Dataset Features

The dataset consists of the following patient attributes:

- Age
- Sex
- Chest Pain Type (cp)
- Resting Blood Pressure (trestbps)
- Cholesterol (chol)
- Fasting Blood Sugar (fbs)
- Rest ECG (restecg)
- Maximum Heart Rate Achieved (thalach)
- Exercise Induced Angina (exang)
- ST Depression (oldpeak)
- Slope
- Number of Major Vessels (ca)
- Thalassemia (thal)
- Target (0 = No disease, 1 = Disease)

---

## Machine Learning Pipeline

1. Data Understanding & Cleaning
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing (Scaling, Train-Test Split)
4. Model Building:
   - Logistic Regression
   - Decision Tree (GridSearchCV)
   - Random Forest (GridSearchCV)
   - Feedforward Neural Network
5. Model Evaluation using Accuracy, Precision, Recall, F1-Score, ROC-AUC
6. Feature Importance Analysis
7. Streamlit Web App Deployment

---

## Results

Random Forest and Neural Network demonstrated superior performance due to their ability to capture complex relationships between medical features.

---

## Feature Importance Insight

The most important predictors identified were:
- Chest Pain Type (cp)
- Maximum Heart Rate (thalach)
- ST Depression (oldpeak)
- Number of Major Vessels (ca)

These align with real clinical diagnostic indicators.

---

## Streamlit App

A Streamlit application was created to input patient parameters and predict heart disease risk in real-time.

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
# heart-disease-ml-classification
