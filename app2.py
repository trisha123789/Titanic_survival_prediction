import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# =====================================================
# TITLE
# =====================================================

st.title("🚢 Titanic Survival Prediction")

st.write(
    "Predict passenger survival using Neural Network"
)

# =====================================================
# LOAD MODEL & SCALER
# =====================================================

model = joblib.load("titanic_model.pkl")

scaler = joblib.load("titanic_model.pkl")

# =====================================================
# LOAD DATASET
# =====================================================

data = pd.read_csv("Titanic-Dataset.csv")

# =====================================================
# PREPROCESSING
# =====================================================

X = data[['Pclass', 'Age', 'Fare']].copy()

y = data['Survived']

X['Age'] = X['Age'].fillna(X['Age'].mean())

X_scaled = scaler.transform(X)

# =====================================================
# MODEL EVALUATION
# =====================================================

y_pred = model.predict(X_scaled)

accuracy = accuracy_score(y, y_pred)

report = classification_report(y, y_pred)

cm = confusion_matrix(y, y_pred)

# =====================================================
# DISPLAY PERFORMANCE
# =====================================================

st.subheader("📊 Model Performance")

st.write(f"Accuracy: {accuracy:.4f}")

st.text("Classification Report")
st.text(report)

st.text("Confusion Matrix")
st.write(cm)

# =====================================================
# USER INPUTS
# =====================================================

st.subheader("🧾 Enter Passenger Details")

pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

age = st.slider(
    "Age",
    1,
    80,
    25
)

fare = st.slider(
    "Fare",
    0,
    600,
    100
)

# =====================================================
# PREDICTION
# =====================================================

if st.button("Predict Survival"):

    sample_df = pd.DataFrame({
        'Pclass': [pclass],
        'Age': [age],
        'Fare': [fare]
    })

    sample_scaled = scaler.transform(sample_df)

    prediction = model.predict(sample_scaled)

    probability = model.predict_proba(sample_scaled)

    st.subheader("🎯 Prediction Result")

    st.write(
        f"Survival Probability: "
        f"{probability[0][1]:.4f}"
    )

    if prediction[0] == 1:
        st.success("✅ Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")
