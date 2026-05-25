
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model

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
st.write("Predict passenger survival using a trained Neural Network")

# =====================================================
# LOAD MODEL & SCALER
# =====================================================

model = load_model("titanic.keras")

scaler = joblib.load("scaler.pkl")

# =====================================================
# LOAD DATASET
# =====================================================

data = pd.read_csv("Titanic-Dataset.csv")

# =====================================================
# PREPROCESS DATA
# =====================================================

X = data[['Pclass', 'Age', 'Fare']].copy()

y = data['Survived']

# Fill missing values
X['Age'] = X['Age'].fillna(X['Age'].mean())

# Scale data
X_scaled = scaler.transform(X)

# =====================================================
# MODEL EVALUATION
# =====================================================

predictions = model.predict(X_scaled)

# Convert probabilities to binary
y_pred = (predictions > 0.5).astype(int)

# Accuracy
accuracy = accuracy_score(y, y_pred)

# Classification Report
report = classification_report(y, y_pred)

# Confusion Matrix
cm = confusion_matrix(y, y_pred)

# =====================================================
# DISPLAY MODEL PERFORMANCE
# =====================================================

st.subheader("📊 Model Performance")

st.write(f"### Accuracy: {accuracy:.4f}")

st.text("Classification Report")
st.text(report)

st.text("Confusion Matrix")
st.write(cm)

# =====================================================
# USER INPUT SECTION
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
# PREDICTION BUTTON
# =====================================================

if st.button("Predict Survival"):

    # Create dataframe
    sample_df = pd.DataFrame({
        'Pclass': [pclass],
        'Age': [age],
        'Fare': [fare]
    })

    # Normalize input
    sample_scaled = scaler.transform(sample_df)

    # Prediction
    prediction = model.predict(sample_scaled)

    probability = prediction[0][0]

    # =================================================
    # DISPLAY RESULT
    # =================================================

    st.subheader("🎯 Prediction Result")

    st.write(f"Survival Probability: {probability:.4f}")

    if probability >= 0.5:
        st.success("✅ Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")
