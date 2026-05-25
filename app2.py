import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

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
    "Predict passenger survival using Artificial Neural Network"
)

# =====================================================
# LOAD DATASET
# =====================================================

data = pd.read_csv("Titanic-Dataset.csv")

# =====================================================
# PREPROCESSING
# =====================================================

X = data[['Pclass', 'Age', 'Fare']].copy()

y = data['Survived']

# Fill Missing Values
X['Age'] = X['Age'].fillna(X['Age'].mean())

# =====================================================
# SCALING
# =====================================================

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# CREATE ANN MODEL
# =====================================================

model = MLPClassifier(
    hidden_layer_sizes=(2,),
    activation='logistic',
    learning_rate_init=0.01,
    max_iter=1000,
    random_state=42
)

# =====================================================
# TRAIN MODEL
# =====================================================

model.fit(X_train, y_train)

# =====================================================
# MODEL EVALUATION
# =====================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

# =====================================================
# DISPLAY PERFORMANCE
# =====================================================

st.subheader("📊 Model Performance")

st.write(f"### Accuracy: {accuracy:.4f}")

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

    # Create DataFrame
    sample_df = pd.DataFrame({
        'Pclass': [pclass],
        'Age': [age],
        'Fare': [fare]
    })

    # Scale Input
    sample_scaled = scaler.transform(sample_df)

    # Prediction
    prediction = model.predict(sample_scaled)

    # Probability
    probability = model.predict_proba(sample_scaled)

    # =================================================
    # DISPLAY RESULT
    # =================================================

    st.subheader("🎯 Prediction Result")

    st.write(
        f"Survival Probability: "
        f"{probability[0][1]:.4f}"
    )

    if prediction[0] == 1:
        st.success("✅ Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")
