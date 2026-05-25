# =====================================================
# TITANIC SURVIVAL PREDICTION APP
# STREAMLIT + TENSORFLOW
# =====================================================

# Run using:
# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# =====================================================
# PAGE TITLE
# =====================================================

st.title("Titanic Survival Prediction")
st.write("Predict whether a passenger survived or not using Neural Networks")

# =====================================================
# LOAD DATASET
# =====================================================

data = pd.read_csv("Titanic-Dataset.csv")

# =====================================================
# SELECT FEATURES
# =====================================================

X = data[['Pclass', 'Age', 'Fare']].copy()

y = data['Survived']

# =====================================================
# HANDLE MISSING VALUES
# =====================================================

X['Age'] = X['Age'].fillna(X['Age'].mean())

# =====================================================
# NORMALIZATION
# =====================================================

scaler = MinMaxScaler()

X = scaler.fit_transform(X)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# CREATE MODEL
# =====================================================

model = Sequential([

    Dense(
        units=2,
        activation='sigmoid',
        input_shape=(3,)
    ),

    Dense(
        units=1,
        activation='sigmoid'
    )
])

# =====================================================
# COMPILE MODEL
# =====================================================

model.compile(
    optimizer=SGD(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =====================================================
# TRAIN MODEL
# =====================================================

model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    verbose=0
)

# =====================================================
# MODEL EVALUATION
# =====================================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

st.subheader("Model Accuracy")

st.write(f"Accuracy: {accuracy:.2f}")

# =====================================================
# USER INPUTS
# =====================================================

st.subheader("Enter Passenger Details")

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

    # Create DataFrame
    sample_df = pd.DataFrame({
        'Pclass': [pclass],
        'Age': [age],
        'Fare': [fare]
    })

    # Normalize
    sample_scaled = scaler.transform(sample_df)

    # Convert to NumPy Array
    sample_scaled = np.array(sample_scaled)

    # Prediction
    prediction = model.predict(sample_scaled)

    probability = prediction[0][0]

    st.subheader("Prediction Result")

    st.write(f"Survival Probability: {probability:.4f}")

    if probability >= 0.5:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Did Not Survive")