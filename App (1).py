import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

def predict_hours(target, previous, sleep, practice):
    best_hours = None
    best_error = float("inf")

    for hours in range(1, 13):

        features = np.array([[hours, previous, sleep, practice]])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]

        error = abs(prediction - target)

        if prediction > target:
            error += 5

        if error < best_error or (error == best_error and (best_hours is None or hours < best_hours)):
            best_error = error
            best_hours = hours

    return best_hours


st.title("Study Hours Predictor System")

target = st.text_input("Target Performance Index")
sleep = st.text_input("Sleep Hours")
previous = st.text_input("Previous Score")
practice = st.text_input("Practice Papers Solved")

if st.button("Predict Required Study Hours"):
    if target and sleep and previous and practice:

        hours = predict_hours(
            int(target),
            int(previous),
            int(sleep),
            int(practice)
        )

        st.success(f"You need approximately {hours} hours of study")
