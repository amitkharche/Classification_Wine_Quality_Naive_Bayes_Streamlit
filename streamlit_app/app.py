
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Wine Quality Prediction - Naive Bayes", layout="wide")
st.title("🍷 Wine Quality Prediction using Naive Bayes + SHAP")

model = joblib.load("models/nb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
            'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

tabs = st.tabs(["🔮 Predict", "📊 Performance"])

with tabs[0]:
    st.header("🔢 Input Wine Features")
    input_data = []
    for feat in features:
        val = st.number_input(f"{feat.title()}", value=5.0, step=0.1)
        input_data.append(val)
    X_input = np.array(input_data).reshape(1, -1)
    X_scaled = scaler.transform(X_input)

    if st.button("Predict"):
        pred = model.predict(X_scaled)
        st.success(f"Predicted Wine Quality: {int(pred[0])}")

        st.subheader("SHAP Explanation")
        df = pd.read_csv("data/winequality.csv")
        X_train = df.drop("quality", axis=1)
        X_train_scaled = scaler.fit_transform(X_train)

        explainer = shap.Explainer(model.predict, X_train_scaled)
        shap_values = explainer(X_scaled)

        shap.initjs()
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=11, show=False)
        st.pyplot(fig)

with tabs[1]:
    st.header("📊 Model Evaluation")
    with open("output/evaluation_report.txt") as f:
        report = f.read()
        st.code(report)

    st.image("output/confusion_matrix.png", caption="Confusion Matrix")
