import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.preprocess import load_data, scale_data  # ✅ Absolute import works now

def evaluate_model(data_path, model_path='models/nb_model.pkl', output_path='output'):
    X, y = load_data(data_path)
    X_scaled, _ = scale_data(X)

    model = joblib.load(model_path)
    y_pred = model.predict(X_scaled)

    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred)

    os.makedirs(output_path, exist_ok=True)
    with open(f"{output_path}/evaluation_report.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{output_path}/confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model(data_path="data/winequality.csv")
