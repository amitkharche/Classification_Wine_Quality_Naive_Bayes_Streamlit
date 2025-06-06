
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(data_path, model_path='models/nb_model.pkl', output_path='output'):
    df = pd.read_csv(data_path)
    X = df.drop('quality', axis=1)
    y = df['quality']

    model = joblib.load(model_path)
    y_pred = model.predict(X)

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

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{output_path}/confusion_matrix.png")
