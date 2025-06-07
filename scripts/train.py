import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import joblib
import logging

from scripts.preprocess import load_data, scale_data  # ✅ Absolute import works now

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)

def train_model(data_path, model_path, test_size=0.2, random_state=42):
    X, y = load_data(data_path)
    X_scaled, scaler = scale_data(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    model = GaussianNB()
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, 'models/scaler.pkl')

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)

    os.makedirs('output', exist_ok=True)
    with open('output/evaluation_report.txt', 'w') as f:
        f.write(report)

    logging.info("✅ Model trained and saved. Evaluation report generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/winequality.csv", help="Path to the input data CSV")
    parser.add_argument("--model_path", default="models/nb_model.pkl", help="Path to save the trained model")
    args = parser.parse_args()

    train_model(args.data_path, args.model_path)
