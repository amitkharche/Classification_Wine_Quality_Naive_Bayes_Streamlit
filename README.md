
---

# 🍷 Wine Quality Prediction using Naive Bayes

This end-to-end machine learning project predicts **wine quality scores** using the **Naive Bayes** algorithm. It features:

✅ Clean EDA & Preprocessing
✅ Modular Training & Evaluation Scripts
✅ SHAP Explainability for Predictions
✅ Streamlit App with Real-Time Dashboard
✅ Confusion Matrix & Metrics Report
✅ Docker & GitHub CI/CD Support

---

## Project Structure

```
wine-quality-naive_bayes-project/
├── data/                    # Dataset CSV
├── models/                  # Saved Naive Bayes model and scaler
├── output/                  # Evaluation report and confusion matrix
├── scripts/                 # Modular ML scripts
│   ├── preprocess.py        # Data loading and scaling
│   ├── train.py             # Train and save model + scaler
│   ├── evaluate.py          # Evaluate model and save results
├── streamlit_app/
│   └── app.py               # Streamlit UI for prediction and dashboard
├── requirements.txt         # Dependencies
├── Dockerfile               # Docker build file
├── docker-compose.yml       # Docker Compose config
├── .github/workflows/ci.yml# GitHub CI config
└── README.md                # Project overview
```

---

## Setup Instructions

### 1. Clone the repository
   ```bash
   git clone https://github.com/amitkharche/Classification_Wine_Quality_Naive_Bayes_Streamlit.git
   cd Classification_Wine_Quality_Naive_Bayes_Streamlit

   ```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python scripts/train.py --data_path data/winequality.csv --model_path models/nb_model.pkl
```

### 4. Evaluate the Model

```bash
python scripts/evaluate.py
```

### 5. Launch the Streamlit App

```bash
streamlit run streamlit_app/app.py
```

---

## Streamlit App Features

### Predict Tab

* Inputs 11 wine features in a **responsive 4-column layout**
* Predicts wine **quality score**
* Shows **SHAP waterfall plot** with top feature impacts (compact size)

### Performance Tab

* Displays **classification report**
* Visualizes **confusion matrix heatmap**

---

## SHAP Explainability

* SHAP helps understand **why** the model made its prediction
* Uses `shap.Explainer` with waterfall plots
* Limited to **top 8 features** for clarity
* Compact figure size for clean dashboard integration

---

## Docker Setup

### Build and Run Manually

```bash
docker build -t wine-predictor .
docker run -p 8501:8501 wine-predictor
```

### With Docker Compose

```bash
docker-compose up --build
```

---

## GitHub Actions (CI/CD)

* Basic CI workflow: `.github/workflows/ci.yml`
* Automatically runs training/evaluation on push
* Extendable to include unit tests, linting, or deployment

---

## Dataset Source

* [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
* Features: acidity, sugar, alcohol, etc.
* Target: wine `quality` score (integer)

---

## Future Enhancements

* [ ] Add RandomForest & LightGBM comparison
* [ ] Integrate MLflow for model tracking
* [ ] Enable Streamlit Cloud/Spaces deployment
* [ ] Add SHAP bar plots and LIME explanations

---

## Deploy on Streamlit Cloud

1. Push repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Set:

   * Main file: `streamlit_app/app.py`
   * Requirements file: `requirements.txt`

---

## License

Licensed under the MIT License. See `LICENSE` for details.

---

## Contributing

Found a bug or have suggestions?
Open an issue or submit a PR. Contributions are welcome!

---

## 📬 Contact

If you have questions or want to collaborate, feel free to connect with me on
- [LinkedIn](https://www.linkedin.com/in/amit-kharche)  
- [Medium](https://medium.com/@amitkharche14)  
- [GitHub](https://github.com/amitkharche)

---