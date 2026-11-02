import os, joblib, pandas as pd
from src.preprocessing import engineer_features
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "data", "loan_data_set.csv")).drop(columns=["Loan_ID","Loan_Status"]).head(10)
df = engineer_features(df)
model = joblib.load(os.path.join(BASE_DIR, "models", "best_model.joblib"))
preds = model.predict(df)
probs = model.predict_proba(df)[:,1]
print("Predictions:", preds)
print("Probabilities:", probs.round(3))
