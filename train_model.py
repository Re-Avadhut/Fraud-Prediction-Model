import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

from utils.config import CHOSEN_THRESHOLD

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "fraud_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print("=== Default threshold (0.5) ===")
    print(classification_report(y_test, y_pred))

    y_probs = model.predict_proba(X_test_scaled)[:, 1]

    for threshold in [0.5, 0.7, 0.9, CHOSEN_THRESHOLD, 0.99]:
        y_pred_thresh = (y_probs >= threshold).astype(int)
        print(f"\n--- Threshold: {threshold} ---")
        print(classification_report(y_test, y_pred_thresh))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)


if __name__ == '__main__':
    main()
