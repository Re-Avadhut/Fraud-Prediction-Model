from flask import Flask, request, jsonify, render_template
import joblib
from pathlib import Path
from utils.preprocess import build_transaction_frame

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

# Load the trained model and the scaler used during training
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
FEATURE_COLUMNS = list(scaler.feature_names_in_)

CHOSEN_THRESHOLD = 0.95


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/check-transaction', methods=['POST'])
def check_transaction():
    data = request.get_json()
    df, missing_fields = build_transaction_frame(data, FEATURE_COLUMNS)

    if missing_fields:
        return jsonify({
            "error": "Missing required transaction fields",
            "missing_fields": missing_fields
        }), 400

    # Scale the incoming data the same way training data was scaled
    df_scaled = scaler.transform(df)

    fraud_probability = model.predict_proba(df_scaled)[0][1]
    is_fraud = bool(fraud_probability >= CHOSEN_THRESHOLD)
    prediction_confidence = fraud_probability if is_fraud else 1 - fraud_probability

    return jsonify({
        "fraud": is_fraud,
        "fraud_probability": round(float(fraud_probability), 4),
        "prediction_confidence": round(float(prediction_confidence), 4),
        "threshold_used": CHOSEN_THRESHOLD
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
