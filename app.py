from pathlib import Path

import joblib
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

from utils.config import CHOSEN_THRESHOLD
from utils.preprocess import build_transaction_frame

app = FastAPI(title="Credit Card Fraud Detection")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
templates = Jinja2Templates(directory=BASE_DIR / "templates")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Load the trained model and the scaler used during training
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
FEATURE_COLUMNS = list(scaler.feature_names_in_)


@app.get('/')
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get('/feature-schema')
def feature_schema():
    return {"features": FEATURE_COLUMNS}


@app.post('/check-transaction')
def check_transaction(data: dict[str, float]):
    df, missing_fields, invalid_fields = build_transaction_frame(
        data, FEATURE_COLUMNS)

    if missing_fields or invalid_fields:
        return JSONResponse(status_code=400, content={
            "error": "Missing or invalid transaction fields",
            "missing_fields": missing_fields,
            "invalid_fields": invalid_fields
        })

    # Scale the incoming data the same way training data was scaled
    df_scaled = scaler.transform(df)

    fraud_probability = model.predict_proba(df_scaled)[0][1]
    is_fraud = bool(fraud_probability >= CHOSEN_THRESHOLD)
    prediction_confidence = fraud_probability if is_fraud else 1 - fraud_probability

    return {
        "fraud": is_fraud,
        "fraud_probability": round(float(fraud_probability), 4),
        "prediction_confidence": round(float(prediction_confidence), 4),
        "threshold_used": CHOSEN_THRESHOLD
    }


if __name__ == '__main__':
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)
