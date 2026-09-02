# Credit Card Fraud Detection

Flask web app for checking whether a credit-card transaction is likely fraudulent using a trained scikit-learn model.

## Project Structure

```text
Fraud Detection/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   ├── .gitkeep
│   └── creditcard.csv
├── models/
│   ├── fraud_model.pkl
│   └── scaler.pkl
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
└── utils/
    └── preprocess.py
```

## Dataset

This project uses the Credit Card Fraud Detection dataset from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download `creditcard.csv` from Kaggle and place it here:

```text
data/creditcard.csv
```

The dataset is intentionally ignored by Git because it is large and should be downloaded from the original Kaggle source.

## Run The App

```powershell
venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:5000/`.

## Retrain The Model

```powershell
venv\Scripts\python.exe train_model.py
```
