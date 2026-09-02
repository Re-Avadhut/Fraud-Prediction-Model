import pandas as pd


def build_transaction_frame(data, feature_columns):
    missing_fields = [field for field in feature_columns if field not in data]

    if missing_fields:
        return None, missing_fields

    return pd.DataFrame([data], columns=feature_columns), []
