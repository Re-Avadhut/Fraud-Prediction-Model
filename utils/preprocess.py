import pandas as pd
import math


def build_transaction_frame(data, feature_columns):
    missing_fields = [field for field in feature_columns if field not in data]

    if missing_fields:
        return None, missing_fields, []

    invalid_fields = []
    for field in feature_columns:
        try:
            if not math.isfinite(float(data[field])):
                invalid_fields.append(field)
        except (TypeError, ValueError):
            invalid_fields.append(field)

    if invalid_fields:
        return None, [], invalid_fields

    return pd.DataFrame([data], columns=feature_columns), [], []
