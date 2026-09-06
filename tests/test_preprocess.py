import math

from utils.preprocess import build_transaction_frame


def test_build_transaction_frame_preserves_feature_order():
    frame, missing, invalid = build_transaction_frame(
        {"V1": 2.0, "Time": 1.0}, ["Time", "V1"]
    )

    assert list(frame.columns) == ["Time", "V1"]
    assert missing == []
    assert invalid == []


def test_build_transaction_frame_rejects_non_finite_values():
    frame, missing, invalid = build_transaction_frame(
        {"Time": math.nan, "V1": 2.0}, ["Time", "V1"]
    )

    assert frame is None
    assert missing == []
    assert invalid == ["Time"]
