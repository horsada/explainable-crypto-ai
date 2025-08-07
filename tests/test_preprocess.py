import pandas as pd
from src.preprocess import engineer_features

def test_engineer_features():
    # Sample mini-DataFrame
    df = pd.DataFrame({
        "close": [100, 102, 101, 105, 107, 110, 109],
    })
    df["return"] = df["close"].pct_change()
    result = engineer_features(df)
    assert "sma_diff" in result.columns
    assert not result.isna().any().any()
