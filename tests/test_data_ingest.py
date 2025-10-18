from scripts.data_ingest import fetch_binance_ohlcv

def test_fetch_binance_ohlcv():
    df = fetch_binance_ohlcv(limit=10)
    assert not df.empty
    assert "close" in df.columns
    assert df.shape[0] == 10
