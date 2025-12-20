import json, os, pandas as pd
from excrypto.data.snapshot import run as snapshot_run

def test_snapshot_writes(tmp_path, monkeypatch):
    out = tmp_path / "data" / "raw"
    # small universe to keep fast
    symbols = "BTC/USDT,ETH/USDT"
    written = snapshot_run(snapshot="2099-01-01", exchange="binance", symbols=symbols, ohlcv_limit=100, root=str(out))
    base = out / "2099-01-01" / "binance"
    assert set(written) <= {"BTC/USDT","ETH/USDT"}
    assert (base / "_snapshot_meta.json").exists()
    assert (base / "_universe.json").exists()
    for s in written:
        p = base / s.replace("/","_") / "ohlcv.parquet"
        assert p.exists()
        df = pd.read_parquet(p)
        assert {"timestamp","open","high","low","close","volume"}.issubset(df.columns)
        assert df["timestamp"].dt.tz is not None  # UTC

def test_snapshot_idempotent(tmp_path):
    out = tmp_path / "data" / "raw"
    snapshot_run(snapshot="2099-01-01", exchange="binance", symbols="BTC/USDT",
        timeframe="1m", ohlcv_limit=100, root=str(out),
    )

    # run again: files still there and readable
    snapshot_run(snapshot="2099-01-01", exchange="binance", symbols="BTC/USDT",
        timeframe="1m", ohlcv_limit=100, root=str(out),
    )
    assert (out / "2099-01-01" / "binance" / "BTC_USDT" / "ohlcv.parquet").exists()
