# src/excrypto/agents/daily.py
from datetime import datetime, timezone
from excrypto.agents.tools import sh

def run_daily(snapshot: str | None, symbols: str, exchange="binance"):
    snap = snapshot or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # 1) snapshot
    sh(["excrypto","pipeline","snapshot","--snapshot",snap,"--exchange",exchange,"--symbols",symbols])
    # 2) signals (momentum)
    out = f"runs/{snap}/momentum"
    sh(["excrypto","baseline","momentum","--snapshot",snap,"--exchange",exchange,"--symbols",symbols,"--out-dir",out])
    # 3) backtest
    if "," in symbols:
        single_or_multi = 'multi'
    else:
        single_or_multi = 'single'
    sh(["excrypto","backtest",single_or_multi,"--data-path",f"{out}/panel.parquet","--out-path",f"{out}/backtest.parquet"])
    # 4) report
    sh(["excrypto","risk","report",f"{out}/backtest.parquet","--title",f"Momentum | {snap}","--out-dir",out])

def run_range(start: str, end: str, timeframe: str, symbols: str, exchange="binance"):
    # 1) snapshot
    sh(["excrypto","pipeline","snapshot","--start",start,"--end",end,"--timeframe",timeframe,"--exchange",exchange,"--symbols",symbols])
    # 2) signals (momentum)
    data_name = f"{start}_to_{end}"

    for strat in ['momentum', 'hodl']:
        out = f"runs/{data_name}/{strat}"
        sh(["excrypto","baseline",f"{strat}","--snapshot",data_name,"--exchange",exchange,"--symbols",symbols,"--out-dir",out])
        # 3) backtest
        if "," in symbols:
            single_or_multi = 'multi'
        else:
            single_or_multi = 'single'
        sh(["excrypto","backtest",single_or_multi,"--data-path",f"{out}/panel.parquet","--out-path",f"{out}/backtest.parquet"])
        # 4) report
        sh(["excrypto","risk","report",f"{out}/backtest.parquet","--title",f"{strat} | {data_name}","--out-dir",out])