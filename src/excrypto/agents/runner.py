# src/excrypto/agents/daily.py
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from excrypto.agents.tools import sh
from excrypto.utils.registry import find  # registry helper for snapshots
from excrypto.utils.paths import RunPaths

# --------- helpers ---------
def _snapshot_exists(snapshot: str, exchange: str) -> bool:
    df = find(snapshot_id=snapshot)
    return (not df.empty) and (df["exchange"].nunique() == 1) and (df["exchange"].iloc[0] == exchange)

def _has(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0

def _parse_symbols(csv: str) -> List[str]:
    return [s.strip() for s in csv.split(",") if s.strip()]

def _params_str(params: Optional[Dict[str, str]]) -> str:
    if not params:
        return ""
    return ",".join(f"{k}={v}" for k,v in sorted(params.items()))

# --------- agents ---------
def run_daily(
    snapshot: Optional[str],
    symbols: str,
    exchange: str = "binance",
    override: bool = False,
    # default momentum params; change if you want a different run key
    mom_params: Optional[Dict[str, str]] = None,  # e.g. {"fast":"20","slow":"60"}
):
    snap = snapshot or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    syms = _parse_symbols(symbols)
    mom_params = mom_params or {"fast": "20", "slow": "60"}

    # 1) Snapshot (idempotent)
    if _snapshot_exists(snap, exchange) and not override:
        print(f"[agent] snapshot {snap}@{exchange} exists — skipping fetch")
    else:
        sh(["excrypto","pipeline","snapshot","--snapshot",snap,"--exchange",exchange,"--symbols",symbols])

    # Build paths for momentum
    m_paths = RunPaths(snapshot=snap, strategy="momentum", symbols=tuple(syms), timeframe='1m', params=mom_params)
    m_paths.ensure()

    # 2) Signals (momentum)
    if _has(m_paths.signals) and _has(m_paths.panel) and not override:
        print(f"[agent] momentum signals exist — skipping")
    else:
        args = ["excrypto","baseline","momentum","--snapshot",snap,"--symbols",symbols]
        # pass params to baseline so it saves under the same RunPaths params hash
        args += ["--fast", mom_params.get("fast","20"), "--slow", mom_params.get("slow","60")]
        sh(args)

    # 3) Backtest (momentum)
    if _has(m_paths.backtest) and not override:
        print(f"[agent] momentum backtest exists — skipping")
    else:
        sh([
            "excrypto","backtest","run",
            snap, "momentum", symbols,
            "--params", _params_str(mom_params),
            "--config", "config/backtest.yaml",
        ])

    # 4) Report (momentum)
    if _has(m_paths.report_md) and not override:
        print(f"[agent] momentum report exists — skipping")
    else:
        sh([
            "excrypto","risk","report",
            snap, "momentum", symbols,
            "--params", _params_str(mom_params),
            "--pnl-col","pnl_net",
            "--title", f"Momentum | {snap}",
        ])

def run_range(
    start: str,
    end: str,
    timeframe: str,
    symbols: str,
    exchange: str = "binance",
    override: bool = False,
    mom_params: Optional[Dict[str, str]] = None,  # e.g. {"fast":"20","slow":"60"}
):
    snap = f"{start}_to_{end}"
    syms = _parse_symbols(symbols)
    mom_params = mom_params or {"fast": "20", "slow": "60"}

    # 1) Combined snapshot (range)
    if _snapshot_exists(snap, exchange) and not override:
        print(f"[agent] combined snapshot {snap}@{exchange} exists — skipping fetch")
    else:
        sh([
            "excrypto","pipeline","snapshot",
            "--start", start, "--end", end,
            "--timeframe", timeframe,
            "--exchange", exchange, "--symbols", symbols
        ])

    # --- Momentum ---
    m_paths = RunPaths(snapshot=snap, strategy="momentum", symbols=tuple(syms), timeframe=timeframe, params=mom_params)
    m_paths.ensure()

    # 2) Signals
    if _has(m_paths.signals) and _has(m_paths.panel) and not override:
        print(f"[agent] momentum signals exist — skipping")
    else:
        sh([
            "excrypto","baseline","momentum",
            "--snapshot", snap, "--symbols", symbols,
            "--fast", mom_params.get("fast","20"), "--slow", mom_params.get("slow","60")
        ])

    # 3) Backtest
    if _has(m_paths.backtest) and not override:
        print(f"[agent] momentum backtest exists — skipping")
    else:
        sh([
            "excrypto","backtest","run",
            snap, "momentum", symbols,
            "--params", _params_str(mom_params),
            "--config", "config/backtest.yaml",
        ])

    # 4) Report
    if _has(m_paths.report_md) and not override:
        print(f"[agent] momentum report exists — skipping")
    else:
        sh([
            "excrypto","risk","report",
            snap, "momentum", symbols,
            "--params", _params_str(mom_params),
            "--pnl-col","pnl_net",
            "--title", f"momentum | {snap}",
        ])

    # --- HODL (no params) ---
    h_paths = RunPaths(snapshot=snap, strategy="hodl", symbols=tuple(syms), timeframe=timeframe, params=None)
    h_paths.ensure()

    # 2) Signals
    if _has(h_paths.signals) and _has(h_paths.panel) and not override:
        print(f"[agent] hodl signals exist — skipping")
    else:
        sh(["excrypto","baseline","hodl","--snapshot", snap, "--symbols", symbols])

    # 3) Backtest
    if _has(h_paths.backtest) and not override:
        print(f"[agent] hodl backtest exists — skipping")
    else:
        sh([
            "excrypto","backtest","run",
            snap, "hodl", symbols,
            "--config", "config/backtest.yaml",
        ])

    # 4) Report
    if _has(h_paths.report_md) and not override:
        print(f"[agent] hodl report exists — skipping")
    else:
        sh([
            "excrypto","risk","report",
            snap, "hodl", symbols,
            "--pnl-col","pnl_net",
            "--title", f"hodl | {snap}",
        ])
