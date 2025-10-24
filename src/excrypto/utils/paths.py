# src/excrypto/utils/paths.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib

def sym_slug(s: str) -> str: return s.replace("/", "_")

def universe_id(symbols: list[str]) -> str:
    return hashlib.md5(",".join(sorted(symbols)).encode()).hexdigest()[:8]

def params_id(params: dict | None) -> str:
    if not params: 
        return "p-default"
    key = "|".join(f"{k}={params[k]}" for k in sorted(params))
    return "p-" + hashlib.md5(key.encode()).hexdigest()[:6]

@dataclass(frozen=True)
class RunPaths:
    snapshot: str
    strategy: str
    symbols: tuple[str, ...]               # keep stable ordering
    params: dict | None = None
    runs_root: Path = Path("runs")

    @property
    def universe(self) -> str:
        return sym_slug(self.symbols[0]) if len(self.symbols)==1 else f"u-{universe_id(list(self.symbols))}"

    @property
    def base(self) -> Path:
        return self.runs_root / self.snapshot / self.strategy / self.universe / params_id(self.params)

    # common artifacts
    @property
    def signals(self) -> Path:   return self.base / "signals.parquet"
    @property
    def panel(self) -> Path:     return self.base / "panel.parquet"
    @property
    def backtest(self) -> Path:  return self.base / "backtest.parquet"
    @property
    def report_dir(self) -> Path:return self.base / "report"
    @property
    def report_md(self) -> Path: return self.report_dir / "risk_report.md"

    def ensure(self) -> None:
        self.base.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
