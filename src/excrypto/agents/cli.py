# src/excrypto/agents/cli.py
from __future__ import annotations
import typer
from pathlib import Path
from typing_extensions import Annotated

from excrypto.agents.runner import run_daily, run_range
from excrypto.agents.orchestrator import run_plan
from excrypto.agents.catalog import summarize

app = typer.Typer(help="Agentic workflows")

@app.command("daily")
def daily(
    snapshot: Annotated[str, typer.Option(help="YYYY-MM-DD UTC; empty = today")] = "",
    symbols: str = "BTC/USDT,ETH/USDT",
    exchange: str = "binance",
    override: Annotated[bool, typer.Option("--override")] = False,
    # momentum params
    fast: Annotated[int, typer.Option(help="Momentum fast window")] = 20,
    slow: Annotated[int, typer.Option(help="Momentum slow window")] = 60,
    # optional ML pipeline configs
    features_cfg: Annotated[Path | None, typer.Option("--features-cfg", exists=True, dir_okay=False)] = None,
    labels_cfg:   Annotated[Path | None, typer.Option("--labels-cfg",   exists=True, dir_okay=False)] = None,
    ml_train_cfg: Annotated[Path | None, typer.Option("--ml-train-cfg", exists=True, dir_okay=False)] = None,
    runs_root:    Annotated[Path, typer.Option("--runs-root")] = Path("runs"),
):
    run_daily(
        snapshot or None,
        symbols,
        exchange,
        override,
        mom_params={"fast": str(fast), "slow": str(slow)},
        features_cfg=features_cfg,
        labels_cfg=labels_cfg,
        ml_train_cfg=ml_train_cfg,
        runs_root=runs_root,
    )

@app.command("range")
def range(
    start: Annotated[str, typer.Option(help="YYYY-MM-DD UTC")] = "",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD UTC (inclusive)")] = "",
    timeframe: str = "1m",
    symbols: str = "BTC/USDT,ETH/USDT",
    exchange: str = "binance",
    override: Annotated[bool, typer.Option("--override")] = False,
    # momentum params
    fast: Annotated[int, typer.Option(help="Momentum fast window")] = 20,
    slow: Annotated[int, typer.Option(help="Momentum slow window")] = 60,
    # optional ML pipeline configs
    features_cfg: Annotated[Path | None, typer.Option("--features-cfg", exists=True, dir_okay=False)] = None,
    labels_cfg:   Annotated[Path | None, typer.Option("--labels-cfg",   exists=True, dir_okay=False)] = None,
    ml_train_cfg: Annotated[Path | None, typer.Option("--ml-train-cfg", exists=True, dir_okay=False)] = None,
    runs_root:    Annotated[Path, typer.Option("--runs-root")] = Path("runs"),
):
    run_range(
        start,
        end,
        timeframe,
        symbols,
        exchange,
        override,
        mom_params={"fast": str(fast), "slow": str(slow)},
        features_cfg=features_cfg,
        labels_cfg=labels_cfg,
        ml_train_cfg=ml_train_cfg,
        runs_root=runs_root,
    )

@app.command("run")
def run(config: Annotated[Path, typer.Option("--config","-c", exists=True, dir_okay=False)]):
    run_plan(config)

@app.command("catalog")
def catalog(snapshot: str = "", out_dir: str = "runs/data_audit"):
    df = summarize(snapshot or None)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    p_csv = Path(out_dir) / "catalog.csv"
    p_md  = Path(out_dir) / "catalog.md"
    df.to_csv(p_csv, index=False)
    p_md.write_text(df.to_markdown(index=False))
    typer.echo(f"Wrote {p_csv} and {p_md}")
