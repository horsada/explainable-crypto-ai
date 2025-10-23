# src/excrypto/risk/cli.py
import typer, pandas as pd
from pathlib import Path
from excrypto.risk.report import write_risk_report_md

app = typer.Typer(help="Risk reports from PnL series")

@app.command("report")
def report(
    backtest_path: str = typer.Argument(..., help="Parquet with 'pnl_net'"),
    title: str = "Strategy",
    out_dir: str = "runs/report",
):
    bt = pd.read_parquet(backtest_path)
    if "pnl_net" not in bt.columns:
        raise typer.BadParameter("Need a 'pnl_net' column in the parquet.")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    md = write_risk_report_md(bt["pnl_net"], None, title, out_dir)
    typer.echo(f"Wrote {md}")
