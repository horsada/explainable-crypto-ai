from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd
import typer

from excrypto.data import FeaturePipeline, FeatureConfig
from excrypto.training.trainer import ModelTrainer, TrainerConfig

app = typer.Typer(help="Train model and save artifact + metadata")

@app.command()
def main(config: str = typer.Option(..., "--config", "-c", help="YAML config path")):
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))

    df = pd.read_csv(cfg["data_path"], parse_dates=["open_time"])
    pipe = FeaturePipeline(FeatureConfig(**cfg["features"]))

    tcfg = cfg.get("trainer", {})
    trainer = ModelTrainer(pipe, TrainerConfig(
        output_dir=Path(tcfg.get("output_dir", "models")),
        random_state=tcfg.get("random_state", 42),
        lgbm_params=tcfg.get("lgbm_params"),
    ))

    trainer.train(df)
    path = trainer.save_model()
    typer.echo(f"Done. Saved: {path}")

if __name__ == "__main__":
    app()
