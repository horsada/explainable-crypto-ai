from pathlib import Path
import yaml
import typer

from excrypto.inference.predictor import CryptoPredictor

app = typer.Typer(help="Load model, build features, predict + explain, log JSONL")

@app.command()
def main(config: str = typer.Option(..., "--config", "-c", help="YAML config path")):
    # CryptoPredictor already loads YAML internally, but we verify path exists
    p = Path(config)
    if not p.exists():
        raise typer.BadParameter(f"Config not found: {p}")
    predictor = CryptoPredictor(config_path=str(p))
    predictor.run()

if __name__ == "__main__":
    app()
