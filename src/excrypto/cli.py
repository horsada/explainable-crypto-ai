import typer
from excrypto.pipeline.snapshot import run as snapshot_run
from excrypto.pipeline import run as features_run
from excrypto.training.runner import run as train_run
from excrypto.inference.runner import run as predict_run

app = typer.Typer(help="Explainable Crypto AI CLI")

@app.command()
def snapshot(snapshot: str, exchange: str = "binance", symbols: str = "BTC/USDT,ETH/USDT"):
    snapshot_run(snapshot=snapshot, exchange=exchange, symbols=symbols)

@app.command()
def features(snapshot: str, exchange: str = "binance"):
    features_run(snapshot=snapshot, exchange=exchange)

@app.command()
def train(config: str):
    train_run(config)

@app.command()
def predict(checkpoint: str, snapshot: str):
    predict_run(checkpoint, snapshot)

if __name__ == "__main__":
    app()
