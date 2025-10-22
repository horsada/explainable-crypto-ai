import typer
# from excrypto.training.runner import run as train_run  # implement this

app = typer.Typer(help="Model training")

@app.command()
def run(config: str = typer.Argument(..., help="YAML config for training")):
    # train_run(config)  # call your training loop here
    typer.echo(f"[stub] Training with {config}")
