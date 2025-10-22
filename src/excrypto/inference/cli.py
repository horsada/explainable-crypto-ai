import typer
# from excrypto.inference.runner import run as infer_run  # implement this

app = typer.Typer(help="Inference / prediction")

@app.command()
def run(
    checkpoint: str = typer.Argument(..., help="Model checkpoint"),
    snapshot: str = typer.Option(..., help="Snapshot to score on"),
):
    # infer_run(checkpoint=checkpoint, snapshot=snapshot)
    typer.echo(f"[stub] Predict with {checkpoint} on snapshot {snapshot}")
