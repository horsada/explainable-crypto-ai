# src/excrypto/cli.py
import typer
from excrypto.pipeline.cli import app as pipeline_app
from excrypto.training.cli import app as training_app
from excrypto.inference.cli import app as inference_app
from excrypto.baseline.cli import app as baseline_app
from excrypto.backtest.cli import app as backtest_app
from excrypto.risk.cli import app as risk_app
from excrypto.agents.cli import app as agents_app
from excrypto.features.cli import app as feature_app
#from excrypto.runner.cli import app as runner_app

app = typer.Typer(help="Explainable Crypto AI")
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(training_app, name="train")
app.add_typer(inference_app, name="predict")
app.add_typer(baseline_app, name="baseline")
app.add_typer(backtest_app, name="backtest")
app.add_typer(risk_app, name="risk")
app.add_typer(agents_app, name="agents")
app.add_typer(feature_app, name='features')
#app.add_typer(runner_app, name="run")

if __name__ == "__main__":
    app()
