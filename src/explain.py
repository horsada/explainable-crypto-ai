import shap
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv("data/features.csv", index_col="open_time", parse_dates=True)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna()
    return df

def explain_model(df, model):
    features = ["return", "volatility", "sma_5", "sma_10", "sma_diff"]
    X = df[features]
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("plots/shap_summary.png", bbox_inches="tight")
    plt.clf()

    shap.plots.bar(shap_values, show=False)
    plt.savefig("plots/shap_bar.png", bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    df = load_data()
    model = joblib.load("models/price_predictor.pkl")
    explain_model(df, model)
