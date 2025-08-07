import pandas as pd
import joblib
import shap

def load_latest_features(path="data/features.csv"):
    features = ["return", "volatility", "sma_5", "sma_10", "sma_diff"]
    df = pd.read_csv(path, index_col="open_time", parse_dates=True)
    df = df.dropna()
    latest = df[features].iloc[[-1]]
    return latest


def run_inference(model, X):
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0][prediction]
    return prediction, proba

def explain_prediction(model, X):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.plots.bar(shap_values, show=False)
    
    import matplotlib.pyplot as plt
    plt.savefig("plots/latest_shap_explanation.png", bbox_inches="tight")
    print("Explanation saved to plots/latest_shap_explanation.png")

if __name__ == "__main__":
    model = joblib.load("models/price_predictor.pkl")
    X = load_latest_features()
    pred, prob = run_inference(model, X)

    print(f"Prediction: {'UP' if pred else 'DOWN'} (confidence: {prob:.2f})")
    explain_prediction(model, X)
