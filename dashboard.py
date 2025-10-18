import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
from PIL import Image
from src.predictor.predictor import CryptoPredictor
from src.eval.evaluator import Evaluator

# --- Load logs ---
@st.cache_data
def load_logs(path="logs/inference_log.jsonl"):
    with open(path, "r") as f:
        return pd.DataFrame([json.loads(line) for line in f])

# --- Load latest prediction ---
@st.cache_resource
def load_predictor():
    return CryptoPredictor()

# --- UI ---
st.title("🧠 Explainable Crypto AI Dashboard")

# Latest prediction
predictor = load_predictor()
X = predictor.load_latest_features()
pred, prob = predictor.predict(X)
sentiment = X.get("sentiment_score", [None])[0]

st.subheader("🔮 Latest Prediction")
st.write(f"**Prediction:** {'🟢 UP' if pred else '🔴 DOWN'}")
st.write(f"**Confidence:** {prob:.2f}")
st.write(f"**Sentiment Score:** {sentiment:.2f}" if sentiment is not None else "No sentiment score available")

# SHAP explanation
st.subheader("📊 SHAP Explanation")
predictor.explain(X)
st.image("plots/latest_shap_explanation.png")

# Accuracy trend
st.subheader("📈 Accuracy Trend")
st.image("plots/accuracy_trend.png")

# SHAP value trends over time
st.subheader("📉 SHAP Value Trends")

shap_csv_path = "plots/shap_values.csv"
if os.path.exists(shap_csv_path):
    df_shap = pd.read_csv(shap_csv_path, index_col=0, parse_dates=True)
    st.line_chart(df_shap)
else:
    st.warning("No SHAP values found. Run batch explanation to generate them.")


# Metrics
st.subheader("📋 Evaluation Metrics")
df_logs = load_logs()
df_eval = df_logs.dropna(subset=["actual"])
if not df_eval.empty:
    evaluator = Evaluator()
    metrics = evaluator.evaluate(df_eval["actual"], df_eval["prediction"])
    st.write(metrics)
else:
    st.warning("No predictions with actuals to evaluate yet.")
