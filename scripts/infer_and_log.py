import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Load model and data
model = joblib.load("models/price_predictor.pkl")
df = pd.read_csv("data/features.csv")
latest_row = df.tail(1).drop(columns=["open_time"], errors="ignore")

# Run prediction
pred = model.predict(latest_row)[0]
prob = model.predict_proba(latest_row)[0].max() if hasattr(model, "predict_proba") else None

# SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(latest_row)

# Get top 3 contributing features (for class 1 if binary)
class_idx = 1 if hasattr(shap_values, '__len__') and isinstance(shap_values, list) else 0
shap_contribs = shap_values[class_idx][0] if isinstance(shap_values, list) else shap_values[0]
top_features = sorted(
    zip(latest_row.columns, shap_contribs),
    key=lambda x: abs(x[1]),
    reverse=True
)[:3]

# Format output
timestamp = datetime.now().isoformat(timespec="seconds")
result = {
    "timestamp": timestamp,
    "prediction": int(pred),
    "confidence": round(prob, 3) if prob else "N/A",
    "top_features": ", ".join([f"{f} ({round(s, 3)})" for f, s in top_features])
}

# Print to console
print(f"🧠 {timestamp} → Pred: {result['prediction']} | Conf: {result['confidence']}")
print(f"📊 Top features: {result['top_features']}")

# Optional: save SHAP plot
shap_dir = "plots/shap"
os.makedirs(shap_dir, exist_ok=True)
plot_path = f"{shap_dir}/shap_{timestamp.replace(':', '-')}.png"
shap.summary_plot(shap_values, latest_row, show=False)
plt.title(f"SHAP: {timestamp}")
plt.savefig(plot_path)
plt.close()

# Log to CSV
log_path = "data/predictions_log.csv"
pd.DataFrame([result]).to_csv(log_path, mode="a", index=False, header=not os.path.exists(log_path))
