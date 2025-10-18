import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_shap_trends(csv_path="plots/shap_values.csv", out_path="plots/shap_trends.png"):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    plt.figure(figsize=(10, 6))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)

    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.legend(loc="upper right")
    plt.title("SHAP Value Trends Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("SHAP Value")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"✅ Saved SHAP trends plot to {out_path}")
