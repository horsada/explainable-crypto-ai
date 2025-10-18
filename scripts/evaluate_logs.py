import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.eval.evaluator import Evaluator

def load_logs(path="logs/inference_log.jsonl"):
    with open(path, "r") as f:
        lines = [json.loads(line) for line in f]
    return pd.DataFrame(lines)

if __name__ == "__main__":
    df = load_logs()
    df = df.dropna(subset=["actual"])  # only evaluate entries with actuals

    if df.empty:
        print("❗ No predictions with actual values to evaluate.")
    else:
        evaluator = Evaluator()
        metrics = evaluator.evaluate(y_true=df["actual"], y_pred=df["prediction"])
        print("📊 Evaluation metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    # === Accuracy trend plot ===
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    daily_acc = df.groupby("date").apply(lambda x: (x["prediction"] == x["actual"]).mean())
    rolling = daily_acc.rolling(window=3).mean()

    plt.figure(figsize=(10, 4))
    plt.plot(daily_acc.index, daily_acc.values, label="Daily Accuracy")
    plt.plot(rolling.index, rolling.values, label="3-Day Avg", linestyle="--")
    plt.title("📈 Daily Prediction Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Date")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/accuracy_trend.png")
    print("✅ Saved accuracy trend → plots/accuracy_trend.png")

