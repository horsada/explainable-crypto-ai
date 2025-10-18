import json
import pandas as pd
from datetime import datetime

LOG_PATH = "logs/inference_log.jsonl"
FEATURES_PATH = "data/features.csv"

def load_logs(path=LOG_PATH):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def save_logs(logs, path=LOG_PATH):
    with open(path, "w") as f:
        for entry in logs:
            f.write(json.dumps(entry) + "\n")

def match_actuals(logs, df):
    df = df.copy()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna()

    for log in logs:
        if log.get("actual") is not None:
            continue  # skip already filled

        ts = pd.to_datetime(log["timestamp"])
        closest = df.index[df.index.get_loc(ts, method="nearest")]

        if abs((ts - closest).total_seconds()) > 3600:
            continue  # skip if too far

        log["actual"] = int(df.loc[closest, "target"])

    return logs

if __name__ == "__main__":
    df = pd.read_csv(FEATURES_PATH, index_col="open_time", parse_dates=True)
    logs = load_logs()
    logs = match_actuals(logs, df)
    save_logs(logs)
    print("✅ Filled actual values in inference logs.")
