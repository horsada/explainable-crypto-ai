import json
import pandas as pd

def load_logs(path="logs/inference_log.jsonl"):
    with open(path, "r") as f:
        lines = [json.loads(line) for line in f]
    return pd.DataFrame(lines)

if __name__ == "__main__":
    df = load_logs()
    pd.set_option("display.max_columns", None)
    print(df.tail())  # show latest predictions
