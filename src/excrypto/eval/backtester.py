import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from src.eval.evaluator import Evaluator

class Backtester:
    def __init__(self, model, features, log_path="logs/backtest_metrics.json"):
        self.model = model
        self.features = features
        self.evaluator = Evaluator(log_path)

    def run(self, df, test_window=30):
        preds, targets, indices = [], [], []

        for i in range(test_window, len(df) - 1):
            train = df.iloc[:i]
            test = df.iloc[i:i+1]

            X_train = train[self.features]
            y_train = train["target"]

            X_test = test[self.features]
            y_test = test["target"]

            self.model.fit(X_train, y_train)
            pred = self.model.predict(X_test)[0]

            preds.append(pred)
            targets.append(y_test.values[0])
            indices.append(test.index[0])

        return pd.DataFrame({"timestamp": indices, "prediction": preds, "actual": targets})

    def evaluate(self, results_df):
        return self.evaluator.evaluate(
            y_true=results_df["actual"],
            y_pred=results_df["prediction"]
        )
