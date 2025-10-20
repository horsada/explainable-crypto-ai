import json
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

class Evaluator:
    def __init__(self, output_path="logs/eval_metrics.json"):
        self.output_path = output_path

    def evaluate(self, y_true, y_pred, y_prob=None):
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
        }

        if y_prob is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

        self._save(metrics)
        return metrics

    def _save(self, metrics):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"✅ Metrics saved to {self.output_path}")
