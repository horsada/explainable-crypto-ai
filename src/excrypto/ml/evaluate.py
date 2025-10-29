from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def cls_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int) if set(np.unique(y_true)) <= {0,1} else np.sign(y_score)
    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1":  float(f1_score(y_true, y_pred, average="macro")),
    }
    return out
