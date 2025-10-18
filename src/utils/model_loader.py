import os
import json
from glob import glob

def get_latest_model_metadata(models_dir="models/"):
    meta_files = sorted(
        glob(os.path.join(models_dir, "*_meta.json")),
        reverse=True
    )
    if not meta_files:
        raise FileNotFoundError("No model metadata files found.")
    with open(meta_files[0]) as f:
        return json.load(f)

def load_latest_model():
    metadata = get_latest_model_metadata()
    import joblib
    model = joblib.load(metadata["model_path"])
    return model, metadata
