from src.infer import run_inference, load_latest_features
import joblib

def test_inference_output_structure():
    model = joblib.load("models/price_predictor.pkl")
    X = load_latest_features()
    pred, prob = run_inference(model, X)
    assert pred in [0, 1]
    assert 0.0 <= prob <= 1.0
