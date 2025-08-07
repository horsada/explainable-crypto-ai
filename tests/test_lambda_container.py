import requests
import subprocess
import time

def test_lambda_docker_response():
    # Start container in background
    proc = subprocess.Popen([
        "docker", "run", "-p", "9000:8080", "xai-predictor-lambda"
    ])

    time.sleep(3)  # wait for container to boot

    try:
        res = requests.post(
            "http://localhost:9000/2015-03-31/functions/function/invocations",
            json={}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["prediction"] in ["UP", "DOWN"]
        assert 0.0 <= data["confidence"] <= 1.0
    finally:
        proc.terminate()
