"""Tests the CI pipeline runs and the gate logic behaves."""
import json
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_training_writes_a_good_model():
    subprocess.run([sys.executable, "src/train.py"], cwd=ROOT, check=True)
    with open(os.path.join(ROOT, "outputs", "metrics.json")) as f:
        m = json.load(f)
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert m["roc_auc"] > 0.9          # breast cancer is an easy, stable benchmark
    assert os.path.exists(os.path.join(ROOT, "outputs", "model.joblib"))


def test_quality_gate_fails_on_bad_metrics(tmp_path):
    bad = tmp_path / "metrics.json"
    bad.write_text(json.dumps({"roc_auc": 0.5, "f1": 0.4}))
    r = subprocess.run([sys.executable, "src/quality_gate.py", "--metrics", str(bad)], cwd=ROOT)
    assert r.returncode == 1            # gate should reject a weak model


if __name__ == "__main__":
    test_training_writes_a_good_model()
    import tempfile, pathlib
    test_quality_gate_fails_on_bad_metrics(pathlib.Path(tempfile.mkdtemp()))
    print("all tests passed")
