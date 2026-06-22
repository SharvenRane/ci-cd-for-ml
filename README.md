# CI/CD for ML

A continuous integration pipeline for a model, the kind that should gate every
change before it ships. On each push, GitHub Actions installs, runs the unit
tests, trains the model, and then checks it against a quality bar. If the model
comes in under the bar, the run fails and nothing gets promoted.

The example trains a classifier on the breast cancer dataset that comes with
scikit-learn, so it runs in seconds with nothing to download, but the structure
is the point and it carries straight over to a heavier model.

## The pipeline

`.github/workflows/ml.yml` runs on every push: install, test, train, then the
quality gate. `src/train.py` trains and writes `outputs/metrics.json`.
`src/quality_gate.py` reads those metrics and exits with an error if ROC-AUC or
F1 fall below the thresholds, which is what stops a regression from reaching
production.

## Run it locally

```
pip install -r requirements.txt
pytest tests/ -q
python src/train.py
python src/quality_gate.py --metrics outputs/metrics.json
```

## Tests

The tests confirm training produces a model and a sensible metric, and that the
gate genuinely fails when handed weak numbers, so a bad model cannot sneak
through.
