"""Train a small classifier and write metrics. Fast and deterministic so CI can
run it on every push.

Uses the breast cancer dataset that ships with scikit-learn (no download), which
keeps the example honest and medical adjacent.
"""
from __future__ import annotations

import json
import os

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

OUT = "outputs"


def main():
    os.makedirs(OUT, exist_ok=True)
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_tr, y_tr)

    prob = model.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = {"roc_auc": float(roc_auc_score(y_te, prob)),
               "f1": float(f1_score(y_te, pred)),
               "n_test": int(len(y_te))}

    joblib.dump(model, os.path.join(OUT, "model.joblib"))
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"ROC-AUC {metrics['roc_auc']:.4f}  F1 {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
