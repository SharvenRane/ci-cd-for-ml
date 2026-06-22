"""Quality gate: the deploy step only passes if the model clears the bar.

Reads outputs/metrics.json and exits non-zero when a metric falls below its
threshold, so a regression stops the pipeline instead of shipping.
"""
from __future__ import annotations

import argparse
import json
import sys

THRESHOLDS = {"roc_auc": 0.95, "f1": 0.93}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="outputs/metrics.json")
    args = ap.parse_args()

    with open(args.metrics) as f:
        metrics = json.load(f)

    failed = []
    for name, floor in THRESHOLDS.items():
        value = metrics.get(name)
        ok = value is not None and value >= floor
        print(f"{'PASS' if ok else 'FAIL'}  {name} = {value:.4f}  (floor {floor})")
        if not ok:
            failed.append(name)

    if failed:
        print(f"\nQuality gate failed on: {', '.join(failed)}")
        sys.exit(1)
    print("\nQuality gate passed.")


if __name__ == "__main__":
    main()
