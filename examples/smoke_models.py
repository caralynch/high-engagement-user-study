"""Run the account study model-fitting procedure on invented data."""

import csv
import math
import tempfile
from pathlib import Path

from run_models import ROOT, fit_models
from synthetic_fixture import create_synthetic_panel


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    expected_features = [row["feature_name"] for row in read_rows(ROOT / "config/selected_features.csv")]
    runs = ROOT / "runs" / "synthetic"
    runs.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="run-", dir=runs))
    print(f"Saving synthetic inputs and results in {workspace}", flush=True)
    panel = workspace / "synthetic_panel.parquet"
    output = workspace / "results"
    create_synthetic_panel(panel)
    fit_models(panel, ROOT / "config/feature_schema.csv", output, 1, "1GB", synthetic=True)

    for model, candidate_count in (("logistic", 48), ("lightgbm", 72)):
        model_root = output / model
        feature_rows = read_rows(model_root / "selected_features.csv")
        selected = [row["feature_name"] for row in feature_rows if row["included"].lower() in {"true", "1"}]
        candidates = read_rows(model_root / f"{model}_validation_metrics_by_candidate.csv")
        metrics = read_rows(model_root / f"{model}_metrics_by_fold.csv")
        if selected != expected_features or len(candidates) != candidate_count:
            raise RuntimeError(f"Unexpected features or candidate count for {model}")
        if [int(row["fold_id"]) for row in metrics] != list(range(1, 7)):
            raise RuntimeError(f"Missing or reordered test folds for {model}")
        if any(row["split_role"] != "test" or not math.isfinite(float(row["average_precision"])) for row in metrics):
            raise RuntimeError(f"Invalid held-out metrics for {model}")
    print("Synthetic check passed for both models across all six folds.")


if __name__ == "__main__":
    main()
