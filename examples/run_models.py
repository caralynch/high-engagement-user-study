"""Fit and validate the two high-engagement user models."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FEATURE_SET = "activity_plus_historical_engagement_reduced_collinearity_logged_rate_only"
SPLITS = ROOT / "config" / "rolling_origin_splits.csv"


def run_stage(script, *arguments):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT), env.get("PYTHONPATH"))))
    subprocess.run(
        [sys.executable, "-B", str(ROOT / "Modelling" / script), *map(str, arguments)],
        check=True,
        cwd=ROOT,
        env=env,
    )


def fit_models(panel, schema, output, threads, memory, *, synthetic=False):
    output.mkdir(parents=True, exist_ok=False)
    shared = (
        "--panel-path", panel,
        "--feature-schema-csv", schema,
        "--feature-set", FEATURE_SET,
        "--reference-splits-csv", SPLITS,
        "--target-score-col", "target_score",
        "--selection-metric", "average_precision",
        "--duckdb-threads", threads,
        "--duckdb-memory-limit", memory,
        "--duckdb-temp-dir", output / "spill",
    )
    smoke = ("--smoke",) if synthetic else ()
    logistic = output / "logistic"
    lightgbm = output / "lightgbm"
    run_stage(
        "1_1_run_logistic_baseline.py", *shared, *smoke,
        "--out-root", logistic, "--random-seed", 20260529,
        "--c-grid", "0.01,0.1,1.0,10.0", "--class-weight-options", "none,balanced",
    )
    run_stage(
        "1_1_validate_logistic_outputs.py", "--out-root", logistic,
        "--expected-feature-set", FEATURE_SET, "--expected-panel-path", panel,
        "--expected-reference-splits-csv", SPLITS, "--expected-target-score-col", "target_score",
        "--expected-random-seed", 20260529, "--expected-validation-candidates", 48,
    )
    run_stage(
        "1_4_run_lightgbm_classifier.py", *shared, *smoke,
        "--out-root", lightgbm, "--random-seed", 2021, "--lightgbm-threads", 1,
    )
    run_stage(
        "1_4_validate_lightgbm_outputs.py", "--out-root", lightgbm, "--panel-path", panel,
        "--expected-feature-set", FEATURE_SET, "--expected-random-seed", 2021,
        "--expected-validation-candidates", 72,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-path", required=True, type=Path)
    parser.add_argument("--feature-schema-csv", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path, help="A new run directory")
    parser.add_argument("--threads", type=int, default=1, help="DuckDB threads")
    parser.add_argument("--memory", default="4GB", help="DuckDB memory limit")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    for path in (args.panel_path, args.feature_schema_csv):
        if not path.is_file():
            parser.error(f"Input file does not exist: {path}")
    fit_models(
        args.panel_path.resolve(), args.feature_schema_csv.resolve(),
        args.out_root.resolve(), args.threads, args.memory,
    )


if __name__ == "__main__":
    main()
