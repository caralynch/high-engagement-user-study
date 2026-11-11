#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Cara Lynch
# See the LICENSE file for details.
"""Run the Study 2 LightGBM binary classifier.

This script evaluates a LightGBM classifier for the prospective
``label_high_engagement`` target. It uses the locked 2021 expanding-window
rolling-origin design, uses the validation month for Average-Precision early
stopping and structural-candidate selection, then refits the selected model on
training plus validation rows before evaluating each test month once.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import platform
import socket
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

try:
    import duckdb
except Exception:  # pragma: no cover - exercised only when duckdb is absent
    duckdb = None

try:
    from lightgbm import LGBMClassifier
    import lightgbm
except Exception:  # pragma: no cover - exercised only when lightgbm is absent
    LGBMClassifier = None
    lightgbm = None


logger = logging.getLogger(__name__)

SCRIPT_PATH = Path(__file__).resolve()
LOGISTIC_SCRIPT = SCRIPT_PATH.with_name("1_1_run_logistic_baseline.py")


def load_logistic_helpers():
    """Load the logistic runner as the shared modelling helper source."""
    spec = importlib.util.spec_from_file_location("study2_logistic_helpers_for_lightgbm", LOGISTIC_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load logistic helper script: {LOGISTIC_SCRIPT}")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


helpers = load_logistic_helpers()

EXPECTED_PREDICTION_MONTHS_2021 = helpers.EXPECTED_PREDICTION_MONTHS_2021
PRIMARY_TEST_MONTHS_2021 = helpers.PRIMARY_TEST_MONTHS_2021
TOP_10_FRACTION = helpers.TOP_10_FRACTION
PR_AUC_DEFINITION = helpers.PR_AUC_DEFINITION
AUROC_DEFINITION = helpers.AUROC_DEFINITION
SUPPORTED_FEATURE_SETS = helpers.SUPPORTED_FEATURE_SETS
SELECTED_FEATURE_FIELDS = helpers.SELECTED_FEATURE_FIELDS

MODEL_NAME = "lightgbm_classifier"
MODEL_TYPE = "lightgbm_classifier"
NUM_LEAVES_CANDIDATES = (15, 31, 63)
REG_LAMBDA_CANDIDATES = (0.0, 1.0)
IS_UNBALANCE_CANDIDATES = (False, True)
FIXED_LEARNING_RATE = 0.05
FIXED_MIN_CHILD_SAMPLES = 50
FIXED_SUBSAMPLE = 1.0
FIXED_SUBSAMPLE_FREQ = 0
FIXED_COLSAMPLE_BYTREE = 1.0
MAX_N_ESTIMATORS = 1000
EARLY_STOPPING_ROUNDS = 50
EARLY_STOPPING_METRIC = "average_precision"
FEATURE_IMPORTANCE_NOTE = (
    "diagnostic_model_importance_not_causal; importances can be affected by correlated predictors"
)
TIE_HANDLING = (
    "Sort by predicted positive-class probability DESC, then deterministic "
    "seeded tie_break_score DESC, then user_id ASC."
)

KNOWN_OUTPUTS = (
    "run_metadata.json",
    "resolved_config.json",
    "selected_features.csv",
    "rolling_origin_splits.csv",
    "lightgbm_hyperparameter_grid.csv",
    "lightgbm_validation_metrics_by_candidate.csv",
    "lightgbm_selected_hyperparameters_by_fold.csv",
    "lightgbm_metrics_by_fold.csv",
    "lightgbm_metrics_summary.csv",
    "feature_importance.csv",
    "fitted_model_summary.json",
    "lightgbm_ranked_predictions.parquet",
)

HYPERPARAMETER_GRID_FIELDS = (
    "candidate_id",
    "model_name",
    "model_type",
    "boosting_type",
    "objective",
    "num_leaves",
    "reg_lambda",
    "is_unbalance",
    "imbalance_mode",
    "learning_rate",
    "min_child_samples",
    "subsample",
    "subsample_freq",
    "colsample_bytree",
    "max_n_estimators",
    "early_stopping_rounds",
    "early_stopping_metric",
    "imputer_strategy",
    "random_state",
    "n_jobs",
)

VALIDATION_METRIC_FIELDS = (
    "model_name",
    "model_type",
    "feature_set",
    "fold_id",
    "candidate_id",
    "num_leaves",
    "reg_lambda",
    "is_unbalance",
    "imbalance_mode",
    "learning_rate",
    "min_child_samples",
    "subsample",
    "subsample_freq",
    "colsample_bytree",
    "max_n_estimators",
    "early_stopping_rounds",
    "early_stopping_metric",
    "best_iteration",
    "early_stopping_validation_metric_name",
    "early_stopping_validation_metric_value",
    "selection_fit_start_prediction_month",
    "selection_fit_end_prediction_month",
    "selection_fit_includes_validation",
    "selection_fit_n_rows",
    "selection_fit_n_positive",
    "selection_fit_n_negative",
    "selection_fit_positive_prevalence",
    "selection_fit_effective_positive_weight",
    "prediction_month",
    "feature_month",
    "split_role",
    "metric_scope",
    "train_start_prediction_month",
    "train_end_prediction_month",
    "validation_prediction_month",
    "test_prediction_month",
    "n_users",
    "n_positive",
    "n_negative",
    "positive_prevalence",
    "k_at_10pct",
    "top_k_positive_count",
    "tp_at_10pct",
    "fp_at_10pct",
    "fn_at_10pct",
    "tn_at_10pct",
    "precision_at_10pct",
    "recall_at_10pct",
    "mcc_at_10pct",
    "average_precision",
    "auprc_ap",
    "auroc",
    "mean_target_score_top_k",
    "mean_target_score_all",
    "selection_metric",
    "selection_rank_within_fold",
    "selected_by_validation",
    "selection_tie_break_order",
    "top_k_rule",
    "pr_auc_definition",
    "auroc_definition",
)

SELECTED_HYPERPARAMETER_FIELDS = (
    "model_name",
    "model_type",
    "feature_set",
    "fold_id",
    "selected_candidate_id",
    "num_leaves",
    "reg_lambda",
    "is_unbalance",
    "imbalance_mode",
    "learning_rate",
    "min_child_samples",
    "subsample",
    "subsample_freq",
    "colsample_bytree",
    "max_n_estimators",
    "early_stopping_rounds",
    "early_stopping_metric",
    "best_iteration",
    "early_stopping_validation_metric_name",
    "early_stopping_validation_metric_value",
    "selection_fit_start_prediction_month",
    "selection_fit_end_prediction_month",
    "selection_fit_includes_validation",
    "selection_fit_n_rows",
    "selection_fit_n_positive",
    "selection_fit_n_negative",
    "selection_fit_positive_prevalence",
    "selection_fit_effective_positive_weight",
    "final_refit_start_prediction_month",
    "final_refit_end_prediction_month",
    "final_refit_includes_validation",
    "final_refit_n_rows",
    "final_refit_n_positive",
    "final_refit_n_negative",
    "final_refit_positive_prevalence",
    "final_refit_effective_positive_weight",
    "final_refit_n_estimators",
    "final_model_relative_path",
    "selection_metric",
    "selected_validation_metric_value",
    "validation_average_precision",
    "validation_auroc",
    "validation_precision_at_10pct",
    "validation_recall_at_10pct",
    "validation_mcc_at_10pct",
    "selection_rank_within_fold",
    "selection_tie_break_order",
    "train_start_prediction_month",
    "train_end_prediction_month",
    "validation_prediction_month",
    "test_prediction_month",
)

FOLD_METRIC_FIELDS = (
    "model_name",
    "model_type",
    "feature_set",
    "fold_id",
    "prediction_month",
    "test_prediction_month",
    "feature_month",
    "split_role",
    "metric_scope",
    "train_start_prediction_month",
    "train_end_prediction_month",
    "validation_prediction_month",
    "selected_candidate_id",
    "num_leaves",
    "reg_lambda",
    "is_unbalance",
    "imbalance_mode",
    "learning_rate",
    "min_child_samples",
    "subsample",
    "subsample_freq",
    "colsample_bytree",
    "max_n_estimators",
    "early_stopping_rounds",
    "early_stopping_metric",
    "best_iteration",
    "early_stopping_validation_metric_name",
    "early_stopping_validation_metric_value",
    "selection_fit_start_prediction_month",
    "selection_fit_end_prediction_month",
    "selection_fit_includes_validation",
    "selection_fit_n_rows",
    "selection_fit_n_positive",
    "selection_fit_n_negative",
    "selection_fit_positive_prevalence",
    "selection_fit_effective_positive_weight",
    "final_refit_start_prediction_month",
    "final_refit_end_prediction_month",
    "final_refit_includes_validation",
    "final_refit_n_rows",
    "final_refit_n_positive",
    "final_refit_n_negative",
    "final_refit_positive_prevalence",
    "final_refit_effective_positive_weight",
    "final_refit_n_estimators",
    "final_model_relative_path",
    "n_users",
    "n_positive",
    "n_negative",
    "positive_prevalence",
    "k_at_10pct",
    "top_k_positive_count",
    "tp_at_10pct",
    "fp_at_10pct",
    "fn_at_10pct",
    "tn_at_10pct",
    "precision_at_10pct",
    "recall_at_10pct",
    "mcc_at_10pct",
    "average_precision",
    "auprc_ap",
    "auroc",
    "mean_target_score_top_k",
    "mean_target_score_all",
    "top_k_rule",
    "pr_auc_definition",
    "auroc_definition",
)

SUMMARY_FIELDS = (
    "model_name",
    "model_type",
    "feature_set",
    "metric_scope",
    "source_split_role",
    "n_folds",
    "test_prediction_month_start",
    "test_prediction_month_end",
    "n_users_pooled",
    "n_positive_pooled",
    "k_at_10pct_pooled",
    "top_k_positive_count_pooled",
    "tp_at_10pct_pooled",
    "fp_at_10pct_pooled",
    "fn_at_10pct_pooled",
    "tn_at_10pct_pooled",
    "mcc_at_10pct_pooled",
    "n_users_macro_mean",
    "n_users_macro_stddev",
    "n_positive_macro_mean",
    "n_positive_macro_stddev",
    "positive_prevalence_macro_mean",
    "positive_prevalence_macro_stddev",
    "precision_at_10pct_macro_mean",
    "precision_at_10pct_macro_stddev",
    "recall_at_10pct_macro_mean",
    "recall_at_10pct_macro_stddev",
    "mcc_at_10pct_macro_mean",
    "mcc_at_10pct_macro_stddev",
    "average_precision_macro_mean",
    "average_precision_macro_stddev",
    "auprc_ap_macro_mean",
    "auprc_ap_macro_stddev",
    "auroc_macro_mean",
    "auroc_macro_stddev",
    "mean_target_score_top_k_macro_mean",
    "mean_target_score_top_k_macro_stddev",
    "mean_target_score_all_macro_mean",
    "mean_target_score_all_macro_stddev",
    "aggregation_note",
    "pr_auc_definition",
    "auroc_definition",
)

FEATURE_IMPORTANCE_FIELDS = (
    "model_name",
    "model_type",
    "feature_set",
    "fold_id",
    "test_prediction_month",
    "selected_candidate_id",
    "is_unbalance",
    "imbalance_mode",
    "best_iteration",
    "final_refit_start_prediction_month",
    "final_refit_end_prediction_month",
    "final_refit_includes_validation",
    "final_refit_n_positive",
    "final_refit_n_negative",
    "final_refit_n_rows",
    "final_refit_positive_prevalence",
    "final_refit_effective_positive_weight",
    "final_refit_n_estimators",
    "final_model_relative_path",
    "feature_name",
    "importance_split",
    "importance_gain",
    "importance_note",
)


@dataclass(frozen=True)
class DuckDBConfig:
    """DuckDB runtime configuration."""

    memory_limit: str
    threads: int
    temp_dir: str


@dataclass(frozen=True)
class LightGBMConfig:
    """Resolved LightGBM run configuration."""

    panel_path: str
    feature_schema_csv: str
    out_root: str
    feature_set: str
    reference_splits_csv: Optional[str]
    num_leaves_candidates: tuple[int, ...]
    reg_lambda_candidates: tuple[float, ...]
    is_unbalance_candidates: tuple[bool, ...]
    learning_rate: float
    min_child_samples: int
    subsample: float
    subsample_freq: int
    colsample_bytree: float
    max_n_estimators: int
    early_stopping_rounds: int
    early_stopping_metric: str
    selection_metric: str
    random_seed: int
    target_score_col: str
    overwrite: bool
    write_ranked_predictions: bool
    top_k_fraction: float
    smoke: bool
    smoke_max_rows: int
    min_train_months: int
    validation_window_months: int
    test_window_months: int
    lightgbm_threads: int


@dataclass(frozen=True)
class ModelConfig:
    """Static LightGBM model configuration."""

    boosting_type: str
    objective: str
    imputer_strategy: str


def validate_cli(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate CLI arguments."""
    if not Path(args.panel_path).exists():
        parser.error(f"--panel-path not found: {args.panel_path}")
    if not Path(args.feature_schema_csv).exists():
        parser.error(f"--feature-schema-csv not found: {args.feature_schema_csv}")
    if args.reference_splits_csv and not Path(args.reference_splits_csv).exists():
        parser.error(f"--reference-splits-csv not found: {args.reference_splits_csv}")
    if not (0.0 < float(args.top_k_fraction) < 1.0):
        parser.error("--top-k-fraction must be greater than 0 and less than 1.")
    if not math.isclose(float(args.top_k_fraction), TOP_10_FRACTION, rel_tol=0.0, abs_tol=1e-12):
        parser.error("--top-k-fraction is currently fixed at 0.10 for the constructed top-10% label.")
    if int(args.smoke_max_rows) < 0:
        parser.error("--smoke-max-rows cannot be negative.")
    if int(args.min_train_months) < 1:
        parser.error("--min-train-months must be positive.")
    if int(args.validation_window_months) != 1:
        parser.error("--validation-window-months is currently fixed at 1.")
    if int(args.test_window_months) != 1:
        parser.error("--test-window-months is currently fixed at 1.")
    if int(args.lightgbm_threads) < 1:
        parser.error("--lightgbm-threads must be positive.")


def known_output_paths(out_root: Path) -> list[Path]:
    """Return deterministic output paths."""
    return [out_root / name for name in KNOWN_OUTPUTS]


def check_output_collisions(paths: Sequence[Path], overwrite: bool) -> None:
    """Fail if outputs already exist and overwrite is disabled."""
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing output(s). Pass --overwrite to replace:\n  - "
            + "\n  - ".join(str(path) for path in existing)
        )


def remove_outputs(paths: Sequence[Path], overwrite: bool) -> None:
    """Remove deterministic file outputs when overwrite is enabled."""
    if not overwrite:
        return
    for path in paths:
        if path.exists() and path.is_file():
            path.unlink()


def package_versions() -> dict[str, Optional[str]]:
    """Return package versions relevant to this script."""
    versions = helpers.package_versions()
    versions["lightgbm"] = getattr(lightgbm, "__version__", None) if lightgbm is not None else None
    return versions


def imbalance_mode(is_unbalance: bool) -> str:
    """Return the auditable imbalance-mode label."""
    return "automatic_is_unbalance" if is_unbalance else "none"


def label_count_summary(labels) -> dict[str, Any]:
    """Return row counts, prevalence, and binary class counts."""
    n_rows = int(len(labels))
    n_positive = int(labels.astype(int).sum())
    n_negative = n_rows - n_positive
    if n_rows <= 0 or n_positive <= 0 or n_negative <= 0:
        raise ValueError(
            "LightGBM fitting requires non-empty binary labels with both classes; "
            f"n_rows={n_rows}, n_positive={n_positive}, n_negative={n_negative}."
        )
    return {
        "n_rows": n_rows,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_prevalence": float(n_positive) / float(n_rows),
    }


def effective_positive_weight(counts: dict[str, Any], is_unbalance: bool) -> float:
    """Return LightGBM's effective positive-to-negative weight ratio."""
    if not is_unbalance:
        return 1.0
    return float(counts["n_negative"]) / float(counts["n_positive"])


def fit_provenance(
    fold: dict[str, object],
    selection_counts: dict[str, Any],
    final_counts: dict[str, Any],
    *,
    is_unbalance: bool,
    best_iteration: int,
    final_model_relative_path: str,
) -> dict[str, Any]:
    """Return selection-fit and final-refit class/window provenance."""
    return {
        "selection_fit_start_prediction_month": str(fold["train_start_prediction_month"]),
        "selection_fit_end_prediction_month": str(fold["train_end_prediction_month"]),
        "selection_fit_includes_validation": False,
        "selection_fit_n_rows": selection_counts["n_rows"],
        "selection_fit_n_positive": selection_counts["n_positive"],
        "selection_fit_n_negative": selection_counts["n_negative"],
        "selection_fit_positive_prevalence": selection_counts["positive_prevalence"],
        "selection_fit_effective_positive_weight": effective_positive_weight(selection_counts, is_unbalance),
        "final_refit_start_prediction_month": str(fold["train_start_prediction_month"]),
        "final_refit_end_prediction_month": str(fold["validation_prediction_month"]),
        "final_refit_includes_validation": True,
        "final_refit_n_rows": final_counts["n_rows"],
        "final_refit_n_positive": final_counts["n_positive"],
        "final_refit_n_negative": final_counts["n_negative"],
        "final_refit_positive_prevalence": final_counts["positive_prevalence"],
        "final_refit_effective_positive_weight": effective_positive_weight(final_counts, is_unbalance),
        "final_refit_n_estimators": int(best_iteration),
        "final_model_relative_path": final_model_relative_path,
    }


def make_hyperparameter_grid(cfg: LightGBMConfig, model_cfg: ModelConfig) -> list[dict[str, Any]]:
    """Return LightGBM candidate hyperparameter rows."""
    rows: list[dict[str, Any]] = []
    candidate_id = 1
    for num_leaves in cfg.num_leaves_candidates:
        for reg_lambda in cfg.reg_lambda_candidates:
            for is_unbalance in cfg.is_unbalance_candidates:
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "model_name": MODEL_NAME,
                        "model_type": MODEL_TYPE,
                        "boosting_type": model_cfg.boosting_type,
                        "objective": model_cfg.objective,
                        "num_leaves": int(num_leaves),
                        "reg_lambda": float(reg_lambda),
                        "is_unbalance": bool(is_unbalance),
                        "imbalance_mode": imbalance_mode(bool(is_unbalance)),
                        "learning_rate": float(cfg.learning_rate),
                        "min_child_samples": int(cfg.min_child_samples),
                        "subsample": float(cfg.subsample),
                        "subsample_freq": int(cfg.subsample_freq),
                        "colsample_bytree": float(cfg.colsample_bytree),
                        "max_n_estimators": int(cfg.max_n_estimators),
                        "early_stopping_rounds": int(cfg.early_stopping_rounds),
                        "early_stopping_metric": cfg.early_stopping_metric,
                        "imputer_strategy": model_cfg.imputer_strategy,
                        "random_state": int(cfg.random_seed),
                        "n_jobs": int(cfg.lightgbm_threads),
                    }
                )
                candidate_id += 1
    if len(rows) != 12:
        raise ValueError(f"The locked LightGBM design must contain exactly 12 candidates; observed {len(rows)}.")
    return rows


def make_pipeline(candidate: dict[str, Any], model_cfg: ModelConfig, *, n_estimators: Optional[int] = None):
    """Create a fresh sklearn Pipeline for one LightGBM candidate."""
    if LGBMClassifier is None:
        raise RuntimeError("lightgbm is required to run the LightGBM classifier.")
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    imputer = SimpleImputer(strategy=model_cfg.imputer_strategy)
    if hasattr(imputer, "set_output"):
        imputer = imputer.set_output(transform="pandas")

    return Pipeline(
        steps=[
            ("imputer", imputer),
            (
                "lightgbm",
                LGBMClassifier(
                    boosting_type=model_cfg.boosting_type,
                    objective=model_cfg.objective,
                    num_leaves=int(candidate["num_leaves"]),
                    learning_rate=float(candidate["learning_rate"]),
                    n_estimators=int(n_estimators if n_estimators is not None else candidate["max_n_estimators"]),
                    min_child_samples=int(candidate["min_child_samples"]),
                    subsample=float(candidate["subsample"]),
                    subsample_freq=int(candidate["subsample_freq"]),
                    colsample_bytree=float(candidate["colsample_bytree"]),
                    reg_lambda=float(candidate["reg_lambda"]),
                    is_unbalance=bool(candidate["is_unbalance"]),
                    metric="None",
                    random_state=int(candidate["random_state"]),
                    n_jobs=int(candidate["n_jobs"]),
                    verbosity=-1,
                ),
            ),
        ]
    )


def validation_average_precision_metric(y_true, y_pred) -> tuple[str, float, bool]:
    """LightGBM callback metric used exclusively for validation early stopping."""
    from sklearn.metrics import average_precision_score

    return EARLY_STOPPING_METRIC, float(average_precision_score(y_true, y_pred)), True


def fit_selection_pipeline(pipeline, x_train, y_train, x_validation, y_validation, candidate: dict[str, Any]):
    """Fit a candidate on training rows with validation-only AP early stopping."""
    if lightgbm is None:
        raise RuntimeError("lightgbm is required for validation-based early stopping.")
    imputer = pipeline.named_steps["imputer"]
    model = pipeline.named_steps["lightgbm"]
    transformed_train = imputer.fit_transform(x_train)
    transformed_validation = imputer.transform(x_validation)
    model.fit(
        transformed_train,
        y_train,
        eval_set=[(transformed_validation, y_validation)],
        eval_names=["validation"],
        eval_metric=validation_average_precision_metric,
        callbacks=[
            lightgbm.early_stopping(
                stopping_rounds=int(candidate["early_stopping_rounds"]),
                first_metric_only=True,
                verbose=False,
            )
        ],
    )
    best_iteration = int(getattr(model, "best_iteration_", 0) or 0)
    if best_iteration < 1 or best_iteration > int(candidate["max_n_estimators"]):
        raise RuntimeError(
            "LightGBM did not expose a valid best_iteration_ after Average-Precision early stopping; "
            f"observed={best_iteration}, max={candidate['max_n_estimators']}."
        )
    best_scores = getattr(model, "best_score_", {}) or {}
    validation_scores = best_scores.get("validation", {})
    metric_value = validation_scores.get(EARLY_STOPPING_METRIC)
    if metric_value is None or not math.isfinite(float(metric_value)):
        raise RuntimeError(
            "LightGBM best_score_ does not contain validation average_precision; "
            "the installed version may not support the required callback semantics."
        )
    return best_iteration, float(metric_value)


def predict_proba_at_iteration(pipeline, frame, best_iteration: int):
    """Predict through the fitted imputer and exactly the selected tree count."""
    transformed = pipeline.named_steps["imputer"].transform(frame)
    return pipeline.named_steps["lightgbm"].predict_proba(
        transformed,
        num_iteration=int(best_iteration),
    )[:, 1]


def selection_sort_key(row: dict[str, Any], cfg: LightGBMConfig) -> tuple[float, float, int, int, int, float, int]:
    """Sort key for selecting candidates; lower tuple is better."""
    metric_value = float(row[cfg.selection_metric])
    auroc = float(row["auroc"])
    return (
        -metric_value,
        -auroc,
        int(row["num_leaves"]),
        int(row["best_iteration"]),
        int(bool(row["is_unbalance"])),
        float(row["reg_lambda"]),
        int(row["candidate_id"]),
    )


def selected_prefix(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return selected-candidate fields for metric output rows."""
    return {
        "selected_candidate_id": candidate["candidate_id"],
        "num_leaves": candidate["num_leaves"],
        "reg_lambda": candidate["reg_lambda"],
        "is_unbalance": candidate["is_unbalance"],
        "imbalance_mode": candidate["imbalance_mode"],
        "learning_rate": candidate["learning_rate"],
        "min_child_samples": candidate["min_child_samples"],
        "subsample": candidate["subsample"],
        "subsample_freq": candidate["subsample_freq"],
        "colsample_bytree": candidate["colsample_bytree"],
        "max_n_estimators": candidate["max_n_estimators"],
        "early_stopping_rounds": candidate["early_stopping_rounds"],
        "early_stopping_metric": candidate["early_stopping_metric"],
    }


def rank_and_score(
    frame,
    scores,
    cfg: LightGBMConfig,
    *,
    fold: dict[str, object],
    split_role: str,
    metric_scope: str,
    candidate: Optional[dict[str, Any]],
    selected_candidate: Optional[dict[str, Any]],
    include_ranked_rows: bool,
    fit_metadata: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Rank scores within a month and compute ranking metrics."""
    rank_frame = frame[
        [
            "user_id",
            "feature_month",
            "prediction_month",
            "label_high_engagement",
            "target_score",
            "target_score_for_summary",
        ]
    ].copy()
    rank_frame["ranking_score"] = scores
    rank_frame["tie_break_score"] = [
        helpers.stable_hash_score(user_id, month, cfg.random_seed, f"{MODEL_NAME}_{split_role}")
        for user_id, month in zip(rank_frame["user_id"], rank_frame["prediction_month"])
    ]
    rank_frame = rank_frame.sort_values(
        ["ranking_score", "tie_break_score", "user_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    rank_frame["rank_within_month"] = range(1, len(rank_frame) + 1)
    rank_frame["label_high_engagement"] = rank_frame["label_high_engagement"].astype(int)
    rank_frame["cumulative_positive"] = rank_frame["label_high_engagement"].cumsum()

    n_users = int(len(rank_frame))
    n_positive = int(rank_frame["label_high_engagement"].sum())
    n_negative = n_users - n_positive
    if n_users <= 0 or n_positive <= 0 or n_negative <= 0:
        raise ValueError(
            f"Cannot score fold={fold['fold_id']} split={split_role}; "
            f"n_users={n_users}, n_positive={n_positive}, n_negative={n_negative}."
        )

    k_at_10pct = n_positive
    top_k = rank_frame.iloc[:k_at_10pct]
    top_k_positive_count = int(top_k["label_high_engagement"].sum())
    tp_at_10pct = top_k_positive_count
    fp_at_10pct = k_at_10pct - tp_at_10pct
    fn_at_10pct = n_positive - tp_at_10pct
    tn_at_10pct = n_negative - fp_at_10pct
    mcc_at_10pct = helpers.matthews_corrcoef_from_counts(tp_at_10pct, fp_at_10pct, fn_at_10pct, tn_at_10pct)
    positive_rows = rank_frame[rank_frame["label_high_engagement"] == 1]
    average_precision = float((positive_rows["cumulative_positive"] / positive_rows["rank_within_month"]).sum())
    average_precision /= float(n_positive)
    positive_rank_sum = float((n_users - positive_rows["rank_within_month"] + 1).sum())
    auroc = (
        positive_rank_sum - (float(n_positive) * (float(n_positive) + 1.0) / 2.0)
    ) / (float(n_positive) * float(n_negative))

    selected = selected_candidate or candidate or {}
    candidate_values = candidate or selected
    prediction_month = str(rank_frame["prediction_month"].iloc[0])
    feature_month = str(rank_frame["feature_month"].iloc[0])
    metric: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "model_type": MODEL_TYPE,
        "feature_set": cfg.feature_set,
        "fold_id": int(fold["fold_id"]),
        "candidate_id": candidate_values.get("candidate_id", ""),
        "num_leaves": candidate_values.get("num_leaves", ""),
        "reg_lambda": candidate_values.get("reg_lambda", ""),
        "is_unbalance": candidate_values.get("is_unbalance", ""),
        "imbalance_mode": candidate_values.get("imbalance_mode", ""),
        "learning_rate": candidate_values.get("learning_rate", ""),
        "min_child_samples": candidate_values.get("min_child_samples", ""),
        "subsample": candidate_values.get("subsample", ""),
        "subsample_freq": candidate_values.get("subsample_freq", ""),
        "colsample_bytree": candidate_values.get("colsample_bytree", ""),
        "max_n_estimators": candidate_values.get("max_n_estimators", ""),
        "early_stopping_rounds": candidate_values.get("early_stopping_rounds", ""),
        "early_stopping_metric": candidate_values.get("early_stopping_metric", ""),
        "best_iteration": candidate_values.get("best_iteration", ""),
        "early_stopping_validation_metric_name": candidate_values.get(
            "early_stopping_validation_metric_name", ""
        ),
        "early_stopping_validation_metric_value": candidate_values.get(
            "early_stopping_validation_metric_value", ""
        ),
        "prediction_month": prediction_month,
        "test_prediction_month": str(fold["test_prediction_month"]),
        "feature_month": feature_month,
        "split_role": split_role,
        "metric_scope": metric_scope,
        "train_start_prediction_month": str(fold["train_start_prediction_month"]),
        "train_end_prediction_month": str(fold["train_end_prediction_month"]),
        "validation_prediction_month": str(fold["validation_prediction_month"]),
        "n_users": n_users,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_prevalence": float(n_positive) / float(n_users),
        "k_at_10pct": k_at_10pct,
        "top_k_positive_count": top_k_positive_count,
        "tp_at_10pct": tp_at_10pct,
        "fp_at_10pct": fp_at_10pct,
        "fn_at_10pct": fn_at_10pct,
        "tn_at_10pct": tn_at_10pct,
        "precision_at_10pct": float(top_k_positive_count) / float(k_at_10pct),
        "recall_at_10pct": float(top_k_positive_count) / float(n_positive),
        "mcc_at_10pct": mcc_at_10pct,
        "average_precision": average_precision,
        "auprc_ap": average_precision,
        "auroc": auroc,
        "mean_target_score_top_k": float(top_k["target_score_for_summary"].mean()),
        "mean_target_score_all": float(rank_frame["target_score_for_summary"].mean()),
        "top_k_rule": "k_at_10pct_equals_monthly_n_positive",
        "pr_auc_definition": PR_AUC_DEFINITION,
        "auroc_definition": AUROC_DEFINITION,
    }
    if fit_metadata:
        metric.update(fit_metadata)

    ranked_rows: list[dict[str, Any]] = []
    if include_ranked_rows:
        selected_candidate_id = selected.get("candidate_id", candidate_values.get("candidate_id", ""))
        for row in rank_frame.itertuples(index=False):
            ranked_rows.append(
                {
                    "model_name": MODEL_NAME,
                    "model_type": MODEL_TYPE,
                    "feature_set": cfg.feature_set,
                    "fold_id": int(fold["fold_id"]),
                    "prediction_month": row.prediction_month,
                    "test_prediction_month": str(fold["test_prediction_month"]),
                    "feature_month": row.feature_month,
                    "split_role": split_role,
                    "metric_scope": metric_scope,
                    "user_id": row.user_id,
                    "selected_candidate_id": selected_candidate_id,
                    "is_unbalance": selected.get("is_unbalance", candidate_values.get("is_unbalance", "")),
                    "imbalance_mode": selected.get("imbalance_mode", candidate_values.get("imbalance_mode", "")),
                    "best_iteration": selected.get("best_iteration", candidate_values.get("best_iteration", "")),
                    "ranking_score": float(row.ranking_score),
                    "tie_break_score": float(row.tie_break_score),
                    "rank_within_month": int(row.rank_within_month),
                    "label_high_engagement": int(row.label_high_engagement),
                    "target_score": float(row.target_score),
                    **(fit_metadata or {}),
                }
            )
    return metric, ranked_rows


def macro_mean(rows: Sequence[dict[str, Any]], column: str) -> float:
    """Return a float mean for a metric column."""
    return sum(float(row[column]) for row in rows) / float(len(rows))


def macro_stddev(rows: Sequence[dict[str, Any]], column: str) -> Optional[float]:
    """Return sample standard deviation for a metric column."""
    if len(rows) < 2:
        return None
    mean = macro_mean(rows, column)
    variance = sum((float(row[column]) - mean) ** 2 for row in rows) / float(len(rows) - 1)
    return math.sqrt(variance)


def summary_rows(fold_metrics: Sequence[dict[str, Any]], cfg: LightGBMConfig) -> list[dict[str, Any]]:
    """Aggregate test metrics across folds."""
    test_rows = [row for row in fold_metrics if row["split_role"] == "test"]
    if len(test_rows) != len(PRIMARY_TEST_MONTHS_2021):
        raise ValueError(f"Expected {len(PRIMARY_TEST_MONTHS_2021)} test metric rows, observed {len(test_rows)}.")
    tp_pooled = sum(int(row["tp_at_10pct"]) for row in test_rows)
    fp_pooled = sum(int(row["fp_at_10pct"]) for row in test_rows)
    fn_pooled = sum(int(row["fn_at_10pct"]) for row in test_rows)
    tn_pooled = sum(int(row["tn_at_10pct"]) for row in test_rows)
    return [
        {
            "model_name": MODEL_NAME,
            "model_type": MODEL_TYPE,
            "feature_set": cfg.feature_set,
            "metric_scope": "primary_rolling_origin_test",
            "source_split_role": "test",
            "n_folds": len(test_rows),
            "test_prediction_month_start": min(str(row["test_prediction_month"]) for row in test_rows),
            "test_prediction_month_end": max(str(row["test_prediction_month"]) for row in test_rows),
            "n_users_pooled": sum(int(row["n_users"]) for row in test_rows),
            "n_positive_pooled": sum(int(row["n_positive"]) for row in test_rows),
            "k_at_10pct_pooled": sum(int(row["k_at_10pct"]) for row in test_rows),
            "top_k_positive_count_pooled": sum(int(row["top_k_positive_count"]) for row in test_rows),
            "tp_at_10pct_pooled": tp_pooled,
            "fp_at_10pct_pooled": fp_pooled,
            "fn_at_10pct_pooled": fn_pooled,
            "tn_at_10pct_pooled": tn_pooled,
            "mcc_at_10pct_pooled": helpers.matthews_corrcoef_from_counts(tp_pooled, fp_pooled, fn_pooled, tn_pooled),
            "n_users_macro_mean": macro_mean(test_rows, "n_users"),
            "n_users_macro_stddev": macro_stddev(test_rows, "n_users"),
            "n_positive_macro_mean": macro_mean(test_rows, "n_positive"),
            "n_positive_macro_stddev": macro_stddev(test_rows, "n_positive"),
            "positive_prevalence_macro_mean": macro_mean(test_rows, "positive_prevalence"),
            "positive_prevalence_macro_stddev": macro_stddev(test_rows, "positive_prevalence"),
            "precision_at_10pct_macro_mean": macro_mean(test_rows, "precision_at_10pct"),
            "precision_at_10pct_macro_stddev": macro_stddev(test_rows, "precision_at_10pct"),
            "recall_at_10pct_macro_mean": macro_mean(test_rows, "recall_at_10pct"),
            "recall_at_10pct_macro_stddev": macro_stddev(test_rows, "recall_at_10pct"),
            "mcc_at_10pct_macro_mean": macro_mean(test_rows, "mcc_at_10pct"),
            "mcc_at_10pct_macro_stddev": macro_stddev(test_rows, "mcc_at_10pct"),
            "average_precision_macro_mean": macro_mean(test_rows, "average_precision"),
            "average_precision_macro_stddev": macro_stddev(test_rows, "average_precision"),
            "auprc_ap_macro_mean": macro_mean(test_rows, "auprc_ap"),
            "auprc_ap_macro_stddev": macro_stddev(test_rows, "auprc_ap"),
            "auroc_macro_mean": macro_mean(test_rows, "auroc"),
            "auroc_macro_stddev": macro_stddev(test_rows, "auroc"),
            "mean_target_score_top_k_macro_mean": macro_mean(test_rows, "mean_target_score_top_k"),
            "mean_target_score_top_k_macro_stddev": macro_stddev(test_rows, "mean_target_score_top_k"),
            "mean_target_score_all_macro_mean": macro_mean(test_rows, "mean_target_score_all"),
            "mean_target_score_all_macro_stddev": macro_stddev(test_rows, "mean_target_score_all"),
            "aggregation_note": "macro month-wise means over rolling-origin test folds; pooled totals are counts only",
            "pr_auc_definition": PR_AUC_DEFINITION,
            "auroc_definition": AUROC_DEFINITION,
        }
    ]


def validate_output_rows(
    validation_metrics: Sequence[dict[str, Any]],
    selected_rows: Sequence[dict[str, Any]],
    fold_metrics: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    hyperparameter_grid: Sequence[dict[str, Any]],
) -> None:
    """Validate output rows before writing files."""
    if len(hyperparameter_grid) != 12:
        raise ValueError(f"LightGBM hyperparameter grid must contain exactly 12 candidates, observed {len(hyperparameter_grid)}.")
    for candidate in hyperparameter_grid:
        if "class_weight" in candidate or "scale_pos_weight" in candidate or "n_estimators" in candidate:
            raise ValueError("LightGBM grid contains an obsolete or prohibited weighting/iteration field.")
        if float(candidate["subsample"]) != 1.0 or int(candidate["subsample_freq"]) != 0:
            raise ValueError("LightGBM bagging must be disabled with subsample=1.0 and subsample_freq=0.")
        if float(candidate["colsample_bytree"]) != 1.0:
            raise ValueError("LightGBM must make all selected features available to every tree.")
    expected_validation = len(PRIMARY_TEST_MONTHS_2021) * len(hyperparameter_grid)
    if len(validation_metrics) != expected_validation:
        raise ValueError(
            f"Unexpected validation metric row count: observed={len(validation_metrics)}, expected={expected_validation}."
        )
    if len(selected_rows) != len(PRIMARY_TEST_MONTHS_2021):
        raise ValueError(
            f"Unexpected selected hyperparameter row count: observed={len(selected_rows)}, "
            f"expected={len(PRIMARY_TEST_MONTHS_2021)}."
        )
    if len(fold_metrics) != len(PRIMARY_TEST_MONTHS_2021):
        raise ValueError(
            f"Unexpected test metric row count: observed={len(fold_metrics)}, "
            f"expected={len(PRIMARY_TEST_MONTHS_2021)}."
        )
    observed_test_months = tuple(sorted(str(row["test_prediction_month"]) for row in fold_metrics))
    if observed_test_months != PRIMARY_TEST_MONTHS_2021:
        raise ValueError(f"Unexpected test months: observed={observed_test_months}; expected={PRIMARY_TEST_MONTHS_2021}.")
    if any(row["split_role"] != "test" or row["metric_scope"] != "primary_rolling_origin_test" for row in fold_metrics):
        raise ValueError("lightgbm_metrics_by_fold.csv must contain only primary rolling-origin test rows.")
    if len(summaries) != 1 or summaries[0]["source_split_role"] != "test":
        raise ValueError("lightgbm_metrics_summary.csv must aggregate only test rows.")
    for row in validation_metrics:
        if str(row["early_stopping_validation_metric_name"]) != EARLY_STOPPING_METRIC:
            raise ValueError("Every candidate must early-stop on validation average_precision.")
        best_iteration = int(row["best_iteration"])
        if best_iteration < 1 or best_iteration > int(row["max_n_estimators"]):
            raise ValueError("Candidate best_iteration is outside [1, max_n_estimators].")
        if row["selection_fit_includes_validation"] is not False:
            raise ValueError("Candidate selection fits must exclude validation rows.")
        expected_weight = (
            float(row["selection_fit_n_negative"]) / float(row["selection_fit_n_positive"])
            if bool(row["is_unbalance"])
            else 1.0
        )
        if not math.isclose(
            float(row["selection_fit_effective_positive_weight"]), expected_weight, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("Candidate selection effective positive weight is inconsistent with class counts.")
    selected_by_fold = {int(row["fold_id"]): row for row in selected_rows}
    for row in fold_metrics:
        selected = selected_by_fold[int(row["fold_id"])]
        if row.get("final_refit_includes_validation") is not True:
            raise ValueError("Final LightGBM models must be refit with validation rows included.")
        if str(row["final_refit_end_prediction_month"]) != str(row["validation_prediction_month"]):
            raise ValueError("Final LightGBM refit must end at the validation month.")
        if str(row["final_refit_end_prediction_month"]) >= str(row["test_prediction_month"]):
            raise ValueError("Final LightGBM refit must not reach the held-out test month.")
        if int(row["best_iteration"]) != int(selected["best_iteration"]):
            raise ValueError("Test metric best_iteration does not match the selected candidate.")
        expected_weight = (
            float(row["final_refit_n_negative"]) / float(row["final_refit_n_positive"])
            if bool(row["is_unbalance"])
            else 1.0
        )
        if not math.isclose(
            float(row["final_refit_effective_positive_weight"]), expected_weight, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("Final-refit effective positive weight is inconsistent with combined class counts.")
    for collection_name, rows in {"validation": validation_metrics, "test": fold_metrics}.items():
        for row in rows:
            for column in ("precision_at_10pct", "recall_at_10pct", "average_precision", "auprc_ap", "auroc"):
                value = float(row[column])
                if value < -1e-12 or value > 1.0 + 1e-12:
                    raise ValueError(f"{collection_name} metric {column} outside [0, 1]: {value}")
            mcc = float(row["mcc_at_10pct"])
            if not math.isfinite(mcc) or mcc < -1.0 - 1e-12 or mcc > 1.0 + 1e-12:
                raise ValueError(f"{collection_name} metric mcc_at_10pct outside [-1, 1] or non-finite: {mcc}")
            tp = int(row["tp_at_10pct"])
            fp = int(row["fp_at_10pct"])
            fn = int(row["fn_at_10pct"])
            tn = int(row["tn_at_10pct"])
            top_k_positive = int(row["top_k_positive_count"])
            k_at_10pct = int(row["k_at_10pct"])
            n_positive = int(row["n_positive"])
            n_negative = int(row["n_negative"])
            if tp != top_k_positive or fp != k_at_10pct - tp or fn != n_positive - tp or tn != n_negative - fp:
                raise ValueError(f"{collection_name} top-k confusion counts are inconsistent.")
            if any(value < 0 for value in (tp, fp, fn, tn)):
                raise ValueError(f"{collection_name} top-k confusion counts must be non-negative.")
            denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            if denominator <= 0:
                raise ValueError(f"{collection_name} MCC denominator is zero.")


def feature_importance_rows(
    pipeline,
    selected_features: Sequence[str],
    cfg: LightGBMConfig,
    fold: dict[str, object],
    selected_candidate: dict[str, Any],
    fit_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return LightGBM feature-importance diagnostics for one selected fold model."""
    model = pipeline.named_steps["lightgbm"]
    split_importance = model.booster_.feature_importance(importance_type="split")
    gain_importance = model.booster_.feature_importance(importance_type="gain")
    rows = []
    for feature_name, split_value, gain_value in zip(selected_features, split_importance, gain_importance):
        rows.append(
            {
                "model_name": MODEL_NAME,
                "model_type": MODEL_TYPE,
                "feature_set": cfg.feature_set,
                "fold_id": int(fold["fold_id"]),
                "test_prediction_month": str(fold["test_prediction_month"]),
                "selected_candidate_id": int(selected_candidate["candidate_id"]),
                "is_unbalance": selected_candidate["is_unbalance"],
                "imbalance_mode": selected_candidate["imbalance_mode"],
                "best_iteration": selected_candidate["best_iteration"],
                "final_refit_start_prediction_month": fit_metadata["final_refit_start_prediction_month"],
                "final_refit_end_prediction_month": fit_metadata["final_refit_end_prediction_month"],
                "final_refit_includes_validation": fit_metadata["final_refit_includes_validation"],
                "final_refit_n_positive": fit_metadata["final_refit_n_positive"],
                "final_refit_n_negative": fit_metadata["final_refit_n_negative"],
                "final_refit_n_rows": fit_metadata["final_refit_n_rows"],
                "final_refit_positive_prevalence": fit_metadata["final_refit_positive_prevalence"],
                "final_refit_effective_positive_weight": fit_metadata[
                    "final_refit_effective_positive_weight"
                ],
                "final_refit_n_estimators": fit_metadata["final_refit_n_estimators"],
                "final_model_relative_path": fit_metadata["final_model_relative_path"],
                "feature_name": feature_name,
                "importance_split": int(split_value),
                "importance_gain": float(gain_value),
                "importance_note": FEATURE_IMPORTANCE_NOTE,
            }
        )
    return rows


def write_ranked_predictions(con: duckdb.DuckDBPyConnection, out_root: Path, ranked_rows: Sequence[dict[str, Any]]) -> None:
    """Write optional ranked predictions to Parquet."""
    import pandas as pd

    ranked_path = out_root / "lightgbm_ranked_predictions.parquet"
    ranked_df = pd.DataFrame(ranked_rows)
    con.register("lightgbm_ranked_predictions_output", ranked_df)
    con.execute(
        f"""
        COPY (
          SELECT *
          FROM lightgbm_ranked_predictions_output
          ORDER BY fold_id, prediction_month, rank_within_month
        )
        TO {helpers.quote_sql_string(ranked_path)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    con.unregister("lightgbm_ranked_predictions_output")


def feature_set_metadata(feature_set_def: dict[str, Any], selected_features: Sequence[str]) -> dict[str, Any]:
    """Return feature-set metadata required for LightGBM run provenance."""
    dropped = list(feature_set_def.get("curated_excluded_feature_names", []))
    return {
        "resolved_feature_set": feature_set_def.get("feature_set"),
        "selected_features": list(selected_features),
        "selected_feature_count": len(selected_features),
        "dropped_features": dropped,
        "dropped_feature_count": len(dropped),
        "schema_equivalent_to_feature_set": feature_set_def.get("equivalent_to_feature_set_under_current_schema", ""),
        "schema_equivalence_note": feature_set_def.get("equivalence_note", ""),
        "interpretation_caveat": FEATURE_IMPORTANCE_NOTE,
    }


def metadata_payload(
    cfg: LightGBMConfig,
    model_cfg: ModelConfig,
    duck: DuckDBConfig,
    selected_features: Sequence[str],
    feature_set_def: dict[str, Any],
    hyperparameter_grid: Sequence[dict[str, Any]],
    outputs: dict[str, str],
    run_started_at: str,
    status: str,
    panel_row_count: Optional[int] = None,
) -> dict[str, Any]:
    """Build run metadata."""
    feature_meta = feature_set_metadata(feature_set_def, selected_features)
    return {
        "script_name": "Modelling/1_4_run_lightgbm_classifier.py",
        "run_started_at_utc": run_started_at,
        "run_finished_at_utc": helpers.utc_now_iso(),
        "status": status,
        "host": {"hostname": socket.gethostname(), "platform": platform.platform()},
        "git": helpers.git_metadata(),
        "package_versions": package_versions(),
        "panel_path": cfg.panel_path,
        "feature_schema_csv": cfg.feature_schema_csv,
        "output_root": cfg.out_root,
        "target_column": "label_high_engagement",
        "target_score_col_for_descriptive_summaries": cfg.target_score_col,
        "model_name": MODEL_NAME,
        "model_type": MODEL_TYPE,
        "model_fitting_performed": True,
        "model_config": asdict(model_cfg),
        "ranking_score": "LGBMClassifier.predict_proba(X)[:, 1]",
        "feature_set": cfg.feature_set,
        "feature_set_definition": feature_set_def,
        "feature_set_metadata": feature_meta,
        "selected_feature_count": len(selected_features),
        "selected_features": list(selected_features),
        "dropped_features": feature_meta["dropped_features"],
        "schema_equivalent_to_feature_set": feature_meta["schema_equivalent_to_feature_set"],
        "schema_equivalence_note": feature_meta["schema_equivalence_note"],
        "hyperparameter_grid": list(hyperparameter_grid),
        "selection_metric": cfg.selection_metric,
        "hyperparameter_selection_protocol": (
            "Each structural candidate is fitted on training rows only; the validation month supplies the "
            "sole early-stopping eval_set and the canonical Average-Precision selection metrics. After "
            "selection, a fresh pipeline is refitted on training plus validation rows for exactly the "
            "selected best_iteration and evaluated once on the test month."
        ),
        "class_balance_handling": {
            "strategy": "is_unbalance_grid",
            "candidate_values": list(cfg.is_unbalance_candidates),
            "smote_or_synthetic_resampling": False,
            "scale_pos_weight": "not_used",
            "class_weight": "not_used",
            "effective_positive_weight": (
                "1.0 when is_unbalance=false; n_negative/n_positive for the relevant fit labels when true"
            ),
        },
        "early_stopping": {
            "enabled_for_candidate_selection": True,
            "eval_set": "validation_month_only",
            "metric": cfg.early_stopping_metric,
            "max_n_estimators": cfg.max_n_estimators,
            "stopping_rounds": cfg.early_stopping_rounds,
            "default_binary_logloss_disabled": True,
        },
        "final_refit_protocol": {
            "fresh_preprocessing_and_model_pipeline": True,
            "refit_months": "training_prediction_months_plus_validation_prediction_month",
            "final_refit_includes_validation": True,
            "n_estimators_source": "selected_candidate_best_iteration",
            "early_stopping_during_final_refit": False,
            "feature_importance_source": "final_train_plus_validation_refit",
            "ranked_prediction_source": "final_train_plus_validation_refit",
        },
        "calibration": {
            "performed": False,
            "note": "Predicted probabilities are used for ranking only; calibration is outside this ranking-focused scope.",
        },
        "feature_importance": {
            "written": "feature_importance.csv" in outputs,
            "importance_types": ["split", "gain"],
            "interpretation_caveat": FEATURE_IMPORTANCE_NOTE,
            "shap_performed": False,
        },
        "ranking_objective_model": False,
        "target_definition_changed": False,
        "eligibility_definition_changed": False,
        "split_definition_changed": False,
        "metric_definition_changed": False,
        "evaluation_design": {
            "prediction_months_expected": list(EXPECTED_PREDICTION_MONTHS_2021),
            "primary_test_months": list(PRIMARY_TEST_MONTHS_2021),
            "primary_metric_filter": 'split_role = "test"',
            "top_k_fraction": cfg.top_k_fraction,
        },
        "metric_definitions": {
            "precision_at_10pct": "Top-k precision where k equals the monthly positive label count.",
            "recall_at_10pct": "Top-k recall where k equals the monthly positive label count.",
            "mcc_at_10pct": "Matthews correlation coefficient at the same top-k cutoff.",
            "average_precision": "Ranked-list Average Precision within prediction month.",
            "auprc_ap": "Same value as average_precision.",
            "auroc": "Deterministic tie-broken rank-sum AUROC.",
        },
        "tie_handling": TIE_HANDLING,
        "panel_row_count": panel_row_count,
        "smoke": cfg.smoke,
        "smoke_max_rows": cfg.smoke_max_rows,
        "duckdb": asdict(duck),
        "outputs": outputs,
    }


def fitted_model_summary_payload(
    cfg: LightGBMConfig,
    selected_features: Sequence[str],
    selected_hyperparameters: Sequence[dict[str, Any]],
    feature_set_def: dict[str, Any],
) -> dict[str, Any]:
    """Return a compact fitted-model summary diagnostic."""
    return {
        "model_name": MODEL_NAME,
        "model_type": MODEL_TYPE,
        "feature_set": cfg.feature_set,
        "feature_set_metadata": feature_set_metadata(feature_set_def, selected_features),
        "n_folds": len(selected_hyperparameters),
        "selected_hyperparameters_by_fold": list(selected_hyperparameters),
        "ranking_score": "LGBMClassifier.predict_proba(X)[:, 1]",
        "feature_importance_caveat": FEATURE_IMPORTANCE_NOTE,
        "shap_performed": False,
    }


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run the Study 2 LightGBM binary classifier with the locked 12-candidate design.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--panel-path", required=True, help="Path to modelling_panel.parquet.")
    parser.add_argument("--feature-schema-csv", required=True, help="Path to feature_schema.csv.")
    parser.add_argument("--out-root", required=True, help="Output root for LightGBM artefacts.")
    parser.add_argument(
        "--feature-set",
        required=True,
        choices=SUPPORTED_FEATURE_SETS,
        help="Pre-specified feature set to evaluate.",
    )
    parser.add_argument("--reference-splits-csv", default=None, help="Optional rolling_origin_splits.csv reference.")
    parser.add_argument(
        "--selection-metric",
        default="average_precision",
        choices=["average_precision"],
        help="Locked validation metric used for structural-candidate selection.",
    )
    parser.add_argument("--random-seed", type=int, default=2021, help="Random seed for model fitting and tie-breaking.")
    parser.add_argument("--target-score-col", default="target_score", help="Target-score column for descriptive summaries.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite deterministic LightGBM outputs.")
    parser.add_argument(
        "--write-ranked-predictions",
        action="store_true",
        help="Write full selected test-month lightgbm_ranked_predictions.parquet.",
    )
    parser.add_argument(
        "--top-k-fraction",
        type=float,
        default=TOP_10_FRACTION,
        help="Reserved top-k fraction; currently fixed at 0.10 to match the constructed label.",
    )
    parser.add_argument("--smoke", action="store_true", help="Enable smoke-mode metadata and optional row limiting.")
    parser.add_argument(
        "--smoke-max-rows",
        type=int,
        default=0,
        help="If >0 and --smoke, keep up to this many rows per prediction month using label-stratified sampling.",
    )
    parser.add_argument("--min-train-months", type=int, default=4, help="Minimum rolling-origin training months.")
    parser.add_argument("--validation-window-months", type=int, default=1, help="Rolling-origin validation window.")
    parser.add_argument("--test-window-months", type=int, default=1, help="Rolling-origin test window.")
    parser.add_argument("--lightgbm-threads", type=int, default=1, help="Threads used by each LightGBM fit.")
    parser.add_argument(
        "--duckdb-memory-limit",
        default=os.environ.get("DUCKDB_MEMORY_LIMIT", "16GB"),
        help="DuckDB memory limit.",
    )
    parser.add_argument(
        "--duckdb-threads",
        type=int,
        default=int(os.environ.get("DUCKDB_THREADS", "4")),
        help="DuckDB threads.",
    )
    parser.add_argument(
        "--duckdb-temp-dir",
        default=os.environ.get("DUCKDB_TEMP_DIR", "duckdb_tmp"),
        help="DuckDB temp_directory.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run LightGBM classifier evaluations."""
    parser = build_argparser()
    args = parser.parse_args(argv)
    validate_cli(args, parser)
    helpers.setup_logging(args.log_level)
    if duckdb is None:
        raise RuntimeError("duckdb is required to run LightGBM classifier metrics.")
    if LGBMClassifier is None:
        raise RuntimeError("lightgbm is required to run the LightGBM classifier.")

    cfg = LightGBMConfig(
        panel_path=str(args.panel_path),
        feature_schema_csv=str(args.feature_schema_csv),
        out_root=str(args.out_root),
        feature_set=str(args.feature_set),
        reference_splits_csv=str(args.reference_splits_csv) if args.reference_splits_csv else None,
        num_leaves_candidates=NUM_LEAVES_CANDIDATES,
        reg_lambda_candidates=REG_LAMBDA_CANDIDATES,
        is_unbalance_candidates=IS_UNBALANCE_CANDIDATES,
        learning_rate=FIXED_LEARNING_RATE,
        min_child_samples=FIXED_MIN_CHILD_SAMPLES,
        subsample=FIXED_SUBSAMPLE,
        subsample_freq=FIXED_SUBSAMPLE_FREQ,
        colsample_bytree=FIXED_COLSAMPLE_BYTREE,
        max_n_estimators=MAX_N_ESTIMATORS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        early_stopping_metric=EARLY_STOPPING_METRIC,
        selection_metric=str(args.selection_metric),
        random_seed=int(args.random_seed),
        target_score_col=str(args.target_score_col),
        overwrite=bool(args.overwrite),
        write_ranked_predictions=bool(args.write_ranked_predictions),
        top_k_fraction=float(args.top_k_fraction),
        smoke=bool(args.smoke),
        smoke_max_rows=int(args.smoke_max_rows),
        min_train_months=int(args.min_train_months),
        validation_window_months=int(args.validation_window_months),
        test_window_months=int(args.test_window_months),
        lightgbm_threads=int(args.lightgbm_threads),
    )
    model_cfg = ModelConfig(boosting_type="gbdt", objective="binary", imputer_strategy="median")
    duck = DuckDBConfig(
        memory_limit=str(args.duckdb_memory_limit),
        threads=int(args.duckdb_threads),
        temp_dir=str(args.duckdb_temp_dir),
    )
    run_started_at = helpers.utc_now_iso()

    out_root = Path(cfg.out_root)
    output_paths = known_output_paths(out_root)
    check_output_collisions(output_paths, cfg.overwrite)
    remove_outputs(output_paths, cfg.overwrite)
    helpers.ensure_dir(out_root)

    schema_rows = helpers.read_feature_schema(cfg.feature_schema_csv)
    selected_features, selected_feature_rows = helpers.resolve_feature_set(cfg.feature_set, schema_rows)
    feature_set_def = helpers.feature_set_definition(cfg.feature_set)
    hyperparameter_grid = make_hyperparameter_grid(cfg, model_cfg)

    outputs: dict[str, str] = {
        "resolved_config.json": str(out_root / "resolved_config.json"),
        "run_metadata.json": str(out_root / "run_metadata.json"),
        "selected_features.csv": str(out_root / "selected_features.csv"),
        "lightgbm_hyperparameter_grid.csv": str(out_root / "lightgbm_hyperparameter_grid.csv"),
    }
    helpers.safe_json_dump(
        {
            "config": asdict(cfg),
            "model": asdict(model_cfg),
            "duckdb": asdict(duck),
            "feature_set_definition": feature_set_def,
            "feature_set_metadata": feature_set_metadata(feature_set_def, selected_features),
            "feature_selection": {
                "feature_set": cfg.feature_set,
                "selected_feature_count": len(selected_features),
                "selected_features": list(selected_features),
            },
        },
        out_root / "resolved_config.json",
    )
    helpers.write_csv_rows(out_root / "selected_features.csv", selected_feature_rows, SELECTED_FEATURE_FIELDS)
    helpers.write_csv_rows(out_root / "lightgbm_hyperparameter_grid.csv", hyperparameter_grid, HYPERPARAMETER_GRID_FIELDS)

    logger.info("Reading panel: %s", cfg.panel_path)
    logger.info("Feature set: %s (%s features)", cfg.feature_set, len(selected_features))
    logger.info("Output root: %s", out_root)

    con = duckdb.connect()
    try:
        helpers.configure_duckdb(con, duck)
        helpers.create_panel_source_view(con, cfg)
        columns = helpers.relation_columns(con, "panel_source")
        helpers.validate_panel_preconditions(con, cfg, columns, selected_features)

        feature_map = helpers.prediction_month_feature_map(con)
        split_rows, primary_test_rows = helpers.build_rolling_origin_splits(feature_map, cfg)
        if cfg.reference_splits_csv:
            helpers.compare_reference_splits(split_rows, cfg.reference_splits_csv)
        helpers.write_csv_rows(out_root / "rolling_origin_splits.csv", split_rows, helpers.split_fieldnames())
        outputs["rolling_origin_splits.csv"] = str(out_root / "rolling_origin_splits.csv")

        validation_metrics: list[dict[str, Any]] = []
        selected_hyperparameters: list[dict[str, Any]] = []
        fold_metrics: list[dict[str, Any]] = []
        feature_importances: list[dict[str, Any]] = []
        ranked_predictions: list[dict[str, Any]] = []

        all_months = list(EXPECTED_PREDICTION_MONTHS_2021)
        for fold in primary_test_rows:
            fold_id = int(fold["fold_id"])
            train_end = str(fold["train_end_prediction_month"])
            train_months = all_months[: all_months.index(train_end) + 1]
            validation_month = str(fold["validation_prediction_month"])
            test_month = str(fold["test_prediction_month"])
            logger.info(
                "Fold %s: train=%s--%s validation=%s test=%s",
                fold_id,
                train_months[0],
                train_months[-1],
                validation_month,
                test_month,
            )

            train_df = helpers.fetch_partition_frame(con, cfg, selected_features, train_months)
            validation_df = helpers.fetch_partition_frame(con, cfg, selected_features, [validation_month])
            x_train = train_df[list(selected_features)]
            y_train = train_df["label_high_engagement"].astype(int)
            x_validation = validation_df[list(selected_features)]
            y_validation = validation_df["label_high_engagement"].astype(int)
            selection_counts = label_count_summary(y_train)
            label_count_summary(y_validation)
            selection_fit_metadata = {
                "selection_fit_start_prediction_month": str(fold["train_start_prediction_month"]),
                "selection_fit_end_prediction_month": str(fold["train_end_prediction_month"]),
                "selection_fit_includes_validation": False,
                "selection_fit_n_rows": selection_counts["n_rows"],
                "selection_fit_n_positive": selection_counts["n_positive"],
                "selection_fit_n_negative": selection_counts["n_negative"],
                "selection_fit_positive_prevalence": selection_counts["positive_prevalence"],
            }

            fold_validation_metrics: list[dict[str, Any]] = []
            for candidate in hyperparameter_grid:
                pipeline = make_pipeline(candidate, model_cfg)
                best_iteration, early_stopping_value = fit_selection_pipeline(
                    pipeline,
                    x_train,
                    y_train,
                    x_validation,
                    y_validation,
                    candidate,
                )
                evaluated_candidate = {
                    **candidate,
                    "best_iteration": best_iteration,
                    "early_stopping_validation_metric_name": cfg.early_stopping_metric,
                    "early_stopping_validation_metric_value": early_stopping_value,
                }
                candidate_fit_metadata = {
                    **selection_fit_metadata,
                    "selection_fit_effective_positive_weight": effective_positive_weight(
                        selection_counts, bool(candidate["is_unbalance"])
                    ),
                }
                validation_scores = predict_proba_at_iteration(pipeline, x_validation, best_iteration)
                metric, _ = rank_and_score(
                    validation_df,
                    validation_scores,
                    cfg,
                    fold=fold,
                    split_role="validation",
                    metric_scope="validation_model_selection",
                    candidate=evaluated_candidate,
                    selected_candidate=None,
                    include_ranked_rows=False,
                    fit_metadata=candidate_fit_metadata,
                )
                metric["selection_metric"] = cfg.selection_metric
                metric["selection_tie_break_order"] = (
                    f"{cfg.selection_metric} desc, auroc desc, num_leaves asc, best_iteration asc, "
                    "is_unbalance false before true, reg_lambda asc, candidate_id asc"
                )
                fold_validation_metrics.append(metric)

            ranked_candidates = sorted(fold_validation_metrics, key=lambda row: selection_sort_key(row, cfg))
            selected_metric = ranked_candidates[0]
            for rank, metric in enumerate(ranked_candidates, start=1):
                metric["selection_rank_within_fold"] = rank
                metric["selected_by_validation"] = rank == 1
            validation_metrics.extend(sorted(ranked_candidates, key=lambda row: int(row["candidate_id"])))

            selected_candidate_id = int(selected_metric["candidate_id"])
            selected_structure = next(
                row for row in hyperparameter_grid if int(row["candidate_id"]) == selected_candidate_id
            )
            selected_candidate = {
                **selected_structure,
                "best_iteration": int(selected_metric["best_iteration"]),
                "early_stopping_validation_metric_name": selected_metric[
                    "early_stopping_validation_metric_name"
                ],
                "early_stopping_validation_metric_value": selected_metric[
                    "early_stopping_validation_metric_value"
                ],
            }
            final_model_relative_path = f"final_models/fold_{fold_id}.txt"
            import pandas as pd

            train_plus_validation_df = pd.concat([train_df, validation_df], ignore_index=True)
            x_train_plus_validation = train_plus_validation_df[list(selected_features)]
            y_train_plus_validation = train_plus_validation_df["label_high_engagement"].astype(int)
            final_counts = label_count_summary(y_train_plus_validation)
            fit_metadata = fit_provenance(
                fold,
                selection_counts,
                final_counts,
                is_unbalance=bool(selected_candidate["is_unbalance"]),
                best_iteration=int(selected_candidate["best_iteration"]),
                final_model_relative_path=final_model_relative_path,
            )
            selected_pipeline = make_pipeline(
                selected_candidate,
                model_cfg,
                n_estimators=int(selected_candidate["best_iteration"]),
            )
            selected_pipeline.fit(x_train_plus_validation, y_train_plus_validation)
            observed_final_iterations = int(
                selected_pipeline.named_steps["lightgbm"].booster_.current_iteration()
            )
            if observed_final_iterations != int(selected_candidate["best_iteration"]):
                raise RuntimeError(
                    "Final LightGBM refit tree count does not match the selected best_iteration: "
                    f"observed={observed_final_iterations}, selected={selected_candidate['best_iteration']}."
                )
            final_model_path = out_root / final_model_relative_path
            final_model_path.parent.mkdir(parents=True, exist_ok=True)
            selected_pipeline.named_steps["lightgbm"].booster_.save_model(
                str(final_model_path),
                num_iteration=int(selected_candidate["best_iteration"]),
            )
            outputs[final_model_relative_path] = str(final_model_path)
            selected_hyperparameters.append(
                {
                    "model_name": MODEL_NAME,
                    "model_type": MODEL_TYPE,
                    "feature_set": cfg.feature_set,
                    "fold_id": fold_id,
                    **selected_prefix(selected_candidate),
                    "best_iteration": selected_candidate["best_iteration"],
                    "early_stopping_validation_metric_name": selected_candidate[
                        "early_stopping_validation_metric_name"
                    ],
                    "early_stopping_validation_metric_value": selected_candidate[
                        "early_stopping_validation_metric_value"
                    ],
                    **fit_metadata,
                    "selection_metric": cfg.selection_metric,
                    "selected_validation_metric_value": selected_metric[cfg.selection_metric],
                    "validation_average_precision": selected_metric["average_precision"],
                    "validation_auroc": selected_metric["auroc"],
                    "validation_precision_at_10pct": selected_metric["precision_at_10pct"],
                    "validation_recall_at_10pct": selected_metric["recall_at_10pct"],
                    "validation_mcc_at_10pct": selected_metric["mcc_at_10pct"],
                    "selection_rank_within_fold": 1,
                    "selection_tie_break_order": selected_metric["selection_tie_break_order"],
                    "train_start_prediction_month": str(fold["train_start_prediction_month"]),
                    "train_end_prediction_month": str(fold["train_end_prediction_month"]),
                    "validation_prediction_month": str(fold["validation_prediction_month"]),
                    "test_prediction_month": str(fold["test_prediction_month"]),
                }
            )
            feature_importances.extend(
                feature_importance_rows(
                    selected_pipeline,
                    selected_features,
                    cfg,
                    fold,
                    selected_candidate,
                    fit_metadata,
                )
            )

            test_df = helpers.fetch_partition_frame(con, cfg, selected_features, [test_month])
            x_test = test_df[list(selected_features)]
            test_scores = selected_pipeline.predict_proba(x_test)[:, 1]
            test_metric, test_ranked_rows = rank_and_score(
                test_df,
                test_scores,
                cfg,
                fold=fold,
                split_role="test",
                metric_scope="primary_rolling_origin_test",
                candidate=None,
                selected_candidate=selected_candidate,
                include_ranked_rows=cfg.write_ranked_predictions,
                fit_metadata=fit_metadata,
            )
            test_metric.update(
                {
                    **selected_prefix(selected_candidate),
                    "best_iteration": selected_candidate["best_iteration"],
                    "early_stopping_validation_metric_name": selected_candidate[
                        "early_stopping_validation_metric_name"
                    ],
                    "early_stopping_validation_metric_value": selected_candidate[
                        "early_stopping_validation_metric_value"
                    ],
                }
            )
            fold_metrics.append(test_metric)
            ranked_predictions.extend(test_ranked_rows)

        summaries = summary_rows(fold_metrics, cfg)
        validate_output_rows(validation_metrics, selected_hyperparameters, fold_metrics, summaries, hyperparameter_grid)

        helpers.write_csv_rows(
            out_root / "lightgbm_validation_metrics_by_candidate.csv",
            validation_metrics,
            VALIDATION_METRIC_FIELDS,
        )
        outputs["lightgbm_validation_metrics_by_candidate.csv"] = str(
            out_root / "lightgbm_validation_metrics_by_candidate.csv"
        )
        helpers.write_csv_rows(
            out_root / "lightgbm_selected_hyperparameters_by_fold.csv",
            selected_hyperparameters,
            SELECTED_HYPERPARAMETER_FIELDS,
        )
        outputs["lightgbm_selected_hyperparameters_by_fold.csv"] = str(
            out_root / "lightgbm_selected_hyperparameters_by_fold.csv"
        )
        helpers.write_csv_rows(out_root / "lightgbm_metrics_by_fold.csv", fold_metrics, FOLD_METRIC_FIELDS)
        outputs["lightgbm_metrics_by_fold.csv"] = str(out_root / "lightgbm_metrics_by_fold.csv")
        helpers.write_csv_rows(out_root / "lightgbm_metrics_summary.csv", summaries, SUMMARY_FIELDS)
        outputs["lightgbm_metrics_summary.csv"] = str(out_root / "lightgbm_metrics_summary.csv")
        helpers.write_csv_rows(out_root / "feature_importance.csv", feature_importances, FEATURE_IMPORTANCE_FIELDS)
        outputs["feature_importance.csv"] = str(out_root / "feature_importance.csv")

        helpers.safe_json_dump(
            fitted_model_summary_payload(cfg, selected_features, selected_hyperparameters, feature_set_def),
            out_root / "fitted_model_summary.json",
        )
        outputs["fitted_model_summary.json"] = str(out_root / "fitted_model_summary.json")

        if cfg.write_ranked_predictions:
            write_ranked_predictions(con, out_root, ranked_predictions)
            outputs["lightgbm_ranked_predictions.parquet"] = str(out_root / "lightgbm_ranked_predictions.parquet")

        panel_row_count = int(con.execute("SELECT COUNT(*) FROM panel_source").fetchone()[0])
        helpers.safe_json_dump(
            metadata_payload(
                cfg=cfg,
                model_cfg=model_cfg,
                duck=duck,
                selected_features=selected_features,
                feature_set_def=feature_set_def,
                hyperparameter_grid=hyperparameter_grid,
                outputs=outputs,
                run_started_at=run_started_at,
                status="finished",
                panel_row_count=panel_row_count,
            ),
            out_root / "run_metadata.json",
        )
    finally:
        con.close()

    logger.info("Done. Wrote LightGBM classifier outputs under %s", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
