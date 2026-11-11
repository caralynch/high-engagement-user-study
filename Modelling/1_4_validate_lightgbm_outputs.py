#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Cara Lynch
# See the LICENSE file for details.
"""Validate completed Study 2 LightGBM classifier outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

try:
    import duckdb
except Exception:  # pragma: no cover - exercised only when duckdb is absent
    duckdb = None

try:
    import lightgbm
except Exception:  # pragma: no cover - exercised only when lightgbm is absent
    lightgbm = None

try:
    import pandas as pd
except Exception:  # pragma: no cover - exercised only when pandas is absent
    pd = None


REQUIRED_OUTPUTS = (
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
)
OPTIONAL_RANKED_PREDICTIONS = "lightgbm_ranked_predictions.parquet"
DEFAULT_EXPECTED_TEST_MONTHS = "2021-07,2021-08,2021-09,2021-10,2021-11,2021-12"

SUBMISSION_REPRESENTATION_EXPECTATIONS = {
    "activity_plus_historical_engagement_reduced_collinearity_logged_rate_only_submission_count_only": {
        "features": (
            "feature_n_submissions_prev",
            "feature_n_total_activity_prev",
            "feature_n_active_days_prev",
            "feature_mean_activity_per_active_day_prev",
            "feature_received_engagement_total_prev",
            "feature_log_received_engagement_rate_prev",
            "feature_received_replies_to_comments_prev",
            "feature_overlap_submission_and_comment_reply_events_prev",
        ),
        "representation": "count",
        "retained": "feature_n_submissions_prev",
        "removed": "feature_submission_share_prev",
    },
    "activity_plus_historical_engagement_reduced_collinearity_logged_rate_only_submission_share_only": {
        "features": (
            "feature_n_total_activity_prev",
            "feature_submission_share_prev",
            "feature_n_active_days_prev",
            "feature_mean_activity_per_active_day_prev",
            "feature_received_engagement_total_prev",
            "feature_log_received_engagement_rate_prev",
            "feature_received_replies_to_comments_prev",
            "feature_overlap_submission_and_comment_reply_events_prev",
        ),
        "representation": "composition",
        "retained": "feature_submission_share_prev",
        "removed": "feature_n_submissions_prev",
    },
}

METRIC_COLUMNS = (
    "precision_at_10pct",
    "recall_at_10pct",
    "average_precision",
    "auprc_ap",
    "auroc",
)
MCC_COLUMNS = ("mcc_at_10pct",)
FOLD_REQUIRED_COLUMNS = (
    "model_name",
    "model_type",
    "feature_set",
    "fold_id",
    "prediction_month",
    "test_prediction_month",
    "split_role",
    "metric_scope",
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
    "k_at_10pct",
    "top_k_positive_count",
    "tp_at_10pct",
    "fp_at_10pct",
    "fn_at_10pct",
    "tn_at_10pct",
    *METRIC_COLUMNS,
    *MCC_COLUMNS,
)
SUMMARY_REQUIRED_COLUMNS = (
    "model_name",
    "model_type",
    "feature_set",
    "source_split_role",
    "n_folds",
    "test_prediction_month_start",
    "test_prediction_month_end",
    "n_users_pooled",
    "n_positive_pooled",
    "average_precision_macro_mean",
    "auprc_ap_macro_mean",
    "auroc_macro_mean",
    "precision_at_10pct_macro_mean",
    "recall_at_10pct_macro_mean",
    "mcc_at_10pct_pooled",
)
SELECTED_REQUIRED_COLUMNS = (
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
    "test_prediction_month",
)
VALIDATION_REQUIRED_COLUMNS = (
    "model_name",
    "model_type",
    "feature_set",
    "fold_id",
    "candidate_id",
    "num_leaves",
    "reg_lambda",
    "is_unbalance",
    "imbalance_mode",
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
    "selection_rank_within_fold",
    "selected_by_validation",
    "prediction_month",
    "split_role",
    "metric_scope",
    "test_prediction_month",
    *METRIC_COLUMNS,
    *MCC_COLUMNS,
)
FEATURE_IMPORTANCE_REQUIRED_COLUMNS = (
    "model_name",
    "model_type",
    "feature_set",
    "fold_id",
    "selected_candidate_id",
    "is_unbalance",
    "best_iteration",
    "final_refit_end_prediction_month",
    "final_refit_includes_validation",
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

GRID_REQUIRED_COLUMNS = (
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
)


@dataclass
class ValidationState:
    """Mutable validation state."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_files: dict[str, dict[str, Any]] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    search_boundary_diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def utc_now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_expected_months(value: str) -> list[str]:
    """Parse comma-separated YYYY-MM expected test months."""
    months = [part.strip() for part in str(value).split(",") if part.strip()]
    if not months:
        raise argparse.ArgumentTypeError("--expected-test-months must contain at least one month.")
    bad = [month for month in months if len(month) != 7 or month[4] != "-"]
    if bad:
        raise argparse.ArgumentTypeError("--expected-test-months must use YYYY-MM values: " + ", ".join(bad))
    return months


def coerce_float(value: Any) -> Optional[float]:
    """Return value as float when possible."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def coerce_int(value: Any) -> Optional[int]:
    """Return an integer-like value as int when possible."""
    number = coerce_float(value)
    if number is None or not float(number).is_integer():
        return None
    return int(number)


def is_truthy(value: Any) -> bool:
    """Return whether a CSV/JSON value represents true."""
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def mark_file(path: Path, *, required: bool, state: ValidationState, label: str) -> bool:
    """Record file existence and non-empty status."""
    info = {"path": str(path), "required": required, "exists": path.exists(), "is_file": path.is_file()}
    info["size_bytes"] = path.stat().st_size if path.exists() and path.is_file() else 0
    state.checked_files[label] = info
    if not path.exists():
        if required:
            state.fail(f"Missing required file: {path}")
        return False
    if not path.is_file():
        state.fail(f"Expected a file but found a non-file path: {path}")
        return False
    if required and info["size_bytes"] <= 0:
        state.fail(f"Required file is empty: {path}")
        return False
    return info["size_bytes"] > 0


def read_json(path: Path, label: str, state: ValidationState) -> Optional[dict[str, Any]]:
    """Read JSON object or record validation error."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        state.fail(f"Could not parse {label} at {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        state.fail(f"{label} must contain a JSON object.")
        return None
    return payload


def read_csv(path: Path, label: str, state: ValidationState):
    """Read CSV as strings or record validation error."""
    if pd is None:
        state.fail("pandas is required to parse LightGBM CSV outputs.")
        return None
    try:
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:
        state.fail(f"Could not parse {label} at {path}: {exc}")
        return None
    state.row_counts[label] = len(table)
    return table


def require_columns(table: Any, table_name: str, columns: Sequence[str], state: ValidationState) -> bool:
    """Record an error if a table lacks required columns."""
    if table is None:
        return False
    missing = [column for column in columns if column not in table.columns]
    if missing:
        state.fail(f"{table_name} is missing required column(s): {', '.join(missing)}")
        return False
    return True


def load_outputs(out_root: Path, state: ValidationState) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load required JSON and CSV outputs where possible."""
    json_outputs: dict[str, Any] = {}
    csv_outputs: dict[str, Any] = {}
    for name in REQUIRED_OUTPUTS:
        path = out_root / name
        if not mark_file(path, required=True, state=state, label=name):
            continue
        if name.endswith(".json"):
            json_outputs[name] = read_json(path, name, state)
        elif name.endswith(".csv"):
            csv_outputs[name] = read_csv(path, name, state)
    present = mark_file(out_root / OPTIONAL_RANKED_PREDICTIONS, required=False, state=state, label=OPTIONAL_RANKED_PREDICTIONS)
    state.checked_files[OPTIONAL_RANKED_PREDICTIONS]["required_by_validator"] = False
    if present:
        if pd is None:
            state.fail("pandas is required to validate ranked-prediction Parquet output.")
        else:
            try:
                ranked = pd.read_parquet(out_root / OPTIONAL_RANKED_PREDICTIONS)
                csv_outputs[OPTIONAL_RANKED_PREDICTIONS] = ranked
                state.row_counts[OPTIONAL_RANKED_PREDICTIONS] = len(ranked)
                state.checked_files[OPTIONAL_RANKED_PREDICTIONS]["schema_checked"] = True
            except Exception as exc:
                state.fail(f"Could not parse {OPTIONAL_RANKED_PREDICTIONS}: {exc}")
    return json_outputs, csv_outputs


def observed_test_months_from_fold_metrics(fold_metrics: Any) -> list[str]:
    """Return observed test months in chronological order."""
    if fold_metrics is None or "test_prediction_month" not in fold_metrics.columns:
        return []
    return sorted(str(value) for value in set(fold_metrics["test_prediction_month"].tolist()) if value)


def validate_shapes(
    csv_outputs: dict[str, Any],
    *,
    expected_n_folds: int,
    expected_validation_candidates: int,
    state: ValidationState,
) -> None:
    """Validate core row counts."""
    expected = {
        "lightgbm_metrics_by_fold.csv": expected_n_folds,
        "lightgbm_selected_hyperparameters_by_fold.csv": expected_n_folds,
        "lightgbm_validation_metrics_by_candidate.csv": expected_validation_candidates,
        "lightgbm_metrics_summary.csv": 1,
    }
    for name, n_expected in expected.items():
        table = csv_outputs.get(name)
        if table is not None and len(table) != n_expected:
            state.fail(f"{name} has the wrong number of rows: observed={len(table)}, expected={n_expected}.")


def validate_metric_sanity(csv_outputs: dict[str, Any], expected_test_months: Sequence[str], state: ValidationState) -> None:
    """Validate metric columns, fold structure, and top-k count consistency."""
    fold_metrics = csv_outputs.get("lightgbm_metrics_by_fold.csv")
    if require_columns(fold_metrics, "lightgbm_metrics_by_fold.csv", FOLD_REQUIRED_COLUMNS, state):
        if set(fold_metrics["split_role"].astype(str)) != {"test"}:
            state.fail("lightgbm_metrics_by_fold.csv must contain only split_role='test' rows.")
        if set(fold_metrics["metric_scope"].astype(str)) != {"primary_rolling_origin_test"}:
            state.fail("lightgbm_metrics_by_fold.csv must contain only primary rolling-origin test rows.")
        observed = observed_test_months_from_fold_metrics(fold_metrics)
        if observed != list(expected_test_months):
            state.fail(f"Fold metric test months do not match expected months: observed={observed}; expected={list(expected_test_months)}.")
        for row_index, row in enumerate(fold_metrics.to_dict("records"), start=1):
            for column in METRIC_COLUMNS:
                value = coerce_float(row.get(column))
                if value is None or value < -1e-12 or value > 1.0 + 1e-12:
                    state.fail(f"lightgbm_metrics_by_fold.csv row {row_index} {column} is not numeric in [0, 1]: {row.get(column)!r}.")
            for column in MCC_COLUMNS:
                value = coerce_float(row.get(column))
                if value is None or value < -1.0 - 1e-12 or value > 1.0 + 1e-12:
                    state.fail(f"lightgbm_metrics_by_fold.csv row {row_index} {column} is not numeric in [-1, 1]: {row.get(column)!r}.")
            try:
                tp = int(row["tp_at_10pct"])
                fp = int(row["fp_at_10pct"])
                fn = int(row["fn_at_10pct"])
                tn = int(row["tn_at_10pct"])
                n_positive = int(row["n_positive"])
                n_negative = int(row["n_negative"])
                k_at_10pct = int(row["k_at_10pct"])
                top_k_positive = int(row["top_k_positive_count"])
            except ValueError:
                state.fail(f"lightgbm_metrics_by_fold.csv row {row_index} has non-integer count columns.")
                continue
            if tp != top_k_positive or fp != k_at_10pct - tp or fn != n_positive - tp or tn != n_negative - fp:
                state.fail(f"lightgbm_metrics_by_fold.csv row {row_index} top-k confusion counts are inconsistent.")

    validation = csv_outputs.get("lightgbm_validation_metrics_by_candidate.csv")
    if require_columns(validation, "lightgbm_validation_metrics_by_candidate.csv", VALIDATION_REQUIRED_COLUMNS, state):
        if set(validation["split_role"].astype(str)) != {"validation"}:
            state.fail("lightgbm_validation_metrics_by_candidate.csv must contain only validation rows.")
        if set(validation["metric_scope"].astype(str)) != {"validation_model_selection"}:
            state.fail("lightgbm_validation_metrics_by_candidate.csv must contain only validation model-selection rows.")

    selected = csv_outputs.get("lightgbm_selected_hyperparameters_by_fold.csv")
    if require_columns(selected, "lightgbm_selected_hyperparameters_by_fold.csv", SELECTED_REQUIRED_COLUMNS, state):
        selected_pairs = {
            (str(row["fold_id"]), str(row["selected_candidate_id"]))
            for row in selected.to_dict("records")
        }
        candidate_pairs = set()
        if validation is not None and "fold_id" in validation.columns and "candidate_id" in validation.columns:
            candidate_pairs = {
                (str(row["fold_id"]), str(row["candidate_id"]))
                for row in validation.to_dict("records")
            }
        missing = sorted(selected_pairs - candidate_pairs)
        if missing:
            state.fail(f"Selected candidate does not appear in validation rows for fold/candidate pair(s): {missing}.")

    summary = csv_outputs.get("lightgbm_metrics_summary.csv")
    if require_columns(summary, "lightgbm_metrics_summary.csv", SUMMARY_REQUIRED_COLUMNS, state) and len(summary) == 1:
        row = summary.iloc[0]
        if str(row.get("source_split_role", "")) != "test":
            state.fail("lightgbm_metrics_summary.csv must be based on test rows.")
        if str(row.get("test_prediction_month_start", "")) != expected_test_months[0]:
            state.fail("lightgbm_metrics_summary.csv test_prediction_month_start does not match expected first test month.")
        if str(row.get("test_prediction_month_end", "")) != expected_test_months[-1]:
            state.fail("lightgbm_metrics_summary.csv test_prediction_month_end does not match expected final test month.")

    importance = csv_outputs.get("feature_importance.csv")
    require_columns(importance, "feature_importance.csv", FEATURE_IMPORTANCE_REQUIRED_COLUMNS, state)


def month_index(value: object) -> Optional[int]:
    """Return a sortable month index for an ISO ``YYYY-MM`` value."""
    text = str(value)
    if len(text) != 7 or text[4] != "-":
        return None
    try:
        year = int(text[:4])
        month = int(text[5:])
    except ValueError:
        return None
    if month < 1 or month > 12:
        return None
    return year * 12 + month


def split_expectations(
    csv_outputs: dict[str, Any],
    *,
    expected_n_folds: int,
    expected_test_months: Sequence[str],
    state: ValidationState,
) -> dict[str, dict[str, Any]]:
    """Validate exact fold agreement and return trusted split expectations."""
    error_count = len(state.errors)
    splits = csv_outputs.get("rolling_origin_splits.csv")
    required = (
        "fold_id",
        "prediction_month",
        "split_role",
        "train_start_prediction_month",
        "train_end_prediction_month",
        "validation_prediction_month",
        "test_prediction_month",
    )
    candidate_table = csv_outputs.get("lightgbm_validation_metrics_by_candidate.csv")
    selected = csv_outputs.get("lightgbm_selected_hyperparameters_by_fold.csv")
    fold_metrics = csv_outputs.get("lightgbm_metrics_by_fold.csv")
    if not (
        require_columns(splits, "rolling_origin_splits.csv", required, state)
        and require_columns(candidate_table, "lightgbm_validation_metrics_by_candidate.csv", ("fold_id",), state)
        and require_columns(selected, "lightgbm_selected_hyperparameters_by_fold.csv", ("fold_id",), state)
        and require_columns(fold_metrics, "lightgbm_metrics_by_fold.csv", ("fold_id", "test_prediction_month"), state)
    ):
        return {}

    valid_roles = {"train", "validation", "test"}
    ignored_roles = {"diagnostic_only"}
    invalid_roles = sorted(set(splits["split_role"].astype(str)) - valid_roles - ignored_roles)
    if invalid_roles:
        state.fail("rolling_origin_splits.csv contains invalid split_role value(s): " + ", ".join(invalid_roles))
    fold_splits = splits[splits["split_role"].astype(str).isin(valid_roles)]
    result: dict[str, dict[str, Any]] = {}
    for fold_id, group in fold_splits.groupby("fold_id", sort=False):
        fold_key = str(fold_id)
        train = group[group["split_role"].astype(str) == "train"]
        validation_rows = group[group["split_role"].astype(str) == "validation"]
        test_rows = group[group["split_role"].astype(str) == "test"]
        if len(train) == 0 or len(validation_rows) != 1 or len(test_rows) != 1:
            state.fail(f"rolling_origin_splits.csv fold {fold_key} has invalid train/validation/test rows.")
            continue
        train_months = sorted(str(value) for value in train["prediction_month"].tolist())
        validation_month = str(validation_rows.iloc[0]["prediction_month"])
        test_month = str(test_rows.iloc[0]["prediction_month"])
        metadata = {
            field: {str(value) for value in group[field].tolist()}
            for field in ("train_start_prediction_month", "train_end_prediction_month", "validation_prediction_month", "test_prediction_month")
        }
        if any(len(values) != 1 for values in metadata.values()):
            state.fail(f"rolling_origin_splits.csv fold {fold_key} has inconsistent split metadata.")
            continue
        train_start = next(iter(metadata["train_start_prediction_month"]))
        train_end = next(iter(metadata["train_end_prediction_month"]))
        declared_validation = next(iter(metadata["validation_prediction_month"]))
        declared_test = next(iter(metadata["test_prediction_month"]))
        month_values = [*train_months, validation_month, test_month, train_start, train_end]
        indices = [month_index(value) for value in month_values]
        if any(value is None for value in indices):
            state.fail(f"rolling_origin_splits.csv fold {fold_key} contains malformed YYYY-MM month value(s).")
            continue
        train_indices = [month_index(value) for value in train_months]
        assert all(value is not None for value in train_indices)
        train_indices = [int(value) for value in train_indices]
        if (
            len(set(train_months)) != len(train_months)
            or train_start != train_months[0]
            or train_end != train_months[-1]
            or train_indices != list(range(train_indices[0], train_indices[-1] + 1))
            or declared_validation != validation_month
            or declared_test != test_month
            or train_indices[-1] >= int(month_index(validation_month))
            or int(month_index(validation_month)) >= int(month_index(test_month))
        ):
            state.fail(f"rolling_origin_splits.csv fold {fold_key} has malformed or overlapping train/validation/test months.")
            continue
        result[fold_key] = {
            "train_start": train_start,
            "train_end": train_end,
            "train_months": tuple(train_months),
            "validation": validation_month,
            "test": test_month,
        }

    split_ids = set(result)
    table_ids = {
        "rolling_origin_splits.csv": split_ids,
        "lightgbm_validation_metrics_by_candidate.csv": {str(value) for value in candidate_table["fold_id"].tolist()},
        "lightgbm_selected_hyperparameters_by_fold.csv": {str(value) for value in selected["fold_id"].tolist()},
        "lightgbm_metrics_by_fold.csv": {str(value) for value in fold_metrics["fold_id"].tolist()},
    }
    if len(split_ids) != int(expected_n_folds):
        state.fail(
            f"rolling_origin_splits.csv has {len(split_ids)} structurally valid fold(s); expected {expected_n_folds}."
        )
    for name, fold_ids in table_ids.items():
        if fold_ids != split_ids:
            state.fail(f"Fold-ID set mismatch for {name}: observed={sorted(fold_ids)}, expected={sorted(split_ids)}.")
    observed_tests = {str(value) for value in fold_metrics["test_prediction_month"].tolist()}
    if observed_tests != set(expected_test_months):
        state.fail(
            "Fold test-month set does not match expected months: "
            f"observed={sorted(observed_tests)}, expected={list(expected_test_months)}."
        )
    for fold_id, expected in result.items():
        metric_months = {
            str(row["test_prediction_month"])
            for row in fold_metrics.to_dict("records")
            if str(row.get("fold_id")) == fold_id
        }
        if metric_months != {expected["test"]}:
            state.fail(
                f"Fold {fold_id} test metric month(s) {sorted(metric_months)} do not match split test month "
                f"{expected['test']}."
            )
    return result if len(state.errors) == error_count else {}


def validate_weight_row(row: Any, *, prefix: str, label: str, state: ValidationState) -> None:
    """Validate class counts and effective positive weight for one fit."""
    n_rows = coerce_int(row.get(f"{prefix}_n_rows"))
    n_positive = coerce_int(row.get(f"{prefix}_n_positive"))
    n_negative = coerce_int(row.get(f"{prefix}_n_negative"))
    observed_weight = coerce_float(row.get(f"{prefix}_effective_positive_weight"))
    if None in (n_rows, n_positive, n_negative) or min(n_rows, n_positive, n_negative) <= 0:
        state.fail(f"{label} has invalid {prefix} class counts.")
        return
    if n_rows != n_positive + n_negative:
        state.fail(f"{label} {prefix} class counts do not sum to {prefix}_n_rows.")
    observed_prevalence = coerce_float(row.get(f"{prefix}_positive_prevalence"))
    expected_prevalence = float(n_positive) / float(n_rows)
    if observed_prevalence is None or not math.isclose(observed_prevalence, expected_prevalence, rel_tol=1e-9, abs_tol=1e-9):
        state.fail(f"{label} {prefix}_positive_prevalence is inconsistent with class counts.")
    expected_weight = float(n_negative) / float(n_positive) if is_truthy(row.get("is_unbalance")) else 1.0
    if observed_weight is None or not math.isclose(observed_weight, expected_weight, rel_tol=1e-9, abs_tol=1e-9):
        state.fail(
            f"{label} {prefix}_effective_positive_weight={row.get(f'{prefix}_effective_positive_weight')!r}; "
            f"expected {expected_weight} from is_unbalance and class counts."
        )


def validation_selection_sort_key(row: Any) -> Optional[tuple[float, float, int, int, int, float, int]]:
    """Return the locked independent validation-winner ordering key."""
    average_precision = coerce_float(row.get("average_precision"))
    auroc = coerce_float(row.get("auroc"))
    num_leaves = coerce_int(row.get("num_leaves"))
    best_iteration = coerce_int(row.get("best_iteration"))
    reg_lambda = coerce_float(row.get("reg_lambda"))
    candidate_id = coerce_int(row.get("candidate_id"))
    if None in (average_precision, auroc, num_leaves, best_iteration, reg_lambda, candidate_id):
        return None
    return (
        -float(average_precision),
        -float(auroc),
        int(num_leaves),
        int(best_iteration),
        int(is_truthy(row.get("is_unbalance"))),
        float(reg_lambda),
        int(candidate_id),
    )


def expected_structural_grid() -> set[tuple[int, float, bool]]:
    """Return the locked 3 x 2 x 2 structural candidate product."""
    return {(leaves, reg, unbalanced) for leaves in (15, 31, 63) for reg in (0.0, 1.0) for unbalanced in (False, True)}


def quote_sql_string(value: object) -> str:
    """Return a DuckDB-safe SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def smoke_panel_config(json_outputs: dict[str, Any], state: ValidationState) -> dict[str, Any]:
    """Read and cross-check the runner's smoke-view configuration."""
    resolved = json_outputs.get("resolved_config.json") or {}
    metadata = json_outputs.get("run_metadata.json") or {}
    config = resolved.get("config", {}) if isinstance(resolved, dict) else {}
    if not isinstance(config, dict):
        state.fail("resolved_config.json lacks a config object for smoke provenance.")
        config = {}
    if not isinstance(metadata, dict):
        state.fail("run_metadata.json is not an object for smoke provenance.")
        metadata = {}
    for source_name, source in (("resolved_config.json config", config), ("run_metadata.json", metadata)):
        for field in ("smoke", "smoke_max_rows"):
            if field not in source:
                state.fail(f"{source_name} is missing required smoke provenance field {field!r}.")
    smoke_value = config.get("smoke")
    smoke = smoke_value if isinstance(smoke_value, bool) else is_truthy(smoke_value)
    if not isinstance(smoke_value, bool):
        state.fail("resolved_config.json records a non-boolean smoke value.")
    try:
        smoke_max_rows = int(config.get("smoke_max_rows"))
    except (TypeError, ValueError):
        state.fail("resolved_config.json records a non-integer smoke_max_rows value.")
        smoke_max_rows = 0
    if smoke_max_rows < 0:
        state.fail("resolved_config.json records a negative smoke_max_rows value.")
    random_seed = config.get("random_seed")
    for field, expected in (("smoke", smoke_value), ("smoke_max_rows", config.get("smoke_max_rows"))):
        observed = metadata.get(field)
        if observed != expected or type(observed) is not type(expected):
            state.fail(f"Smoke configuration mismatch between resolved_config.json and run_metadata.json for {field}.")
    metadata_seed = metadata.get("random_seed")
    if random_seed is not None and metadata_seed is not None and str(metadata_seed) != str(random_seed):
        state.fail("Smoke configuration mismatch between resolved_config.json and run_metadata.json for random_seed.")
    return {
        "enabled": smoke and smoke_max_rows > 0,
        "smoke_max_rows": smoke_max_rows,
        "random_seed": random_seed,
        # The runner's label is fixed rather than configurable.  Its smoke
        # view intentionally orders by user_id; random_seed is recorded for
        # run provenance but is not part of this deterministic selection.
        "label_column": "label_high_engagement",
    }


def effective_panel_relation(panel_path: Path, smoke_config: dict[str, Any]) -> tuple[str, str]:
    """Return raw and runner-equivalent effective Parquet relations."""
    raw_relation = f"read_parquet({quote_sql_string(panel_path)})"
    relation = raw_relation
    if smoke_config["enabled"]:
        per_label_limit = max(1, int(smoke_config["smoke_max_rows"]) // 2)
        relation = f"""
        (
          SELECT *
          FROM (
            SELECT *, ROW_NUMBER() OVER (
              PARTITION BY CAST(prediction_month AS VARCHAR), CAST({smoke_config['label_column']} AS INTEGER)
              ORDER BY CAST(user_id AS VARCHAR) ASC
            ) AS _smoke_row_num
            FROM {raw_relation}
          )
          WHERE _smoke_row_num <= {per_label_limit}
        )
        """
    return raw_relation, relation


def validate_panel_contract(
    con: Any,
    *,
    raw_relation: str,
    effective_relation: str,
    required_months: Sequence[str],
    label_column: str,
    state: ValidationState,
) -> bool:
    """Validate raw duplicate keys and the runner-equivalent monthly label contract."""
    error_count = len(state.errors)
    duplicate_rows = con.execute(
        f"""
        SELECT CAST(user_id AS VARCHAR), CAST(prediction_month AS VARCHAR), COUNT(*)
        FROM {raw_relation}
        GROUP BY 1, 2
        HAVING COUNT(*) > 1
        ORDER BY 2, 1
        LIMIT 5
        """
    ).fetchall()
    if duplicate_rows:
        duplicate_key_count = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM (
                  SELECT 1 FROM {raw_relation}
                  GROUP BY CAST(user_id AS VARCHAR), CAST(prediction_month AS VARCHAR)
                  HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
        )
        examples = ", ".join(f"({row[0]!r}, {row[1]!r}; rows={int(row[2])})" for row in duplicate_rows)
        state.fail(
            "--panel-path violates the one-user-month contract: "
            f"duplicate (user_id, prediction_month) key count={duplicate_key_count}; examples={examples}."
        )
    rows = con.execute(
        f"""
        SELECT
          CAST(prediction_month AS VARCHAR) AS prediction_month,
          COUNT(*) AS n_rows,
          COUNT(DISTINCT CAST({label_column} AS INTEGER)) AS n_distinct_labels,
           SUM(
             CASE
               WHEN {label_column} IS NULL
                 OR CAST({label_column} AS INTEGER) NOT IN (0, 1)
               THEN 1
               ELSE 0
             END
           ) AS n_non_binary
        FROM {effective_relation}
        GROUP BY 1
        """
    ).fetchall()
    by_month = {str(row[0]): (int(row[1]), int(row[2]), int(row[3] or 0)) for row in rows}
    for month in required_months:
        n_rows, n_distinct, n_non_binary = by_month.get(str(month), (0, 0, 0))
        if n_rows <= 0:
            state.fail(f"--panel-path has zero rows for required prediction month {month} in the effective panel view.")
        elif n_non_binary:
            state.fail(
                f"--panel-path has {n_non_binary} NULL or non-binary {label_column} value(s) for required prediction month {month}."
            )
        elif n_distinct != 2:
            state.fail(
                f"--panel-path has only one label class for required prediction month {month} in the effective panel view."
            )
    return len(state.errors) == error_count


def panel_fit_counts(
    panel_path: Path,
    expectations: dict[str, dict[str, str]],
    smoke_config: dict[str, Any],
    state: ValidationState,
) -> dict[str, dict[str, dict[str, float]]]:
    """Reconstruct fit populations from the declared Parquet panel using DuckDB."""
    if duckdb is None:
        state.fail("duckdb is required to reconstruct LightGBM fitting populations from --panel-path.")
        return {}
    result: dict[str, dict[str, dict[str, float]]] = {}
    con = duckdb.connect()
    try:
        raw_relation, relation = effective_panel_relation(panel_path, smoke_config)
        required_months = sorted(
            {month for windows in expectations.values() for month in (*windows["train_months"], windows["validation"], windows["test"])}
        )
        if not validate_panel_contract(
            con,
            raw_relation=raw_relation,
            effective_relation=relation,
            required_months=required_months,
            label_column=smoke_config["label_column"],
            state=state,
        ):
            return {}
        for fold_id, windows in expectations.items():
            result[fold_id] = {}
            for prefix, end_month in (("selection_fit", windows["train_end"]), ("final_refit", windows["validation"])):
                row = con.execute(
                    f"""
                    SELECT
                      COUNT(*) AS n_rows,
                      SUM(CAST({smoke_config['label_column']} AS INTEGER)) AS n_positive
                    FROM {relation}
                    WHERE CAST(prediction_month AS VARCHAR) >= ?
                      AND CAST(prediction_month AS VARCHAR) <= ?
                    """,
                    [windows["train_start"], end_month],
                ).fetchone()
                n_rows = int(row[0] or 0)
                n_positive = int(row[1] or 0)
                n_negative = n_rows - n_positive
                if n_rows <= 0 or n_positive <= 0 or n_negative <= 0:
                    state.fail(
                        f"--panel-path has invalid {prefix} binary population for fold {fold_id}: "
                        f"n_rows={n_rows}, n_positive={n_positive}, n_negative={n_negative}."
                    )
                    return {}
                result[fold_id][prefix] = {
                    "n_rows": float(n_rows),
                    "n_positive": float(n_positive),
                    "n_negative": float(n_negative),
                    "positive_prevalence": float(n_positive) / float(n_rows),
                }
    except Exception as exc:
        state.fail(f"Could not reconstruct fitting populations from --panel-path {panel_path}: {exc}")
    finally:
        con.close()
    return result


def validate_row_against_panel_counts(
    row: Any,
    *,
    prefix: str,
    expected_counts: dict[str, float],
    label: str,
    state: ValidationState,
) -> None:
    """Compare recorded fit provenance directly with the declared panel population."""
    for field in ("n_rows", "n_positive", "n_negative"):
        observed = coerce_int(row.get(f"{prefix}_{field}"))
        expected = int(expected_counts[field])
        if observed != expected:
            state.fail(f"{label} {prefix}_{field}={observed!r} disagrees with --panel-path ({expected}).")
    observed_prevalence = coerce_float(row.get(f"{prefix}_positive_prevalence"))
    if observed_prevalence is not None and not math.isclose(
        observed_prevalence, expected_counts["positive_prevalence"], rel_tol=1e-9, abs_tol=1e-9
    ):
        state.fail(
            f"{label} {prefix}_positive_prevalence={observed_prevalence!r} disagrees with --panel-path "
            f"({expected_counts['positive_prevalence']})."
        )
    expected_weight = (
        expected_counts["n_negative"] / expected_counts["n_positive"]
        if is_truthy(row.get("is_unbalance"))
        else 1.0
    )
    observed_weight = coerce_float(row.get(f"{prefix}_effective_positive_weight"))
    if observed_weight is not None and not math.isclose(observed_weight, expected_weight, rel_tol=1e-9, abs_tol=1e-9):
        state.fail(
            f"{label} {prefix}_effective_positive_weight={observed_weight!r} disagrees with --panel-path "
            f"({expected_weight})."
        )


def validate_iteration_weighting_and_refit(
    csv_outputs: dict[str, Any], expectations: dict[str, dict[str, Any]], state: ValidationState
) -> None:
    """Validate the locked grid, AP early stopping, weighting, and refit provenance."""
    grid = csv_outputs.get("lightgbm_hyperparameter_grid.csv")
    if require_columns(grid, "lightgbm_hyperparameter_grid.csv", GRID_REQUIRED_COLUMNS, state):
        obsolete = [column for column in ("class_weight", "selected_class_weight", "n_estimators", "scale_pos_weight") if column in grid.columns]
        if obsolete:
            state.fail("lightgbm_hyperparameter_grid.csv contains obsolete/prohibited column(s): " + ", ".join(obsolete))
        if len(grid) != 12:
            state.fail(f"lightgbm_hyperparameter_grid.csv must contain exactly 12 rows, observed {len(grid)}.")
        observed = {
            (coerce_int(row["num_leaves"]), coerce_float(row["reg_lambda"]), is_truthy(row["is_unbalance"]))
            for row in grid.to_dict("records")
        }
        expected = {(leaves, reg, unbalanced) for leaves in (15, 31, 63) for reg in (0.0, 1.0) for unbalanced in (False, True)}
        if observed != expected:
            state.fail("lightgbm_hyperparameter_grid.csv does not match the locked 3 x 2 x 2 structural design.")
        for index, row in enumerate(grid.to_dict("records"), start=1):
            if coerce_float(row["learning_rate"]) != 0.05 or coerce_int(row["min_child_samples"]) != 50:
                state.fail(f"lightgbm_hyperparameter_grid.csv row {index} has inconsistent fixed learning/min-child parameters.")
            if coerce_float(row["subsample"]) != 1.0 or coerce_int(row["subsample_freq"]) != 0:
                state.fail(f"lightgbm_hyperparameter_grid.csv row {index} activates or misrecords bagging.")
            if coerce_float(row["colsample_bytree"]) != 1.0:
                state.fail(f"lightgbm_hyperparameter_grid.csv row {index} does not expose all features to each tree.")
            if coerce_int(row["max_n_estimators"]) != 1000 or coerce_int(row["early_stopping_rounds"]) != 50:
                state.fail(f"lightgbm_hyperparameter_grid.csv row {index} has inconsistent iteration bounds.")
            if str(row["early_stopping_metric"]) != "average_precision":
                state.fail(f"lightgbm_hyperparameter_grid.csv row {index} does not record Average-Precision early stopping.")

    validation = csv_outputs.get("lightgbm_validation_metrics_by_candidate.csv")
    validation_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    if require_columns(validation, "lightgbm_validation_metrics_by_candidate.csv", VALIDATION_REQUIRED_COLUMNS, state):
        for index, row in enumerate(validation.to_dict("records"), start=1):
            label = f"lightgbm_validation_metrics_by_candidate.csv row {index}"
            if any(column in validation.columns for column in ("class_weight", "selected_class_weight", "n_estimators", "scale_pos_weight")):
                state.fail("lightgbm_validation_metrics_by_candidate.csv contains obsolete/prohibited weighting or fixed-estimator columns.")
                break
            best_iteration = coerce_int(row.get("best_iteration"))
            maximum = coerce_int(row.get("max_n_estimators"))
            if best_iteration is None or maximum is None or not (1 <= best_iteration <= maximum):
                state.fail(f"{label} has inconsistent best_iteration/max_n_estimators.")
            if str(row.get("early_stopping_metric")) != "average_precision" or str(row.get("early_stopping_validation_metric_name")) != "average_precision":
                state.fail(f"{label} does not record validation Average-Precision early stopping.")
            early_value = coerce_float(row.get("early_stopping_validation_metric_value"))
            if early_value is None or not (0.0 <= early_value <= 1.0):
                state.fail(f"{label} has invalid early-stopping validation AP.")
            if is_truthy(row.get("selection_fit_includes_validation")):
                state.fail(f"{label} records candidate fitting as including validation rows.")
            validate_weight_row(row, prefix="selection_fit", label=label, state=state)
            expected = expectations.get(str(row.get("fold_id")), {})
            if expected and (
                str(row.get("selection_fit_start_prediction_month")) != expected["train_start"]
                or str(row.get("selection_fit_end_prediction_month")) != expected["train_end"]
            ):
                state.fail(f"{label} selection-fit window is inconsistent with rolling_origin_splits.csv.")
            validation_by_pair[(str(row.get("fold_id")), str(row.get("candidate_id")))] = row

    selected = csv_outputs.get("lightgbm_selected_hyperparameters_by_fold.csv")
    selected_by_fold: dict[str, dict[str, Any]] = {}
    if require_columns(selected, "lightgbm_selected_hyperparameters_by_fold.csv", SELECTED_REQUIRED_COLUMNS, state):
        for index, row in enumerate(selected.to_dict("records"), start=1):
            fold_id = str(row.get("fold_id"))
            label = f"lightgbm_selected_hyperparameters_by_fold.csv fold {fold_id}"
            candidate = validation_by_pair.get((fold_id, str(row.get("selected_candidate_id"))))
            if candidate:
                for column in ("num_leaves", "reg_lambda", "is_unbalance", "imbalance_mode", "best_iteration", "max_n_estimators"):
                    if str(row.get(column)) != str(candidate.get(column)):
                        state.fail(f"{label} {column} does not match its selected validation candidate.")
                if not math.isclose(float(row["selected_validation_metric_value"]), float(candidate["average_precision"]), rel_tol=1e-9, abs_tol=1e-9):
                    state.fail(f"{label} selected validation metric does not match candidate canonical Average Precision.")
            validate_weight_row(row, prefix="selection_fit", label=label, state=state)
            validate_weight_row(row, prefix="final_refit", label=label, state=state)
            if coerce_int(row.get("final_refit_n_estimators")) != coerce_int(row.get("best_iteration")):
                state.fail(f"{label} final_refit_n_estimators does not equal selected best_iteration.")
            expected = expectations.get(fold_id, {})
            if is_truthy(row.get("selection_fit_includes_validation")):
                state.fail(f"{label} records candidate fitting as including validation rows.")
            if not is_truthy(row.get("final_refit_includes_validation")):
                state.fail(f"{label} was not recorded as refit on training plus validation rows.")
            if expected:
                if str(row.get("selection_fit_start_prediction_month")) != expected["train_start"] or str(row.get("selection_fit_end_prediction_month")) != expected["train_end"]:
                    state.fail(f"{label} selection-fit window is inconsistent with rolling_origin_splits.csv.")
                if str(row.get("final_refit_start_prediction_month")) != expected["train_start"] or str(row.get("final_refit_end_prediction_month")) != expected["validation"]:
                    state.fail(f"{label} final-refit window is inconsistent with rolling_origin_splits.csv.")
                if str(row.get("final_refit_end_prediction_month")) >= expected["test"]:
                    state.fail(f"{label} final refit reaches the held-out test month.")
            final_rows = coerce_int(row.get("final_refit_n_rows"))
            selection_rows = coerce_int(row.get("selection_fit_n_rows"))
            if final_rows is not None and selection_rows is not None and final_rows <= selection_rows:
                state.fail(f"{label} final-refit row count must exceed the selection-fit row count.")
            selected_by_fold[fold_id] = row

    for table_name in ("lightgbm_metrics_by_fold.csv", "feature_importance.csv", OPTIONAL_RANKED_PREDICTIONS):
        table = csv_outputs.get(table_name)
        if table is None:
            continue
        if table_name == OPTIONAL_RANKED_PREDICTIONS:
            required = (
                "fold_id", "split_role", "prediction_month", "test_prediction_month", "selected_candidate_id", "is_unbalance", "best_iteration",
                "final_refit_end_prediction_month", "final_refit_includes_validation",
                "final_refit_effective_positive_weight", "final_refit_n_estimators",
            )
            if not require_columns(table, table_name, required, state):
                continue
            if set(table["split_role"].astype(str)) != {"test"}:
                state.fail("lightgbm_ranked_predictions.parquet must contain only test rows.")
            expected_pairs = {(fold_id, values["test"]) for fold_id, values in expectations.items()}
            observed_pairs = {
                (str(row.get("fold_id")), str(row.get("prediction_month")))
                for row in table.to_dict("records")
            }
            if observed_pairs != expected_pairs:
                state.fail(
                    "lightgbm_ranked_predictions.parquet fold/month combinations do not match validated test folds: "
                    f"observed={sorted(observed_pairs)}, expected={sorted(expected_pairs)}."
                )
        for index, row in enumerate(table.to_dict("records"), start=1):
            fold_id = str(row.get("fold_id"))
            selected_row = selected_by_fold.get(fold_id)
            if not selected_row:
                continue
            label = f"{table_name} row {index}"
            for column in ("selected_candidate_id", "is_unbalance", "best_iteration", "final_refit_end_prediction_month", "final_refit_includes_validation", "final_refit_effective_positive_weight", "final_refit_n_estimators"):
                if str(row.get(column)) != str(selected_row.get(column)):
                    state.fail(f"{label} {column} does not match the selected final-refit metadata for fold {fold_id}.")
            if table_name == OPTIONAL_RANKED_PREDICTIONS:
                expected = expectations.get(fold_id)
                if expected is None:
                    state.fail(f"{label} has an unknown fold_id {fold_id!r}.")
                elif (
                    str(row.get("prediction_month")) != str(row.get("test_prediction_month"))
                    or str(row.get("prediction_month")) != expected["test"]
                ):
                    state.fail(f"{label} does not belong to its validated fold test month {expected['test']}.")


def validate_independent_winners_populations_and_models(
    csv_outputs: dict[str, Any], *, expectations: dict[str, dict[str, Any]], json_outputs: dict[str, Any], out_root: Path, panel_path: Path, state: ValidationState
) -> None:
    """Independently validate selection, fit populations, and native final boosters."""
    if not expectations:
        return
    validation = csv_outputs.get("lightgbm_validation_metrics_by_candidate.csv")
    selected = csv_outputs.get("lightgbm_selected_hyperparameters_by_fold.csv")
    if not (
        require_columns(validation, "lightgbm_validation_metrics_by_candidate.csv", VALIDATION_REQUIRED_COLUMNS, state)
        and require_columns(selected, "lightgbm_selected_hyperparameters_by_fold.csv", SELECTED_REQUIRED_COLUMNS, state)
    ):
        return
    panel_counts = panel_fit_counts(panel_path, expectations, smoke_panel_config(json_outputs, state), state)
    selected_by_fold = {str(row["fold_id"]): row for row in selected.to_dict("records")}
    candidate_rows = validation.to_dict("records")
    for fold_id, expected in expectations.items():
        rows = [row for row in candidate_rows if str(row.get("fold_id")) == fold_id]
        structures = {
            (coerce_int(row.get("num_leaves")), coerce_float(row.get("reg_lambda")), is_truthy(row.get("is_unbalance")))
            for row in rows
        }
        if len(rows) != 12 or len({str(row.get("candidate_id")) for row in rows}) != 12 or structures != expected_structural_grid():
            state.fail(f"Fold {fold_id} must contain exactly the locked 12 distinct structural candidates.")
            continue
        for row in rows:
            if coerce_int(row.get("best_iteration")) == 1000:
                candidate_id = str(row.get("candidate_id"))
                state.warn(f"Fold {fold_id} candidate {candidate_id} reached the 1000-tree search boundary.")
                state.search_boundary_diagnostics.append(
                    {"fold_id": fold_id, "type": "candidate_best_iteration_1000", "candidate_id": candidate_id}
                )
        keys = [(validation_selection_sort_key(row), row) for row in rows]
        if any(key is None for key, _ in keys):
            state.fail(f"Fold {fold_id} has candidate rows that cannot be sorted by the locked selection key.")
            continue
        winner = min(keys, key=lambda item: item[0])[1]
        winner_id = str(winner["candidate_id"])
        marked = [row for row in rows if is_truthy(row.get("selected_by_validation"))]
        rank_one = [row for row in rows if coerce_int(row.get("selection_rank_within_fold")) == 1]
        if len(marked) != 1 or str(marked[0].get("candidate_id")) != winner_id:
            state.fail(f"Fold {fold_id} stored selected_by_validation marker disagrees with the independently computed winner.")
        if len(rank_one) != 1 or str(rank_one[0].get("candidate_id")) != winner_id:
            state.fail(f"Fold {fold_id} stored rank-one marker disagrees with the independently computed winner.")
        selected_row = selected_by_fold.get(fold_id)
        if selected_row is None:
            state.fail(f"Fold {fold_id} has no selected-hyperparameter row.")
            continue
        if str(selected_row.get("selected_candidate_id")) != winner_id:
            state.fail(f"Fold {fold_id} selected candidate is not the deterministic validation winner.")
        for column in (
            "num_leaves", "reg_lambda", "is_unbalance", "imbalance_mode", "learning_rate", "min_child_samples",
            "subsample", "subsample_freq", "colsample_bytree", "max_n_estimators", "early_stopping_rounds",
            "early_stopping_metric", "best_iteration",
        ):
            if str(selected_row.get(column)) != str(winner.get(column)):
                state.fail(f"Fold {fold_id} selected {column} does not match the independently computed winner.")
        fit_counts = panel_counts.get(fold_id)
        if fit_counts and "selection_fit" in fit_counts and "final_refit" in fit_counts:
            for row in rows:
                validate_row_against_panel_counts(
                    row, prefix="selection_fit", expected_counts=fit_counts["selection_fit"],
                    label=f"Fold {fold_id} candidate {row.get('candidate_id')}", state=state,
                )
            validate_row_against_panel_counts(
                selected_row, prefix="selection_fit", expected_counts=fit_counts["selection_fit"],
                label=f"Fold {fold_id} selected row", state=state,
            )
            validate_row_against_panel_counts(
                selected_row, prefix="final_refit", expected_counts=fit_counts["final_refit"],
                label=f"Fold {fold_id} selected row", state=state,
            )
        model_relative_path = str(selected_row.get("final_model_relative_path", ""))
        model_path = out_root / model_relative_path
        if not model_relative_path or Path(model_relative_path).is_absolute() or model_path.parent != out_root / "final_models":
            state.fail(f"Fold {fold_id} has an invalid final_model_relative_path: {model_relative_path!r}.")
        elif lightgbm is None:
            state.fail("lightgbm is required to inspect final native booster artifacts.")
        elif not model_path.is_file():
            state.fail(f"Fold {fold_id} final model file is missing: {model_relative_path}.")
        else:
            try:
                tree_count = int(lightgbm.Booster(model_file=str(model_path)).current_iteration())
                if tree_count != coerce_int(selected_row.get("best_iteration")) or tree_count != coerce_int(selected_row.get("final_refit_n_estimators")):
                    state.fail(f"Fold {fold_id} native final model tree count does not match selected best_iteration/final_refit_n_estimators.")
            except Exception as exc:
                state.fail(f"Fold {fold_id} final model file is unreadable: {exc}")
        if coerce_int(winner.get("num_leaves")) == 63:
            state.warn(f"Fold {fold_id} selected num_leaves=63 search boundary.")
            state.search_boundary_diagnostics.append({"fold_id": fold_id, "type": "selected_num_leaves_63"})
        lower = [coerce_float(row.get("average_precision")) for row in rows if (coerce_int(row.get("num_leaves")) or 0) < (coerce_int(winner.get("num_leaves")) or 0)]
        lower = [value for value in lower if value is not None]
        if lower:
            gap = float(winner["average_precision"]) - max(lower)
            state.warn(f"Fold {fold_id} selected-versus-best-lower-leaf validation AP gap: {gap:.6f}.")
            state.search_boundary_diagnostics.append({"fold_id": fold_id, "type": "selected_vs_lower_leaf_ap_gap", "average_precision_gap": gap})
    for table_name in ("lightgbm_metrics_by_fold.csv", "feature_importance.csv", OPTIONAL_RANKED_PREDICTIONS):
        table = csv_outputs.get(table_name)
        if table is None:
            continue
        for index, row in enumerate(table.to_dict("records"), start=1):
            fold_id = str(row.get("fold_id"))
            selected_row = selected_by_fold.get(fold_id)
            if selected_row is None:
                continue
            for column in ("final_model_relative_path", "final_refit_n_estimators", "best_iteration"):
                if str(row.get(column)) != str(selected_row.get(column)):
                    state.fail(f"{table_name} row {index} {column} does not match selected final-refit provenance.")
            fit_counts = panel_counts.get(fold_id)
            if fit_counts and "final_refit" in fit_counts:
                validate_row_against_panel_counts(
                    row, prefix="final_refit", expected_counts=fit_counts["final_refit"],
                    label=f"{table_name} row {index}", state=state,
                )


def validate_config_metadata(
    json_outputs: dict[str, Any],
    *,
    expected_feature_set: Optional[str],
    expected_panel_path: Optional[str],
    expected_random_seed: Optional[int],
    state: ValidationState,
) -> None:
    """Validate config and metadata provenance."""
    resolved = json_outputs.get("resolved_config.json") or {}
    metadata = json_outputs.get("run_metadata.json") or {}
    fitted_summary = json_outputs.get("fitted_model_summary.json") or {}
    config = resolved.get("config", {}) if isinstance(resolved, dict) else {}
    feature_meta = metadata.get("feature_set_metadata", {}) if isinstance(metadata, dict) else {}

    for source_name, source in (("resolved_config.json", config), ("run_metadata.json", metadata), ("fitted_model_summary.json", fitted_summary)):
        if not source:
            continue
        model_type = source.get("model_type")
        if model_type and str(model_type) != "lightgbm_classifier":
            state.fail(f"{source_name} model_type must be lightgbm_classifier, observed {model_type!r}.")

    if expected_feature_set is not None:
        observed_values = [
            config.get("feature_set"),
            metadata.get("feature_set"),
            feature_meta.get("resolved_feature_set"),
            fitted_summary.get("feature_set"),
        ]
        bad = [value for value in observed_values if value not in (None, "", expected_feature_set)]
        if bad:
            state.fail(f"Feature-set metadata does not match expected {expected_feature_set!r}: {bad}.")

    if expected_panel_path is not None:
        observed = config.get("panel_path") or metadata.get("panel_path")
        if observed and str(Path(str(observed))) != str(Path(str(expected_panel_path))):
            state.fail(f"panel_path does not match expected value: observed={observed!r}, expected={expected_panel_path!r}.")

    if expected_random_seed is not None:
        observed = config.get("random_seed") or metadata.get("random_seed")
        if observed not in (None, "") and int(observed) != int(expected_random_seed):
            state.fail(f"random_seed does not match expected value: observed={observed!r}, expected={expected_random_seed!r}.")

    required_metadata_fields = (
        "resolved_feature_set",
        "selected_features",
        "dropped_features",
        "schema_equivalent_to_feature_set",
        "interpretation_caveat",
    )
    if feature_meta:
        missing = [field for field in required_metadata_fields if field not in feature_meta]
        if missing:
            state.fail("run_metadata.json feature_set_metadata missing field(s): " + ", ".join(missing))
    else:
        state.fail("run_metadata.json lacks feature_set_metadata.")

    class_balance = metadata.get("class_balance_handling", {}) if isinstance(metadata, dict) else {}
    if class_balance:
        if str(class_balance.get("strategy")) != "is_unbalance_grid":
            state.fail("run_metadata.json class_balance_handling.strategy must be is_unbalance_grid.")
        if str(class_balance.get("scale_pos_weight")) != "not_used":
            state.fail("run_metadata.json must record scale_pos_weight as not used.")
    early_stopping = metadata.get("early_stopping", {}) if isinstance(metadata, dict) else {}
    if early_stopping:
        if str(early_stopping.get("metric")) != "average_precision":
            state.fail("run_metadata.json early stopping metric must be average_precision.")
        if str(early_stopping.get("eval_set")) != "validation_month_only":
            state.fail("run_metadata.json early stopping eval_set must be validation_month_only.")
    final_refit = metadata.get("final_refit_protocol", {}) if isinstance(metadata, dict) else {}
    if final_refit and not bool(final_refit.get("final_refit_includes_validation")):
        state.fail("run_metadata.json final-refit protocol must include validation rows.")

    git = metadata.get("git", {}) if isinstance(metadata, dict) else {}
    if git.get("dirty"):
        state.warn("run_metadata.json records a dirty git worktree.")


def validate_submission_representation_inventory(
    selected_features: Any,
    json_outputs: dict[str, Any],
    expected_feature_set: Optional[str],
    state: ValidationState,
) -> None:
    """Enforce the exact eight-feature submission-sensitivity contract."""
    resolved = json_outputs.get("resolved_config.json") or {}
    metadata = json_outputs.get("run_metadata.json") or {}
    config = resolved.get("config", {}) if isinstance(resolved, dict) else {}
    feature_set = expected_feature_set or metadata.get("feature_set") or config.get("feature_set")
    expectation = SUBMISSION_REPRESENTATION_EXPECTATIONS.get(str(feature_set))
    if expectation is None:
        return
    if selected_features is None:
        state.fail("Submission-representation sensitivity lacks selected_features.csv.")
        return
    required = {"feature_set", "feature_name", "included", "exclusion_reason"}
    missing = sorted(required - set(selected_features.columns))
    if missing:
        state.fail("selected_features.csv missing submission-sensitivity field(s): " + ", ".join(missing))
        return
    rows = selected_features.to_dict("records")
    included = [str(row["feature_name"]) for row in rows if is_truthy(row["included"])]
    expected_features = list(expectation["features"])
    if included != expected_features:
        state.fail(
            "Submission-representation selected feature inventory differs: "
            f"observed={included}, expected={expected_features}."
        )
    labels = {str(row["feature_set"]) for row in rows}
    if labels != {str(feature_set)}:
        state.fail(f"selected_features.csv feature_set labels differ: {sorted(labels)}.")
    removed_rows = [row for row in rows if str(row["feature_name"]) == expectation["removed"]]
    if len(removed_rows) != 1 or is_truthy(removed_rows[0].get("included", "")):
        state.fail(f"Removed submission feature is not uniquely excluded: {expectation['removed']}.")
    elif removed_rows[0].get("exclusion_reason") != "dropped_for_submission_representation_sensitivity":
        state.fail("Removed submission feature lacks the sensitivity-specific exclusion reason.")
    definition = metadata.get("feature_set_definition", {}) if isinstance(metadata, dict) else {}
    expected_fields = {
        "sensitivity_counterpart_feature_set": (
            "activity_plus_historical_engagement_reduced_collinearity_logged_rate_only"
        ),
        "sensitivity_axis": "prior_month_submission_representation",
        "retained_submission_representation": expectation["representation"],
        "retained_submission_feature": expectation["retained"],
        "removed_submission_feature": expectation["removed"],
        "automatically_replaces_headline_model": False,
        "author_decision_required": True,
        "ready_for_thesis_use": False,
    }
    for field, expected in expected_fields.items():
        if definition.get(field) != expected:
            state.fail(
                f"run_metadata.json feature_set_definition.{field} differs: "
                f"observed={definition.get(field)!r}, expected={expected!r}."
            )
    feature_meta = metadata.get("feature_set_metadata", {}) if isinstance(metadata, dict) else {}
    if feature_meta.get("selected_features") != expected_features:
        state.fail("run_metadata.json feature_set_metadata.selected_features differs from the exact inventory.")
    if metadata.get("selected_features") != expected_features:
        state.fail("run_metadata.json selected_features differs from the exact eight-feature inventory.")
    resolved_definition = resolved.get("feature_set_definition", {}) if isinstance(resolved, dict) else {}
    if resolved_definition != definition:
        state.fail("Resolved-config and run-metadata feature-set definitions differ.")


def capture_summary_metrics(csv_outputs: dict[str, Any]) -> dict[str, Any]:
    """Copy key summary metrics into the validation summary."""
    summary = csv_outputs.get("lightgbm_metrics_summary.csv")
    if summary is None or len(summary) != 1:
        return {}
    row = summary.iloc[0].to_dict()
    keys = (
        "average_precision_macro_mean",
        "auprc_ap_macro_mean",
        "auroc_macro_mean",
        "precision_at_10pct_macro_mean",
        "recall_at_10pct_macro_mean",
        "mcc_at_10pct_pooled",
    )
    return {key: row.get(key, "") for key in keys}


def build_validation_summary(
    state: ValidationState,
    *,
    expected_test_months: Sequence[str],
    observed_test_months: Sequence[str],
    key_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build validation summary JSON."""
    return {
        "validation_status": "failed" if state.errors else "passed",
        "validated_at_utc": utc_now_iso(),
        "expected_test_months": list(expected_test_months),
        "observed_test_months": list(observed_test_months),
        "checked_files": state.checked_files,
        "row_counts": state.row_counts,
        "key_metrics": key_metrics,
        "search_boundary_diagnostics": state.search_boundary_diagnostics,
        "warnings": state.warnings,
        "errors": state.errors,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Validate completed Study 2 LightGBM classifier outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-root", required=True, help="Completed LightGBM output directory.")
    parser.add_argument("--panel-path", required=True, help="Modelling-panel Parquet used by the completed run.")
    parser.add_argument("--expected-feature-set", default=None, help="Expected feature set recorded in outputs.")
    parser.add_argument("--expected-panel-path", default=None, help="Expected modelling_panel.parquet path.")
    parser.add_argument("--expected-random-seed", type=int, default=None, help="Expected deterministic random seed.")
    parser.add_argument("--expected-n-folds", type=int, default=6, help="Expected number of test folds.")
    parser.add_argument(
        "--expected-validation-candidates",
        type=int,
        default=72,
        help="Expected rows in lightgbm_validation_metrics_by_candidate.csv.",
    )
    parser.add_argument(
        "--expected-test-months",
        type=parse_expected_months,
        default=DEFAULT_EXPECTED_TEST_MONTHS,
        help="Comma-separated expected rolling-origin test months.",
    )
    parser.add_argument(
        "--write-validation-summary",
        default=None,
        help="Optional path for validation summary JSON; defaults under --out-root.",
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate validator CLI args."""
    if int(args.expected_n_folds) < 1:
        parser.error("--expected-n-folds must be at least 1.")
    if int(args.expected_validation_candidates) < 1:
        parser.error("--expected-validation-candidates must be at least 1.")
    if not Path(args.panel_path).is_file():
        parser.error(f"--panel-path is not a file: {args.panel_path}")
    if isinstance(args.expected_test_months, str):
        args.expected_test_months = parse_expected_months(args.expected_test_months)


def run_validation(args: argparse.Namespace) -> tuple[dict[str, Any], Path, bool]:
    """Run validation and return summary, path, and success flag."""
    out_root = Path(args.out_root)
    summary_path = Path(args.write_validation_summary) if args.write_validation_summary else out_root / "lightgbm_output_validation_summary.json"
    state = ValidationState()
    if not out_root.exists() or not out_root.is_dir():
        state.fail(f"--out-root is not a directory: {out_root}")
        summary = build_validation_summary(
            state,
            expected_test_months=args.expected_test_months,
            observed_test_months=[],
            key_metrics={},
        )
        write_json(summary_path, summary)
        return summary, summary_path, False

    json_outputs, csv_outputs = load_outputs(out_root, state)
    validate_shapes(
        csv_outputs,
        expected_n_folds=int(args.expected_n_folds),
        expected_validation_candidates=int(args.expected_validation_candidates),
        state=state,
    )
    validate_metric_sanity(csv_outputs, args.expected_test_months, state)
    expectations = split_expectations(
        csv_outputs,
        expected_n_folds=int(args.expected_n_folds),
        expected_test_months=args.expected_test_months,
        state=state,
    )
    validate_iteration_weighting_and_refit(csv_outputs, expectations, state)
    validate_independent_winners_populations_and_models(
        csv_outputs,
        expectations=expectations,
        json_outputs=json_outputs,
        out_root=out_root,
        panel_path=Path(args.panel_path),
        state=state,
    )
    validate_config_metadata(
        json_outputs,
        expected_feature_set=args.expected_feature_set,
        expected_panel_path=args.expected_panel_path,
        expected_random_seed=args.expected_random_seed,
        state=state,
    )
    validate_submission_representation_inventory(
        csv_outputs.get("selected_features.csv"),
        json_outputs,
        args.expected_feature_set,
        state,
    )
    observed_test_months = observed_test_months_from_fold_metrics(csv_outputs.get("lightgbm_metrics_by_fold.csv"))
    summary = build_validation_summary(
        state,
        expected_test_months=args.expected_test_months,
        observed_test_months=observed_test_months,
        key_metrics=capture_summary_metrics(csv_outputs),
    )
    write_json(summary_path, summary)
    return summary, summary_path, not state.errors


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""
    parser = build_argparser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    summary, summary_path, ok = run_validation(args)
    print(f"Validation status: {summary['validation_status']}")
    print(f"Summary: {summary_path}")
    if summary["warnings"]:
        print(f"Warnings: {len(summary['warnings'])}")
    if summary["errors"]:
        print(f"Errors: {len(summary['errors'])}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
