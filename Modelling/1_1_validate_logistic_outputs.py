#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Cara Lynch
# See the LICENSE file for details.
"""Validate completed Study 2 logistic baseline outputs."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

try:
    import pandas as pd
except Exception:  # pragma: no cover - exercised only when pandas is absent
    pd = None


logger = logging.getLogger(__name__)


REQUIRED_OUTPUTS = (
    "run_metadata.json",
    "resolved_config.json",
    "selected_features.csv",
    "rolling_origin_splits.csv",
    "logistic_hyperparameter_grid.csv",
    "logistic_validation_metrics_by_candidate.csv",
    "logistic_selected_hyperparameters_by_fold.csv",
    "logistic_metrics_by_fold.csv",
    "logistic_metrics_summary.csv",
    "logistic_coefficients_by_fold.csv",
)

OPTIONAL_RANKED_PREDICTIONS = "logistic_ranked_predictions.parquet"
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
VALIDATION_SUMMARY_NAME = "logistic_output_validation_summary.json"
PROPORTION_COLUMNS = (
    "precision_at_10pct",
    "recall_at_10pct",
    "average_precision",
    "auprc_ap",
    "auroc",
)
MCC_COLUMNS = (
    "mcc_at_10pct",
)
SUMMARY_METRIC_COLUMNS = (
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
    "positive_prevalence_macro_mean",
    "precision_at_10pct_macro_mean",
    "recall_at_10pct_macro_mean",
    "mcc_at_10pct_macro_mean",
    "average_precision_macro_mean",
    "auprc_ap_macro_mean",
    "auroc_macro_mean",
)
REFIT_PROVENANCE_COLUMNS = (
    "candidate_fit_start_prediction_month",
    "candidate_fit_end_prediction_month",
    "candidate_fit_includes_validation",
    "candidate_fit_n_rows",
    "candidate_fit_n_positive",
    "candidate_fit_n_negative",
    "final_refit_start_prediction_month",
    "final_refit_end_prediction_month",
    "final_refit_includes_validation",
    "final_refit_n_rows",
    "final_refit_n_positive",
    "final_refit_n_negative",
)
SELECTED_PARAMETER_COLUMNS = (
    "selected_candidate_id",
    "selected_C",
    "selected_class_weight",
)
ITERATION_PROVENANCE_COLUMNS = (
    "configured_max_iter",
    "actual_n_iter",
    "reached_max_iter",
)


@dataclass
class ValidationState:
    """Mutable validation findings and summary material."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_files: dict[str, dict[str, Any]] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    observed_test_months: list[str] = field(default_factory=list)
    summary_metrics: dict[str, Any] = field(default_factory=dict)
    ranked_predictions: dict[str, Any] = field(default_factory=dict)

    def fail(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def setup_logging(level: str) -> None:
    """Configure console logging."""
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def utc_now_iso() -> str:
    """Return a stable UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_expected_months(value: str) -> list[str]:
    """Parse comma-separated YYYY-MM expected test months."""
    months = [part.strip() for part in value.split(",") if part.strip()]
    if not months:
        raise argparse.ArgumentTypeError("--expected-test-months must contain at least one month.")
    bad = [month for month in months if not re.fullmatch(r"\d{4}-\d{2}", month)]
    if bad:
        raise argparse.ArgumentTypeError(
            "--expected-test-months must use YYYY-MM values. Problem value(s): " + ", ".join(bad)
        )
    return months


def normalise_identifier(value: Any) -> str:
    """Normalise fold and candidate identifiers read from CSV."""
    text = str(value).strip()
    if re.fullmatch(r"-?\d+\.0+", text):
        return str(int(float(text)))
    return text


def normalise_path_text(value: Any) -> str:
    """Normalise path text without resolving host-specific absolute paths."""
    return str(value).strip().replace("\\", "/").rstrip("/")


def values_match(expected: Any, observed: Any, *, is_path: bool = False) -> bool:
    """Return whether two expected/observed values match."""
    if is_path:
        return normalise_path_text(expected) == normalise_path_text(observed)
    return str(expected).strip() == str(observed).strip()


def is_truthy(value: Any) -> bool:
    """Parse CSV truthy values used by the model scripts."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "t"}


def coerce_float(value: Any) -> Optional[float]:
    """Convert a CSV value to float, returning None for missing/non-finite values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def coerce_int_like(value: Any) -> Optional[int]:
    """Convert an integer-like CSV value to int."""
    number = coerce_float(value)
    if number is None or not float(number).is_integer():
        return None
    return int(number)


def validation_selection_sort_key(row: Any, selection_metric: str) -> Optional[tuple[float, float, float, int, int]]:
    """Mirror the logistic runner's deterministic validation-candidate ordering."""
    metric_value = coerce_float(row.get(selection_metric))
    auroc = coerce_float(row.get("auroc"))
    c_value = coerce_float(row.get("C"))
    candidate_id = coerce_int_like(row.get("candidate_id"))
    class_weight = str(row.get("class_weight", "")).strip()
    if None in (metric_value, auroc, c_value, candidate_id) or not class_weight:
        return None
    return (
        -metric_value,
        -auroc,
        c_value,
        1 if class_weight == "none" else 2,
        candidate_id,
    )


def require_columns(table: Any, table_name: str, columns: Sequence[str], state: ValidationState) -> bool:
    """Record an error if a loaded CSV lacks required columns."""
    missing = [column for column in columns if column not in table.columns]
    if missing:
        state.fail(f"{table_name} is missing required column(s): {', '.join(missing)}")
        return False
    return True


def mark_file(path: Path, *, required: bool, state: ValidationState, label: str) -> bool:
    """Record basic existence and non-empty status for one expected file."""
    status: dict[str, Any] = {
        "path": str(path),
        "required": required,
        "exists": path.exists(),
        "non_empty": False,
    }
    state.checked_files[label] = status

    if not path.exists():
        if required:
            state.fail(f"Missing required file: {path}")
        return False
    if not path.is_file():
        state.fail(f"Expected a file but found a non-file path: {path}")
        return False

    try:
        size = path.stat().st_size
    except OSError as exc:
        state.fail(f"Could not stat {path}: {exc}")
        return False

    status["size_bytes"] = size
    status["non_empty"] = size > 0
    if required and size <= 0:
        state.fail(f"Required file is empty: {path}")
        return False
    return size > 0


def read_json(path: Path, label: str, state: ValidationState) -> Optional[dict[str, Any]]:
    """Read a JSON object, recording parse errors."""
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        state.fail(f"Could not parse JSON file {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        state.fail(f"{label} must contain a JSON object.")
        return None
    return payload


def read_csv(path: Path, label: str, state: ValidationState) -> Optional[Any]:
    """Read a CSV with pandas, recording parse errors."""
    if pd is None:
        state.fail("pandas is required to parse logistic CSV outputs.")
        return None
    try:
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:
        state.fail(f"Could not parse CSV file {path}: {exc}")
        return None
    state.row_counts[label] = int(len(table))
    return table


def load_outputs(out_root: Path, state: ValidationState) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load required CSV/JSON outputs where possible."""
    tables: dict[str, Any] = {}
    json_payloads: dict[str, Any] = {}

    for name in REQUIRED_OUTPUTS:
        path = out_root / name
        if not mark_file(path, required=True, state=state, label=name):
            continue
        if name.endswith(".json"):
            payload = read_json(path, name, state)
            if payload is not None:
                json_payloads[name] = payload
        elif name.endswith(".csv"):
            table = read_csv(path, name, state)
            if table is not None:
                tables[name] = table

    return tables, json_payloads


def check_ranked_predictions(out_root: Path, require_ranked: bool, state: ValidationState) -> None:
    """Record optional ranked-prediction presence, failing only when requested."""
    path = out_root / OPTIONAL_RANKED_PREDICTIONS
    present = mark_file(path, required=False, state=state, label=OPTIONAL_RANKED_PREDICTIONS)
    info = dict(state.checked_files[OPTIONAL_RANKED_PREDICTIONS])
    info["present"] = bool(path.exists())
    info["required_by_validator"] = bool(require_ranked)
    state.ranked_predictions = info
    if require_ranked and not present:
        state.fail(f"Ranked predictions were required but absent or empty: {path}")


def included_feature_count(selected_features: Any, state: ValidationState) -> Optional[int]:
    """Return the number of included features in selected_features.csv."""
    if not require_columns(selected_features, "selected_features.csv", ("feature_name", "included"), state):
        return None
    included_mask = selected_features["included"].map(is_truthy)
    count = int(included_mask.sum())
    if count <= 0:
        state.fail("selected_features.csv contains no included features.")
    return count


def validate_submission_representation_inventory(
    selected_features: Any,
    payloads: dict[str, Any],
    expected_feature_set: Optional[str],
    state: ValidationState,
) -> None:
    """Enforce the exact eight-feature submission-sensitivity contract."""
    metadata = payloads.get("run_metadata.json") or {}
    resolved = payloads.get("resolved_config.json") or {}
    config = get_config_object(resolved, state) if isinstance(resolved, dict) else {}
    feature_set = expected_feature_set or metadata.get("feature_set") or config.get("feature_set")
    expectation = SUBMISSION_REPRESENTATION_EXPECTATIONS.get(str(feature_set))
    if expectation is None:
        return
    if selected_features is None:
        state.fail("Submission-representation sensitivity lacks selected_features.csv.")
        return
    if not require_columns(
        selected_features,
        "selected_features.csv",
        ("feature_set", "feature_name", "included", "exclusion_reason"),
        state,
    ):
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
    if metadata.get("selected_features") != expected_features:
        state.fail("run_metadata.json selected_features differs from the exact eight-feature inventory.")
    resolved_definition = resolved.get("feature_set_definition", {}) if isinstance(resolved, dict) else {}
    if resolved_definition != definition:
        state.fail("Resolved-config and run-metadata feature-set definitions differ.")


def validate_required_shapes(
    tables: dict[str, Any],
    *,
    expected_n_folds: int,
    expected_validation_candidates: int,
    state: ValidationState,
) -> Optional[int]:
    """Validate required table row counts and selected feature count."""
    fold_metrics = tables.get("logistic_metrics_by_fold.csv")
    if fold_metrics is not None and len(fold_metrics) != expected_n_folds:
        state.fail(
            "logistic_metrics_by_fold.csv has the wrong number of rows: "
            f"observed={len(fold_metrics)}, expected={expected_n_folds}."
        )

    selected = tables.get("logistic_selected_hyperparameters_by_fold.csv")
    if selected is not None and len(selected) != expected_n_folds:
        state.fail(
            "logistic_selected_hyperparameters_by_fold.csv has the wrong number of rows: "
            f"observed={len(selected)}, expected={expected_n_folds}."
        )

    validation = tables.get("logistic_validation_metrics_by_candidate.csv")
    if validation is not None and len(validation) != expected_validation_candidates:
        state.fail(
            "logistic_validation_metrics_by_candidate.csv has the wrong number of rows: "
            f"observed={len(validation)}, expected={expected_validation_candidates}."
        )

    summary = tables.get("logistic_metrics_summary.csv")
    if summary is not None and len(summary) != 1:
        state.fail(
            "logistic_metrics_summary.csv must have exactly one row: "
            f"observed={len(summary)}."
        )

    n_included_features: Optional[int] = None
    selected_features = tables.get("selected_features.csv")
    if selected_features is not None:
        n_included_features = included_feature_count(selected_features, state)

    coefficients = tables.get("logistic_coefficients_by_fold.csv")
    if coefficients is not None and n_included_features is not None:
        expected_rows = int(expected_n_folds) * int(n_included_features)
        if len(coefficients) != expected_rows:
            state.fail(
                "logistic_coefficients_by_fold.csv has the wrong number of rows: "
                f"observed={len(coefficients)}, expected={expected_rows} "
                f"({expected_n_folds} folds x {n_included_features} included features)."
            )

    return n_included_features


def sorted_fold_ids(table: Any) -> list[str]:
    """Return fold identifiers sorted numerically where possible."""
    fold_ids = {normalise_identifier(value) for value in table["fold_id"] if normalise_identifier(value)}

    def sort_key(value: str) -> tuple[int, Any]:
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    return sorted(fold_ids, key=sort_key)


def validate_fold_month_structure(
    tables: dict[str, Any],
    *,
    expected_test_months: Sequence[str],
    expected_n_folds: int,
    state: ValidationState,
) -> None:
    """Validate rolling-origin month and split-role structure."""
    fold_metrics = tables.get("logistic_metrics_by_fold.csv")
    if fold_metrics is not None:
        month_col = "test_prediction_month" if "test_prediction_month" in fold_metrics.columns else "prediction_month"
        if month_col not in fold_metrics.columns:
            state.fail("logistic_metrics_by_fold.csv lacks test_prediction_month and prediction_month columns.")
        else:
            observed = [str(value) for value in fold_metrics[month_col].tolist()]
            state.observed_test_months = observed
            if observed != list(expected_test_months):
                state.fail(
                    "Fold metric test months do not match expected months in order: "
                    f"observed={observed}, expected={list(expected_test_months)}."
                )
        require_columns(fold_metrics, "logistic_metrics_by_fold.csv", ("fold_id", "split_role"), state)
        if "split_role" in fold_metrics.columns:
            bad_roles = sorted({str(role) for role in fold_metrics["split_role"] if str(role) != "test"})
            if bad_roles:
                state.fail(
                    "logistic_metrics_by_fold.csv must contain only test rows; "
                    f"observed non-test role(s): {', '.join(bad_roles)}."
                )

    splits = tables.get("rolling_origin_splits.csv")
    if splits is None:
        return
    required_columns = (
        "fold_id",
        "prediction_month",
        "split_role",
        "validation_prediction_month",
        "test_prediction_month",
    )
    if not require_columns(splits, "rolling_origin_splits.csv", required_columns, state):
        return

    roles = {str(role) for role in splits["split_role"]}
    missing_roles = sorted({"train", "validation", "test"} - roles)
    if missing_roles:
        state.fail("rolling_origin_splits.csv is missing split role(s): " + ", ".join(missing_roles))

    fold_rows = splits[splits["fold_id"].map(lambda value: bool(normalise_identifier(value)))]
    fold_ids = sorted_fold_ids(fold_rows) if len(fold_rows) else []
    if len(fold_ids) != expected_n_folds:
        state.fail(
            "rolling_origin_splits.csv has the wrong number of fold ids: "
            f"observed={len(fold_ids)}, expected={expected_n_folds}."
        )

    for fold_id in fold_ids:
        group = fold_rows[fold_rows["fold_id"].map(normalise_identifier) == fold_id]
        train_months = set(str(value) for value in group[group["split_role"] == "train"]["prediction_month"])
        validation_rows = group[group["split_role"] == "validation"]
        test_rows = group[group["split_role"] == "test"]
        validation_months = set(str(value) for value in validation_rows["prediction_month"])
        test_months = set(str(value) for value in test_rows["prediction_month"])

        if len(validation_rows) != 1 or len(validation_months) != 1:
            state.fail(
                f"rolling_origin_splits.csv fold {fold_id} must have exactly one validation row/month; "
                f"observed_rows={len(validation_rows)}, observed_months={sorted(validation_months)}."
            )
        if len(test_rows) != 1 or len(test_months) != 1:
            state.fail(
                f"rolling_origin_splits.csv fold {fold_id} must have exactly one test row/month; "
                f"observed_rows={len(test_rows)}, observed_months={sorted(test_months)}."
            )

        if "is_primary_test_month" in group.columns and len(test_rows):
            bad_primary = [
                str(row.get("prediction_month", ""))
                for _, row in test_rows.iterrows()
                if not is_truthy(row.get("is_primary_test_month", ""))
            ]
            if bad_primary:
                state.fail(
                    f"rolling_origin_splits.csv fold {fold_id} has test row(s) without "
                    f"is_primary_test_month=True: {', '.join(bad_primary)}."
                )

        overlap_test_train = sorted(test_months & train_months)
        if overlap_test_train:
            state.fail(
                f"rolling_origin_splits.csv fold {fold_id} has test month(s) also in train: "
                + ", ".join(overlap_test_train)
            )
        overlap_validation_train = sorted(validation_months & train_months)
        if overlap_validation_train:
            state.fail(
                f"rolling_origin_splits.csv fold {fold_id} has validation month(s) also in train: "
                + ", ".join(overlap_validation_train)
            )
        overlap_validation_test = sorted(validation_months & test_months)
        if overlap_validation_test:
            state.fail(
                f"rolling_origin_splits.csv fold {fold_id} has validation month(s) also in test: "
                + ", ".join(overlap_validation_test)
            )


def records_for_comparison(table: Any) -> list[dict[str, str]]:
    """Convert a pandas table to stable string records for split comparisons."""
    return [
        {str(column): str(row[column]).strip() for column in table.columns}
        for _, row in table.iterrows()
    ]


def validate_reference_splits(
    generated_splits: Optional[Any],
    reference_path: Optional[str],
    state: ValidationState,
) -> None:
    """Compare generated rolling-origin splits against an optional reference CSV."""
    if not reference_path:
        return
    path = Path(reference_path)
    if not mark_file(path, required=True, state=state, label="expected_reference_splits_csv"):
        return
    reference = read_csv(path, "expected_reference_splits_csv", state)
    if generated_splits is None or reference is None:
        return
    if list(generated_splits.columns) != list(reference.columns):
        state.fail(
            "rolling_origin_splits.csv columns do not match expected reference splits: "
            f"observed={list(generated_splits.columns)}, expected={list(reference.columns)}."
        )
        return
    if records_for_comparison(generated_splits) != records_for_comparison(reference):
        state.fail("rolling_origin_splits.csv does not match --expected-reference-splits-csv exactly.")


def validate_hyperparameters(tables: dict[str, Any], state: ValidationState) -> None:
    """Validate grid, validation-candidate, and selected-candidate consistency."""
    grid = tables.get("logistic_hyperparameter_grid.csv")
    validation = tables.get("logistic_validation_metrics_by_candidate.csv")
    selected = tables.get("logistic_selected_hyperparameters_by_fold.csv")
    if grid is None or validation is None or selected is None:
        return
    if not require_columns(grid, "logistic_hyperparameter_grid.csv", ("candidate_id",), state):
        return
    if not require_columns(
        validation,
        "logistic_validation_metrics_by_candidate.csv",
        ("fold_id", "candidate_id"),
        state,
    ):
        return
    if not require_columns(
        selected,
        "logistic_selected_hyperparameters_by_fold.csv",
        (
            "fold_id",
            "selected_candidate_id",
            "selected_C",
            "selected_class_weight",
            "selected_validation_metric_value",
        ),
        state,
    ):
        return

    grid_candidate_ids = [normalise_identifier(value) for value in grid["candidate_id"]]
    if len(grid_candidate_ids) != len(set(grid_candidate_ids)):
        state.fail("logistic_hyperparameter_grid.csv candidate_id values are not unique.")
    grid_candidate_set = set(grid_candidate_ids)

    for fold_id in sorted_fold_ids(validation):
        fold_validation = validation[validation["fold_id"].map(normalise_identifier) == fold_id]
        observed = [normalise_identifier(value) for value in fold_validation["candidate_id"]]
        observed_set = set(observed)
        if len(observed) != len(observed_set):
            state.fail(
                f"logistic_validation_metrics_by_candidate.csv fold {fold_id} has duplicate candidate_id rows."
            )
        if observed_set != grid_candidate_set:
            state.fail(
                f"logistic_validation_metrics_by_candidate.csv fold {fold_id} candidate ids do not match grid: "
                f"observed={sorted(observed_set)}, expected={sorted(grid_candidate_set)}."
            )

    for _, selected_row in selected.iterrows():
        fold_id = normalise_identifier(selected_row["fold_id"])
        selected_candidate_id = normalise_identifier(selected_row["selected_candidate_id"])
        fold_candidates = validation[validation["fold_id"].map(normalise_identifier) == fold_id]
        selected_validation = fold_candidates[
            fold_candidates["candidate_id"].map(normalise_identifier) == selected_candidate_id
        ]
        if len(selected_validation) == 0:
            state.fail(
                "Selected candidate does not appear in validation rows: "
                f"fold_id={fold_id}, selected_candidate_id={selected_candidate_id}."
            )
            continue
        validation_row = selected_validation.iloc[0]
        selected_c = coerce_float(selected_row.get("selected_C"))
        validation_c = coerce_float(validation_row.get("C"))
        if selected_c is None or validation_c is None or not math.isclose(selected_c, validation_c, rel_tol=1e-12, abs_tol=1e-12):
            state.fail(
                "selected_C does not match the selected validation candidate: "
                f"fold_id={fold_id}, candidate_id={selected_candidate_id}, "
                f"selected_C={selected_row.get('selected_C')!r}, validation_C={validation_row.get('C')!r}."
            )
        selected_class_weight = str(selected_row.get("selected_class_weight", "")).strip()
        validation_class_weight = str(validation_row.get("class_weight", "")).strip()
        if selected_class_weight != validation_class_weight:
            state.fail(
                "selected_class_weight does not match the selected validation candidate: "
                f"fold_id={fold_id}, candidate_id={selected_candidate_id}, "
                f"selected_class_weight={selected_class_weight!r}, validation_class_weight={validation_class_weight!r}."
            )

        validation_metric_names = set()
        if "selection_metric" in fold_candidates.columns:
            validation_metric_names = {
                str(value).strip()
                for value in fold_candidates["selection_metric"].tolist()
                if str(value).strip()
            }
        if len(validation_metric_names) > 1:
            state.fail(
                f"Validation candidates disagree on selection_metric for fold {fold_id}: "
                f"{sorted(validation_metric_names)}."
            )
            continue
        metric_name = next(iter(validation_metric_names), "average_precision")
        selected_metric_name = str(selected_row.get("selection_metric", "")).strip()
        if selected_metric_name and selected_metric_name != metric_name:
            state.fail(
                f"Selected selection_metric does not match validation candidates for fold {fold_id}: "
                f"selected={selected_metric_name!r}, validation={metric_name!r}."
            )
        if metric_name not in fold_candidates.columns:
            state.fail(
                f"Selected fold {fold_id} references selection metric {metric_name!r}, "
                "but that column is absent from logistic_validation_metrics_by_candidate.csv."
            )
            continue

        candidate_sort_keys: list[tuple[tuple[float, float, float, int, int], Any]] = []
        for _, candidate_row in fold_candidates.iterrows():
            sort_key = validation_selection_sort_key(candidate_row, metric_name)
            if sort_key is None:
                state.fail(
                    "Could not determine the validation selection order for "
                    f"fold_id={fold_id}, candidate_id={candidate_row.get('candidate_id')!r}; "
                    f"expected numeric {metric_name!r}, 'auroc', 'C', and candidate_id, plus class_weight."
                )
                continue
            candidate_sort_keys.append((sort_key, candidate_row))
        if len(candidate_sort_keys) != len(fold_candidates):
            continue
        winner_row = min(candidate_sort_keys, key=lambda item: item[0])[1]
        winner_candidate_id = normalise_identifier(winner_row.get("candidate_id", ""))
        if selected_candidate_id != winner_candidate_id:
            state.fail(
                "Selected candidate is not the deterministic validation winner: "
                f"fold_id={fold_id}, selected_candidate_id={selected_candidate_id}, "
                f"validation_winner_candidate_id={winner_candidate_id}."
            )

        selected_value = coerce_float(selected_row["selected_validation_metric_value"])
        candidate_value = coerce_float(validation_row[metric_name])
        if selected_value is None or candidate_value is None:
            state.fail(
                f"Could not compare selected validation metric for fold {fold_id}, "
                f"candidate {selected_candidate_id}, metric {metric_name}."
            )
            continue
        if not math.isclose(selected_value, candidate_value, rel_tol=1e-9, abs_tol=1e-9):
            state.fail(
                "selected_validation_metric_value does not equal the selected candidate metric: "
                f"fold_id={fold_id}, candidate_id={selected_candidate_id}, metric={metric_name}, "
                f"selected={selected_value}, validation_row={candidate_value}."
            )


def split_expectations_by_fold(splits: Any, state: ValidationState) -> dict[str, dict[str, str]]:
    """Return expected train/validation/test month metadata by fold."""
    expectations: dict[str, dict[str, str]] = {}
    if splits is None:
        return expectations
    required = (
        "fold_id",
        "prediction_month",
        "split_role",
        "train_start_prediction_month",
        "train_end_prediction_month",
        "validation_prediction_month",
        "test_prediction_month",
    )
    if not require_columns(splits, "rolling_origin_splits.csv", required, state):
        return expectations
    fold_rows = splits[splits["fold_id"].map(lambda value: bool(normalise_identifier(value)))]
    for fold_id in sorted_fold_ids(fold_rows) if len(fold_rows) else []:
        group = fold_rows[fold_rows["fold_id"].map(normalise_identifier) == fold_id]
        train_rows = group[group["split_role"] == "train"]
        validation_rows = group[group["split_role"] == "validation"]
        test_rows = group[group["split_role"] == "test"]
        if len(train_rows) == 0 or len(validation_rows) != 1 or len(test_rows) != 1:
            continue
        expectations[fold_id] = {
            "train_start_prediction_month": str(train_rows["prediction_month"].min()),
            "train_end_prediction_month": str(train_rows["prediction_month"].max()),
            "validation_prediction_month": str(validation_rows.iloc[0]["prediction_month"]),
            "test_prediction_month": str(test_rows.iloc[0]["prediction_month"]),
        }
    return expectations


def validate_refit_row(
    row: Any,
    *,
    table_name: str,
    fold_id: str,
    expected: dict[str, str],
    state: ValidationState,
) -> None:
    """Validate one row's final train-plus-validation refit provenance."""
    label = f"{table_name} fold {fold_id}"
    train_start = expected.get("train_start_prediction_month", str(row.get("train_start_prediction_month", "")).strip())
    train_end = expected.get("train_end_prediction_month", str(row.get("train_end_prediction_month", "")).strip())
    validation_month = expected.get("validation_prediction_month", str(row.get("validation_prediction_month", "")).strip())
    test_month = expected.get("test_prediction_month", str(row.get("test_prediction_month", "")).strip())

    if not is_truthy(row.get("final_refit_includes_validation", "")):
        state.fail(f"{label} was not recorded as refit on training plus validation rows.")
    if is_truthy(row.get("candidate_fit_includes_validation", "")):
        state.fail(f"{label} records candidate-selection fitting as including validation rows.")

    comparisons = (
        ("candidate_fit_start_prediction_month", train_start),
        ("candidate_fit_end_prediction_month", train_end),
        ("final_refit_start_prediction_month", train_start),
        ("final_refit_end_prediction_month", validation_month),
    )
    for column, expected_value in comparisons:
        observed = str(row.get(column, "")).strip()
        if expected_value and observed != expected_value:
            state.fail(f"{label} {column}={observed!r}; expected {expected_value!r}.")

    final_end = str(row.get("final_refit_end_prediction_month", "")).strip()
    if final_end and test_month and final_end >= test_month:
        state.fail(
            f"{label} final refit month range includes or reaches the held-out test month: "
            f"final_refit_end={final_end}, test_month={test_month}."
        )

    candidate_rows = coerce_int_like(row.get("candidate_fit_n_rows"))
    candidate_pos = coerce_int_like(row.get("candidate_fit_n_positive"))
    candidate_neg = coerce_int_like(row.get("candidate_fit_n_negative"))
    final_rows = coerce_int_like(row.get("final_refit_n_rows"))
    final_pos = coerce_int_like(row.get("final_refit_n_positive"))
    final_neg = coerce_int_like(row.get("final_refit_n_negative"))
    for column, value in (
        ("candidate_fit_n_rows", candidate_rows),
        ("candidate_fit_n_positive", candidate_pos),
        ("candidate_fit_n_negative", candidate_neg),
        ("final_refit_n_rows", final_rows),
        ("final_refit_n_positive", final_pos),
        ("final_refit_n_negative", final_neg),
    ):
        if value is None or value < 0:
            state.fail(f"{label} has invalid {column}: {row.get(column)!r}.")
    if None not in (candidate_rows, candidate_pos, candidate_neg) and candidate_rows != candidate_pos + candidate_neg:
        state.fail(f"{label} candidate training-only label counts do not sum to candidate_fit_n_rows.")
    if None not in (final_rows, final_pos, final_neg) and final_rows != final_pos + final_neg:
        state.fail(f"{label} final train-plus-validation label counts do not sum to final_refit_n_rows.")
    if None not in (candidate_rows, final_rows) and final_rows <= candidate_rows:
        state.fail(f"{label} final_refit_n_rows must be greater than candidate_fit_n_rows.")
    if None not in (candidate_pos, final_pos) and final_pos < candidate_pos:
        state.fail(f"{label} final_refit_n_positive must be at least candidate_fit_n_positive.")
    if None not in (candidate_neg, final_neg) and final_neg < candidate_neg:
        state.fail(f"{label} final_refit_n_negative must be at least candidate_fit_n_negative.")


def validate_refit_provenance(tables: dict[str, Any], state: ValidationState) -> None:
    """Validate train-only candidate fitting and final train-plus-validation refit metadata."""
    expectations = split_expectations_by_fold(tables.get("rolling_origin_splits.csv"), state)

    validation = tables.get("logistic_validation_metrics_by_candidate.csv")
    if validation is not None and require_columns(
        validation,
        "logistic_validation_metrics_by_candidate.csv",
        (
            "fold_id",
            "candidate_id",
            "candidate_fit_start_prediction_month",
            "candidate_fit_end_prediction_month",
            "candidate_fit_includes_validation",
            "candidate_fit_n_rows",
            "candidate_fit_n_positive",
            "candidate_fit_n_negative",
        ),
        state,
    ):
        for _, row in validation.iterrows():
            fold_id = normalise_identifier(row.get("fold_id", ""))
            expected = expectations.get(fold_id, {})
            label = f"logistic_validation_metrics_by_candidate.csv fold {fold_id} candidate {normalise_identifier(row.get('candidate_id', ''))}"
            if is_truthy(row.get("candidate_fit_includes_validation", "")):
                state.fail(f"{label} records candidate fitting as including validation rows.")
            if expected:
                if str(row.get("candidate_fit_start_prediction_month", "")).strip() != expected["train_start_prediction_month"]:
                    state.fail(f"{label} candidate_fit_start_prediction_month is inconsistent with rolling_origin_splits.csv.")
                if str(row.get("candidate_fit_end_prediction_month", "")).strip() != expected["train_end_prediction_month"]:
                    state.fail(f"{label} candidate_fit_end_prediction_month is inconsistent with rolling_origin_splits.csv.")

    selected = tables.get("logistic_selected_hyperparameters_by_fold.csv")
    fold_metrics = tables.get("logistic_metrics_by_fold.csv")
    coefficients = tables.get("logistic_coefficients_by_fold.csv")
    selected_by_fold: dict[str, dict[str, str]] = {}
    if selected is not None and require_columns(
        selected,
        "logistic_selected_hyperparameters_by_fold.csv",
        ("fold_id", "test_prediction_month", *SELECTED_PARAMETER_COLUMNS, *REFIT_PROVENANCE_COLUMNS),
        state,
    ):
        for _, row in selected.iterrows():
            fold_id = normalise_identifier(row.get("fold_id", ""))
            selected_by_fold[fold_id] = {column: str(row.get(column, "")).strip() for column in SELECTED_PARAMETER_COLUMNS}
            validate_refit_row(
                row,
                table_name="logistic_selected_hyperparameters_by_fold.csv",
                fold_id=fold_id,
                expected=expectations.get(fold_id, {}),
                state=state,
            )

    for table_name, table in (
        ("logistic_metrics_by_fold.csv", fold_metrics),
        ("logistic_coefficients_by_fold.csv", coefficients),
    ):
        if table is None:
            continue
        required = ("fold_id", *SELECTED_PARAMETER_COLUMNS, *REFIT_PROVENANCE_COLUMNS)
        if table_name == "logistic_metrics_by_fold.csv":
            required = ("fold_id", "test_prediction_month", *SELECTED_PARAMETER_COLUMNS, *REFIT_PROVENANCE_COLUMNS)
        if not require_columns(table, table_name, required, state):
            continue
        for _, row in table.iterrows():
            fold_id = normalise_identifier(row.get("fold_id", ""))
            validate_refit_row(
                row,
                table_name=table_name,
                fold_id=fold_id,
                expected=expectations.get(fold_id, {}),
                state=state,
            )
            expected_selected = selected_by_fold.get(fold_id)
            if expected_selected:
                for column in SELECTED_PARAMETER_COLUMNS:
                    observed = str(row.get(column, "")).strip()
                    if observed != expected_selected[column]:
                        state.fail(
                            f"{table_name} fold {fold_id} {column}={observed!r}; "
                            f"expected selected-hyperparameter value {expected_selected[column]!r}."
                        )

    ranked_path = None
    if state.ranked_predictions.get("present"):
        ranked_path = Path(str(state.ranked_predictions.get("path", "")))
    if ranked_path and pd is not None and ranked_path.exists():
        try:
            ranked = pd.read_parquet(ranked_path)
        except Exception as exc:
            state.fail(f"Could not parse ranked predictions Parquet {ranked_path}: {exc}")
            return
        if not require_columns(
            ranked,
            "logistic_ranked_predictions.parquet",
            ("fold_id", "split_role", *SELECTED_PARAMETER_COLUMNS, *REFIT_PROVENANCE_COLUMNS),
            state,
        ):
            return
        bad_roles = sorted({str(role) for role in ranked["split_role"] if str(role) != "test"})
        if bad_roles:
            state.fail("logistic_ranked_predictions.parquet must contain only test rows.")
        for _, row in ranked.iterrows():
            fold_id = normalise_identifier(row.get("fold_id", ""))
            validate_refit_row(
                row,
                table_name="logistic_ranked_predictions.parquet",
                fold_id=fold_id,
                expected=expectations.get(fold_id, {}),
                state=state,
            )
            expected_selected = selected_by_fold.get(fold_id)
            if expected_selected:
                for column in SELECTED_PARAMETER_COLUMNS:
                    observed = str(row.get(column, "")).strip()
                    if observed != expected_selected[column]:
                        state.fail(
                            f"logistic_ranked_predictions.parquet fold {fold_id} {column}={observed!r}; "
                            f"expected selected-hyperparameter value {expected_selected[column]!r}."
                        )


def validate_iteration_provenance(tables: dict[str, Any], state: ValidationState) -> None:
    """Validate observed final-fit iterations, while flagging legacy roots without them."""
    table_names = (
        "logistic_selected_hyperparameters_by_fold.csv",
        "logistic_metrics_by_fold.csv",
        "logistic_coefficients_by_fold.csv",
    )
    available = {name: tables.get(name) for name in table_names if tables.get(name) is not None}
    if not available:
        return

    columns_present = {
        name: set(ITERATION_PROVENANCE_COLUMNS).issubset(set(table.columns))
        for name, table in available.items()
    }
    if not any(columns_present.values()):
        state.warn(
            "Observed logistic convergence iterations are not persisted in this legacy output root; "
            "configured max_iter is not an observed iteration count."
        )
        return
    for name, present in columns_present.items():
        if not present:
            state.fail(
                f"{name} must include all iteration provenance columns once any output records them: "
                f"{', '.join(ITERATION_PROVENANCE_COLUMNS)}."
            )
    if not all(columns_present.values()):
        return

    expected_by_fold: dict[str, tuple[int, int, bool]] = {}
    for table_name, table in available.items():
        for _, row in table.iterrows():
            fold_id = normalise_identifier(row.get("fold_id", ""))
            configured = coerce_int_like(row.get("configured_max_iter"))
            actual = coerce_int_like(row.get("actual_n_iter"))
            reached_text = str(row.get("reached_max_iter", "")).strip().lower()
            if reached_text not in {"true", "false", "1", "0", "yes", "no", "y", "n", "t", "f"}:
                state.fail(f"{table_name} fold {fold_id} has invalid reached_max_iter={row.get('reached_max_iter')!r}.")
                continue
            reached = is_truthy(row.get("reached_max_iter"))
            if configured is None or configured < 1 or actual is None or actual < 1 or actual > configured:
                state.fail(
                    f"{table_name} fold {fold_id} has invalid iteration provenance: "
                    f"actual_n_iter={row.get('actual_n_iter')!r}, "
                    f"configured_max_iter={row.get('configured_max_iter')!r}."
                )
                continue
            if reached != (actual >= configured):
                state.fail(
                    f"{table_name} fold {fold_id} reached_max_iter={row.get('reached_max_iter')!r} "
                    "is inconsistent with actual_n_iter and configured_max_iter."
                )
            observed = (configured, actual, reached)
            expected = expected_by_fold.setdefault(fold_id, observed)
            if observed != expected:
                state.fail(
                    f"{table_name} fold {fold_id} iteration provenance {observed!r} "
                    f"does not match the other fold-level outputs {expected!r}."
                )


def validate_metric_sanity(
    tables: dict[str, Any],
    state: ValidationState,
    *,
    smoke_mode: bool = False,
) -> None:
    """Validate metric ranges and top-10% label conventions.

    Smoke runners use label-stratified row limiting, so their observed test
    prevalence is intentionally not the production 0.10 prevalence.  The
    smoke mode exception is narrowly scoped to that expected-prevalence
    assertion; all other metric checks remain active.
    """
    fold_metrics = tables.get("logistic_metrics_by_fold.csv")
    if fold_metrics is None:
        return
    required_columns = (
        "fold_id",
        "n_users",
        "n_positive",
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
        "auroc",
    )
    if not require_columns(fold_metrics, "logistic_metrics_by_fold.csv", required_columns, state):
        return

    for _, row in fold_metrics.iterrows():
        fold_id = normalise_identifier(row.get("fold_id", ""))
        test_month = str(row.get("test_prediction_month", row.get("prediction_month", ""))).strip()
        label = f"fold {fold_id}" + (f" ({test_month})" if test_month else "")

        n_users = coerce_float(row["n_users"])
        n_positive = coerce_float(row["n_positive"])
        if n_users is None or n_users <= 0:
            state.fail(f"logistic_metrics_by_fold.csv {label} has non-positive n_users: {row['n_users']}.")
        if n_positive is None or n_positive <= 0:
            state.fail(f"logistic_metrics_by_fold.csv {label} has non-positive n_positive: {row['n_positive']}.")

        prevalence = coerce_float(row["positive_prevalence"])
        if prevalence is None:
            state.fail(f"logistic_metrics_by_fold.csv {label} has invalid positive_prevalence.")
        elif not smoke_mode and abs(prevalence - 0.10) > 0.001:
            state.fail(
                f"logistic_metrics_by_fold.csv {label} positive_prevalence is not approximately 0.10: "
                f"{prevalence}."
            )

        for column in PROPORTION_COLUMNS:
            if column not in fold_metrics.columns:
                continue
            value = coerce_float(row[column])
            if value is None or value < -1e-12 or value > 1.0 + 1e-12:
                state.fail(
                    f"logistic_metrics_by_fold.csv {label} metric {column} is outside [0, 1]: "
                    f"{row[column]}."
                )

        for column in MCC_COLUMNS:
            if column not in fold_metrics.columns:
                continue
            value = coerce_float(row[column])
            if value is None or value < -1.0 - 1e-12 or value > 1.0 + 1e-12:
                state.fail(
                    f"logistic_metrics_by_fold.csv {label} metric {column} is outside [-1, 1] "
                    f"or non-finite: {row[column]}."
                )

        k_at_10pct = coerce_float(row["k_at_10pct"])
        top_k_positive = coerce_float(row["top_k_positive_count"])
        n_negative = coerce_float(row.get("n_negative"))
        tp = coerce_float(row["tp_at_10pct"])
        fp = coerce_float(row["fp_at_10pct"])
        fn = coerce_float(row["fn_at_10pct"])
        tn = coerce_float(row["tn_at_10pct"])
        precision = coerce_float(row["precision_at_10pct"])
        recall = coerce_float(row["recall_at_10pct"])
        if k_at_10pct is not None and n_positive is not None:
            if not math.isclose(k_at_10pct, n_positive, rel_tol=0.0, abs_tol=1e-9):
                state.warn(
                    f"logistic_metrics_by_fold.csv {label} has k_at_10pct={k_at_10pct} "
                    f"but n_positive={n_positive}; check top-k convention."
                )
            elif precision is not None and recall is not None and not math.isclose(
                precision, recall, rel_tol=1e-9, abs_tol=1e-9
            ):
                state.warn(
                    f"logistic_metrics_by_fold.csv {label} has k_at_10pct == n_positive "
                    f"but precision_at_10pct ({precision}) != recall_at_10pct ({recall})."
                )
        if None not in (tp, fp, fn, tn, top_k_positive, k_at_10pct, n_positive, n_negative):
            if not math.isclose(tp, top_k_positive, rel_tol=0.0, abs_tol=1e-9):
                state.fail(f"logistic_metrics_by_fold.csv {label} has tp_at_10pct inconsistent with top_k_positive_count.")
            if not math.isclose(fp, k_at_10pct - tp, rel_tol=0.0, abs_tol=1e-9):
                state.fail(f"logistic_metrics_by_fold.csv {label} has fp_at_10pct inconsistent with top-k counts.")
            if not math.isclose(fn, n_positive - tp, rel_tol=0.0, abs_tol=1e-9):
                state.fail(f"logistic_metrics_by_fold.csv {label} has fn_at_10pct inconsistent with positive counts.")
            if not math.isclose(tn, n_negative - fp, rel_tol=0.0, abs_tol=1e-9):
                state.fail(f"logistic_metrics_by_fold.csv {label} has tn_at_10pct inconsistent with negative counts.")
            denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            if denominator <= 0:
                state.fail(f"logistic_metrics_by_fold.csv {label} has zero MCC denominator.")


def get_config_object(resolved_config: dict[str, Any], state: ValidationState) -> dict[str, Any]:
    """Return the nested config object from resolved_config.json."""
    config = resolved_config.get("config")
    if not isinstance(config, dict):
        state.fail("resolved_config.json is missing a top-level config object.")
        return {}
    return config


def metadata_target_score_col(metadata: dict[str, Any]) -> Optional[Any]:
    """Return the target score column from metadata under any current key."""
    for key in ("target_score_col", "target_score_col_for_descriptive_summaries"):
        if key in metadata:
            return metadata.get(key)
    return None


def validate_agreement(
    *,
    label: str,
    left_source: str,
    left_value: Any,
    right_source: str,
    right_value: Any,
    state: ValidationState,
    is_path: bool = False,
) -> None:
    """Validate that two present metadata/config values agree."""
    if left_value in (None, "") or right_value in (None, ""):
        return
    if not values_match(left_value, right_value, is_path=is_path):
        state.fail(
            f"{left_source} and {right_source} disagree on {label}: "
            f"{left_value!r} != {right_value!r}."
        )


def validate_expected_value(
    *,
    label: str,
    expected: Any,
    observed: Sequence[tuple[str, Any]],
    state: ValidationState,
    is_path: bool = False,
) -> None:
    """Validate an optional CLI expected value against present config/metadata values."""
    if expected is None:
        return
    present = [(source, value) for source, value in observed if value not in (None, "")]
    if not present:
        state.warn(f"No config or metadata field was available to validate expected {label}={expected!r}.")
        return
    for source, value in present:
        if not values_match(expected, value, is_path=is_path):
            state.fail(f"{source} {label} does not match expected value: observed={value!r}, expected={expected!r}.")


def validate_config_metadata(
    payloads: dict[str, Any],
    *,
    expected_feature_set: Optional[str],
    expected_panel_path: Optional[str],
    expected_reference_splits_csv: Optional[str],
    expected_target_score_col: str,
    expected_random_seed: Optional[int],
    state: ValidationState,
) -> None:
    """Validate resolved config and run metadata consistency."""
    resolved = payloads.get("resolved_config.json")
    metadata = payloads.get("run_metadata.json")
    if resolved is None or metadata is None:
        return
    config = get_config_object(resolved, state)
    if not isinstance(metadata, dict):
        state.fail("run_metadata.json must contain a JSON object.")
        return

    validate_agreement(
        label="feature_set",
        left_source="resolved_config.json",
        left_value=config.get("feature_set"),
        right_source="run_metadata.json",
        right_value=metadata.get("feature_set"),
        state=state,
    )
    validate_agreement(
        label="panel_path",
        left_source="resolved_config.json",
        left_value=config.get("panel_path"),
        right_source="run_metadata.json",
        right_value=metadata.get("panel_path"),
        state=state,
        is_path=True,
    )
    validate_agreement(
        label="feature_schema_csv",
        left_source="resolved_config.json",
        left_value=config.get("feature_schema_csv"),
        right_source="run_metadata.json",
        right_value=metadata.get("feature_schema_csv"),
        state=state,
        is_path=True,
    )
    validate_agreement(
        label="output_root",
        left_source="resolved_config.json",
        left_value=config.get("out_root"),
        right_source="run_metadata.json",
        right_value=metadata.get("output_root"),
        state=state,
        is_path=True,
    )
    validate_agreement(
        label="target_score_col",
        left_source="resolved_config.json",
        left_value=config.get("target_score_col"),
        right_source="run_metadata.json",
        right_value=metadata_target_score_col(metadata),
        state=state,
    )
    validate_agreement(
        label="random_seed",
        left_source="resolved_config.json",
        left_value=config.get("random_seed"),
        right_source="run_metadata.json",
        right_value=metadata.get("random_seed"),
        state=state,
    )

    validate_expected_value(
        label="feature_set",
        expected=expected_feature_set,
        observed=(("resolved_config.json", config.get("feature_set")), ("run_metadata.json", metadata.get("feature_set"))),
        state=state,
    )
    validate_expected_value(
        label="panel_path",
        expected=expected_panel_path,
        observed=(("resolved_config.json", config.get("panel_path")), ("run_metadata.json", metadata.get("panel_path"))),
        state=state,
        is_path=True,
    )
    validate_expected_value(
        label="reference_splits_csv",
        expected=expected_reference_splits_csv,
        observed=(("resolved_config.json", config.get("reference_splits_csv")),),
        state=state,
        is_path=True,
    )
    validate_expected_value(
        label="target_score_col",
        expected=expected_target_score_col,
        observed=(
            ("resolved_config.json", config.get("target_score_col")),
            ("run_metadata.json", metadata_target_score_col(metadata)),
        ),
        state=state,
    )
    validate_expected_value(
        label="random_seed",
        expected=expected_random_seed,
        observed=(("resolved_config.json", config.get("random_seed")), ("run_metadata.json", metadata.get("random_seed"))),
        state=state,
    )

    package_versions = metadata.get("package_versions")
    if not isinstance(package_versions, dict) or not package_versions:
        state.fail("run_metadata.json is missing non-empty package_versions metadata.")

    git = metadata.get("git")
    if not isinstance(git, dict):
        state.fail("run_metadata.json is missing git metadata.")
    else:
        if "dirty" not in git:
            state.warn("run_metadata.json git metadata lacks a dirty flag.")
        elif bool(git.get("dirty")):
            state.warn("run_metadata.json records a dirty git worktree; outputs may still be valid.")
        if "commit" not in git:
            state.warn("run_metadata.json git metadata lacks a commit field.")


def artifact_smoke_mode(payloads: dict[str, Any]) -> bool:
    """Return whether the artefact explicitly records that smoke mode was used."""
    metadata = payloads.get("run_metadata.json")
    return isinstance(metadata, dict) and is_truthy(metadata.get("smoke", False))


def capture_summary_metrics(tables: dict[str, Any], state: ValidationState) -> None:
    """Copy key metrics from logistic_metrics_summary.csv into validation summary."""
    summary = tables.get("logistic_metrics_summary.csv")
    if summary is None or len(summary) != 1:
        return
    row = summary.iloc[0].to_dict()
    state.summary_metrics = {
        column: row.get(column, "")
        for column in SUMMARY_METRIC_COLUMNS
        if column in summary.columns
    }


def build_validation_summary(
    *,
    out_root: Path,
    expected_test_months: Sequence[str],
    state: ValidationState,
) -> dict[str, Any]:
    """Build machine-readable validation summary."""
    return {
        "validation_status": "failed" if state.errors else "passed",
        "output_root": str(out_root),
        "checked_files": state.checked_files,
        "row_counts": state.row_counts,
        "expected_test_months": list(expected_test_months),
        "observed_test_months": state.observed_test_months,
        "warnings": state.warnings,
        "errors": state.errors,
        "key_metrics": state.summary_metrics,
        "ranked_predictions": state.ranked_predictions,
        "timestamp": utc_now_iso(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def print_human_summary(summary: dict[str, Any], summary_path: Path, wrote_summary: bool) -> None:
    """Print a concise CLI summary."""
    status = str(summary["validation_status"]).upper()
    print(f"Logistic output validation: {status}")
    print(f"Output root: {summary['output_root']}")
    print(f"Observed test months: {', '.join(summary.get('observed_test_months') or ['<unavailable>'])}")
    row_counts = summary.get("row_counts", {})
    if row_counts:
        compact_counts = ", ".join(f"{name}={count}" for name, count in sorted(row_counts.items()))
        print(f"Row counts: {compact_counts}")
    ranked = summary.get("ranked_predictions") or {}
    print(
        "Ranked predictions: "
        + ("present" if ranked.get("present") else "absent")
        + (" (required)" if ranked.get("required_by_validator") else "")
    )
    if summary.get("warnings"):
        print("Warnings:")
        for warning in summary["warnings"]:
            print(f"  WARN: {warning}")
    if summary.get("errors"):
        print("Errors:")
        for error in summary["errors"]:
            print(f"  FAIL: {error}")
    if wrote_summary:
        print(f"Validation summary: {summary_path}")
    else:
        print(f"Validation summary not written: {summary_path}")


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Validate completed Study 2 logistic baseline output artefacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-root", required=True, help="Completed logistic baseline output directory.")
    parser.add_argument("--expected-feature-set", default=None, help="Expected feature set recorded in outputs.")
    parser.add_argument("--expected-panel-path", default=None, help="Expected modelling_panel.parquet path.")
    parser.add_argument(
        "--expected-reference-splits-csv",
        default=None,
        help="Expected rolling_origin_splits.csv reference to compare exactly.",
    )
    parser.add_argument(
        "--expected-target-score-col",
        default="target_score",
        help="Expected target-score column recorded in config/metadata.",
    )
    parser.add_argument("--expected-random-seed", type=int, default=None, help="Expected deterministic random seed.")
    parser.add_argument("--expected-n-folds", type=int, default=6, help="Expected number of test folds.")
    parser.add_argument(
        "--expected-validation-candidates",
        type=int,
        default=48,
        help="Expected rows in logistic_validation_metrics_by_candidate.csv.",
    )
    parser.add_argument(
        "--expected-test-months",
        type=parse_expected_months,
        default=DEFAULT_EXPECTED_TEST_MONTHS,
        help="Comma-separated expected rolling-origin test months.",
    )
    parser.add_argument(
        "--require-ranked-predictions",
        action="store_true",
        help="Fail if logistic_ranked_predictions.parquet is absent or empty.",
    )
    parser.add_argument(
        "--write-validation-summary",
        default=None,
        help=f"Validation summary JSON path. Defaults to <out-root>/{VALIDATION_SUMMARY_NAME}.",
    )
    parser.add_argument(
        "--overwrite-summary",
        action="store_true",
        help="Overwrite an existing validation summary JSON.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate CLI argument combinations."""
    if int(args.expected_n_folds) < 1:
        parser.error("--expected-n-folds must be at least 1.")
    if int(args.expected_validation_candidates) < 1:
        parser.error("--expected-validation-candidates must be at least 1.")
    if isinstance(args.expected_test_months, str):
        try:
            args.expected_test_months = parse_expected_months(args.expected_test_months)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))


def run_validation(args: argparse.Namespace) -> tuple[dict[str, Any], Path, bool]:
    """Run validation and return summary plus write status."""
    out_root = Path(args.out_root)
    summary_path = (
        Path(args.write_validation_summary)
        if args.write_validation_summary
        else out_root / VALIDATION_SUMMARY_NAME
    )
    state = ValidationState()

    if summary_path.exists() and not args.overwrite_summary:
        state.fail(f"Validation summary already exists; pass --overwrite-summary to replace it: {summary_path}")

    if not out_root.exists():
        state.fail(f"Output root does not exist: {out_root}")
    elif not out_root.is_dir():
        state.fail(f"--out-root is not a directory: {out_root}")

    tables, payloads = load_outputs(out_root, state)
    check_ranked_predictions(out_root, bool(args.require_ranked_predictions), state)
    n_included_features = validate_required_shapes(
        tables,
        expected_n_folds=int(args.expected_n_folds),
        expected_validation_candidates=int(args.expected_validation_candidates),
        state=state,
    )
    if n_included_features is not None:
        state.row_counts["included_features"] = int(n_included_features)

    validate_fold_month_structure(
        tables,
        expected_test_months=args.expected_test_months,
        expected_n_folds=int(args.expected_n_folds),
        state=state,
    )
    validate_reference_splits(
        tables.get("rolling_origin_splits.csv"),
        args.expected_reference_splits_csv,
        state,
    )
    validate_hyperparameters(tables, state)
    validate_refit_provenance(tables, state)
    validate_iteration_provenance(tables, state)
    validate_config_metadata(
        payloads,
        expected_feature_set=args.expected_feature_set,
        expected_panel_path=args.expected_panel_path,
        expected_reference_splits_csv=args.expected_reference_splits_csv,
        expected_target_score_col=args.expected_target_score_col,
        expected_random_seed=args.expected_random_seed,
        state=state,
    )
    validate_submission_representation_inventory(
        tables.get("selected_features.csv"),
        payloads,
        args.expected_feature_set,
        state,
    )
    smoke_mode = artifact_smoke_mode(payloads)
    if smoke_mode:
        state.warn("Smoke artefact detected; production prevalence assertion is not applied.")
    validate_metric_sanity(tables, state, smoke_mode=smoke_mode)
    capture_summary_metrics(tables, state)

    summary = build_validation_summary(
        out_root=out_root,
        expected_test_months=args.expected_test_months,
        state=state,
    )
    wrote_summary = False
    if not summary_path.exists() or args.overwrite_summary:
        write_json(summary_path, summary)
        wrote_summary = True
    return summary, summary_path, wrote_summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = build_argparser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    setup_logging(args.log_level)

    summary, summary_path, wrote_summary = run_validation(args)
    print_human_summary(summary, summary_path, wrote_summary)
    return 1 if summary.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
