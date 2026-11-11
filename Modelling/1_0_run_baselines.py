#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Cara Lynch
# See the LICENSE file for details.
"""Run non-learned Study 2 heuristic baseline ranking evaluations.

The baselines in this script do not fit model parameters. They rank each
eligible user-month prediction population using pre-existing feature-month
signals, then evaluate those rankings against ``label_high_engagement``.

Primary comparison metrics are restricted to the accepted expanding-window
rolling-origin test months. Optional all-month metrics are written separately
and labelled as diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import duckdb


logger = logging.getLogger(__name__)


EXPECTED_PREDICTION_MONTHS_2021 = tuple(f"2021-{month:02d}" for month in range(2, 13))
PRIMARY_TEST_MONTHS_2021 = tuple(f"2021-{month:02d}" for month in range(7, 13))
TOP_10_FRACTION = 0.10
HASH_DOUBLE_DENOMINATOR = 18446744073709551616.0
PR_AUC_DEFINITION = "average_precision_ranked_list"
AUROC_DEFINITION = "deterministic_tie_broken_mann_whitney_rank_sum"
TIE_HANDLING = (
    "Sort by ranking_score DESC, then deterministic seeded tie_break_score DESC, "
    "then user_id ASC. random_ranking uses its deterministic pseudo-random score "
    "as ranking_score and user_id as the final collision tie-breaker."
)

BASELINE_DEFINITIONS = (
    {
        "baseline_name": "random_ranking",
        "status": "implemented",
        "ranking_signal": "deterministic pseudo-random score by user_id, prediction_month, and --random-seed",
        "primary_score_column": "",
        "auxiliary_score_column": "",
        "description": "Ranks eligible users randomly within each prediction month using a reproducible hash score.",
    },
    {
        "baseline_name": "persistence_binary",
        "status": "implemented",
        "ranking_signal": "prior top-10% indicator from feature_received_engagement_per_activity_prev",
        "primary_score_column": "feature_received_engagement_per_activity_prev_top10_indicator",
        "auxiliary_score_column": "feature_received_engagement_per_activity_prev",
        "description": (
            "Ranks users by whether their feature-month received-engagement-per-activity "
            "score is in the top 10% among the current prediction population."
        ),
    },
    {
        "baseline_name": "historical_engagement_rate",
        "status": "implemented",
        "ranking_signal": "feature_received_engagement_per_activity_prev",
        "primary_score_column": "feature_received_engagement_per_activity_prev",
        "auxiliary_score_column": "",
        "description": "Ranks users by prior received engagement per prior activity.",
    },
    {
        "baseline_name": "historical_engagement_total",
        "status": "implemented",
        "ranking_signal": "feature_received_engagement_total_prev",
        "primary_score_column": "feature_received_engagement_total_prev",
        "auxiliary_score_column": "",
        "description": "Ranks users by total received engagement in the feature month.",
    },
    {
        "baseline_name": "activity_total",
        "status": "implemented",
        "ranking_signal": "feature_n_total_activity_prev",
        "primary_score_column": "feature_n_total_activity_prev",
        "auxiliary_score_column": "",
        "description": "Ranks users by total authored activity in the feature month.",
    },
    {
        "baseline_name": "activity_frequency",
        "status": "implemented",
        "ranking_signal": "feature_n_active_days_prev",
        "primary_score_column": "feature_n_active_days_prev",
        "auxiliary_score_column": "",
        "description": "Ranks users by number of active days in the feature month.",
    },
)

BASELINE_DEFINITION_FIELDS = (
    "baseline_name",
    "status",
    "ranking_signal",
    "primary_score_column",
    "auxiliary_score_column",
    "tie_handling",
    "description",
)

REQUIRED_PANEL_COLUMNS = (
    "user_id",
    "feature_month",
    "prediction_month",
    "label_high_engagement",
    "target_score",
    "feature_received_engagement_per_activity_prev",
    "feature_received_engagement_total_prev",
    "feature_n_total_activity_prev",
    "feature_n_active_days_prev",
)

RANKING_SOURCE_COLUMNS = (
    "feature_received_engagement_per_activity_prev",
    "feature_received_engagement_total_prev",
    "feature_n_total_activity_prev",
    "feature_n_active_days_prev",
)

KNOWN_OUTPUTS = (
    "run_metadata.json",
    "resolved_config.json",
    "baseline_definitions.csv",
    "rolling_origin_splits.csv",
    "baseline_metrics_by_month_all.csv",
    "baseline_metrics_by_fold.csv",
    "baseline_metrics_summary.csv",
    "baseline_topk_by_month.csv",
    "baseline_ranked_predictions.parquet",
)


@dataclass(frozen=True)
class DuckDBConfig:
    """DuckDB runtime configuration."""

    memory_limit: str
    threads: int
    temp_dir: str


@dataclass(frozen=True)
class BaselineConfig:
    """Resolved baseline evaluation configuration."""

    panel_path: str
    feature_schema_csv: Optional[str]
    out_root: str
    random_seed: int
    target_score_col: str
    describe_only: bool
    overwrite: bool
    write_ranked_predictions: bool
    top_k_fraction: float
    write_all_month_diagnostics: bool
    smoke: bool
    smoke_max_rows: int
    min_train_months: int
    validation_window_months: int
    test_window_months: int


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


def quote_sql_string(value: str | Path) -> str:
    """Return a single-quoted SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def quote_identifier(name: str) -> str:
    """Return a double-quoted SQL identifier."""
    return '"' + str(name).replace('"', '""') + '"'


def sql_string_list(values: Sequence[str]) -> str:
    """Return a SQL tuple of string literals."""
    if not values:
        raise ValueError("Cannot build SQL string list from an empty sequence.")
    return "(" + ", ".join(quote_sql_string(value) for value in values) + ")"


def ensure_dir(path: str | Path) -> None:
    """Create a directory if needed."""
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_json_dump(obj: dict, path: str | Path) -> None:
    """Write stable JSON."""
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def write_csv_rows(path: str | Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    """Write dictionaries as CSV."""
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def copy_query_to_csv(con: duckdb.DuckDBPyConnection, query: str, path: str | Path) -> None:
    """Write a DuckDB query result to CSV."""
    ensure_dir(Path(path).parent)
    con.execute(
        f"""
        COPY (
          {query}
        )
        TO {quote_sql_string(path)}
        (FORMAT CSV, HEADER, DELIMITER ',')
        """
    )


def git_value(args: Sequence[str]) -> Optional[str]:
    """Return a best-effort git command value."""
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def git_metadata() -> dict[str, object]:
    """Return best-effort git provenance."""
    status = git_value(["status", "--short"])
    return {
        "commit": git_value(["rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_short": status,
    }


def package_versions() -> dict[str, Optional[str]]:
    """Return package versions relevant to this script."""
    return {
        "python": sys.version.replace("\n", " "),
        "duckdb": getattr(duckdb, "__version__", "installed"),
    }


def known_output_paths(out_root: Path) -> list[Path]:
    """Return all known output paths for collision checks."""
    return [out_root / name for name in KNOWN_OUTPUTS]


def check_output_collisions(paths: Sequence[Path], overwrite: bool) -> None:
    """Fail if outputs exist and overwrite is disabled."""
    if overwrite:
        return
    existing = sorted(path for path in paths if path.exists())
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing output(s). Pass --overwrite to replace:\n  - "
            + "\n  - ".join(str(path) for path in existing)
        )


def remove_outputs(paths: Sequence[Path], overwrite: bool) -> None:
    """Remove deterministic file outputs when overwrite is enabled."""
    if not overwrite:
        return
    for path in sorted({p for p in paths if p.exists()}):
        if path.is_dir():
            raise IsADirectoryError(f"Refusing to remove directory output path: {path}")
        logger.info("Removing existing output: %s", path)
        path.unlink()


def configure_duckdb(con: duckdb.DuckDBPyConnection, duck: DuckDBConfig) -> None:
    """Apply DuckDB runtime settings."""
    ensure_dir(duck.temp_dir)
    con.execute(f"SET memory_limit={quote_sql_string(duck.memory_limit)}")
    con.execute(f"SET threads={int(duck.threads)}")
    con.execute(f"SET temp_directory={quote_sql_string(duck.temp_dir)}")
    con.execute("SET preserve_insertion_order=false")
    con.execute("PRAGMA disable_object_cache")


def baseline_definition_rows() -> list[dict[str, str]]:
    """Return baseline definitions with the shared tie-handling note attached."""
    rows: list[dict[str, str]] = []
    for row in BASELINE_DEFINITIONS:
        enriched = dict(row)
        enriched["tie_handling"] = TIE_HANDLING
        rows.append(enriched)
    return rows


def month_add(month: str, offset: int) -> str:
    """Add calendar months to a YYYY-MM string."""
    year_s, month_s = month.split("-")
    total = int(year_s) * 12 + int(month_s) - 1 + offset
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def previous_month(month: str) -> str:
    """Return the previous calendar month for a YYYY-MM string."""
    return month_add(month, -1)


def validate_cli(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate CLI arguments before work starts."""
    if not Path(args.panel_path).exists():
        parser.error(f"--panel-path not found: {args.panel_path}")
    if args.feature_schema_csv and not Path(args.feature_schema_csv).exists():
        parser.error(f"--feature-schema-csv not found: {args.feature_schema_csv}")
    if not (0.0 < float(args.top_k_fraction) < 1.0):
        parser.error("--top-k-fraction must be greater than 0 and less than 1.")
    if not math.isclose(float(args.top_k_fraction), TOP_10_FRACTION, rel_tol=0.0, abs_tol=1e-12):
        parser.error(
            "--top-k-fraction is currently fixed at 0.10 because the constructed "
            "label and Precision@10%/Recall@10% convention use the monthly positive count."
        )
    if int(args.smoke_max_rows) < 0:
        parser.error("--smoke-max-rows cannot be negative.")
    if int(args.min_train_months) < 1:
        parser.error("--min-train-months must be at least 1.")
    if int(args.validation_window_months) != 1:
        parser.error("--validation-window-months currently supports only 1 for the locked 2021 protocol.")
    if int(args.test_window_months) != 1:
        parser.error("--test-window-months currently supports only 1 for the locked 2021 protocol.")
    if int(args.duckdb_threads) < 1:
        parser.error("--duckdb-threads must be at least 1.")


def create_panel_source_view(con: duckdb.DuckDBPyConnection, cfg: BaselineConfig) -> None:
    """Create the input panel view, optionally with a smoke row limit."""
    relation = f"read_parquet({quote_sql_string(cfg.panel_path)})"
    if cfg.smoke and int(cfg.smoke_max_rows) > 0:
        relation = f"(SELECT * FROM {relation} LIMIT {int(cfg.smoke_max_rows)})"
    con.execute(f"CREATE OR REPLACE TEMP VIEW panel_source AS SELECT * FROM {relation}")


def relation_columns(con: duckdb.DuckDBPyConnection, relation_name: str) -> list[str]:
    """Return relation columns."""
    rows = con.execute(f"DESCRIBE SELECT * FROM {quote_identifier(relation_name)} LIMIT 0").fetchall()
    return [str(row[0]) for row in rows]


def observed_prediction_months(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Return distinct prediction months in the panel source."""
    rows = con.execute(
        """
        SELECT DISTINCT CAST(prediction_month AS VARCHAR) AS prediction_month
        FROM panel_source
        ORDER BY 1
        """
    ).fetchall()
    return [str(row[0]) for row in rows]


def prediction_month_feature_map(con: duckdb.DuckDBPyConnection) -> dict[str, str]:
    """Return the single feature month for each prediction month."""
    rows = con.execute(
        """
        SELECT
          CAST(prediction_month AS VARCHAR) AS prediction_month,
          MIN(CAST(feature_month AS VARCHAR)) AS feature_month,
          COUNT(DISTINCT CAST(feature_month AS VARCHAR)) AS n_feature_months
        FROM panel_source
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchall()
    bad = [str(row[0]) for row in rows if int(row[2]) != 1]
    if bad:
        raise ValueError(
            "Each prediction month must map to exactly one feature month. "
            "Problem month(s): " + ", ".join(bad)
        )
    return {str(row[0]): str(row[1]) for row in rows}


def validate_panel_preconditions(
    con: duckdb.DuckDBPyConnection,
    cfg: BaselineConfig,
    columns: Sequence[str],
    evaluated_months: Sequence[str],
) -> None:
    """Validate panel shape and required columns before metrics are computed."""
    column_set = set(columns)
    required = set(REQUIRED_PANEL_COLUMNS) | {cfg.target_score_col}
    missing = sorted(column for column in required if column not in column_set)
    if missing:
        raise ValueError("Panel is missing required column(s): " + ", ".join(missing))

    n_rows = int(con.execute("SELECT COUNT(*) FROM panel_source").fetchone()[0])
    if n_rows < 1:
        raise ValueError("Panel has no rows.")

    observed_months = observed_prediction_months(con)
    missing_expected = [month for month in EXPECTED_PREDICTION_MONTHS_2021 if month not in observed_months]
    if missing_expected:
        raise ValueError(
            "Panel is missing prediction month(s) required for the 2021 rolling-origin protocol: "
            + ", ".join(missing_expected)
        )

    evaluated_sql = sql_string_list(evaluated_months)
    invalid_label_count = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM panel_source
            WHERE CAST(prediction_month AS VARCHAR) IN {evaluated_sql}
              AND CAST(label_high_engagement AS INTEGER) NOT IN (0, 1)
            """
        ).fetchone()[0]
    )
    if invalid_label_count:
        raise ValueError(f"label_high_engagement must be binary 0/1. Invalid row count: {invalid_label_count}")

    month_balance = con.execute(
        f"""
        SELECT
          CAST(prediction_month AS VARCHAR) AS prediction_month,
          COUNT(*) AS n_users,
          SUM(CAST(label_high_engagement AS INTEGER)) AS n_positive
        FROM panel_source
        WHERE CAST(prediction_month AS VARCHAR) IN {evaluated_sql}
        GROUP BY 1
        HAVING SUM(CAST(label_high_engagement AS INTEGER)) <= 0
          OR SUM(CAST(label_high_engagement AS INTEGER)) >= COUNT(*)
        ORDER BY 1
        """
    ).fetchall()
    if month_balance:
        details = ", ".join(f"{row[0]} (n={row[1]}, positives={row[2]})" for row in month_balance)
        raise ValueError("Every evaluated prediction month must have at least one positive and one negative: " + details)

    missing_ranking: list[str] = []
    for column in RANKING_SOURCE_COLUMNS:
        n_missing = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM panel_source
                WHERE CAST(prediction_month AS VARCHAR) IN {evaluated_sql}
                  AND {quote_identifier(column)} IS NULL
                """
            ).fetchone()[0]
        )
        if n_missing:
            missing_ranking.append(f"{column}={n_missing}")
    if missing_ranking:
        raise ValueError("Ranking source columns must not contain missing values: " + ", ".join(missing_ranking))

    feature_map = prediction_month_feature_map(con)
    wrong_feature_months = [
        f"{prediction_month}->{feature_month}"
        for prediction_month, feature_month in feature_map.items()
        if prediction_month in EXPECTED_PREDICTION_MONTHS_2021 and feature_month != previous_month(prediction_month)
    ]
    if wrong_feature_months:
        raise ValueError(
            "Expected one-month-lag feature_month values for the 2021 protocol. Problem mapping(s): "
            + ", ".join(wrong_feature_months)
        )

    logger.info("Panel preconditions passed for %s row(s) and %s evaluated month(s).", n_rows, len(evaluated_months))


def build_rolling_origin_splits(
    feature_map: dict[str, str],
    cfg: BaselineConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build fold split rows and a compact primary-test lookup."""
    months = list(EXPECTED_PREDICTION_MONTHS_2021)
    if int(cfg.min_train_months) != 4:
        logger.warning(
            "Using non-default --min-train-months=%s; the locked 2021 protocol uses 4.",
            int(cfg.min_train_months),
        )
    split_rows: list[dict[str, object]] = []
    primary_test_rows: list[dict[str, object]] = []

    first_test_index = int(cfg.min_train_months) + int(cfg.validation_window_months)
    fold_id = 1
    for test_index in range(first_test_index, len(months), int(cfg.test_window_months)):
        train_months = months[: test_index - int(cfg.validation_window_months)]
        validation_month = months[test_index - int(cfg.validation_window_months)]
        test_month = months[test_index]
        train_start = train_months[0]
        train_end = train_months[-1]
        fold_common = {
            "fold_id": fold_id,
            "train_start_prediction_month": train_start,
            "train_end_prediction_month": train_end,
            "validation_prediction_month": validation_month,
            "test_prediction_month": test_month,
        }

        for prediction_month in train_months:
            split_rows.append(
                {
                    **fold_common,
                    "prediction_month": prediction_month,
                    "feature_month": feature_map[prediction_month],
                    "split_role": "train",
                    "is_primary_test_month": False,
                }
            )
        split_rows.append(
            {
                **fold_common,
                "prediction_month": validation_month,
                "feature_month": feature_map[validation_month],
                "split_role": "validation",
                "is_primary_test_month": False,
            }
        )
        split_rows.append(
            {
                **fold_common,
                "prediction_month": test_month,
                "feature_month": feature_map[test_month],
                "split_role": "test",
                "is_primary_test_month": True,
            }
        )
        primary_test_rows.append(
            {
                **fold_common,
                "prediction_month": test_month,
                "feature_month": feature_map[test_month],
                "split_role": "test",
                "is_primary_test_month": True,
            }
        )
        fold_id += 1

    observed_tests = tuple(row["test_prediction_month"] for row in primary_test_rows)
    if observed_tests != PRIMARY_TEST_MONTHS_2021:
        raise ValueError(
            "Rolling-origin test months do not match the locked 2021 protocol. "
            f"Observed={observed_tests}; expected={PRIMARY_TEST_MONTHS_2021}"
        )

    if cfg.write_all_month_diagnostics:
        for prediction_month, feature_month in sorted(feature_map.items()):
            split_rows.append(
                {
                    "fold_id": "",
                    "prediction_month": prediction_month,
                    "feature_month": feature_month,
                    "split_role": "diagnostic_only",
                    "train_start_prediction_month": "",
                    "train_end_prediction_month": "",
                    "validation_prediction_month": "",
                    "test_prediction_month": "",
                    "is_primary_test_month": False,
                }
            )

    return split_rows, primary_test_rows


def split_fieldnames() -> tuple[str, ...]:
    """Return the rolling-origin split artefact columns."""
    return (
        "fold_id",
        "prediction_month",
        "feature_month",
        "split_role",
        "train_start_prediction_month",
        "train_end_prediction_month",
        "validation_prediction_month",
        "test_prediction_month",
        "is_primary_test_month",
    )


def register_primary_test_months(con: duckdb.DuckDBPyConnection, rows: Sequence[dict[str, object]]) -> None:
    """Register primary test fold metadata in DuckDB."""
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE primary_test_months (
          baseline_join_key INTEGER,
          fold_id INTEGER,
          prediction_month VARCHAR,
          feature_month VARCHAR,
          train_start_prediction_month VARCHAR,
          train_end_prediction_month VARCHAR,
          validation_prediction_month VARCHAR,
          test_prediction_month VARCHAR
        )
        """
    )
    con.executemany(
        """
        INSERT INTO primary_test_months VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                1,
                int(row["fold_id"]),
                str(row["prediction_month"]),
                str(row["feature_month"]),
                str(row["train_start_prediction_month"]),
                str(row["train_end_prediction_month"]),
                str(row["validation_prediction_month"]),
                str(row["test_prediction_month"]),
            )
            for row in rows
        ],
    )


def deterministic_score_sql(seed: int, scope: str, user_expr: str = "user_id", month_expr: str = "prediction_month") -> str:
    """Return a deterministic hash score in [0, 1)."""
    payload = (
        f"CAST({user_expr} AS VARCHAR) || '|' || "
        f"CAST({month_expr} AS VARCHAR) || '|' || "
        f"{quote_sql_string(scope)} || '|' || {quote_sql_string(str(seed))}"
    )
    return f"(CAST(hash({payload}) AS DOUBLE) / {HASH_DOUBLE_DENOMINATOR:.1f})"


def create_ranked_baseline_tables(
    con: duckdb.DuckDBPyConnection,
    cfg: BaselineConfig,
    evaluated_months: Sequence[str],
) -> None:
    """Materialise scored and ranked baseline rows in DuckDB."""
    months_sql = sql_string_list(evaluated_months)
    target_score_col_sql = quote_identifier(cfg.target_score_col)
    random_score = deterministic_score_sql(cfg.random_seed, "random_ranking")
    prior_tie_score = deterministic_score_sql(cfg.random_seed, "persistence_prior_top10")
    persistence_tie = deterministic_score_sql(cfg.random_seed, "persistence_binary")
    rate_tie = deterministic_score_sql(cfg.random_seed, "historical_engagement_rate")
    total_tie = deterministic_score_sql(cfg.random_seed, "historical_engagement_total")
    activity_total_tie = deterministic_score_sql(cfg.random_seed, "activity_total")
    activity_frequency_tie = deterministic_score_sql(cfg.random_seed, "activity_frequency")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE eval_panel AS
        SELECT
          CAST(user_id AS VARCHAR) AS user_id,
          CAST(feature_month AS VARCHAR) AS feature_month,
          CAST(prediction_month AS VARCHAR) AS prediction_month,
          CAST(label_high_engagement AS INTEGER) AS label_high_engagement,
          CAST(target_score AS DOUBLE) AS target_score,
          CAST({target_score_col_sql} AS DOUBLE) AS target_score_for_summary,
          CAST(feature_received_engagement_per_activity_prev AS DOUBLE)
            AS feature_received_engagement_per_activity_prev,
          CAST(feature_received_engagement_total_prev AS DOUBLE)
            AS feature_received_engagement_total_prev,
          CAST(feature_n_total_activity_prev AS DOUBLE)
            AS feature_n_total_activity_prev,
          CAST(feature_n_active_days_prev AS DOUBLE)
            AS feature_n_active_days_prev
        FROM panel_source
        WHERE CAST(prediction_month AS VARCHAR) IN {months_sql}
        """
    )

    logger.info("Deriving persistence prior-high-engagement indicator.")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE eval_panel_with_persistence AS
        SELECT
          *,
          CASE WHEN persistence_prior_rank <= persistence_prior_k THEN 1.0 ELSE 0.0 END
            AS persistence_prior_high_engagement,
          feature_received_engagement_per_activity_prev AS persistence_prior_continuous_score
        FROM (
          SELECT
            *,
            ROW_NUMBER() OVER (
              PARTITION BY prediction_month, feature_month
              ORDER BY feature_received_engagement_per_activity_prev DESC,
                       {prior_tie_score} DESC,
                       user_id ASC
            ) AS persistence_prior_rank,
            CAST(CEIL(
              COUNT(*) OVER (PARTITION BY prediction_month, feature_month) * {float(cfg.top_k_fraction):.12f}
            ) AS BIGINT) AS persistence_prior_k
          FROM eval_panel
        )
        """
    )

    logger.info("Scoring heuristic baselines.")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE baseline_scores AS
        SELECT
          prediction_month,
          feature_month,
          user_id,
          'random_ranking' AS baseline_name,
          {random_score} AS ranking_score,
          0.0 AS tie_break_score,
          CAST(NULL AS DOUBLE) AS continuous_prior_score,
          label_high_engagement,
          target_score,
          target_score_for_summary
        FROM eval_panel_with_persistence

        UNION ALL
        SELECT
          prediction_month,
          feature_month,
          user_id,
          'persistence_binary' AS baseline_name,
          persistence_prior_high_engagement AS ranking_score,
          {persistence_tie} AS tie_break_score,
          persistence_prior_continuous_score AS continuous_prior_score,
          label_high_engagement,
          target_score,
          target_score_for_summary
        FROM eval_panel_with_persistence

        UNION ALL
        SELECT
          prediction_month,
          feature_month,
          user_id,
          'historical_engagement_rate' AS baseline_name,
          feature_received_engagement_per_activity_prev AS ranking_score,
          {rate_tie} AS tie_break_score,
          CAST(NULL AS DOUBLE) AS continuous_prior_score,
          label_high_engagement,
          target_score,
          target_score_for_summary
        FROM eval_panel_with_persistence

        UNION ALL
        SELECT
          prediction_month,
          feature_month,
          user_id,
          'historical_engagement_total' AS baseline_name,
          feature_received_engagement_total_prev AS ranking_score,
          {total_tie} AS tie_break_score,
          CAST(NULL AS DOUBLE) AS continuous_prior_score,
          label_high_engagement,
          target_score,
          target_score_for_summary
        FROM eval_panel_with_persistence

        UNION ALL
        SELECT
          prediction_month,
          feature_month,
          user_id,
          'activity_total' AS baseline_name,
          feature_n_total_activity_prev AS ranking_score,
          {activity_total_tie} AS tie_break_score,
          CAST(NULL AS DOUBLE) AS continuous_prior_score,
          label_high_engagement,
          target_score,
          target_score_for_summary
        FROM eval_panel_with_persistence

        UNION ALL
        SELECT
          prediction_month,
          feature_month,
          user_id,
          'activity_frequency' AS baseline_name,
          feature_n_active_days_prev AS ranking_score,
          {activity_frequency_tie} AS tie_break_score,
          CAST(NULL AS DOUBLE) AS continuous_prior_score,
          label_high_engagement,
          target_score,
          target_score_for_summary
        FROM eval_panel_with_persistence
        """
    )

    logger.info("Ranking baseline predictions within baseline x prediction_month.")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE baseline_ranked AS
        SELECT
          prediction_month,
          feature_month,
          p.fold_id,
          CASE WHEN p.fold_id IS NOT NULL THEN 'test' ELSE 'diagnostic_only' END AS split_role,
          CASE
            WHEN p.fold_id IS NOT NULL THEN 'primary_rolling_origin_test'
            ELSE 'diagnostic_all_prediction_months'
          END AS row_scope,
          user_id,
          baseline_name,
          ranking_score,
          tie_break_score,
          continuous_prior_score,
          ROW_NUMBER() OVER (
            PARTITION BY baseline_name, prediction_month
            ORDER BY ranking_score DESC, tie_break_score DESC, user_id ASC
          ) AS rank_within_month,
          CAST(label_high_engagement AS INTEGER) AS label_high_engagement,
          target_score,
          target_score_for_summary,
          COUNT(*) OVER (PARTITION BY baseline_name, prediction_month) AS n_users_window,
          SUM(CAST(label_high_engagement AS INTEGER)) OVER (
            PARTITION BY baseline_name, prediction_month
          ) AS n_positive_window,
          SUM(CAST(label_high_engagement AS INTEGER)) OVER (
            PARTITION BY baseline_name, prediction_month
            ORDER BY ranking_score DESC, tie_break_score DESC, user_id ASC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
          ) AS cumulative_positive
        FROM baseline_scores b
        LEFT JOIN primary_test_months p
          USING (prediction_month, feature_month)
        """
    )


def create_metric_tables(con: duckdb.DuckDBPyConnection, cfg: BaselineConfig) -> None:
    """Create month, fold, and summary metric tables."""
    logger.info("Computing month-wise ranking metrics.")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE baseline_metrics_by_month AS
        WITH grouped AS (
          SELECT
            baseline_name,
            prediction_month,
            MAX(feature_month) AS feature_month,
            MAX(n_users_window) AS n_users,
            MAX(n_positive_window) AS n_positive,
            MAX(n_users_window) - MAX(n_positive_window) AS n_negative,
            SUM(CASE WHEN rank_within_month <= n_positive_window THEN label_high_engagement ELSE 0 END)
              AS top_k_positive_count,
            AVG(target_score_for_summary) AS mean_target_score_all,
            AVG(CASE WHEN rank_within_month <= n_positive_window THEN target_score_for_summary ELSE NULL END)
              AS mean_target_score_top_k,
            SUM(
              CASE
                WHEN label_high_engagement = 1
                THEN CAST(cumulative_positive AS DOUBLE) / CAST(rank_within_month AS DOUBLE)
                ELSE 0.0
              END
            ) / CAST(MAX(n_positive_window) AS DOUBLE) AS average_precision,
            (
              SUM(
                CASE
                  WHEN label_high_engagement = 1
                  THEN CAST(n_users_window - rank_within_month + 1 AS DOUBLE)
                  ELSE 0.0
                END
              )
              - (CAST(MAX(n_positive_window) AS DOUBLE) * (CAST(MAX(n_positive_window) AS DOUBLE) + 1.0) / 2.0)
            )
            / (
              CAST(MAX(n_positive_window) AS DOUBLE)
              * CAST(MAX(n_users_window) - MAX(n_positive_window) AS DOUBLE)
            ) AS auroc
          FROM baseline_ranked
          GROUP BY 1, 2
        ),
        topk_counts AS (
          SELECT
            *,
            n_positive AS k_at_10pct,
            top_k_positive_count AS tp_at_10pct
          FROM grouped
        ),
        confusion_counts AS (
          SELECT
            *,
            k_at_10pct - tp_at_10pct AS fp_at_10pct,
            n_positive - tp_at_10pct AS fn_at_10pct
          FROM topk_counts
        ),
        counts AS (
          SELECT
            *,
            n_users - tp_at_10pct - fp_at_10pct - fn_at_10pct AS tn_at_10pct
          FROM confusion_counts
        )
        SELECT
          baseline_name,
          CAST(NULL AS INTEGER) AS fold_id,
          prediction_month,
          feature_month,
          'diagnostic_only' AS split_role,
          'diagnostic_all_prediction_months' AS metric_scope,
          CAST(n_users AS BIGINT) AS n_users,
          CAST(n_positive AS BIGINT) AS n_positive,
          CAST(n_negative AS BIGINT) AS n_negative,
          CAST(n_positive AS DOUBLE) / CAST(n_users AS DOUBLE) AS positive_prevalence,
          CAST(k_at_10pct AS BIGINT) AS k_at_10pct,
          CAST(top_k_positive_count AS BIGINT) AS top_k_positive_count,
          CAST(tp_at_10pct AS BIGINT) AS tp_at_10pct,
          CAST(fp_at_10pct AS BIGINT) AS fp_at_10pct,
          CAST(fn_at_10pct AS BIGINT) AS fn_at_10pct,
          CAST(tn_at_10pct AS BIGINT) AS tn_at_10pct,
          CAST(top_k_positive_count AS DOUBLE) / CAST(n_positive AS DOUBLE) AS precision_at_10pct,
          CAST(top_k_positive_count AS DOUBLE) / CAST(n_positive AS DOUBLE) AS recall_at_10pct,
          CASE
            WHEN (
              CAST((tp_at_10pct + fp_at_10pct) AS DOUBLE)
              * CAST((tp_at_10pct + fn_at_10pct) AS DOUBLE)
              * CAST((tn_at_10pct + fp_at_10pct) AS DOUBLE)
              * CAST((tn_at_10pct + fn_at_10pct) AS DOUBLE)
            ) = 0.0
            THEN CAST('nan' AS DOUBLE)
            ELSE (
              CAST(tp_at_10pct AS DOUBLE) * CAST(tn_at_10pct AS DOUBLE)
              - CAST(fp_at_10pct AS DOUBLE) * CAST(fn_at_10pct AS DOUBLE)
            ) / SQRT(
              CAST((tp_at_10pct + fp_at_10pct) AS DOUBLE)
              * CAST((tp_at_10pct + fn_at_10pct) AS DOUBLE)
              * CAST((tn_at_10pct + fp_at_10pct) AS DOUBLE)
              * CAST((tn_at_10pct + fn_at_10pct) AS DOUBLE)
            )
          END AS mcc_at_10pct,
          average_precision,
          average_precision AS auprc_ap,
          auroc,
          mean_target_score_top_k,
          mean_target_score_all,
          'k_at_10pct_equals_monthly_n_positive' AS top_k_rule,
          'average_precision_ranked_list' AS pr_auc_definition,
          'deterministic_tie_broken_mann_whitney_rank_sum' AS auroc_definition
        FROM counts
        """
    )

    logger.info("Extracting primary rolling-origin fold metrics.")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE baseline_metrics_by_fold AS
        SELECT
          m.baseline_name,
          p.fold_id,
          m.prediction_month,
          p.test_prediction_month,
          m.feature_month,
          'test' AS split_role,
          'primary_rolling_origin_test' AS metric_scope,
          p.train_start_prediction_month,
          p.train_end_prediction_month,
          p.validation_prediction_month,
          m.n_users,
          m.n_positive,
          m.n_negative,
          m.positive_prevalence,
          m.k_at_10pct,
          m.top_k_positive_count,
          m.tp_at_10pct,
          m.fp_at_10pct,
          m.fn_at_10pct,
          m.tn_at_10pct,
          m.precision_at_10pct,
          m.recall_at_10pct,
          m.mcc_at_10pct,
          m.average_precision,
          m.auprc_ap,
          m.auroc,
          m.mean_target_score_top_k,
          m.mean_target_score_all,
          m.top_k_rule,
          m.pr_auc_definition,
          m.auroc_definition
        FROM baseline_metrics_by_month m
        INNER JOIN primary_test_months p
          ON m.prediction_month = p.test_prediction_month
         AND m.feature_month = p.feature_month
        """
    )

    logger.info("Summarising primary fold metrics.")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE baseline_metrics_summary AS
        SELECT
          baseline_name,
          'primary_rolling_origin_test' AS metric_scope,
          'test' AS source_split_role,
          COUNT(*) AS n_folds,
          MIN(test_prediction_month) AS test_prediction_month_start,
          MAX(test_prediction_month) AS test_prediction_month_end,
          SUM(n_users) AS n_users_pooled,
          SUM(n_positive) AS n_positive_pooled,
          SUM(k_at_10pct) AS k_at_10pct_pooled,
          SUM(top_k_positive_count) AS top_k_positive_count_pooled,
          SUM(tp_at_10pct) AS tp_at_10pct_pooled,
          SUM(fp_at_10pct) AS fp_at_10pct_pooled,
          SUM(fn_at_10pct) AS fn_at_10pct_pooled,
          SUM(tn_at_10pct) AS tn_at_10pct_pooled,
          CASE
            WHEN (
              CAST((SUM(tp_at_10pct) + SUM(fp_at_10pct)) AS DOUBLE)
              * CAST((SUM(tp_at_10pct) + SUM(fn_at_10pct)) AS DOUBLE)
              * CAST((SUM(tn_at_10pct) + SUM(fp_at_10pct)) AS DOUBLE)
              * CAST((SUM(tn_at_10pct) + SUM(fn_at_10pct)) AS DOUBLE)
            ) = 0.0
            THEN CAST('nan' AS DOUBLE)
            ELSE (
              CAST(SUM(tp_at_10pct) AS DOUBLE) * CAST(SUM(tn_at_10pct) AS DOUBLE)
              - CAST(SUM(fp_at_10pct) AS DOUBLE) * CAST(SUM(fn_at_10pct) AS DOUBLE)
            ) / SQRT(
              CAST((SUM(tp_at_10pct) + SUM(fp_at_10pct)) AS DOUBLE)
              * CAST((SUM(tp_at_10pct) + SUM(fn_at_10pct)) AS DOUBLE)
              * CAST((SUM(tn_at_10pct) + SUM(fp_at_10pct)) AS DOUBLE)
              * CAST((SUM(tn_at_10pct) + SUM(fn_at_10pct)) AS DOUBLE)
            )
          END AS mcc_at_10pct_pooled,
          AVG(CAST(n_users AS DOUBLE)) AS n_users_macro_mean,
          STDDEV_SAMP(CAST(n_users AS DOUBLE)) AS n_users_macro_stddev,
          AVG(CAST(n_positive AS DOUBLE)) AS n_positive_macro_mean,
          STDDEV_SAMP(CAST(n_positive AS DOUBLE)) AS n_positive_macro_stddev,
          AVG(positive_prevalence) AS positive_prevalence_macro_mean,
          STDDEV_SAMP(positive_prevalence) AS positive_prevalence_macro_stddev,
          AVG(precision_at_10pct) AS precision_at_10pct_macro_mean,
          STDDEV_SAMP(precision_at_10pct) AS precision_at_10pct_macro_stddev,
          AVG(recall_at_10pct) AS recall_at_10pct_macro_mean,
          STDDEV_SAMP(recall_at_10pct) AS recall_at_10pct_macro_stddev,
          AVG(mcc_at_10pct) AS mcc_at_10pct_macro_mean,
          STDDEV_SAMP(mcc_at_10pct) AS mcc_at_10pct_macro_stddev,
          AVG(average_precision) AS average_precision_macro_mean,
          STDDEV_SAMP(average_precision) AS average_precision_macro_stddev,
          AVG(auprc_ap) AS auprc_ap_macro_mean,
          STDDEV_SAMP(auprc_ap) AS auprc_ap_macro_stddev,
          AVG(auroc) AS auroc_macro_mean,
          STDDEV_SAMP(auroc) AS auroc_macro_stddev,
          AVG(mean_target_score_top_k) AS mean_target_score_top_k_macro_mean,
          STDDEV_SAMP(mean_target_score_top_k) AS mean_target_score_top_k_macro_stddev,
          AVG(mean_target_score_all) AS mean_target_score_all_macro_mean,
          STDDEV_SAMP(mean_target_score_all) AS mean_target_score_all_macro_stddev,
          'macro month-wise means over rolling-origin test folds; pooled totals are counts only' AS aggregation_note,
          'average_precision_ranked_list' AS pr_auc_definition,
          'deterministic_tie_broken_mann_whitney_rank_sum' AS auroc_definition
        FROM baseline_metrics_by_fold
        GROUP BY 1
        """
    )

    logger.info("Preparing top-k audit rows.")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE baseline_topk_by_month AS
        SELECT
          baseline_name,
          prediction_month,
          feature_month,
          fold_id,
          split_role,
          row_scope AS metric_scope,
          user_id,
          ranking_score,
          tie_break_score,
          continuous_prior_score,
          rank_within_month,
          n_positive_window AS k_at_10pct,
          label_high_engagement,
          target_score
        FROM baseline_ranked
        WHERE rank_within_month <= n_positive_window
        """
    )

    if not cfg.write_all_month_diagnostics:
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE baseline_metrics_by_month_primary_only AS
            SELECT *
            FROM baseline_metrics_by_month
            WHERE prediction_month IN (SELECT test_prediction_month FROM primary_test_months)
            """
        )


def validate_metric_outputs(con: duckdb.DuckDBPyConnection, cfg: BaselineConfig) -> None:
    """Validate metric tables before files are written."""
    duplicate_month_rows = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM (
              SELECT baseline_name, prediction_month, COUNT(*) AS n
              FROM baseline_metrics_by_month
              GROUP BY 1, 2
              HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
    )
    if duplicate_month_rows:
        raise ValueError("Metric outputs contain duplicate baseline_name x prediction_month rows.")

    duplicate_fold_rows = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM (
              SELECT baseline_name, fold_id, test_prediction_month, COUNT(*) AS n
              FROM baseline_metrics_by_fold
              GROUP BY 1, 2, 3
              HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
    )
    if duplicate_fold_rows:
        raise ValueError("Fold outputs contain duplicate baseline_name x fold_id x test_prediction_month rows.")

    expected_fold_rows = len(BASELINE_DEFINITIONS) * len(PRIMARY_TEST_MONTHS_2021)
    observed_fold_rows = int(con.execute("SELECT COUNT(*) FROM baseline_metrics_by_fold").fetchone()[0])
    if observed_fold_rows != expected_fold_rows:
        raise ValueError(
            f"Unexpected number of primary fold metric rows: observed={observed_fold_rows}, "
            f"expected={expected_fold_rows}."
        )

    range_failures = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM baseline_metrics_by_fold
            WHERE precision_at_10pct < -1e-12 OR precision_at_10pct > 1.0 + 1e-12
               OR recall_at_10pct < -1e-12 OR recall_at_10pct > 1.0 + 1e-12
               OR mcc_at_10pct < -1.0 - 1e-12 OR mcc_at_10pct > 1.0 + 1e-12
               OR average_precision < -1e-12 OR average_precision > 1.0 + 1e-12
               OR auprc_ap < -1e-12 OR auprc_ap > 1.0 + 1e-12
               OR auroc < -1e-12 OR auroc > 1.0 + 1e-12
            """
        ).fetchone()[0]
    )
    if range_failures:
        raise ValueError("One or more primary fold metrics fall outside [0, 1].")

    if cfg.write_all_month_diagnostics:
        diagnostic_range_failures = int(
            con.execute(
                """
                SELECT COUNT(*)
                FROM baseline_metrics_by_month
                WHERE precision_at_10pct < -1e-12 OR precision_at_10pct > 1.0 + 1e-12
                   OR recall_at_10pct < -1e-12 OR recall_at_10pct > 1.0 + 1e-12
                   OR mcc_at_10pct < -1.0 - 1e-12 OR mcc_at_10pct > 1.0 + 1e-12
                   OR average_precision < -1e-12 OR average_precision > 1.0 + 1e-12
                   OR auprc_ap < -1e-12 OR auprc_ap > 1.0 + 1e-12
                   OR auroc < -1e-12 OR auroc > 1.0 + 1e-12
                """
            ).fetchone()[0]
        )
        if diagnostic_range_failures:
            raise ValueError("One or more all-month diagnostic metrics fall outside [0, 1].")

    formula_failures = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM baseline_metrics_by_fold
            WHERE top_k_positive_count > k_at_10pct
               OR tp_at_10pct <> top_k_positive_count
               OR fp_at_10pct <> k_at_10pct - tp_at_10pct
               OR fn_at_10pct <> n_positive - tp_at_10pct
               OR tn_at_10pct <> n_negative - fp_at_10pct
               OR tp_at_10pct < 0 OR fp_at_10pct < 0 OR fn_at_10pct < 0 OR tn_at_10pct < 0
               OR ABS(recall_at_10pct - (CAST(top_k_positive_count AS DOUBLE) / CAST(n_positive AS DOUBLE))) > 1e-12
               OR ABS(precision_at_10pct - (CAST(top_k_positive_count AS DOUBLE) / CAST(k_at_10pct AS DOUBLE))) > 1e-12
               OR ABS(average_precision - auprc_ap) > 1e-12
            """
        ).fetchone()[0]
    )
    if formula_failures:
        raise ValueError("One or more primary fold metric formula checks failed.")

    mcc_denominator_failures = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM baseline_metrics_by_fold
            WHERE (
              CAST((tp_at_10pct + fp_at_10pct) AS DOUBLE)
              * CAST((tp_at_10pct + fn_at_10pct) AS DOUBLE)
              * CAST((tn_at_10pct + fp_at_10pct) AS DOUBLE)
              * CAST((tn_at_10pct + fn_at_10pct) AS DOUBLE)
            ) <= 0.0
            """
        ).fetchone()[0]
    )
    if mcc_denominator_failures:
        raise ValueError("One or more primary fold MCC denominators are zero.")

    non_test_fold_rows = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM baseline_metrics_by_fold
            WHERE split_role <> 'test' OR metric_scope <> 'primary_rolling_origin_test'
            """
        ).fetchone()[0]
    )
    if non_test_fold_rows:
        raise ValueError("baseline_metrics_by_fold must contain only primary rolling-origin test rows.")

    summary_scope_failures = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM baseline_metrics_summary
            WHERE metric_scope <> 'primary_rolling_origin_test'
               OR source_split_role <> 'test'
            """
        ).fetchone()[0]
    )
    if summary_scope_failures:
        raise ValueError("baseline_metrics_summary.csv must be based on rolling-origin test rows only.")

    summary_rows = int(con.execute("SELECT COUNT(*) FROM baseline_metrics_summary").fetchone()[0])
    if summary_rows != len(BASELINE_DEFINITIONS):
        raise ValueError(
            f"Unexpected baseline summary row count: observed={summary_rows}, expected={len(BASELINE_DEFINITIONS)}."
        )

    logger.info("Metric validation checks passed.")


def write_metric_outputs(con: duckdb.DuckDBPyConnection, cfg: BaselineConfig, out_root: Path) -> dict[str, str]:
    """Write metric and prediction artefacts."""
    outputs: dict[str, str] = {}

    if cfg.write_all_month_diagnostics:
        copy_query_to_csv(
            con,
            "SELECT * FROM baseline_metrics_by_month ORDER BY baseline_name, prediction_month",
            out_root / "baseline_metrics_by_month_all.csv",
        )
        outputs["baseline_metrics_by_month_all.csv"] = str(out_root / "baseline_metrics_by_month_all.csv")

    copy_query_to_csv(
        con,
        "SELECT * FROM baseline_metrics_by_fold ORDER BY baseline_name, fold_id, test_prediction_month",
        out_root / "baseline_metrics_by_fold.csv",
    )
    outputs["baseline_metrics_by_fold.csv"] = str(out_root / "baseline_metrics_by_fold.csv")

    copy_query_to_csv(
        con,
        "SELECT * FROM baseline_metrics_summary ORDER BY baseline_name",
        out_root / "baseline_metrics_summary.csv",
    )
    outputs["baseline_metrics_summary.csv"] = str(out_root / "baseline_metrics_summary.csv")

    copy_query_to_csv(
        con,
        """
        SELECT *
        FROM baseline_topk_by_month
        ORDER BY baseline_name, prediction_month, rank_within_month
        """,
        out_root / "baseline_topk_by_month.csv",
    )
    outputs["baseline_topk_by_month.csv"] = str(out_root / "baseline_topk_by_month.csv")

    if cfg.write_ranked_predictions:
        ranked_path = out_root / "baseline_ranked_predictions.parquet"
        con.execute(
            f"""
            COPY (
              SELECT
                prediction_month,
                feature_month,
                fold_id,
                split_role,
                user_id,
                baseline_name,
                ranking_score,
                tie_break_score,
                continuous_prior_score,
                rank_within_month,
                label_high_engagement,
                target_score
              FROM baseline_ranked
              ORDER BY baseline_name, prediction_month, rank_within_month
            )
            TO {quote_sql_string(ranked_path)}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        outputs["baseline_ranked_predictions.parquet"] = str(ranked_path)

    return outputs


def metadata_payload(
    cfg: BaselineConfig,
    duck: DuckDBConfig,
    outputs: dict[str, str],
    run_started_at: str,
    status: str,
    panel_row_count: Optional[int] = None,
    evaluated_prediction_months: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    """Build run metadata."""
    return {
        "script_name": "Modelling/1_0_run_baselines.py",
        "run_timestamp_utc": run_started_at,
        "git": git_metadata(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "duckdb": asdict(duck),
        "status": status,
        "panel_path": cfg.panel_path,
        "feature_schema_csv": cfg.feature_schema_csv,
        "output_root": cfg.out_root,
        "panel_row_count": panel_row_count,
        "target_score_col_for_descriptive_summaries": cfg.target_score_col,
        "truth_label_col": "label_high_engagement",
        "random_seed": cfg.random_seed,
        "baseline_names": [row["baseline_name"] for row in BASELINE_DEFINITIONS],
        "baseline_count": len(BASELINE_DEFINITIONS),
        "baseline_type": "non_learned_heuristic_rankings",
        "model_fitting_performed": False,
        "rolling_origin_protocol": {
            "prediction_months_required": list(EXPECTED_PREDICTION_MONTHS_2021),
            "minimum_training_window_months": cfg.min_train_months,
            "validation_window_months": cfg.validation_window_months,
            "test_window_months": cfg.test_window_months,
            "primary_test_months": list(PRIMARY_TEST_MONTHS_2021),
            "primary_metric_filter": 'split_role = "test"',
        },
        "all_month_diagnostics": {
            "written": bool(cfg.write_all_month_diagnostics),
            "metric_scope": "diagnostic_all_prediction_months",
            "evaluated_prediction_months": list(evaluated_prediction_months or []),
            "note": "Diagnostic all-month metrics are separate from the primary rolling-origin test summary.",
        },
        "metric_definitions": {
            "k_at_10pct": "monthly n_positive from label_high_engagement; equivalent to the constructed top-10% label count",
            "tp_at_10pct": "top_k_positive_count; true positives among users selected by the top-10% ranked cutoff.",
            "fp_at_10pct": "k_at_10pct - tp_at_10pct; false positives among top-10% selected users.",
            "fn_at_10pct": "n_positive - tp_at_10pct; labelled positives outside the top-10% selected users.",
            "tn_at_10pct": "n_negative - fp_at_10pct; labelled negatives outside the top-10% selected users.",
            "precision_at_10pct": "top_k_positive_count / k_at_10pct",
            "recall_at_10pct": "top_k_positive_count / n_positive",
            "mcc_at_10pct": (
                "Matthews correlation coefficient for the top-10% ranked cutoff using "
                "tp_at_10pct, fp_at_10pct, fn_at_10pct, and tn_at_10pct. "
                "If the denominator is zero, the implementation returns NaN; full Study 2 "
                "validation asserts non-zero denominators for evaluated folds."
            ),
            "average_precision": (
                "After deterministic descending ranking, mean precision at ranks where "
                "label_high_engagement == 1."
            ),
            "auprc_ap": "Same value as average_precision.",
            "pr_auc_definition": PR_AUC_DEFINITION,
            "auroc": (
                "Mann-Whitney/rank-sum AUROC after deterministic tie-breaking; "
                "pooled AUROC is not reported."
            ),
            "auroc_definition": AUROC_DEFINITION,
        },
        "tie_handling": TIE_HANDLING,
        "top_k_fraction": cfg.top_k_fraction,
        "top_k_fraction_note": (
            "The CLI value is fixed at 0.10 in this implementation to match the constructed label; "
            "metric k uses monthly n_positive."
        ),
        "smoke": cfg.smoke,
        "smoke_max_rows": cfg.smoke_max_rows if cfg.smoke else 0,
        "outputs": outputs,
        "known_caveats": [
            "Heuristic baselines are not fitted models; train and validation months are recorded for comparability only.",
            "AUROC uses reproducibly tie-broken ranks rather than average-rank treatment of equal heuristic scores.",
            "Fold summaries are descriptive because users may appear in multiple months.",
        ],
    }


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run Study 2 non-learned heuristic baseline ranking evaluations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--panel-path", required=True, help="Path to modelling_panel.parquet.")
    parser.add_argument("--feature-schema-csv", default=None, help="Optional feature_schema.csv.")
    parser.add_argument("--out-root", required=True, help="Output root for baseline artefacts.")
    parser.add_argument("--random-seed", type=int, default=2021, help="Random seed for deterministic random scores.")
    parser.add_argument(
        "--target-score-col",
        default="target_score",
        help="Target-score column used only for optional mean target-score summaries.",
    )
    parser.add_argument(
        "--describe-only",
        action="store_true",
        help="Write metadata and definitions, then exit successfully without computing metrics.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite deterministic baseline outputs.")
    parser.add_argument(
        "--write-ranked-predictions",
        action="store_true",
        help="Write full baseline_ranked_predictions.parquet. This is six rows per evaluated user-month.",
    )
    parser.add_argument(
        "--top-k-fraction",
        type=float,
        default=TOP_10_FRACTION,
        help="Reserved top-k fraction; currently fixed at 0.10 to match the constructed label.",
    )
    parser.add_argument(
        "--write-all-month-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write baseline_metrics_by_month_all.csv for all available prediction months.",
    )
    parser.add_argument("--smoke", action="store_true", help="Enable smoke-mode metadata and optional row limiting.")
    parser.add_argument("--smoke-max-rows", type=int, default=0, help="If >0 and --smoke, limit panel rows read.")
    parser.add_argument("--min-train-months", type=int, default=4, help="Minimum rolling-origin training months.")
    parser.add_argument("--validation-window-months", type=int, default=1, help="Rolling-origin validation window.")
    parser.add_argument("--test-window-months", type=int, default=1, help="Rolling-origin test window.")
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
    """Run baseline evaluations."""
    parser = build_argparser()
    args = parser.parse_args(argv)
    validate_cli(args, parser)
    setup_logging(args.log_level)

    cfg = BaselineConfig(
        panel_path=str(args.panel_path),
        feature_schema_csv=str(args.feature_schema_csv) if args.feature_schema_csv else None,
        out_root=str(args.out_root),
        random_seed=int(args.random_seed),
        target_score_col=str(args.target_score_col),
        describe_only=bool(args.describe_only),
        overwrite=bool(args.overwrite),
        write_ranked_predictions=bool(args.write_ranked_predictions),
        top_k_fraction=float(args.top_k_fraction),
        write_all_month_diagnostics=bool(args.write_all_month_diagnostics),
        smoke=bool(args.smoke),
        smoke_max_rows=int(args.smoke_max_rows),
        min_train_months=int(args.min_train_months),
        validation_window_months=int(args.validation_window_months),
        test_window_months=int(args.test_window_months),
    )
    duck = DuckDBConfig(
        memory_limit=str(args.duckdb_memory_limit),
        threads=int(args.duckdb_threads),
        temp_dir=str(args.duckdb_temp_dir),
    )
    run_started_at = utc_now_iso()

    out_root = Path(cfg.out_root)
    output_paths = known_output_paths(out_root)
    check_output_collisions(output_paths, cfg.overwrite)
    remove_outputs(output_paths, cfg.overwrite)
    ensure_dir(out_root)

    definition_rows = baseline_definition_rows()
    outputs: dict[str, str] = {
        "resolved_config.json": str(out_root / "resolved_config.json"),
        "run_metadata.json": str(out_root / "run_metadata.json"),
        "baseline_definitions.csv": str(out_root / "baseline_definitions.csv"),
    }
    safe_json_dump({"config": asdict(cfg), "duckdb": asdict(duck)}, out_root / "resolved_config.json")
    write_csv_rows(out_root / "baseline_definitions.csv", definition_rows, BASELINE_DEFINITION_FIELDS)

    if cfg.describe_only:
        safe_json_dump(
            metadata_payload(
                cfg=cfg,
                duck=duck,
                outputs=outputs,
                run_started_at=run_started_at,
                status="describe_only_metric_computation_skipped",
            ),
            out_root / "run_metadata.json",
        )
        logger.info("Wrote baseline definitions under %s; metric computation skipped by --describe-only.", out_root)
        return 0

    logger.info("Reading panel: %s", cfg.panel_path)
    logger.info("Output root: %s", out_root)
    if cfg.smoke:
        logger.warning(
            "Smoke mode enabled. Panel row limit: %s",
            int(cfg.smoke_max_rows) if int(cfg.smoke_max_rows) > 0 else "none",
        )

    con = duckdb.connect()
    try:
        configure_duckdb(con, duck)
        create_panel_source_view(con, cfg)
        columns = relation_columns(con, "panel_source")
        bootstrap_missing = sorted({"prediction_month", "feature_month"} - set(columns))
        if bootstrap_missing:
            raise ValueError("Panel is missing required column(s): " + ", ".join(bootstrap_missing))
        prediction_months = observed_prediction_months(con)
        evaluated_months = prediction_months if cfg.write_all_month_diagnostics else list(PRIMARY_TEST_MONTHS_2021)
        validate_panel_preconditions(con, cfg, columns, evaluated_months)

        feature_map = prediction_month_feature_map(con)
        split_rows, primary_test_rows = build_rolling_origin_splits(feature_map, cfg)
        write_csv_rows(out_root / "rolling_origin_splits.csv", split_rows, split_fieldnames())
        outputs["rolling_origin_splits.csv"] = str(out_root / "rolling_origin_splits.csv")
        register_primary_test_months(con, primary_test_rows)

        create_ranked_baseline_tables(con, cfg, evaluated_months)
        create_metric_tables(con, cfg)
        validate_metric_outputs(con, cfg)
        outputs.update(write_metric_outputs(con, cfg, out_root))

        panel_row_count = int(con.execute("SELECT COUNT(*) FROM panel_source").fetchone()[0])
        safe_json_dump(
            metadata_payload(
                cfg=cfg,
                duck=duck,
                outputs=outputs,
                run_started_at=run_started_at,
                status="finished",
                panel_row_count=panel_row_count,
                evaluated_prediction_months=evaluated_months,
            ),
            out_root / "run_metadata.json",
        )
    finally:
        con.close()

    logger.info("Done. Wrote baseline outputs under %s", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
