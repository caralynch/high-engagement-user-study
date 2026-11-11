#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Cara Lynch
# See the LICENSE file for details.
"""Validate an existing Study 2 modelling panel."""

from __future__ import annotations

import argparse
import csv
import json
import logging
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


TINY_SMOKE_PANEL_ROW_THRESHOLD = 100
OUTPUTS = (
    "panel_validation_summary.csv",
    "label_prevalence_by_month.csv",
    "missingness_by_feature.csv",
    "validation_errors.txt",
    "run_metadata.json",
)

REQUIRED_COLUMNS = (
    "user_id",
    "feature_month",
    "prediction_month",
    "feature_window_start_month",
    "feature_window_end_month",
    "target_threshold_quantile",
    "target_activity",
    "target_received_engagement_total",
    "target_received_comments_on_submissions",
    "target_received_replies_to_comments",
    "target_overlap_submission_and_comment_reply_events",
    "target_distinct_commenters_on_submissions",
    "target_distinct_repliers_to_comments",
    "target_distinct_received_engagers",
    "target_received_engagement_per_activity",
    "target_observed_received_engagement_per_activity",
    "target_active_in_prediction_month",
    "target_rank_within_month",
    "target_threshold_value",
    "label_high_engagement",
    "target_score",
    "target_score_col",
)

TARGET_COLUMNS = tuple(
    column
    for column in REQUIRED_COLUMNS
    if column.startswith("target_") or column == "label_high_engagement"
)


@dataclass(frozen=True)
class DuckDBConfig:
    """DuckDB runtime configuration."""

    memory_limit: str
    threads: int
    temp_dir: str


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


def target_score_policy_sql(target_score_col: str) -> str:
    """Return the configured prospective ranking score policy."""
    return (
        "CASE "
        "WHEN target_active_in_prediction_month "
        f"THEN CAST({quote_identifier(target_score_col)} AS DOUBLE) "
        "ELSE 0.0 "
        "END"
    )


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


def expected_output_paths(out_root: Path) -> list[Path]:
    """Return deterministic validation output paths."""
    return [out_root / name for name in OUTPUTS]


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


def read_feature_schema(path: Optional[str]) -> tuple[list[str], list[str]]:
    """Return feature names and read errors from a feature_schema CSV."""
    if not path:
        return [], []
    source = Path(path)
    if not source.exists():
        return [], [f"Feature schema CSV not found: {source}"]
    try:
        with open(source, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if "feature_name" not in (reader.fieldnames or []):
                return [], [f"Feature schema CSV lacks feature_name column: {source}"]
            return [str(row["feature_name"]) for row in reader if row.get("feature_name")], []
    except Exception as exc:
        return [], [f"Could not read feature schema CSV {source}: {exc}"]


def resolve_panel_metadata_path(panel_path: str | Path, requested: Optional[str]) -> Path:
    """Return the explicit builder metadata path or the panel's sibling resolved config."""
    if requested:
        return Path(requested)
    return Path(panel_path).parent / "resolved_config.json"


def validate_panel_author_identifier(
    metadata_path: str | Path,
    smoke: bool,
) -> tuple[dict[str, object], list[str], Optional[str]]:
    """Require production panel metadata to record author_pid as the resolved identifier."""
    source = Path(metadata_path)
    resolved: Optional[str] = None
    details = str(source)

    if not source.exists():
        observed = "metadata missing"
    else:
        try:
            with open(source, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                raise ValueError("top-level JSON value is not an object")
            config = payload.get("config")
            if not isinstance(config, dict):
                config = payload
            value = config.get("author_col_resolved")
            resolved = str(value) if value is not None else None
            observed = resolved if resolved is not None else "author_col_resolved missing"
        except Exception as exc:
            observed = f"metadata unreadable: {exc}"

    if resolved == "author_pid":
        status = "PASS"
    else:
        status = "WARN" if smoke else "FAIL"

    row = {
        "check_name": "production_author_identifier_is_pseudonymised",
        "status": status,
        "observed_value": observed,
        "expected_value": "author_pid",
        "details": details,
    }
    errors = []
    if status == "FAIL":
        errors.append(
            "production_author_identifier_is_pseudonymised: "
            f"observed={observed}; expected=author_pid; metadata={source}"
        )
    return row, errors, resolved


def relation_columns(con: duckdb.DuckDBPyConnection, panel_path: str | Path) -> list[str]:
    """Return columns in the panel parquet."""
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet({quote_sql_string(panel_path)}) LIMIT 0"
    ).fetchall()
    return [str(row[0]) for row in rows]


def copy_query_to_csv(con: duckdb.DuckDBPyConnection, query: str, path: str | Path) -> None:
    """Write a query result to CSV."""
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


def write_text(path: str | Path, text: str) -> None:
    """Write a UTF-8 text file."""
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def validate_panel(
    con: duckdb.DuckDBPyConnection,
    panel_path: str | Path,
    feature_names: Sequence[str],
    target_threshold_quantile: Optional[float],
    out_root: Path,
    smoke: bool,
) -> tuple[list[dict[str, object]], list[str]]:
    """Run panel checks and write detail tables where possible."""
    summary: list[dict[str, object]] = []
    errors: list[str] = []

    def add(check: str, status: str, observed: object, expected: object, details: str = "") -> None:
        summary.append(
            {
                "check_name": check,
                "status": status,
                "observed_value": observed,
                "expected_value": expected,
                "details": details,
            }
        )
        if status == "FAIL":
            errors.append(f"{check}: observed={observed}; expected={expected}; {details}".strip())

    columns = relation_columns(con, panel_path)
    column_set = set(columns)
    missing_required = [column for column in REQUIRED_COLUMNS if column not in column_set]
    add(
        "required_columns_exist",
        "PASS" if not missing_required else "FAIL",
        ", ".join(missing_required) if missing_required else "none",
        "none",
    )
    if missing_required:
        return summary, errors

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW panel AS
        SELECT *
        FROM read_parquet({quote_sql_string(panel_path)})
        """
    )

    n_rows = int(con.execute("SELECT COUNT(*) FROM panel").fetchone()[0])
    add("panel_file_readable_and_nonempty", "PASS" if n_rows > 0 else "FAIL", n_rows, ">0")

    duplicate_units = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM (
              SELECT user_id, prediction_month, COUNT(*) AS n
              FROM panel
              GROUP BY 1, 2
              HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
    )
    add("one_row_per_user_month", "PASS" if duplicate_units == 0 else "FAIL", duplicate_units, 0)
    add("no_duplicate_user_month_rows", "PASS" if duplicate_units == 0 else "FAIL", duplicate_units, 0)

    non_preceding = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM panel
            WHERE feature_month >= prediction_month
            """
        ).fetchone()[0]
    )
    add("feature_month_precedes_target_month", "PASS" if non_preceding == 0 else "FAIL", non_preceding, 0)

    null_users = int(con.execute("SELECT COUNT(*) FROM panel WHERE user_id IS NULL").fetchone()[0])
    add("user_ids_non_null", "PASS" if null_users == 0 else "FAIL", null_users, 0)

    feature_names_resolved = list(feature_names) if feature_names else [c for c in columns if c.startswith("feature_")]
    target_feature_overlap = sorted(set(feature_names_resolved) & set(TARGET_COLUMNS))
    add(
        "target_columns_not_in_feature_schema",
        "PASS" if not target_feature_overlap else "FAIL",
        ", ".join(target_feature_overlap) if target_feature_overlap else "none",
        "none",
    )

    missing_feature_columns = [feature for feature in feature_names_resolved if feature not in column_set]
    add(
        "feature_schema_columns_exist_in_panel",
        "PASS" if not missing_feature_columns else "FAIL",
        ", ".join(missing_feature_columns) if missing_feature_columns else "none",
        "none",
    )

    threshold_expr = (
        "MAX(target_threshold_quantile)"
        if "target_threshold_quantile" in column_set
        else str(float(target_threshold_quantile))
        if target_threshold_quantile is not None
        else "NULL"
    )
    if threshold_expr == "NULL":
        add("target_threshold_quantile_available", "FAIL", "missing", "panel column or CLI value")
    else:
        add("target_threshold_quantile_available", "PASS", "available", "available")

    copy_query_to_csv(
        con,
        f"""
        SELECT
          prediction_month,
          COUNT(*) AS n_prediction_population,
          SUM(label_high_engagement) AS observed_positives,
          CAST(CEIL(COUNT(*) * (1.0 - {threshold_expr})) AS BIGINT) AS expected_positives,
          SUM(label_high_engagement) = CAST(CEIL(COUNT(*) * (1.0 - {threshold_expr})) AS BIGINT)
            AS positive_count_matches_expected,
          AVG(CAST(label_high_engagement AS DOUBLE)) AS label_prevalence,
          {threshold_expr} AS target_threshold_quantile,
          MAX(target_threshold_value) AS target_threshold_value,
          SUM(CASE WHEN target_active_in_prediction_month THEN 1 ELSE 0 END) AS n_active_in_prediction_month,
          SUM(CASE WHEN NOT target_active_in_prediction_month THEN 1 ELSE 0 END) AS n_inactive_in_prediction_month,
          SUM(CASE
            WHEN target_active_in_prediction_month AND target_received_engagement_total = 0 THEN 1
            ELSE 0
          END) AS n_active_zero_received_engagement,
          SUM(CASE
            WHEN NOT target_active_in_prediction_month AND target_score = 0 THEN 1
            ELSE 0
          END) AS n_inactive_zero_ranking_score,
          SUM(target_received_engagement_total) AS total_target_received_engagement,
          SUM(target_received_comments_on_submissions) AS total_target_received_comments_on_submissions,
          SUM(target_received_replies_to_comments) AS total_target_received_replies_to_comments,
          SUM(target_received_comments_on_submissions + target_received_replies_to_comments)
            AS total_target_received_engagement_component_sum,
          SUM(target_overlap_submission_and_comment_reply_events)
            AS total_target_overlap_submission_and_comment_reply_events,
          SUM(target_distinct_commenters_on_submissions) AS total_target_distinct_commenters_on_submissions,
          SUM(target_distinct_repliers_to_comments) AS total_target_distinct_repliers_to_comments,
          SUM(target_distinct_received_engagers) AS total_target_distinct_received_engagers,
          AVG(target_received_engagement_per_activity)
            AS mean_target_received_engagement_per_activity_among_active,
          AVG(CASE WHEN target_received_engagement_total > 0 THEN 1.0 ELSE 0.0 END)
            AS share_nonzero_received_engagement
        FROM panel
        GROUP BY 1
        ORDER BY 1
        """,
        out_root / "label_prevalence_by_month.csv",
    )

    positive_mismatches = int(
        con.execute(
            f"""
            WITH by_month AS (
              SELECT
                prediction_month,
                SUM(label_high_engagement) AS observed_positives,
                CAST(CEIL(COUNT(*) * (1.0 - {threshold_expr})) AS BIGINT) AS expected_positives
              FROM panel
              GROUP BY 1
            )
            SELECT COUNT(*)
            FROM by_month
            WHERE observed_positives <> expected_positives
            """
        ).fetchone()[0]
    ) if threshold_expr != "NULL" else -1
    add(
        "positive_count_matches_threshold_rule",
        "PASS" if positive_mismatches == 0 else "FAIL",
        positive_mismatches,
        0,
        "expected_positives = ceil(n_prediction_population * (1 - target_threshold_quantile))",
    )

    target_score_source_rows = con.execute(
        "SELECT DISTINCT target_score_col FROM panel ORDER BY target_score_col"
    ).fetchall()
    target_score_sources = [str(row[0]) for row in target_score_source_rows]
    unsupported_sources = [source for source in target_score_sources if source not in column_set]
    add(
        "target_score_source_column_exists",
        "PASS" if not unsupported_sources else "FAIL",
        ", ".join(unsupported_sources) if unsupported_sources else "all present",
        "all present",
    )
    if not unsupported_sources:
        mismatch_count = 0
        for source in target_score_sources:
            target_score_expected = target_score_policy_sql(source)
            mismatch_count += int(
                con.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM panel
                    WHERE target_score_col = {quote_sql_string(source)}
                      AND ABS(COALESCE(target_score, -999999999.0)
                              - COALESCE({target_score_expected}, -999999999.0)) > 1e-12
                    """
                ).fetchone()[0]
            )
        add("target_score_matches_configured_policy", "PASS" if mismatch_count == 0 else "FAIL", mismatch_count, 0)

    target_unique_gt_components = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM panel
            WHERE target_received_engagement_total
              > target_received_comments_on_submissions + target_received_replies_to_comments
            """
        ).fetchone()[0]
    )
    add(
        "target_unique_total_lte_component_sum",
        "PASS" if target_unique_gt_components == 0 else "FAIL",
        target_unique_gt_components,
        0,
    )

    target_overlap_mismatches = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM panel
            WHERE target_overlap_submission_and_comment_reply_events
              <> target_received_comments_on_submissions
                 + target_received_replies_to_comments
                 - target_received_engagement_total
            """
        ).fetchone()[0]
    )
    add(
        "target_overlap_equals_component_sum_minus_unique_total",
        "PASS" if target_overlap_mismatches == 0 else "FAIL",
        target_overlap_mismatches,
        0,
    )
    add(
        "target_received_engagement_total_uses_unique_total",
        "PASS" if target_overlap_mismatches == 0 else "FAIL",
        target_overlap_mismatches,
        0,
        "target total must equal component sum minus overlap",
    )

    feature_overlap_columns = {
        "feature_received_engagement_total_prev",
        "feature_received_comments_on_submissions_prev",
        "feature_received_replies_to_comments_prev",
        "feature_overlap_submission_and_comment_reply_events_prev",
    }
    if feature_overlap_columns.issubset(column_set):
        feature_overlap_mismatches = int(
            con.execute(
                """
                SELECT COUNT(*)
                FROM panel
                WHERE feature_overlap_submission_and_comment_reply_events_prev
                  <> feature_received_comments_on_submissions_prev
                     + feature_received_replies_to_comments_prev
                     - feature_received_engagement_total_prev
                """
            ).fetchone()[0]
        )
        add(
            "feature_overlap_equals_component_sum_minus_unique_total",
            "PASS" if feature_overlap_mismatches == 0 else "FAIL",
            feature_overlap_mismatches,
            0,
        )

    inactive_target_mismatches = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM panel
            WHERE NOT target_active_in_prediction_month
              AND (
                target_received_engagement_per_activity IS NOT NULL
                OR target_observed_received_engagement_per_activity IS NOT NULL
                OR ABS(COALESCE(target_score, -999999999.0) - 0.0) > 1e-12
              )
            """
        ).fetchone()[0]
    )
    add(
        "inactive_targets_have_null_observed_rate_and_zero_score",
        "PASS" if inactive_target_mismatches == 0 else "FAIL",
        inactive_target_mismatches,
        0,
    )

    active_null_rate_mismatches = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM panel
            WHERE target_active_in_prediction_month
              AND target_received_engagement_per_activity IS NULL
            """
        ).fetchone()[0]
    )
    add(
        "active_targets_have_numeric_observed_rate",
        "PASS" if active_null_rate_mismatches == 0 else "FAIL",
        active_null_rate_mismatches,
        0,
    )

    observed_alias_mismatches = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM panel
            WHERE COALESCE(target_received_engagement_per_activity, -999999999.0)
              <> COALESCE(target_observed_received_engagement_per_activity, -999999999.0)
            """
        ).fetchone()[0]
    )
    add(
        "observed_per_activity_alias_matches_target_column",
        "PASS" if observed_alias_mismatches == 0 else "FAIL",
        observed_alias_mismatches,
        0,
    )

    active_zero_mismatches = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM panel
            WHERE target_active_in_prediction_month
              AND target_received_engagement_total = 0
              AND (
                target_received_engagement_per_activity IS NULL
                OR ABS(target_received_engagement_per_activity - 0.0) > 1e-12
              )
            """
        ).fetchone()[0]
    )
    add(
        "active_zero_engagement_targets_have_zero_observed_rate",
        "PASS" if active_zero_mismatches == 0 else "FAIL",
        active_zero_mismatches,
        0,
    )

    total_received_engagement = int(
        con.execute("SELECT COALESCE(SUM(target_received_engagement_total), 0) FROM panel").fetchone()[0]
    )
    all_zero_status = (
        "PASS"
        if total_received_engagement > 0
        else "WARN"
        if smoke or n_rows <= TINY_SMOKE_PANEL_ROW_THRESHOLD
        else "FAIL"
    )
    add(
        "target_received_engagement_not_all_zero",
        all_zero_status,
        total_received_engagement,
        ">0 unless smoke/tiny fixture run",
        f"smoke={bool(smoke)}; tiny_smoke_row_threshold={TINY_SMOKE_PANEL_ROW_THRESHOLD}",
    )

    split_present = "split" in column_set or "split_name" in column_set
    add(
        "split_column_present_if_available",
        "PASS" if split_present else "WARN",
        split_present,
        "optional",
        "Initial panel scaffold does not assign splits." if not split_present else "",
    )

    missingness_rows: list[dict[str, object]] = []
    for feature in feature_names_resolved:
        if feature not in column_set:
            missingness_rows.append(
                {
                    "feature_name": feature,
                    "n_rows": n_rows,
                    "n_missing": "",
                    "missing_share": "",
                    "status": "missing_feature_column",
                }
            )
            continue
        n_missing = int(
            con.execute(
                f"SELECT COUNT(*) FROM panel WHERE {quote_identifier(feature)} IS NULL"
            ).fetchone()[0]
        )
        missingness_rows.append(
            {
                "feature_name": feature,
                "n_rows": n_rows,
                "n_missing": n_missing,
                "missing_share": (float(n_missing) / float(n_rows)) if n_rows else "",
                "status": "ok",
            }
        )
    write_csv_rows(
        out_root / "missingness_by_feature.csv",
        missingness_rows,
        ("feature_name", "n_rows", "n_missing", "missing_share", "status"),
    )

    return summary, errors


def metadata_payload(
    args: argparse.Namespace,
    duck: DuckDBConfig,
    errors: Sequence[str],
    panel_metadata_path: str | Path,
    panel_author_col_resolved: Optional[str],
) -> dict[str, object]:
    """Build validation run metadata."""
    return {
        "script_name": "Modelling/0_1_validate_modelling_panel.py",
        "run_timestamp_utc": utc_now_iso(),
        "git": git_metadata(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "duckdb": asdict(duck),
        "panel_path": str(args.panel_path),
        "feature_schema_csv": str(args.feature_schema_csv) if args.feature_schema_csv else None,
        "panel_metadata_json": str(panel_metadata_path),
        "panel_author_col_resolved": panel_author_col_resolved,
        "output_root": str(args.out_root),
        "smoke": bool(args.smoke),
        "validation_error_count": len(errors),
    }


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Validate a Study 2 modelling_panel.parquet file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--panel-path", required=True, help="Path to modelling_panel.parquet.")
    parser.add_argument("--feature-schema-csv", default=None, help="Optional feature_schema.csv from panel build.")
    parser.add_argument(
        "--panel-metadata-json",
        default=None,
        help="Builder resolved_config.json or run_metadata.json. Defaults to resolved_config.json beside the panel.",
    )
    parser.add_argument("--out-root", required=True, help="Output root for validation artefacts.")
    parser.add_argument(
        "--target-threshold-quantile",
        type=float,
        default=None,
        help="Fallback threshold quantile if the panel lacks target_threshold_quantile.",
    )
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
    parser.add_argument("--overwrite", action="store_true", help="Overwrite deterministic validation outputs.")
    parser.add_argument("--smoke", action="store_true", help="Downgrade smoke-fixture all-zero engagement checks.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run panel validation."""
    parser = build_argparser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    out_root = Path(args.out_root)
    output_paths = expected_output_paths(out_root)
    check_output_collisions(output_paths, bool(args.overwrite))
    remove_outputs(output_paths, bool(args.overwrite))
    ensure_dir(out_root)

    summary: list[dict[str, object]] = []
    errors: list[str] = []
    panel_metadata_path = resolve_panel_metadata_path(args.panel_path, args.panel_metadata_json)
    author_check, author_errors, panel_author_col_resolved = validate_panel_author_identifier(
        panel_metadata_path,
        bool(args.smoke),
    )
    summary.append(author_check)
    errors.extend(author_errors)
    feature_names, feature_schema_errors = read_feature_schema(args.feature_schema_csv)
    errors.extend(feature_schema_errors)

    if not Path(args.panel_path).exists():
        errors.append(f"Panel file not found: {args.panel_path}")
        summary.append(
            {
                "check_name": "panel_file_exists",
                "status": "FAIL",
                "observed_value": "missing",
                "expected_value": "exists",
                "details": str(args.panel_path),
            }
        )
        write_csv_rows(
            out_root / "panel_validation_summary.csv",
            summary,
            ("check_name", "status", "observed_value", "expected_value", "details"),
        )
        write_csv_rows(out_root / "label_prevalence_by_month.csv", [], ("prediction_month",))
        write_csv_rows(out_root / "missingness_by_feature.csv", [], ("feature_name",))
        write_text(out_root / "validation_errors.txt", "\n".join(errors) + "\n")
        duck = DuckDBConfig(str(args.duckdb_memory_limit), int(args.duckdb_threads), str(args.duckdb_temp_dir))
        safe_json_dump(
            metadata_payload(args, duck, errors, panel_metadata_path, panel_author_col_resolved),
            out_root / "run_metadata.json",
        )
        return 1

    duck = DuckDBConfig(
        memory_limit=str(args.duckdb_memory_limit),
        threads=int(args.duckdb_threads),
        temp_dir=str(args.duckdb_temp_dir),
    )
    con = duckdb.connect()
    try:
        configure_duckdb(con, duck)
        try:
            validation_summary, validation_errors = validate_panel(
                con,
                str(args.panel_path),
                feature_names,
                float(args.target_threshold_quantile) if args.target_threshold_quantile is not None else None,
                out_root,
                bool(args.smoke),
            )
            summary.extend(validation_summary)
            errors.extend(validation_errors)
        except Exception as exc:
            errors.append(f"Panel validation could not complete: {exc}")
            summary.append(
                {
                    "check_name": "panel_file_readable",
                    "status": "FAIL",
                    "observed_value": type(exc).__name__,
                    "expected_value": "readable parquet",
                    "details": str(exc),
                }
            )
            write_csv_rows(out_root / "label_prevalence_by_month.csv", [], ("prediction_month",))
            write_csv_rows(out_root / "missingness_by_feature.csv", [], ("feature_name",))
    finally:
        con.close()

    if feature_schema_errors:
        summary.append(
            {
                "check_name": "feature_schema_readable",
                "status": "FAIL",
                "observed_value": "; ".join(feature_schema_errors),
                "expected_value": "readable feature_schema.csv",
                "details": "",
            }
        )

    hard_errors = [err for err in errors if "split_column_present_if_available" not in err]
    write_csv_rows(
        out_root / "panel_validation_summary.csv",
        summary,
        ("check_name", "status", "observed_value", "expected_value", "details"),
    )
    write_text(
        out_root / "validation_errors.txt",
        ("No validation errors.\n" if not hard_errors else "\n".join(hard_errors) + "\n"),
    )
    safe_json_dump(
        metadata_payload(args, duck, hard_errors, panel_metadata_path, panel_author_col_resolved),
        out_root / "run_metadata.json",
    )

    if hard_errors:
        logger.error("Validation failed with %s error(s).", len(hard_errors))
        return 1
    logger.info("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
