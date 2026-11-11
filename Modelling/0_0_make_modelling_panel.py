#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Cara Lynch
# See the LICENSE file for details.
"""
Build the initial Study 2 prospective user-month modelling panel.

This scaffold implements the first global user-month design:

- each row is one eligible user-month prediction unit;
- features come from month t-1;
- targets come from prediction month t;
- the prediction population is users with at least --min-prior-activity
  contributions in month t-1;
- inactive users in month t remain in the denominator with target scores set to
  zero, while observed per-activity engagement is NULL because the denominator
  is zero.

The script deliberately supports only --feature-window-months 1 for now. Longer
windows should be added explicitly rather than inferred from this first pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import platform
import re
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import duckdb


logger = logging.getLogger(__name__)


MONTH_RE = re.compile(r"^20\d{2}-(0[1-9]|1[0-2])$")
TARGET_INACTIVE_USER_SCORE_POLICY = "set_to_zero"
TARGET_OBSERVED_PER_ACTIVITY_INACTIVE_POLICY = "set_to_null"
EXCLUDE_SELF_ENGAGEMENT = True
TINY_SMOKE_PANEL_ROW_THRESHOLD = 100
SPLIT_DEFINITION = "not_assigned_initial_panel_scaffold"
TARGET_SCORE_COLS = (
    "target_received_engagement_per_activity",
    "target_received_engagement_total",
)

PANEL_OUTPUTS = (
    "modelling_panel.parquet",
    "feature_schema.csv",
    "target_summary_by_month.csv",
    "prediction_population_summary_by_month.csv",
    "engagement_attribution_diagnostics.csv",
    "panel_validation_summary.csv",
    "resolved_config.json",
    "run_metadata.json",
    "input_inventory.csv",
    "month_pair_inventory.csv",
)

FEATURE_SCHEMA_ROWS: tuple[dict[str, str], ...] = (
    {
        "feature_name": "feature_n_submissions_prev",
        "feature_group": "activity",
        "source_window": "feature_month_t_minus_1",
        "description": "Number of submissions authored by the user in the feature month.",
    },
    {
        "feature_name": "feature_n_comments_prev",
        "feature_group": "activity",
        "source_window": "feature_month_t_minus_1",
        "description": "Number of comments authored by the user in the feature month.",
    },
    {
        "feature_name": "feature_n_total_activity_prev",
        "feature_group": "activity",
        "source_window": "feature_month_t_minus_1",
        "description": "Submissions plus comments authored by the user in the feature month.",
    },
    {
        "feature_name": "feature_submission_share_prev",
        "feature_group": "activity",
        "source_window": "feature_month_t_minus_1",
        "description": "Share of feature-month activity that was submissions.",
    },
    {
        "feature_name": "feature_comment_share_prev",
        "feature_group": "activity",
        "source_window": "feature_month_t_minus_1",
        "description": "Share of feature-month activity that was comments.",
    },
    {
        "feature_name": "feature_n_active_days_prev",
        "feature_group": "activity",
        "source_window": "feature_month_t_minus_1",
        "description": "Distinct active days in the feature month, where timestamps are available.",
    },
    {
        "feature_name": "feature_mean_activity_per_active_day_prev",
        "feature_group": "activity",
        "source_window": "feature_month_t_minus_1",
        "description": "Mean authored contributions per active day in the feature month.",
    },
    {
        "feature_name": "feature_received_engagement_total_prev",
        "feature_group": "historical_engagement",
        "source_window": "feature_month_t_minus_1",
        "description": "Unique other-user child comments received in the feature month.",
    },
    {
        "feature_name": "feature_received_engagement_per_activity_prev",
        "feature_group": "historical_engagement",
        "source_window": "feature_month_t_minus_1",
        "description": "Feature-month received engagement divided by feature-month activity.",
    },
    {
        "feature_name": "feature_log_received_engagement_rate_prev",
        "feature_group": "historical_engagement",
        "source_window": "feature_month_t_minus_1",
        "description": "log1p(unique feature received engagement) minus log1p(feature activity).",
    },
    {
        "feature_name": "feature_received_comments_on_submissions_prev",
        "feature_group": "historical_engagement",
        "source_window": "feature_month_t_minus_1",
        "description": "Submission-thread component: other-user comments received on the user's submissions in the feature month.",
    },
    {
        "feature_name": "feature_received_replies_to_comments_prev",
        "feature_group": "historical_engagement",
        "source_window": "feature_month_t_minus_1",
        "description": "Direct-reply component: other-user replies received by the user's comments in the feature month.",
    },
    {
        "feature_name": "feature_overlap_submission_and_comment_reply_events_prev",
        "feature_group": "historical_engagement",
        "source_window": "feature_month_t_minus_1",
        "description": "Other-user child comments attributed through both submission-thread and direct-reply routes in the feature month.",
    },
    {
        "feature_name": "feature_n_active_subreddits_prev",
        "feature_group": "subreddit_participation",
        "source_window": "feature_month_t_minus_1",
        "description": "Number of subreddits in which the user authored content in the feature month.",
    },
    {
        "feature_name": "feature_subreddit_hhi_prev",
        "feature_group": "subreddit_participation",
        "source_window": "feature_month_t_minus_1",
        "description": "Herfindahl-Hirschman concentration of feature-month activity across subreddits.",
    },
    {
        "feature_name": "feature_subreddit_entropy_prev",
        "feature_group": "subreddit_participation",
        "source_window": "feature_month_t_minus_1",
        "description": "Entropy of feature-month activity across subreddits.",
    },
    {
        "feature_name": "feature_top_subreddit_activity_share_prev",
        "feature_group": "subreddit_participation",
        "source_window": "feature_month_t_minus_1",
        "description": "Share of feature-month activity in the user's most active subreddit.",
    },
    {
        "feature_name": "feature_is_single_subreddit_user_prev",
        "feature_group": "subreddit_participation",
        "source_window": "feature_month_t_minus_1",
        "description": "Whether all feature-month activity occurred in one subreddit.",
    },
    {
        "feature_name": "feature_month_number",
        "feature_group": "calendar",
        "source_window": "feature_month_t_minus_1",
        "description": "Calendar month number of the feature month.",
    },
)


@dataclass(frozen=True)
class DuckDBConfig:
    """DuckDB runtime configuration."""

    memory_limit: str
    threads: int
    temp_dir: str


@dataclass(frozen=True)
class PanelConfig:
    """Resolved panel-build configuration."""

    in_root: str
    out_root: str
    selected_subreddits_csv: Optional[str]
    year: str
    months: tuple[str, ...]
    feature_window_months: int
    target_threshold_quantile: float
    target_score_col: str
    min_prior_activity: int
    author_col_requested: str
    author_col_resolved: str
    overwrite: bool
    smoke: bool
    smoke_max_rows: int
    target_inactive_user_score_policy: str
    target_observed_per_activity_inactive_policy: str
    exclude_self_engagement: bool
    split_definition: str
    random_seed: Optional[int]
    known_caveats: tuple[str, ...]


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
    """Return a stable UTC timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def quote_sql_string(value: str | Path) -> str:
    """Return a single-quoted SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def quote_identifier(name: str) -> str:
    """Return a double-quoted SQL identifier."""
    return '"' + str(name).replace('"', '""') + '"'


def qcol(alias: str, column: str) -> str:
    """Return a qualified SQL identifier."""
    return f"{alias}.{quote_identifier(column)}"


def clean_varchar_sql(expr: str) -> str:
    """Return SQL that trims an expression and maps empty strings to NULL."""
    return f"NULLIF(trim(CAST({expr} AS VARCHAR)), '')"


def strip_reddit_prefix_sql(expr: str, prefix: str) -> str:
    """Return SQL that trims an ID and strips one expected Reddit fullname prefix."""
    clean_expr = clean_varchar_sql(expr)
    return (
        "CASE "
        f"WHEN {clean_expr} IS NULL THEN NULL "
        f"WHEN starts_with({clean_expr}, {quote_sql_string(prefix)}) "
        f"THEN substr({clean_expr}, {len(prefix) + 1}) "
        f"ELSE {clean_expr} "
        "END"
    )


def target_score_policy_sql(target_score_col: str) -> str:
    """Return the ranking score policy expression for prospective targets."""
    return (
        "CASE "
        "WHEN target_active_in_prediction_month "
        f"THEN CAST({quote_identifier(target_score_col)} AS DOUBLE) "
        "ELSE 0.0 "
        "END"
    )


def ensure_dir(path: str | Path) -> None:
    """Create a directory if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_json_dump(obj: dict, path: str | Path) -> None:
    """Write stable JSON."""
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def write_csv_rows(path: str | Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    """Write dictionaries as a CSV with a stable header."""
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
    """Return deterministic output paths for collision checks."""
    return [out_root / name for name in PANEL_OUTPUTS]


def check_output_collisions(paths: Sequence[Path], overwrite: bool) -> None:
    """Fail if deterministic outputs already exist and overwrite is disabled."""
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


def parse_months(value: str) -> tuple[str, ...]:
    """Parse a comma-separated YYYY-MM month list."""
    months = tuple(part.strip() for part in value.split(",") if part.strip())
    bad = [month for month in months if not MONTH_RE.fullmatch(month)]
    if bad:
        raise ValueError(f"Invalid month value(s): {', '.join(bad)}")
    return tuple(sorted(dict.fromkeys(months)))


def default_year_months(year: str) -> tuple[str, ...]:
    """Return all months for a year."""
    return tuple(f"{year}-{month:02d}" for month in range(1, 13))


def month_add_one(month: str) -> str:
    """Return the next calendar month for a YYYY-MM string."""
    year_s, month_s = month.split("-")
    year_i = int(year_s)
    month_i = int(month_s)
    if month_i == 12:
        return f"{year_i + 1}-01"
    return f"{year_i}-{month_i + 1:02d}"


def month_number(month: str) -> int:
    """Return the numeric month from a YYYY-MM string."""
    return int(month.split("-")[1])


def input_path(in_root: str, kind: str, month: str) -> Path:
    """Return the expected monthly input path."""
    root = Path(in_root)
    if kind == "comments":
        return root / "comments" / f"comments_core_{month}.parquet"
    if kind == "submissions":
        return root / "submissions" / f"submissions_core_{month}.parquet"
    raise ValueError(f"Unknown input kind: {kind}")


def build_input_inventory(in_root: str, months: Sequence[str]) -> list[dict[str, object]]:
    """Build monthly input inventory records."""
    rows: list[dict[str, object]] = []
    for month in months:
        comments_path = input_path(in_root, "comments", month)
        submissions_path = input_path(in_root, "submissions", month)
        rows.append(
            {
                "month": month,
                "comments_core_path": str(comments_path),
                "comments_core_exists": comments_path.exists(),
                "comments_core_rows": "",
                "submissions_core_path": str(submissions_path),
                "submissions_core_exists": submissions_path.exists(),
                "submissions_core_rows": "",
            }
        )
    return rows


def build_month_pair_inventory(inventory: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Build feature-month to prediction-month pair records."""
    by_month = {str(row["month"]): row for row in inventory}
    months = sorted(by_month)
    records: list[dict[str, object]] = []
    for feature_month in months:
        prediction_month = month_add_one(feature_month)
        feature = by_month[feature_month]
        target = by_month.get(prediction_month)
        if target is None:
            used = False
            reason = "prediction_month_not_selected"
            target = {
                "comments_core_path": "",
                "comments_core_exists": False,
                "submissions_core_path": "",
                "submissions_core_exists": False,
            }
        else:
            inputs_complete = all(
                bool(value)
                for value in (
                    feature["comments_core_exists"],
                    feature["submissions_core_exists"],
                    target["comments_core_exists"],
                    target["submissions_core_exists"],
                )
            )
            used = inputs_complete
            reason = "used" if inputs_complete else "missing_input_file"
        records.append(
            {
                "feature_month": feature_month,
                "prediction_month": prediction_month,
                "feature_comments_core_path": feature["comments_core_path"],
                "feature_submissions_core_path": feature["submissions_core_path"],
                "prediction_comments_core_path": target["comments_core_path"],
                "prediction_submissions_core_path": target["submissions_core_path"],
                "feature_comments_core_exists": feature["comments_core_exists"],
                "feature_submissions_core_exists": feature["submissions_core_exists"],
                "prediction_comments_core_exists": target["comments_core_exists"],
                "prediction_submissions_core_exists": target["submissions_core_exists"],
                "used": used,
                "status": reason,
            }
        )
    return records


def used_months_from_pairs(pairs: Sequence[dict[str, object]]) -> tuple[str, ...]:
    """Return months needed by used month pairs."""
    months: set[str] = set()
    for pair in pairs:
        if bool(pair["used"]):
            months.add(str(pair["feature_month"]))
            months.add(str(pair["prediction_month"]))
    return tuple(sorted(months))


def validate_cli(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate cross-argument constraints."""
    if int(args.feature_window_months) != 1:
        parser.error("--feature-window-months currently supports only 1 in this initial scaffold.")
    if not (0.0 < float(args.target_threshold_quantile) < 1.0):
        parser.error("--target-threshold-quantile must be greater than 0 and less than 1.")
    if int(args.min_prior_activity) < 1:
        parser.error("--min-prior-activity must be at least 1 for the recently active population.")
    if int(args.smoke_max_rows) < 0:
        parser.error("--smoke-max-rows cannot be negative.")
    if args.selected_subreddits_csv and not Path(args.selected_subreddits_csv).exists():
        parser.error(f"--selected-subreddits-csv not found: {args.selected_subreddits_csv}")


def configure_duckdb(con: duckdb.DuckDBPyConnection, duck: DuckDBConfig) -> None:
    """Apply DuckDB runtime settings."""
    ensure_dir(duck.temp_dir)
    con.execute(f"SET memory_limit={quote_sql_string(duck.memory_limit)}")
    con.execute(f"SET threads={int(duck.threads)}")
    con.execute(f"SET temp_directory={quote_sql_string(duck.temp_dir)}")
    con.execute("SET preserve_insertion_order=false")
    con.execute("PRAGMA disable_object_cache")


def parquet_relation(path: str | Path, smoke_max_rows: int) -> str:
    """Return a read_parquet relation, optionally row-limited."""
    lit = quote_sql_string(path)
    if smoke_max_rows > 0:
        return f"(SELECT * FROM read_parquet({lit}) LIMIT {int(smoke_max_rows)})"
    return f"read_parquet({lit})"


def create_union_view(
    con: duckdb.DuckDBPyConnection,
    view_name: str,
    inventory: Sequence[dict[str, object]],
    kind: str,
    months: Sequence[str],
    smoke_max_rows: int,
) -> None:
    """Create a temp view over monthly Parquet files for one relation kind."""
    parts: list[str] = []
    month_set = set(months)
    for row in inventory:
        month = str(row["month"])
        if month not in month_set:
            continue
        path_key = f"{kind}_core_path"
        exists_key = f"{kind}_core_exists"
        if not bool(row[exists_key]):
            continue
        relation = parquet_relation(str(row[path_key]), smoke_max_rows)
        parts.append(
            f"""
            SELECT
              src.*,
              {quote_sql_string(month)} AS _input_month,
              {quote_sql_string(str(row[path_key]))} AS _input_path
            FROM {relation} AS src
            """
        )
    if not parts:
        raise FileNotFoundError(f"No existing {kind} files available for used month pairs.")
    con.execute(f"CREATE OR REPLACE TEMP VIEW {quote_identifier(view_name)} AS " + "\nUNION ALL\n".join(parts))


def describe_relation(con: duckdb.DuckDBPyConnection, relation_name: str) -> dict[str, str]:
    """Return relation column types keyed by column name."""
    rows = con.execute(f"DESCRIBE SELECT * FROM {quote_identifier(relation_name)} LIMIT 0").fetchall()
    return {str(row[0]): str(row[1]) for row in rows}


def require_columns(source_label: str, columns: Iterable[str], required: Sequence[str]) -> None:
    """Raise if a source is missing required columns."""
    column_set = set(columns)
    missing = [column for column in required if column not in column_set]
    if missing:
        raise ValueError(
            f"{source_label} is missing required column(s): {', '.join(missing)}. "
            f"Available columns: {', '.join(sorted(column_set))}"
        )


def resolve_author_col(
    comments_columns: Iterable[str],
    submissions_columns: Iterable[str],
    requested: str,
) -> str:
    """Resolve the author identifier column without silently selecting a raw identifier."""
    comments_set = set(comments_columns)
    submissions_set = set(submissions_columns)
    if requested == "auto":
        if "author_pid" in comments_set and "author_pid" in submissions_set:
            return "author_pid"
        unsafe_candidates = [
            candidate
            for candidate in ("author", "user_id")
            if candidate in comments_set and candidate in submissions_set
        ]
        if unsafe_candidates:
            raise RuntimeError(
                "Refusing to resolve --author-col auto to a non-pseudonymised identifier "
                f"({', '.join(unsafe_candidates)}). Production modelling requires author_pid."
            )
        raise RuntimeError(
            "Could not resolve --author-col auto. Expected author_pid to exist in both "
            "comments_core and submissions_core."
        )
    if requested not in comments_set or requested not in submissions_set:
        raise RuntimeError(
            f"Requested --author-col {requested!r} is not present in both comments_core and submissions_core."
        )
    return requested


def enforce_production_author_col(author_col: str, smoke: bool) -> None:
    """Fail closed when a non-smoke panel would use a non-pseudonymised identifier."""
    if not smoke and author_col != "author_pid":
        raise RuntimeError(
            "Production modelling requires --author-col author_pid; "
            f"resolved identifier was {author_col!r}."
        )


def deleted_author_filter(alias: str, columns: Iterable[str]) -> str:
    """Return SQL that excludes deleted authors where the flag is available."""
    if "is_deleted_author" not in set(columns):
        return "TRUE"
    return f"COALESCE({qcol(alias, 'is_deleted_author')}, FALSE) = FALSE"


def valid_author_filter(alias: str, author_col: str, columns: Iterable[str]) -> str:
    """Return SQL for valid author identifiers."""
    expr = f"{qcol(alias, author_col)} IS NOT NULL AND NULLIF(trim(CAST({qcol(alias, author_col)} AS VARCHAR)), '') IS NOT NULL"
    if author_col == "author":
        expr += f" AND lower(CAST({qcol(alias, author_col)} AS VARCHAR)) <> '[deleted]'"
    return f"({expr}) AND ({deleted_author_filter(alias, columns)})"


def created_day_expr(alias: str, schema: dict[str, str]) -> str:
    """Return a best-effort SQL expression for the event date."""
    if "created_utc" not in schema:
        return "CAST(NULL AS DATE)"
    dtype = schema["created_utc"].lower()
    col = qcol(alias, "created_utc")
    if "timestamp" in dtype or dtype == "date":
        return f"CAST({col} AS DATE)"
    if any(token in dtype for token in ("int", "double", "float", "decimal", "hugeint")):
        return f"CAST(to_timestamp(TRY_CAST({col} AS DOUBLE)) AS DATE)"
    return f"TRY_CAST(substr(CAST({col} AS VARCHAR), 1, 10) AS DATE)"


def read_csv_columns(con: duckdb.DuckDBPyConnection, path: str | Path) -> list[str]:
    """Return CSV column names via DuckDB."""
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_csv_auto({quote_sql_string(path)}, all_varchar=true) LIMIT 0"
    ).fetchall()
    return [str(row[0]) for row in rows]


def create_selected_subreddits_table(con: duckdb.DuckDBPyConnection, csv_path: str | Path) -> int:
    """Create a selected_subreddits temp table from a CSV with a subreddit column."""
    columns = read_csv_columns(con, csv_path)
    require_columns("selected subreddit CSV", columns, ["subreddit"])
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE selected_subreddits AS
        SELECT DISTINCT NULLIF(trim(CAST({quote_identifier("subreddit")} AS VARCHAR)), '') AS subreddit
        FROM read_csv_auto({quote_sql_string(csv_path)}, all_varchar=true)
        WHERE NULLIF(trim(CAST({quote_identifier("subreddit")} AS VARCHAR)), '') IS NOT NULL
        """
    )
    n_selected = int(con.execute("SELECT COUNT(*) FROM selected_subreddits").fetchone()[0])
    if n_selected < 1:
        raise ValueError(f"Selected subreddit CSV is empty after cleaning: {csv_path}")
    return n_selected


def validate_observed_subreddits(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Return observed subreddits absent from selected_subreddits."""
    rows = con.execute(
        """
        WITH observed AS (
          SELECT DISTINCT NULLIF(trim(CAST(subreddit AS VARCHAR)), '') AS subreddit
          FROM comments_core_all
          WHERE NULLIF(trim(CAST(subreddit AS VARCHAR)), '') IS NOT NULL
          UNION
          SELECT DISTINCT NULLIF(trim(CAST(subreddit AS VARCHAR)), '') AS subreddit
          FROM submissions_core_all
          WHERE NULLIF(trim(CAST(subreddit AS VARCHAR)), '') IS NOT NULL
        )
        SELECT observed.subreddit
        FROM observed
        LEFT JOIN selected_subreddits selected
          ON observed.subreddit = selected.subreddit
        WHERE selected.subreddit IS NULL
        ORDER BY observed.subreddit
        LIMIT 50
        """
    ).fetchall()
    return [str(row[0]) for row in rows]


def count_file_rows(con: duckdb.DuckDBPyConnection, path: str | Path) -> int:
    """Count rows in a Parquet file."""
    return int(con.execute(f"SELECT COUNT(*) FROM read_parquet({quote_sql_string(path)})").fetchone()[0])


def add_inventory_row_counts(con: duckdb.DuckDBPyConnection, inventory: list[dict[str, object]]) -> None:
    """Populate row-count fields in the inventory records."""
    for row in inventory:
        if bool(row["comments_core_exists"]):
            row["comments_core_rows"] = count_file_rows(con, str(row["comments_core_path"]))
        if bool(row["submissions_core_exists"]):
            row["submissions_core_rows"] = count_file_rows(con, str(row["submissions_core_path"]))


def create_month_pairs_table(con: duckdb.DuckDBPyConnection, pairs: Sequence[dict[str, object]]) -> None:
    """Create the used feature/prediction month-pair table."""
    used = [(str(row["feature_month"]), str(row["prediction_month"])) for row in pairs if bool(row["used"])]
    if not used:
        raise RuntimeError("No usable adjacent feature/prediction month pairs were found.")
    con.execute("CREATE OR REPLACE TEMP TABLE month_pairs(feature_month VARCHAR, prediction_month VARCHAR)")
    con.executemany("INSERT INTO month_pairs VALUES (?, ?)", used)


def build_panel_table(
    con: duckdb.DuckDBPyConnection,
    cfg: PanelConfig,
    comments_schema: dict[str, str],
    submissions_schema: dict[str, str],
) -> None:
    """Materialise the modelling panel as a DuckDB temp table."""
    author_col = cfg.author_col_resolved
    comments_author_filter = valid_author_filter("src", author_col, comments_schema)
    submissions_author_filter = valid_author_filter("src", author_col, submissions_schema)
    comments_day = created_day_expr("src", comments_schema)
    submissions_day = created_day_expr("src", submissions_schema)
    comments_engager_user = (
        f"CASE WHEN {comments_author_filter} "
        f"THEN CAST({qcol('src', author_col)} AS VARCHAR) "
        "ELSE NULL END"
    )
    target_score_expr = target_score_policy_sql(cfg.target_score_col)
    submission_self_count_expr = "child.engager_user_id IS NOT NULL AND child.engager_user_id = som.user_id"
    comment_self_count_expr = "child.engager_user_id IS NOT NULL AND child.engager_user_id = pcom.user_id"
    submission_event_filter = "NOT is_self_engagement" if cfg.exclude_self_engagement else "TRUE"
    comment_reply_event_filter = "NOT is_self_engagement" if cfg.exclude_self_engagement else "TRUE"

    logger.info("Creating user-level source views")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW comments_core_userlevel AS
        SELECT src.*
        FROM comments_core_all AS src
        WHERE {comments_author_filter}
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW submissions_core_userlevel AS
        SELECT src.*
        FROM submissions_core_all AS src
        WHERE {submissions_author_filter}
        """
    )

    logger.info("Materialising authored activity events")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE activity_events AS
        SELECT
          {qcol("src", "_input_month")} AS month,
          CAST({qcol("src", author_col)} AS VARCHAR) AS user_id,
          NULLIF(trim(CAST({qcol("src", "subreddit")} AS VARCHAR)), '') AS subreddit,
          {submissions_day} AS active_day,
          1::BIGINT AS is_submission,
          0::BIGINT AS is_comment
        FROM submissions_core_userlevel AS src
        UNION ALL
        SELECT
          {qcol("src", "_input_month")} AS month,
          CAST({qcol("src", author_col)} AS VARCHAR) AS user_id,
          NULLIF(trim(CAST({qcol("src", "subreddit")} AS VARCHAR)), '') AS subreddit,
          {comments_day} AS active_day,
          0::BIGINT AS is_submission,
          1::BIGINT AS is_comment
        FROM comments_core_userlevel AS src
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE user_month_activity AS
        SELECT
          month,
          user_id,
          CAST(SUM(is_submission) AS BIGINT) AS n_submissions,
          CAST(SUM(is_comment) AS BIGINT) AS n_comments,
          CAST(COUNT(*) AS BIGINT) AS n_total_activity,
          CAST(COUNT(DISTINCT active_day) AS BIGINT) AS n_active_days,
          CASE
            WHEN COUNT(DISTINCT active_day) > 0
            THEN CAST(COUNT(*) AS DOUBLE) / CAST(COUNT(DISTINCT active_day) AS DOUBLE)
            ELSE NULL
          END AS mean_activity_per_active_day
        FROM activity_events
        GROUP BY 1, 2
        """
    )

    logger.info("Materialising subreddit participation features")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE user_month_subreddit_features AS
        WITH user_subreddit_activity AS (
          SELECT
            month,
            user_id,
            subreddit,
            CAST(COUNT(*) AS BIGINT) AS n_activity
          FROM activity_events
          WHERE subreddit IS NOT NULL
          GROUP BY 1, 2, 3
        ),
        totals AS (
          SELECT month, user_id, SUM(n_activity) AS total_activity
          FROM user_subreddit_activity
          GROUP BY 1, 2
        ),
        shares AS (
          SELECT
            usa.month,
            usa.user_id,
            usa.subreddit,
            usa.n_activity,
            t.total_activity,
            CAST(usa.n_activity AS DOUBLE) / CAST(t.total_activity AS DOUBLE) AS activity_share
          FROM user_subreddit_activity usa
          INNER JOIN totals t
            ON usa.month = t.month AND usa.user_id = t.user_id
        )
        SELECT
          month,
          user_id,
          CAST(COUNT(*) AS BIGINT) AS n_active_subreddits,
          SUM(POWER(activity_share, 2)) AS subreddit_hhi,
          -SUM(CASE WHEN activity_share > 0 THEN activity_share * LN(activity_share) ELSE 0 END) AS subreddit_entropy,
          MAX(activity_share) AS top_subreddit_activity_share,
          CAST(CASE WHEN COUNT(*) = 1 THEN TRUE ELSE FALSE END AS BOOLEAN) AS is_single_subreddit_user
        FROM shares
        GROUP BY 1, 2
        """
    )

    logger.info("Materialising received engagement tables")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE submission_owner_map AS
        SELECT DISTINCT
          {clean_varchar_sql(qcol("src", "id"))} AS submission_id_raw,
          {strip_reddit_prefix_sql(qcol("src", "id"), "t3_")} AS submission_id_norm,
          CAST({qcol("src", author_col)} AS VARCHAR) AS user_id
        FROM submissions_core_userlevel AS src
        WHERE {clean_varchar_sql(qcol("src", "id"))} IS NOT NULL
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE parent_comment_owner_map AS
        SELECT DISTINCT
          {clean_varchar_sql(qcol("src", "id"))} AS comment_id_raw,
          {strip_reddit_prefix_sql(qcol("src", "id"), "t1_")} AS comment_id_norm,
          CAST({qcol("src", author_col)} AS VARCHAR) AS user_id
        FROM comments_core_userlevel AS src
        WHERE {clean_varchar_sql(qcol("src", "id"))} IS NOT NULL
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE comments_engagement_events AS
        SELECT
          {qcol("src", "_input_month")} AS month,
          {clean_varchar_sql(qcol("src", "id"))} AS comment_id_raw,
          {strip_reddit_prefix_sql(qcol("src", "id"), "t1_")} AS comment_id_norm,
          {clean_varchar_sql(qcol("src", "link_id"))} AS link_id_raw,
          {strip_reddit_prefix_sql(qcol("src", "link_id"), "t3_")} AS link_submission_id_norm,
          {clean_varchar_sql(qcol("src", "parent_id"))} AS parent_id_raw,
          {strip_reddit_prefix_sql(qcol("src", "parent_id"), "t1_")} AS parent_comment_id_norm,
          {comments_engager_user} AS engager_user_id
        FROM comments_core_all AS src
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE submission_engagement_events_raw AS
        SELECT DISTINCT
          child.month,
          som.user_id,
          child.comment_id_norm AS child_comment_id,
          child.engager_user_id,
          CAST({submission_self_count_expr} AS BOOLEAN) AS is_self_engagement
        FROM comments_engagement_events AS child
        INNER JOIN submission_owner_map som
          ON child.link_id_raw = som.submission_id_raw
        WHERE child.comment_id_norm IS NOT NULL
          AND child.link_id_raw IS NOT NULL
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE submission_engagement_events_normalized_all AS
        SELECT DISTINCT
          child.month,
          som.user_id,
          child.comment_id_norm AS child_comment_id,
          child.engager_user_id,
          CAST({submission_self_count_expr} AS BOOLEAN) AS is_self_engagement
        FROM comments_engagement_events AS child
        INNER JOIN submission_owner_map som
          ON child.link_submission_id_norm = som.submission_id_norm
        WHERE child.comment_id_norm IS NOT NULL
          AND child.link_submission_id_norm IS NOT NULL
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE submission_engagement_events AS
        SELECT DISTINCT
          month,
          user_id,
          child_comment_id,
          engager_user_id
        FROM submission_engagement_events_normalized_all
        WHERE {submission_event_filter}
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE comment_reply_engagement_events_raw AS
        SELECT DISTINCT
          child.month,
          pcom.user_id,
          child.comment_id_norm AS child_comment_id,
          child.engager_user_id,
          CAST({comment_self_count_expr} AS BOOLEAN) AS is_self_engagement
        FROM comments_engagement_events AS child
        INNER JOIN parent_comment_owner_map pcom
          ON child.parent_id_raw = pcom.comment_id_raw
        WHERE child.comment_id_norm IS NOT NULL
          AND child.parent_id_raw IS NOT NULL
          AND NOT starts_with(child.parent_id_raw, 't3_')
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE comment_reply_engagement_events_normalized_all AS
        SELECT DISTINCT
          child.month,
          pcom.user_id,
          child.comment_id_norm AS child_comment_id,
          child.engager_user_id,
          CAST({comment_self_count_expr} AS BOOLEAN) AS is_self_engagement
        FROM comments_engagement_events AS child
        INNER JOIN parent_comment_owner_map pcom
          ON child.parent_comment_id_norm = pcom.comment_id_norm
        WHERE child.comment_id_norm IS NOT NULL
          AND child.parent_comment_id_norm IS NOT NULL
          AND NOT starts_with(child.parent_id_raw, 't3_')
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE comment_reply_engagement_events AS
        SELECT DISTINCT
          month,
          user_id,
          child_comment_id,
          engager_user_id
        FROM comment_reply_engagement_events_normalized_all
        WHERE {comment_reply_event_filter}
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE user_month_engagement AS
        WITH route_events AS (
          SELECT
            month,
            user_id,
            child_comment_id,
            engager_user_id,
            1::BIGINT AS is_submission_thread_route,
            0::BIGINT AS is_comment_reply_route
          FROM submission_engagement_events
          UNION ALL
          SELECT
            month,
            user_id,
            child_comment_id,
            engager_user_id,
            0::BIGINT AS is_submission_thread_route,
            1::BIGINT AS is_comment_reply_route
          FROM comment_reply_engagement_events
        ),
        component_counts AS (
          SELECT
            month,
            user_id,
            CAST(SUM(is_submission_thread_route) AS BIGINT)
              AS n_submission_thread_comments_received_other_users,
            CAST(SUM(is_comment_reply_route) AS BIGINT)
              AS n_comment_replies_received_other_users,
            CAST(COUNT(DISTINCT CASE
              WHEN is_submission_thread_route = 1 THEN engager_user_id
              ELSE NULL
            END) AS BIGINT) AS n_distinct_commenters_on_submissions_other_users,
            CAST(COUNT(DISTINCT CASE
              WHEN is_comment_reply_route = 1 THEN engager_user_id
              ELSE NULL
            END) AS BIGINT) AS n_distinct_repliers_to_comments_other_users,
            CAST(COUNT(DISTINCT engager_user_id) AS BIGINT)
              AS n_distinct_received_engagers_other_users
          FROM route_events
          GROUP BY 1, 2
        ),
        unique_events AS (
          SELECT
            month,
            user_id,
            child_comment_id,
            CAST(MAX(is_submission_thread_route) AS BIGINT) AS in_submission_thread,
            CAST(MAX(is_comment_reply_route) AS BIGINT) AS in_comment_reply
          FROM route_events
          GROUP BY 1, 2, 3
        ),
        unique_counts AS (
          SELECT
            month,
            user_id,
            CAST(COUNT(*) AS BIGINT) AS n_total_engagement_received_unique_other_users,
            CAST(SUM(CASE
              WHEN in_submission_thread = 1 AND in_comment_reply = 1 THEN 1
              ELSE 0
            END) AS BIGINT) AS n_overlap_submission_and_comment_reply_events
          FROM unique_events
          GROUP BY 1, 2
        )
        SELECT
          c.month,
          c.user_id,
          c.n_submission_thread_comments_received_other_users,
          c.n_comment_replies_received_other_users,
          u.n_total_engagement_received_unique_other_users,
          u.n_overlap_submission_and_comment_reply_events,
          c.n_distinct_commenters_on_submissions_other_users,
          c.n_distinct_repliers_to_comments_other_users,
          c.n_distinct_received_engagers_other_users
        FROM component_counts c
        INNER JOIN unique_counts u
          ON c.month = u.month AND c.user_id = u.user_id
        """
    )

    logger.info("Materialising modelling panel")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE modelling_panel AS
        WITH feature_rows AS (
          SELECT
            fb.user_id,
            mp.feature_month,
            mp.prediction_month,
            mp.feature_month AS feature_window_start_month,
            mp.feature_month AS feature_window_end_month,
            {int(cfg.feature_window_months)}::INTEGER AS feature_window_months,
            {float(cfg.target_threshold_quantile)}::DOUBLE AS target_threshold_quantile,
            {int(cfg.min_prior_activity)}::BIGINT AS min_prior_activity,
            a.n_submissions AS feature_n_submissions_prev,
            a.n_comments AS feature_n_comments_prev,
            a.n_total_activity AS feature_n_total_activity_prev,
            CASE
              WHEN a.n_total_activity > 0
              THEN CAST(a.n_submissions AS DOUBLE) / CAST(a.n_total_activity AS DOUBLE)
              ELSE NULL
            END AS feature_submission_share_prev,
            CASE
              WHEN a.n_total_activity > 0
              THEN CAST(a.n_comments AS DOUBLE) / CAST(a.n_total_activity AS DOUBLE)
              ELSE NULL
            END AS feature_comment_share_prev,
            a.n_active_days AS feature_n_active_days_prev,
            a.mean_activity_per_active_day AS feature_mean_activity_per_active_day_prev,
            COALESCE(e.n_total_engagement_received_unique_other_users, 0)
              AS feature_received_engagement_total_prev,
            CASE
              WHEN a.n_total_activity > 0
              THEN CAST(COALESCE(e.n_total_engagement_received_unique_other_users, 0) AS DOUBLE)
                   / CAST(a.n_total_activity AS DOUBLE)
              ELSE NULL
            END AS feature_received_engagement_per_activity_prev,
            LN(1 + CAST(COALESCE(e.n_total_engagement_received_unique_other_users, 0) AS DOUBLE))
              - LN(1 + CAST(a.n_total_activity AS DOUBLE))
              AS feature_log_received_engagement_rate_prev,
            COALESCE(e.n_submission_thread_comments_received_other_users, 0)
              AS feature_received_comments_on_submissions_prev,
            COALESCE(e.n_comment_replies_received_other_users, 0)
              AS feature_received_replies_to_comments_prev,
            COALESCE(e.n_overlap_submission_and_comment_reply_events, 0)
              AS feature_overlap_submission_and_comment_reply_events_prev,
            COALESCE(s.n_active_subreddits, 0) AS feature_n_active_subreddits_prev,
            s.subreddit_hhi AS feature_subreddit_hhi_prev,
            s.subreddit_entropy AS feature_subreddit_entropy_prev,
            s.top_subreddit_activity_share AS feature_top_subreddit_activity_share_prev,
            COALESCE(s.is_single_subreddit_user, FALSE) AS feature_is_single_subreddit_user_prev,
            CAST(substr(mp.feature_month, 6, 2) AS INTEGER) AS feature_month_number
          FROM month_pairs mp
          INNER JOIN user_month_activity a
            ON a.month = mp.feature_month
          INNER JOIN user_month_activity fb
            ON fb.month = a.month AND fb.user_id = a.user_id
          LEFT JOIN user_month_engagement e
            ON e.month = a.month AND e.user_id = a.user_id
          LEFT JOIN user_month_subreddit_features s
            ON s.month = a.month AND s.user_id = a.user_id
          WHERE a.n_total_activity >= {int(cfg.min_prior_activity)}
        ),
        target_joined AS (
          SELECT
            f.*,
            COALESCE(ta.n_total_activity, 0)::BIGINT AS target_activity,
            COALESCE(te.n_total_engagement_received_unique_other_users, 0)::BIGINT
              AS target_received_engagement_total,
            COALESCE(te.n_submission_thread_comments_received_other_users, 0)::BIGINT
              AS target_received_comments_on_submissions,
            COALESCE(te.n_comment_replies_received_other_users, 0)::BIGINT
              AS target_received_replies_to_comments,
            COALESCE(te.n_overlap_submission_and_comment_reply_events, 0)::BIGINT
              AS target_overlap_submission_and_comment_reply_events,
            COALESCE(te.n_distinct_commenters_on_submissions_other_users, 0)::BIGINT
              AS target_distinct_commenters_on_submissions,
            COALESCE(te.n_distinct_repliers_to_comments_other_users, 0)::BIGINT
              AS target_distinct_repliers_to_comments,
            COALESCE(te.n_distinct_received_engagers_other_users, 0)::BIGINT
              AS target_distinct_received_engagers,
            CASE
              WHEN COALESCE(ta.n_total_activity, 0) > 0
              THEN CAST(COALESCE(te.n_total_engagement_received_unique_other_users, 0) AS DOUBLE)
                   / CAST(ta.n_total_activity AS DOUBLE)
              ELSE NULL
            END AS target_received_engagement_per_activity,
            CAST(COALESCE(ta.n_total_activity, 0) > 0 AS BOOLEAN) AS target_active_in_prediction_month
          FROM feature_rows f
          LEFT JOIN user_month_activity ta
            ON ta.month = f.prediction_month AND ta.user_id = f.user_id
          LEFT JOIN user_month_engagement te
            ON te.month = f.prediction_month AND te.user_id = f.user_id
        ),
        scored AS (
          SELECT
            *,
            CAST({quote_sql_string(cfg.target_score_col)} AS VARCHAR) AS target_score_col,
            CAST({target_score_expr} AS DOUBLE) AS target_score
          FROM target_joined
        ),
        ranked AS (
          SELECT
            *,
            ROW_NUMBER() OVER (
              PARTITION BY prediction_month
              ORDER BY target_score DESC, user_id ASC
            ) AS target_rank_within_month,
            CAST(COUNT(*) OVER (PARTITION BY prediction_month) AS BIGINT) AS n_prediction_population
          FROM scored
        ),
        with_expected AS (
          SELECT
            *,
            CAST(CEIL(
              CAST(n_prediction_population AS DOUBLE) * (1.0 - {float(cfg.target_threshold_quantile)})
            ) AS BIGINT) AS expected_positive_count
          FROM ranked
        ),
        thresholds AS (
          SELECT
            prediction_month,
            MAX(CASE
              WHEN target_rank_within_month = expected_positive_count THEN target_score
              ELSE NULL
            END) AS target_threshold_value
          FROM with_expected
          GROUP BY 1
        )
        SELECT
          w.user_id,
          w.feature_month,
          w.prediction_month,
          w.feature_window_start_month,
          w.feature_window_end_month,
          w.feature_window_months,
          w.target_threshold_quantile,
          w.min_prior_activity,
          w.feature_n_submissions_prev,
          w.feature_n_comments_prev,
          w.feature_n_total_activity_prev,
          w.feature_submission_share_prev,
          w.feature_comment_share_prev,
          w.feature_n_active_days_prev,
          w.feature_mean_activity_per_active_day_prev,
          w.feature_received_engagement_total_prev,
          w.feature_received_engagement_per_activity_prev,
          w.feature_log_received_engagement_rate_prev,
          w.feature_received_comments_on_submissions_prev,
          w.feature_received_replies_to_comments_prev,
          w.feature_overlap_submission_and_comment_reply_events_prev,
          w.feature_n_active_subreddits_prev,
          w.feature_subreddit_hhi_prev,
          w.feature_subreddit_entropy_prev,
          w.feature_top_subreddit_activity_share_prev,
          w.feature_is_single_subreddit_user_prev,
          w.feature_month_number,
          w.target_activity,
          w.target_received_engagement_total,
          w.target_received_comments_on_submissions,
          w.target_received_replies_to_comments,
          w.target_overlap_submission_and_comment_reply_events,
          w.target_distinct_commenters_on_submissions,
          w.target_distinct_repliers_to_comments,
          w.target_distinct_received_engagers,
          w.target_received_engagement_per_activity,
          w.target_received_engagement_per_activity AS target_observed_received_engagement_per_activity,
          w.target_active_in_prediction_month,
          w.target_rank_within_month,
          t.target_threshold_value,
          CASE
            WHEN w.target_rank_within_month <= w.expected_positive_count THEN 1
            ELSE 0
          END::INTEGER AS label_high_engagement,
          w.target_score,
          w.target_score_col
        FROM with_expected w
        INNER JOIN thresholds t
          ON w.prediction_month = t.prediction_month
        """
    )


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


def write_feature_schema(path: str | Path) -> None:
    """Write the feature schema CSV."""
    fieldnames = ("feature_name", "feature_group", "source_window", "description")
    write_csv_rows(path, FEATURE_SCHEMA_ROWS, fieldnames)


def write_panel_outputs(con: duckdb.DuckDBPyConnection, out_root: Path) -> None:
    """Write panel and derived summary outputs."""
    panel_path = out_root / "modelling_panel.parquet"
    con.execute(
        f"""
        COPY (
          SELECT *
          FROM modelling_panel
          ORDER BY prediction_month, target_rank_within_month, user_id
        )
        TO {quote_sql_string(panel_path)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    logger.info("Wrote modelling panel: %s", panel_path)

    write_feature_schema(out_root / "feature_schema.csv")

    copy_query_to_csv(
        con,
        """
        SELECT
          m.prediction_month,
          MAX(m.feature_month) AS feature_month,
          COUNT(*) AS n_prediction_population,
          SUM(m.label_high_engagement) AS n_high_engagement,
          AVG(CAST(m.label_high_engagement AS DOUBLE)) AS label_prevalence,
          MAX(m.target_threshold_quantile) AS target_threshold_quantile,
          MAX(m.target_threshold_value) AS target_threshold_value,
          SUM(CASE WHEN m.target_active_in_prediction_month THEN 1 ELSE 0 END) AS n_active_in_prediction_month,
          SUM(CASE WHEN NOT m.target_active_in_prediction_month THEN 1 ELSE 0 END) AS n_inactive_in_prediction_month,
          AVG(CASE WHEN m.target_active_in_prediction_month THEN 1.0 ELSE 0.0 END)
            AS share_active_in_prediction_month,
          SUM(CASE
            WHEN m.target_active_in_prediction_month AND m.target_received_engagement_total = 0 THEN 1
            ELSE 0
          END) AS n_active_zero_received_engagement,
          AVG(CASE
            WHEN m.target_active_in_prediction_month
            THEN CASE WHEN m.target_received_engagement_total = 0 THEN 1.0 ELSE 0.0 END
            ELSE NULL
          END) AS share_active_zero_received_engagement,
          SUM(CASE
            WHEN NOT m.target_active_in_prediction_month AND m.target_score = 0 THEN 1
            ELSE 0
          END) AS n_inactive_zero_ranking_score,
          SUM(CASE
            WHEN NOT m.target_active_in_prediction_month AND m.target_received_engagement_total > 0 THEN 1
            ELSE 0
          END) AS n_inactive_nonzero_observed_received_engagement,
          SUM(m.target_received_engagement_total) AS total_target_received_engagement,
          SUM(m.target_received_comments_on_submissions) AS total_target_received_comments_on_submissions,
          SUM(m.target_received_replies_to_comments) AS total_target_received_replies_to_comments,
          SUM(m.target_received_comments_on_submissions + m.target_received_replies_to_comments)
            AS total_target_received_engagement_component_sum,
          SUM(m.target_overlap_submission_and_comment_reply_events)
            AS total_target_overlap_submission_and_comment_reply_events,
          AVG(CASE
            WHEN m.target_received_comments_on_submissions + m.target_received_replies_to_comments > 0
            THEN CAST(m.target_overlap_submission_and_comment_reply_events AS DOUBLE)
                 / CAST(m.target_received_comments_on_submissions + m.target_received_replies_to_comments AS DOUBLE)
            ELSE NULL
          END) AS mean_user_overlap_share_of_component_total,
          SUM(m.target_distinct_commenters_on_submissions) AS total_target_distinct_commenters_on_submissions,
          SUM(m.target_distinct_repliers_to_comments) AS total_target_distinct_repliers_to_comments,
          SUM(m.target_distinct_received_engagers) AS total_target_distinct_received_engagers,
          AVG(CAST(m.target_activity AS DOUBLE)) AS mean_target_activity,
          MEDIAN(CAST(m.target_activity AS DOUBLE)) AS median_target_activity,
          AVG(CAST(m.target_received_engagement_total AS DOUBLE)) AS mean_target_received_engagement_total,
          MEDIAN(CAST(m.target_received_engagement_total AS DOUBLE)) AS median_target_received_engagement_total,
          QUANTILE_CONT(CAST(m.target_received_engagement_total AS DOUBLE), 0.90) AS p90_target_received_engagement_total,
          MAX(m.target_received_engagement_total) AS max_target_received_engagement_total,
          AVG(m.target_received_engagement_per_activity)
            AS mean_target_received_engagement_per_activity_among_active,
          MEDIAN(m.target_received_engagement_per_activity)
            AS median_target_received_engagement_per_activity_among_active,
          QUANTILE_CONT(m.target_received_engagement_per_activity, 0.90)
            AS p90_target_received_engagement_per_activity_among_active,
          MAX(m.target_received_engagement_per_activity)
            AS max_target_received_engagement_per_activity_among_active,
          SUM(CASE WHEN m.target_received_engagement_total > 0 THEN 1 ELSE 0 END)
            AS n_nonzero_received_engagement,
          AVG(CASE WHEN m.target_received_engagement_total > 0 THEN 1.0 ELSE 0.0 END)
            AS share_nonzero_received_engagement,
          AVG(CASE
            WHEN m.target_active_in_prediction_month
            THEN CASE WHEN m.target_received_engagement_total > 0 THEN 1.0 ELSE 0.0 END
            ELSE NULL
          END) AS share_active_nonzero_received_engagement
        FROM modelling_panel m
        GROUP BY 1
        ORDER BY 1
        """,
        out_root / "target_summary_by_month.csv",
    )

    copy_query_to_csv(
        con,
        """
        WITH months AS (
          SELECT DISTINCT month
          FROM comments_engagement_events
        ),
        submission_raw AS (
          SELECT month, COUNT(*) AS raw_matched_submission_thread_events
          FROM submission_engagement_events_raw
          GROUP BY 1
        ),
        submission_normalized AS (
          SELECT
            month,
            COUNT(*) AS normalised_matched_submission_thread_events,
            SUM(CASE WHEN is_self_engagement THEN 1 ELSE 0 END)
              AS self_submission_thread_events_excluded
          FROM submission_engagement_events_normalized_all
          GROUP BY 1
        ),
        direct_raw AS (
          SELECT month, COUNT(*) AS raw_matched_direct_reply_events
          FROM comment_reply_engagement_events_raw
          GROUP BY 1
        ),
        direct_normalized AS (
          SELECT
            month,
            COUNT(*) AS normalised_matched_direct_reply_events,
            SUM(CASE WHEN is_self_engagement THEN 1 ELSE 0 END)
              AS self_direct_reply_events_excluded
          FROM comment_reply_engagement_events_normalized_all
          GROUP BY 1
        ),
        submission_final AS (
          SELECT
            month,
            COUNT(*) AS submission_thread_events_after_self_exclusion,
            COUNT(DISTINCT engager_user_id) AS distinct_commenters_on_submissions
          FROM submission_engagement_events
          GROUP BY 1
        ),
        direct_final AS (
          SELECT
            month,
            COUNT(*) AS direct_reply_events_after_self_exclusion,
            COUNT(DISTINCT engager_user_id) AS distinct_repliers_to_comments
          FROM comment_reply_engagement_events
          GROUP BY 1
        ),
        received_distinct AS (
          SELECT
            month,
            COUNT(DISTINCT engager_user_id) AS distinct_received_engagers
          FROM (
            SELECT month, engager_user_id FROM submission_engagement_events
            UNION ALL
            SELECT month, engager_user_id FROM comment_reply_engagement_events
          )
          GROUP BY 1
        ),
        unique_totals AS (
          SELECT
            month,
            SUM(n_total_engagement_received_unique_other_users)
              AS unique_total_after_deduplication,
            SUM(n_overlap_submission_and_comment_reply_events)
              AS overlap_submission_and_comment_reply_events
          FROM user_month_engagement
          GROUP BY 1
        )
        SELECT
          m.month,
          COALESCE(sr.raw_matched_submission_thread_events, 0)
            AS raw_matched_submission_thread_events,
          COALESCE(sn.normalised_matched_submission_thread_events, 0)
            AS normalised_matched_submission_thread_events,
          COALESCE(sn.self_submission_thread_events_excluded, 0)
            AS self_submission_thread_events_excluded,
          COALESCE(dr.raw_matched_direct_reply_events, 0)
            AS raw_matched_direct_reply_events,
          COALESCE(dn.normalised_matched_direct_reply_events, 0)
            AS normalised_matched_direct_reply_events,
          COALESCE(dn.self_direct_reply_events_excluded, 0)
            AS self_direct_reply_events_excluded,
          COALESCE(sf.submission_thread_events_after_self_exclusion, 0)
            + COALESCE(df.direct_reply_events_after_self_exclusion, 0)
            AS component_total_after_self_exclusion,
          COALESCE(ut.unique_total_after_deduplication, 0)
            AS unique_total_after_deduplication,
          COALESCE(ut.overlap_submission_and_comment_reply_events, 0)
            AS overlap_submission_and_comment_reply_events,
          CASE
            WHEN COALESCE(sf.submission_thread_events_after_self_exclusion, 0)
                 + COALESCE(df.direct_reply_events_after_self_exclusion, 0) > 0
            THEN CAST(COALESCE(ut.overlap_submission_and_comment_reply_events, 0) AS DOUBLE)
                 / CAST(
                     COALESCE(sf.submission_thread_events_after_self_exclusion, 0)
                     + COALESCE(df.direct_reply_events_after_self_exclusion, 0)
                     AS DOUBLE
                   )
            ELSE NULL
          END AS overlap_share_of_component_total,
          COALESCE(sf.distinct_commenters_on_submissions, 0)
            AS distinct_commenters_on_submissions,
          COALESCE(df.distinct_repliers_to_comments, 0)
            AS distinct_repliers_to_comments,
          COALESCE(rd.distinct_received_engagers, 0)
            AS distinct_received_engagers
        FROM months m
        LEFT JOIN submission_raw sr
          ON sr.month = m.month
        LEFT JOIN submission_normalized sn
          ON sn.month = m.month
        LEFT JOIN direct_raw dr
          ON dr.month = m.month
        LEFT JOIN direct_normalized dn
          ON dn.month = m.month
        LEFT JOIN submission_final sf
          ON sf.month = m.month
        LEFT JOIN direct_final df
          ON df.month = m.month
        LEFT JOIN received_distinct rd
          ON rd.month = m.month
        LEFT JOIN unique_totals ut
          ON ut.month = m.month
        ORDER BY m.month
        """,
        out_root / "engagement_attribution_diagnostics.csv",
    )

    copy_query_to_csv(
        con,
        """
        SELECT
          prediction_month,
          MAX(feature_month) AS feature_month,
          COUNT(*) AS n_prediction_population,
          MIN(feature_n_total_activity_prev) AS min_prior_activity_observed,
          AVG(CAST(feature_n_total_activity_prev AS DOUBLE)) AS mean_prior_activity,
          MEDIAN(CAST(feature_n_total_activity_prev AS DOUBLE)) AS median_prior_activity,
          SUM(feature_n_total_activity_prev) AS total_prior_activity,
          SUM(CASE WHEN target_active_in_prediction_month THEN 1 ELSE 0 END) AS n_active_in_prediction_month,
          SUM(CASE WHEN NOT target_active_in_prediction_month THEN 1 ELSE 0 END) AS n_inactive_in_prediction_month,
          AVG(CASE WHEN target_active_in_prediction_month THEN 1.0 ELSE 0.0 END) AS share_active_in_prediction_month,
          SUM(CASE
            WHEN target_active_in_prediction_month AND target_received_engagement_total = 0 THEN 1
            ELSE 0
          END) AS n_active_zero_received_engagement,
          SUM(CASE
            WHEN NOT target_active_in_prediction_month AND target_score = 0 THEN 1
            ELSE 0
          END) AS n_inactive_zero_ranking_score,
          SUM(CASE WHEN target_score = 0 THEN 1 ELSE 0 END) AS n_zero_target_score,
          AVG(CASE WHEN target_score = 0 THEN 1.0 ELSE 0.0 END) AS share_zero_target_score
        FROM modelling_panel
        GROUP BY 1
        ORDER BY 1
        """,
        out_root / "prediction_population_summary_by_month.csv",
    )


def panel_validation_rows(
    con: duckdb.DuckDBPyConnection,
    target_score_col: str,
    smoke: bool,
) -> list[dict[str, object]]:
    """Return compact validation summary rows for the just-built panel."""
    rows: list[dict[str, object]] = []

    def add(check: str, status: str, observed: object, expected: object, details: str = "") -> None:
        rows.append(
            {
                "check_name": check,
                "status": status,
                "observed_value": observed,
                "expected_value": expected,
                "details": details,
            }
        )

    n_rows = int(con.execute("SELECT COUNT(*) FROM modelling_panel").fetchone()[0])
    add("row_count_positive", "PASS" if n_rows > 0 else "FAIL", n_rows, ">0")

    duplicate_rows = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM (
              SELECT user_id, prediction_month, COUNT(*) AS n
              FROM modelling_panel
              GROUP BY 1, 2
              HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
    )
    add("duplicate_user_prediction_month_rows", "PASS" if duplicate_rows == 0 else "FAIL", duplicate_rows, 0)

    non_preceding = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM modelling_panel
            WHERE feature_month >= prediction_month
            """
        ).fetchone()[0]
    )
    add("feature_month_precedes_prediction_month", "PASS" if non_preceding == 0 else "FAIL", non_preceding, 0)

    null_users = int(
        con.execute("SELECT COUNT(*) FROM modelling_panel WHERE user_id IS NULL").fetchone()[0]
    )
    add("user_id_non_null", "PASS" if null_users == 0 else "FAIL", null_users, 0)

    target_score_expected = target_score_policy_sql(target_score_col)
    mismatch = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM modelling_panel
            WHERE ABS(COALESCE(target_score, -999999999.0) - COALESCE({target_score_expected}, -999999999.0)) > 1e-12
               OR target_score_col <> {quote_sql_string(target_score_col)}
            """
        ).fetchone()[0]
    )
    add("target_score_matches_configured_policy", "PASS" if mismatch == 0 else "FAIL", mismatch, 0)

    target_unique_gt_components = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM modelling_panel
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
            FROM modelling_panel
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

    feature_overlap_mismatches = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM modelling_panel
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
            FROM modelling_panel
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
            FROM modelling_panel
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
            FROM modelling_panel
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
            FROM modelling_panel
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
        con.execute(
            "SELECT COALESCE(SUM(target_received_engagement_total), 0) FROM modelling_panel"
        ).fetchone()[0]
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
        ">0 unless tiny smoke run",
        f"tiny_smoke_row_threshold={TINY_SMOKE_PANEL_ROW_THRESHOLD}",
    )

    positive_mismatches = int(
        con.execute(
            """
            WITH by_month AS (
              SELECT
                prediction_month,
                COUNT(*) AS n_prediction_population,
                SUM(label_high_engagement) AS observed_positives,
                CAST(CEIL(COUNT(*) * (1.0 - MAX(target_threshold_quantile))) AS BIGINT) AS expected_positives
              FROM modelling_panel
              GROUP BY 1
            )
            SELECT COUNT(*)
            FROM by_month
            WHERE observed_positives <> expected_positives
            """
        ).fetchone()[0]
    )
    add("positive_count_matches_threshold_rule", "PASS" if positive_mismatches == 0 else "FAIL", positive_mismatches, 0)

    return rows


def write_panel_validation_summary(
    con: duckdb.DuckDBPyConnection,
    out_root: Path,
    target_score_col: str,
    smoke: bool,
) -> None:
    """Write panel validation summary CSV."""
    rows = panel_validation_rows(con, target_score_col, smoke)
    write_csv_rows(
        out_root / "panel_validation_summary.csv",
        rows,
        ("check_name", "status", "observed_value", "expected_value", "details"),
    )


def write_inventory_outputs(
    out_root: Path,
    inventory: Sequence[dict[str, object]],
    month_pairs: Sequence[dict[str, object]],
) -> None:
    """Write input and month-pair inventories."""
    write_csv_rows(
        out_root / "input_inventory.csv",
        inventory,
        (
            "month",
            "comments_core_path",
            "comments_core_exists",
            "comments_core_rows",
            "submissions_core_path",
            "submissions_core_exists",
            "submissions_core_rows",
        ),
    )
    write_csv_rows(
        out_root / "month_pair_inventory.csv",
        month_pairs,
        (
            "feature_month",
            "prediction_month",
            "feature_comments_core_path",
            "feature_submissions_core_path",
            "prediction_comments_core_path",
            "prediction_submissions_core_path",
            "feature_comments_core_exists",
            "feature_submissions_core_exists",
            "prediction_comments_core_exists",
            "prediction_submissions_core_exists",
            "used",
            "status",
        ),
    )


def package_versions() -> dict[str, Optional[str]]:
    """Return package versions relevant to this script."""
    versions: dict[str, Optional[str]] = {"python": sys.version.replace("\n", " ")}
    for package in ("duckdb",):
        try:
            module = __import__(package)
        except Exception:
            versions[package] = None
        else:
            versions[package] = getattr(module, "__version__", "installed")
    return versions


def metadata_payload(
    cfg: PanelConfig,
    duck: DuckDBConfig,
    run_started_at: str,
    selected_subreddit_count: Optional[int],
    prediction_months: Sequence[str],
) -> dict[str, object]:
    """Build run metadata payload."""
    target_definition = {
        "target_score_col": cfg.target_score_col,
        "target_score_alias": "target_score",
        "available_target_score_columns": list(TARGET_SCORE_COLS),
        "threshold_quantile": cfg.target_threshold_quantile,
        "label": "label_high_engagement",
        "label_denominator": (
            "prediction population within each prediction month after applying prior-activity eligibility"
        ),
        "positive_count_rule": "ceil(n_prediction_population * (1 - target_threshold_quantile))",
        "inactive_user_score_policy": cfg.target_inactive_user_score_policy,
        "observed_per_activity_inactive_policy": cfg.target_observed_per_activity_inactive_policy,
        "target_score_policy": (
            "active users use the configured observed target-score column; inactive users are ranked with score 0"
        ),
        "target_active_column": "target_active_in_prediction_month",
        "received_engagement_components": {
            "submission_thread_comments": "other-user child comments received on the user's submissions",
            "comment_direct_replies": "other-user child comments directly replying to the user's comments",
            "total": (
                "unique other-user child comments per recipient user-month; "
                "submission/direct overlap is de-duplicated"
            ),
            "overlap": "child comments attributed through both component routes",
            "distinct_engagers": "audit/sensitivity counts, not the primary event total",
        },
        "id_prefix_diagnostics": {
            "current_blocker": False,
            "builder_join_policy": "compare both raw and normalised Reddit fullnames for diagnostics; use normalised IDs for attribution",
            "submission_join_normalisation": "strip leading t3_ from submission ids and comment link_id values",
            "comment_join_normalisation": "strip leading t1_ from comment ids and comment parent_id values",
            "diagnostics_output": "engagement_attribution_diagnostics.csv",
        },
        "exclude_self_engagement": cfg.exclude_self_engagement,
    }
    return {
        "script_name": "Modelling/0_0_make_modelling_panel.py",
        "run_timestamp_utc": run_started_at,
        "git": git_metadata(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "duckdb": asdict(duck),
        "input_root": cfg.in_root,
        "selected_subreddits_csv": cfg.selected_subreddits_csv,
        "selected_subreddits_count": selected_subreddit_count,
        "output_root": cfg.out_root,
        "year": cfg.year,
        "months": list(cfg.months),
        "feature_window": {
            "feature_window_months": cfg.feature_window_months,
            "feature_month_relation": "t_minus_1",
        },
        "prediction_months": list(prediction_months),
        "target_definition": target_definition,
        "target_inactive_user_score_policy": cfg.target_inactive_user_score_policy,
        "eligibility_definition": {
            "prediction_population": "users active in feature month t-1",
            "min_prior_activity": cfg.min_prior_activity,
        },
        "split_definition": cfg.split_definition,
        "random_seed": cfg.random_seed,
        "known_caveats": list(cfg.known_caveats),
        "author_col_requested": cfg.author_col_requested,
        "author_col_resolved": cfg.author_col_resolved,
        "author_identifier_pseudonymised": cfg.author_col_resolved == "author_pid",
        "smoke": cfg.smoke,
        "smoke_max_rows": cfg.smoke_max_rows if cfg.smoke else 0,
        "outputs": {name: str(Path(cfg.out_root) / name) for name in PANEL_OUTPUTS},
    }


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Build the initial Study 2 prospective user-month modelling panel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--in-root", required=True, help="Input root containing monthly selected subreddit Parquets.")
    parser.add_argument("--out-root", required=True, help="Output root for modelling panel artefacts.")
    parser.add_argument(
        "--selected-subreddits-csv",
        default=None,
        help="Optional CSV with a subreddit column. Used for validation only; unexpected observed subreddits fail.",
    )
    parser.add_argument("--year", default="2021", help="Dataset year.")
    parser.add_argument(
        "--months",
        "--only-months",
        dest="months",
        default="",
        help="Comma-separated YYYY-MM months to include. Defaults to all months in --year.",
    )
    parser.add_argument("--feature-window-months", type=int, default=1, help="Feature window length in months.")
    parser.add_argument("--target-threshold-quantile", type=float, default=0.90, help="Top-tail threshold quantile.")
    parser.add_argument(
        "--target-score-col",
        default="target_received_engagement_per_activity",
        choices=TARGET_SCORE_COLS,
        help="Target audit column copied into target_score for ranking/evaluation.",
    )
    parser.add_argument("--min-prior-activity", type=int, default=1, help="Minimum t-1 activity for eligibility.")
    parser.add_argument(
        "--author-col",
        default="auto",
        choices=["auto", "author_pid", "author", "user_id"],
        help="Author identifier column to use. Non-smoke production runs require author_pid; auto never falls back to raw identifiers.",
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
        help="DuckDB temp_directory. Prefer node-local scratch on HPC.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite deterministic outputs.")
    parser.add_argument("--smoke", action="store_true", help="Enable smoke-mode metadata and row limiting if set.")
    parser.add_argument("--smoke-max-rows", type=int, default=0, help="If >0 and --smoke, limit rows read per file.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run panel construction."""
    parser = build_argparser()
    args = parser.parse_args(argv)
    validate_cli(args, parser)
    setup_logging(args.log_level)

    months = parse_months(args.months) if str(args.months).strip() else default_year_months(str(args.year))
    non_year_months = [month for month in months if not month.startswith(f"{args.year}-")]
    if non_year_months:
        parser.error(f"All selected months must belong to --year {args.year}: {', '.join(non_year_months)}")

    out_root = Path(args.out_root)
    output_paths = expected_output_paths(out_root)
    check_output_collisions(output_paths, bool(args.overwrite))
    remove_outputs(output_paths, bool(args.overwrite))
    ensure_dir(out_root)

    inventory = build_input_inventory(str(args.in_root), months)
    month_pairs = build_month_pair_inventory(inventory)
    used_pairs = [pair for pair in month_pairs if bool(pair["used"])]
    if not used_pairs:
        write_inventory_outputs(out_root, inventory, month_pairs)
        raise RuntimeError(
            "No usable adjacent feature/prediction month pairs found. "
            "Supply at least two consecutive months with both comments_core and submissions_core files."
        )

    used_months = used_months_from_pairs(month_pairs)
    run_started_at = utc_now_iso()
    duck = DuckDBConfig(
        memory_limit=str(args.duckdb_memory_limit),
        threads=int(args.duckdb_threads),
        temp_dir=str(args.duckdb_temp_dir),
    )

    logger.info("Input root: %s", args.in_root)
    logger.info("Output root: %s", args.out_root)
    logger.info("Selected months: %s", ", ".join(months))
    logger.info("Used month pairs: %s", len(used_pairs))
    if args.smoke:
        logger.warning(
            "Smoke mode enabled. Per-file row limit: %s",
            int(args.smoke_max_rows) if int(args.smoke_max_rows) > 0 else "none",
        )

    selected_subreddit_count: Optional[int] = None
    prediction_months = tuple(str(pair["prediction_month"]) for pair in used_pairs)

    con = duckdb.connect()
    try:
        configure_duckdb(con, duck)
        add_inventory_row_counts(con, inventory)
        write_inventory_outputs(out_root, inventory, month_pairs)

        create_union_view(
            con,
            "comments_core_all",
            inventory,
            "comments",
            used_months,
            int(args.smoke_max_rows) if args.smoke else 0,
        )
        create_union_view(
            con,
            "submissions_core_all",
            inventory,
            "submissions",
            used_months,
            int(args.smoke_max_rows) if args.smoke else 0,
        )

        comments_schema = describe_relation(con, "comments_core_all")
        submissions_schema = describe_relation(con, "submissions_core_all")
        require_columns(
            "comments_core inputs",
            comments_schema,
            ["id", "link_id", "parent_id", "subreddit", "_input_month"],
        )
        require_columns("submissions_core inputs", submissions_schema, ["id", "subreddit", "_input_month"])
        author_col = resolve_author_col(comments_schema, submissions_schema, str(args.author_col))
        enforce_production_author_col(author_col, bool(args.smoke))

        if args.selected_subreddits_csv:
            selected_subreddit_count = create_selected_subreddits_table(con, str(args.selected_subreddits_csv))
            unexpected = validate_observed_subreddits(con)
            if unexpected:
                raise ValueError(
                    "Observed subreddit(s) not present in --selected-subreddits-csv. "
                    "The panel builder validates but does not silently filter. First unexpected values: "
                    + ", ".join(unexpected)
                )
            logger.info("Selected subreddit validation passed for %s subreddit(s).", selected_subreddit_count)

        cfg = PanelConfig(
            in_root=str(args.in_root),
            out_root=str(args.out_root),
            selected_subreddits_csv=str(args.selected_subreddits_csv) if args.selected_subreddits_csv else None,
            year=str(args.year),
            months=months,
            feature_window_months=int(args.feature_window_months),
            target_threshold_quantile=float(args.target_threshold_quantile),
            target_score_col=str(args.target_score_col),
            min_prior_activity=int(args.min_prior_activity),
            author_col_requested=str(args.author_col),
            author_col_resolved=author_col,
            overwrite=bool(args.overwrite),
            smoke=bool(args.smoke),
            smoke_max_rows=int(args.smoke_max_rows),
            target_inactive_user_score_policy=TARGET_INACTIVE_USER_SCORE_POLICY,
            target_observed_per_activity_inactive_policy=TARGET_OBSERVED_PER_ACTIVITY_INACTIVE_POLICY,
            exclude_self_engagement=EXCLUDE_SELF_ENGAGEMENT,
            split_definition=SPLIT_DEFINITION,
            random_seed=None,
            known_caveats=(
                "Initial scaffold only supports one-month feature windows.",
                "No train/validation/test split is assigned by the panel builder.",
                "Selected subreddit CSV is used for validation only, not silent filtering.",
            ),
        )

        create_month_pairs_table(con, month_pairs)
        build_panel_table(con, cfg, comments_schema, submissions_schema)
        write_panel_outputs(con, out_root)
        write_panel_validation_summary(con, out_root, cfg.target_score_col, cfg.smoke)

        metadata = metadata_payload(cfg, duck, run_started_at, selected_subreddit_count, prediction_months)
        safe_json_dump({"config": asdict(cfg), "duckdb": asdict(duck)}, out_root / "resolved_config.json")
        safe_json_dump(metadata, out_root / "run_metadata.json")
    finally:
        con.close()

    logger.info("Done. Wrote modelling outputs under %s", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
