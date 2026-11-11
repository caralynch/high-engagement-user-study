#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Cara Lynch
# See the LICENSE file for details.
"""
Preprocessing step 6.2: build subreddit monthly activity and threshold summaries.

This script creates a month-by-subreddit activity table from cleaned Reddit
Parquet files and evaluates candidate subreddit eligibility thresholds for the
2021 high-engagement user modelling pipeline.

Core definitions
----------------
Activity event
    One authored submission or one authored comment.

Subreddit-month active users
    The number of distinct non-deleted, non-null author identifiers appearing in
    either submissions or comments for a subreddit in a month. This is counted
    exactly across submissions + comments using COUNT(DISTINCT author_key) after
    UNION ALL, not by adding separate distinct counts.

Content volume
    Submission/comment/activity counts retain content rows regardless of deleted
    author status. Deleted/null authors are excluded only from the active-user
    denominator by default.

Two threshold views
-------------------
1. Selected-subreddit / keep-all-data view:
   Select subreddits that satisfy the monthly threshold in at least N selected
   input months. Once selected, all observed monthly activity for those
   subreddits is retained, including months where that subreddit did not satisfy
   the threshold.

2. Qualifying-subreddit-month view:
   Retain only subreddit-month rows that individually satisfy the threshold.

Inputs
------
Expected default layout:

  <in-root>/
    submissions/submissions_core_YYYY-MM.parquet
    comments/comments_core_YYYY-MM.parquet

Default concrete month paths are used, so `.parquet.bak` files are not scanned.

Outputs
-------
Under <out-root>/:

  subreddit_monthly_activity.parquet
  threshold_retention_summary.csv
  threshold_retention_by_month.csv
  run_metadata.json

The script also writes a timestamped copy of the run metadata under
<out-root>/audit/ for provenance.

Command-line arguments
----------------------
--in-root
    Input root containing cleaned monthly Parquet outputs.
--out-root
    Output directory.
--year
    Dataset year (YYYY). Defaults to 2021.
--only-months
    Comma-separated subset of months to process, for example 2021-01,2021-06.
--limit-months
    Limit number of months after applying --only-months; 0 means no limit.
--submissions-glob
    Override submissions Parquet glob/path.
--comments-glob
    Override comments Parquet glob/path.
--author-col
    Author identifier column to use for active-user counts. Choices are auto,
    author_pid, author, and user_id.
--include-deleted-authors-in-user-count
    Count [deleted]/u_deleted as an active user. By default, deleted/null
    authors are excluded from n_users.
--include-user-subreddits
    Do not exclude user-profile subreddits beginning with u_.
--min-users-grid
    Comma-separated monthly active-user thresholds. Include 0 for
    activity-only thresholding.
--min-activity-grid
    Comma-separated monthly activity thresholds. Include 0 for user-only
    thresholding.
--required-months
    Number of selected input months a subreddit must satisfy thresholds. Use all
    for all selected input months.
--overwrite
    Overwrite existing outputs.
--smoke
    Enable smoke mode.
--smoke-max-rows
    If greater than 0 and --smoke is set, limit rows read per relation.
--duckdb-memory-limit
    DuckDB memory limit. Defaults to DUCKDB_MEMORY_LIMIT or 16GB.
--duckdb-threads
    DuckDB threads. Defaults to DUCKDB_THREADS or 4.
--duckdb-temp-dir
    DuckDB temp_directory. Prefer node-local $TMPDIR on shared compute systems. Defaults to
    DUCKDB_TEMP_DIR or duckdb_tmp.
--log-level
    Logging level. Choices are DEBUG, INFO, WARNING, ERROR, and CRITICAL.

Methodological cautions
-----------------------
- Month is parsed from input file basenames by default. This is deliberate
  because the cleaned monthly files are named by month and `created_utc` may be
  an integer epoch, timestamp, or absent depending on cleaning stage. Parent
  directories are ignored so run-date tokens cannot be mistaken for data months.
- User-profile subreddits whose subreddit name begins with `u_` are excluded by
  default because they are not comparable to ordinary topic-based communities.
- If the selected author identifier differs between submissions and comments,
  the script fails by default because cross-relation distinct user counts would
  not be comparable.
- `--required-months all` means all selected input months after `--only-months`
  and `--limit-months` are resolved. It is deliberately not inferred from the
  generated monthly table, so an empty monthly table fails separately instead of
  changing the threshold definition.

Example
-------
python 06_2_make_subreddit_monthly_activity.py \
  --in-root inputs/parquet_clean \
  --out-root runs/subreddit_monthly_activity \
  --year 2021 \
  --min-users-grid 0,10,25,50,100,250,500,1000 \
  --min-activity-grid 0,10,25,50,100,250,500,1000,5000 \
  --required-months all \
  --duckdb-memory-limit 120GB \
  --duckdb-threads 4 \
  --duckdb-temp-dir "${TMPDIR:-/tmp}/duckdb_threshold_audit"
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import platform
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import duckdb


# ----------------------------- Logging ---------------------------------


def setup_logging(level: str, log_path: Optional[Path] = None) -> None:
    """Configure console logging and, optionally, file logging."""
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"Invalid log level: {level}")

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        ensure_dir(log_path.parent)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


logger = logging.getLogger(__name__)


# ----------------------------- Constants ---------------------------------


MONTH_RE = re.compile(r"(?<!\d)(20\d{2})[-_](0[1-9]|1[0-2])(?!\d)")


# ----------------------------- Data classes ------------------------------


@dataclass(frozen=True)
class DuckDBConfig:
    """DuckDB runtime configuration."""

    memory_limit: str = "16GB"
    threads: int = 4
    temp_dir: str = "duckdb_tmp"


@dataclass(frozen=True)
class ActivityConfig:
    """Resolved runtime configuration."""

    in_root: str
    out_root: str
    year: str
    selected_months: tuple[str, ...]
    only_months: tuple[str, ...]
    limit_months: int
    submissions_input: str | list[str]
    comments_input: str | list[str]
    author_col: str
    include_deleted_authors_in_user_count: bool
    include_user_subreddits: bool
    min_users_grid: tuple[int, ...]
    min_activity_grid: tuple[int, ...]
    required_months: str
    overwrite: bool
    smoke: bool
    smoke_max_rows: int
    run_id: str


# ----------------------------- Generic helpers ---------------------------


def parse_int_grid(value: str) -> list[int]:
    """Parse comma-separated non-negative integer grid."""
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        parsed = int(part)
        if parsed < 0:
            raise ValueError(f"Threshold values must be non-negative: {parsed}")
        out.append(parsed)
    if not out:
        raise ValueError("Threshold grid cannot be empty.")
    return sorted(set(out))


def parse_only_months(value: str) -> tuple[str, ...]:
    """Parse comma-separated YYYY-MM list."""
    if not value.strip():
        return tuple()
    months = tuple(part.strip() for part in value.split(",") if part.strip())
    bad = [month for month in months if not MONTH_RE.fullmatch(month)]
    if bad:
        raise ValueError(f"Invalid --only-months value(s): {', '.join(bad)}")
    return tuple(sorted(set(months)))


def sql_list_int(values: Iterable[int]) -> str:
    """Return SQL VALUES list for integers."""
    return ", ".join(f"({int(v)})" for v in values)


def sql_in_list_strings(values: Iterable[str]) -> str:
    """Return a SQL IN-list for strings."""
    values = tuple(values)
    if not values:
        raise ValueError("SQL string IN-list cannot be empty.")
    return "(" + ", ".join(quote_sql_string(value) for value in values) + ")"


def quote_sql_string(value: str | Path) -> str:
    """Return a single-quoted SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def quote_identifier(name: str) -> str:
    """Return a double-quoted SQL identifier."""
    return '"' + name.replace('"', '""') + '"'


def ensure_dir(path: str | Path) -> None:
    """Create a directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_json_dump(obj: dict, path: str | Path) -> None:
    """Write JSON with stable formatting."""
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def utc_now_iso() -> str:
    """Return UTC timestamp in stable ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_run_id(prefix: str, year: str) -> str:
    """Stable-ish run identifier string."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{year}_{ts}"


def build_selected_months(year: str, only_months: Sequence[str], limit_months: int) -> tuple[str, ...]:
    """Resolve the month list requested by CLI controls."""
    if only_months:
        months = tuple(sorted(only_months))
    else:
        months = tuple(f"{year}-{month:02d}" for month in range(1, 13))

    if limit_months > 0:
        months = months[:limit_months]

    return months


def resolve_required_months(value: str, selected_month_count: int) -> int:
    """Resolve --required-months against the selected input month count."""
    if selected_month_count < 1:
        raise ValueError("No selected months were resolved; cannot validate --required-months.")

    raw = value.strip().lower()
    if raw == "all":
        return selected_month_count

    try:
        required_months = int(raw)
    except ValueError as exc:
        raise ValueError("--required-months must be an integer or 'all'.") from exc

    if required_months < 1 or required_months > selected_month_count:
        raise ValueError(
            f"--required-months must be between 1 and {selected_month_count}, or 'all' "
            f"(selected input months resolved: {selected_month_count})."
        )

    return required_months


def build_inputs(
    in_root: str,
    selected_months: Sequence[str],
    submissions_glob: str,
    comments_glob: str,
) -> tuple[str | list[str], str | list[str]]:
    """Build concrete default paths or pass through custom globs."""
    if submissions_glob or comments_glob:
        if not submissions_glob or not comments_glob:
            raise ValueError("--submissions-glob and --comments-glob must be supplied together.")
        return submissions_glob, comments_glob

    root = Path(in_root)
    submissions = [
        str(root / "submissions" / f"submissions_core_{month}.parquet")
        for month in selected_months
    ]
    comments = [
        str(root / "comments" / f"comments_core_{month}.parquet")
        for month in selected_months
    ]
    return submissions, comments


def duckdb_path_arg(paths: str | Sequence[str]) -> str:
    """Build a read_parquet path argument for one glob or many concrete files."""
    if isinstance(paths, str):
        return quote_sql_string(paths)
    if not paths:
        raise ValueError("No Parquet paths supplied.")
    return "[" + ", ".join(quote_sql_string(path) for path in paths) + "]"


def parquet_source(paths: str | Sequence[str], smoke_max_rows: int) -> str:
    """Return a DuckDB FROM source over Parquet files, optionally row-limited."""
    read_expr = f"read_parquet({duckdb_path_arg(paths)}, filename=true, union_by_name=true)"
    if smoke_max_rows > 0:
        return f"(SELECT * FROM {read_expr} LIMIT {int(smoke_max_rows)})"
    return read_expr


def missing_paths(paths: str | Sequence[str]) -> list[str]:
    """Return missing concrete paths, ignoring custom glob strings."""
    if isinstance(paths, str):
        return []
    return [path for path in paths if not Path(path).exists()]


def _has_glob_magic(path: str) -> bool:
    """Return whether a path string contains simple filesystem glob syntax."""
    return any(char in path for char in "*?[")


def existing_input_paths(paths: str | Sequence[str]) -> tuple[str, ...]:
    """Return concrete input paths that exist or match a custom glob."""
    if isinstance(paths, str):
        if _has_glob_magic(paths):
            return tuple(sorted(glob.glob(paths, recursive=True)))
        return (paths,) if Path(paths).exists() else tuple()

    return tuple(path for path in paths if Path(path).exists())


def _input_description(paths: str | Sequence[str]) -> str:
    """Format input paths for error messages."""
    if isinstance(paths, str):
        return paths
    if not paths:
        return "<none>"
    return "\n  - ".join(str(path) for path in paths)


def validate_input_files(submissions_input: str | Sequence[str], comments_input: str | Sequence[str]) -> int:
    """Validate that expected Parquet inputs exist and return the file count."""
    submission_files = existing_input_paths(submissions_input)
    comment_files = existing_input_paths(comments_input)
    input_file_count = len(submission_files) + len(comment_files)

    if input_file_count == 0:
        raise FileNotFoundError(
            "No input Parquet files found for selected months. Checked submissions input(s):\n  - "
            + _input_description(submissions_input)
            + "\nChecked comments input(s):\n  - "
            + _input_description(comments_input)
        )

    if not submission_files:
        raise FileNotFoundError(
            "No submissions Parquet input files found. Checked:\n  - " + _input_description(submissions_input)
        )
    if not comment_files:
        raise FileNotFoundError(
            "No comments Parquet input files found. Checked:\n  - " + _input_description(comments_input)
        )

    missing = missing_paths(submissions_input) + missing_paths(comments_input)
    if missing:
        raise FileNotFoundError("Missing selected Parquet input(s):\n  - " + "\n  - ".join(missing))

    return input_file_count


def validate_monthly_activity_nonempty(
    row_count: int,
    input_file_count: int,
    filtered_event_count: int,
    selected_month_count: int,
    input_row_count: int | None = None,
    diagnostics: dict | None = None,
) -> None:
    """Raise a clear error when monthly activity construction produced no rows."""
    if row_count > 0:
        return

    if input_file_count < 1:
        raise FileNotFoundError("No input Parquet files found; cannot construct subreddit_monthly_activity.")

    input_rows_note = ""
    if input_row_count is not None:
        input_rows_note = f" ({input_row_count} raw rows read from matched inputs)"

    if filtered_event_count == 0:
        first_zero_stage = first_zero_construction_stage(diagnostics)
        stage_note = f" First zero construction stage: {first_zero_stage}." if first_zero_stage else ""
        counts_note = ""
        if diagnostics is not None:
            counts_note = " Diagnostic counts: " + compact_diagnostic_counts_for_error(diagnostics)
        raise ValueError(
            "subreddit_monthly_activity has zero rows: input Parquet files were found"
            f"{input_rows_note}, but no rows matched the year/month/subreddit filters. "
            f"{stage_note}{counts_note} "
            f"Selected input months resolved to {selected_month_count}. Check --year, "
            "--only-months/--limit-months, filename month parsing, created_utc parsing diagnostics, "
            "and --include-user-subreddits."
        )

    raise RuntimeError(
        "subreddit_monthly_activity has zero rows even though "
        f"{filtered_event_count} filtered events were available from {input_file_count} input files; "
        "monthly aggregation unexpectedly produced no rows."
    )


def check_output_collisions(paths: Sequence[Path], overwrite: bool) -> None:
    """Raise if an output exists and overwrite is disabled."""
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output(s). Pass --overwrite to replace:\n  - "
            + "\n  - ".join(str(path) for path in existing)
        )


def remove_outputs(paths: Sequence[Path], overwrite: bool) -> None:
    """Remove existing outputs when overwrite is enabled."""
    if not overwrite:
        return
    for path in paths:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


# ----------------------------- Schema helpers ----------------------------


def detect_schema(con: duckdb.DuckDBPyConnection, paths: str | Sequence[str]) -> list[tuple[str, str]]:
    """Return (column name, DuckDB type) pairs from Parquet input."""
    q = f"DESCRIBE SELECT * FROM read_parquet({duckdb_path_arg(paths)}, union_by_name=true) LIMIT 0"
    return [(str(row[0]), str(row[1])) for row in con.execute(q).fetchall()]


def detect_columns(con: duckdb.DuckDBPyConnection, paths: str | Sequence[str]) -> list[str]:
    """Return column names from Parquet input using DuckDB DESCRIBE."""
    return [name for name, _column_type in detect_schema(con, paths)]


def schema_column_names(schema: Sequence[tuple[str, str]]) -> list[str]:
    """Return column names from a DuckDB schema tuple list."""
    return [name for name, _column_type in schema]


def schema_type(schema: Sequence[tuple[str, str]], column: str) -> str | None:
    """Return a column type from a DuckDB schema tuple list, if present."""
    for name, column_type in schema:
        if name == column:
            return column_type
    return None


def choose_author_col(cols: list[str], preferred: str) -> str:
    """Choose author identifier column."""
    if preferred != "auto":
        if preferred not in cols:
            raise ValueError(f"Requested author column {preferred!r} not found. Available columns: {cols}")
        return preferred

    # Prefer pseudonymised IDs if present; otherwise use raw author. `user_id` is
    # supported because some derived tables use it for per-user summaries.
    for candidate in ["author_pid", "author", "user_id"]:
        if candidate in cols:
            return candidate

    raise ValueError(f"Could not detect author column. Available columns: {cols}")


def choose_subreddit_col(cols: list[str]) -> str:
    """Choose subreddit column."""
    for candidate in ["subreddit", "subreddit_name_prefixed", "subreddit_id"]:
        if candidate in cols:
            return candidate
    raise ValueError(f"Could not detect subreddit column. Available columns: {cols}")


def choose_thread_count_col(cols: list[str]) -> str | None:
    """Try to find a submission-level comment-count column."""
    for candidate in ["num_comments", "n_comments", "comment_count", "comments_count"]:
        if candidate in cols:
            return candidate
    return None


def filename_basename_expr(filename_alias: str = "filename") -> str:
    """Return a SQL expression for the basename of DuckDB's filename column."""
    return f"regexp_extract(replace(CAST({filename_alias} AS VARCHAR), chr(92), '/'), '([^/]+)$', 1)"


def filename_month_expr(filename_alias: str = "filename") -> str:
    """Return a YYYY-MM month parsed from the input file basename."""
    basename_expr = filename_basename_expr(filename_alias)
    return (
        "replace("
        f"regexp_extract({basename_expr}, '(20[0-9]{{2}}[-_](0[1-9]|1[0-2]))', 1), "
        "'_', '-'"
        ")"
    )


def month_expr(filename_alias: str = "filename") -> str:
    """
    Month expression.

    We intentionally parse the month from the file basename because the cleaned
    inputs are monthly files and `created_utc` type can vary across
    Pushshift-derived pipelines. Parsing only the basename avoids accidentally
    matching run-date tokens in parent directories.
    """
    return filename_month_expr(filename_alias)


def _epoch_numeric_month_expr(value_expr: str) -> str:
    """Return YYYY-MM from an epoch numeric expression without guessing tiny numbers."""
    abs_expr = f"abs({value_expr})"
    return f"""
        CASE
            WHEN {value_expr} IS NULL THEN NULL
            WHEN {abs_expr} BETWEEN 100000000 AND 9999999999
                THEN strftime(CAST(to_timestamp({value_expr}) AS TIMESTAMP), '%Y-%m')
            WHEN {abs_expr} BETWEEN 100000000000 AND 9999999999999
                THEN strftime(CAST(to_timestamp({value_expr} / 1000.0) AS TIMESTAMP), '%Y-%m')
            WHEN {abs_expr} BETWEEN 100000000000000 AND 9999999999999999
                THEN strftime(CAST(to_timestamp({value_expr} / 1000000.0) AS TIMESTAMP), '%Y-%m')
            WHEN {abs_expr} BETWEEN 100000000000000000 AND 9999999999999999999
                THEN strftime(CAST(to_timestamp({value_expr} / 1000000000.0) AS TIMESTAMP), '%Y-%m')
            ELSE NULL
        END
    """


def created_utc_month_expr(column_name: str, column_type: str) -> str:
    """
    Return a robust YYYY-MM expression for created_utc diagnostics.

    Timestamp/date columns are formatted directly. String columns are parsed as
    ISO-like timestamps unless they are wholly numeric, in which case they are
    treated as plausible epoch seconds/milliseconds/microseconds/nanoseconds by
    magnitude. Unsupported column types fail loudly instead of being guessed.
    """
    col = quote_identifier(column_name)
    normalized_type = column_type.upper()

    if "TIMESTAMP" in normalized_type or normalized_type == "DATE":
        return f"strftime(CAST({col} AS TIMESTAMP), '%Y-%m')"

    numeric_type_tokens = (
        "BIGINT",
        "INTEGER",
        "SMALLINT",
        "TINYINT",
        "HUGEINT",
        "UBIGINT",
        "UINTEGER",
        "USMALLINT",
        "UTINYINT",
        "DOUBLE",
        "FLOAT",
        "REAL",
        "DECIMAL",
    )
    if any(token in normalized_type for token in numeric_type_tokens):
        return _epoch_numeric_month_expr(f"TRY_CAST({col} AS DOUBLE)")

    string_type_tokens = ("VARCHAR", "CHAR", "STRING")
    if any(token in normalized_type for token in string_type_tokens):
        value_expr = f"NULLIF(trim(CAST({col} AS VARCHAR)), '')"
        numeric_expr = f"TRY_CAST({value_expr} AS DOUBLE)"
        return f"""
            CASE
                WHEN {value_expr} IS NULL THEN NULL
                WHEN regexp_extract({value_expr}, '^[-]?[0-9]+(\\.[0-9]+)?$', 0) <> ''
                    THEN {_epoch_numeric_month_expr(numeric_expr)}
                ELSE strftime(TRY_CAST({value_expr} AS TIMESTAMP), '%Y-%m')
            END
        """

    raise ValueError(
        f"Unsupported created_utc column type {column_type!r}; expected timestamp, date, string, or epoch numeric."
    )


def subreddit_expr(col: str) -> str:
    """
    Normalised subreddit name expression.

    If `subreddit_name_prefixed` is used, remove a leading r/ so filtering and
    grouping remain comparable with the usual Pushshift `subreddit` field.
    """
    return f"regexp_replace(CAST({col} AS VARCHAR), '^r/', '')"


def ordinary_subreddit_filter(subreddit_sql_expr: str, include_user_subreddits: bool) -> str:
    """Return SQL filter for ordinary subreddits."""
    if include_user_subreddits:
        return "TRUE"
    return f"lower({subreddit_sql_expr}) NOT LIKE 'u\\_%' ESCAPE '\\'"


def deleted_author_condition(author_col: str, cols: list[str]) -> str:
    """
    SQL condition identifying deleted/null authors.

    This covers raw-author and pseudonymised cleaned-core conventions without
    assuming every column is present.
    """
    checks = [f"{author_col} IS NULL"]

    if "is_deleted_author" in cols:
        checks.append("COALESCE(CAST(is_deleted_author AS BOOLEAN), FALSE)")

    # Raw Pushshift / cleaned raw author convention.
    checks.append(f"lower(CAST({author_col} AS VARCHAR)) = '[deleted]'")

    # Pseudonymisation convention from the pipeline.
    checks.append(f"CAST({author_col} AS VARCHAR) = 'u_deleted'")

    return "(" + " OR ".join(checks) + ")"


def author_key_expr(author_col: str, cols: list[str], include_deleted: bool) -> str:
    """
    Author key expression for exact distinct counting across submissions+comments.

    Lower/trim normalisation avoids raw-author case/whitespace mismatches. It is
    harmless for pseudonymised IDs.
    """
    key = f"lower(trim(CAST({author_col} AS VARCHAR)))"
    if include_deleted:
        return key
    return f"CASE WHEN {deleted_author_condition(author_col, cols)} THEN NULL ELSE {key} END"


# ----------------------------- DuckDB helpers ----------------------------


def configure_duckdb(con: duckdb.DuckDBPyConnection, cfg: DuckDBConfig) -> None:
    """Apply DuckDB runtime settings."""
    ensure_dir(cfg.temp_dir)
    con.execute(f"SET memory_limit={quote_sql_string(cfg.memory_limit)}")
    con.execute(f"SET threads={int(cfg.threads)}")
    con.execute(f"SET temp_directory={quote_sql_string(cfg.temp_dir)}")
    con.execute("SET preserve_insertion_order=false")
    con.execute("PRAGMA disable_object_cache")


# ----------------------------- CLI --------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    p = argparse.ArgumentParser(
        description="Create month-by-subreddit activity Parquet and threshold retention summaries (DuckDB).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--in-root", required=True, help="Input root containing cleaned monthly Parquet outputs.")
    p.add_argument("--out-root", required=True, help="Output directory.")
    p.add_argument("--year", default="2021", help="Dataset year (YYYY).")
    p.add_argument("--only-months", default="", help="Comma-separated subset like 2021-01,2021-06.")
    p.add_argument("--limit-months", type=int, default=0, help="Limit number of months after whitelist; 0 = no limit.")
    p.add_argument("--submissions-glob", default="", help="Override submissions Parquet glob/path.")
    p.add_argument("--comments-glob", default="", help="Override comments Parquet glob/path.")
    p.add_argument(
        "--author-col",
        default="auto",
        choices=["auto", "author_pid", "author", "user_id"],
        help="Author identifier column to use for active-user counts.",
    )
    p.add_argument(
        "--include-deleted-authors-in-user-count",
        action="store_true",
        help="Count [deleted]/u_deleted as an active user. Default excludes deleted/null authors from n_users.",
    )
    p.add_argument(
        "--include-user-subreddits",
        action="store_true",
        help="Do not exclude user-profile subreddits beginning with u_.",
    )
    p.add_argument(
        "--min-users-grid",
        default="0,10,25,50,100,250,500,1000",
        help="Comma-separated monthly active-user thresholds. Include 0 for activity-only thresholding.",
    )
    p.add_argument(
        "--min-activity-grid",
        default="0,10,25,50,100,250,500,1000,5000",
        help="Comma-separated monthly activity thresholds. Include 0 for user-only thresholding.",
    )
    p.add_argument(
        "--required-months",
        default="all",
        help="Number of selected input months a subreddit must satisfy thresholds. Use 'all' for all selected months.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    p.add_argument("--smoke", action="store_true", help="Enable smoke mode.")
    p.add_argument("--smoke-max-rows", type=int, default=0, help="If >0 and --smoke, limit rows read per relation.")
    p.add_argument(
        "--duckdb-memory-limit",
        default=os.environ.get("DUCKDB_MEMORY_LIMIT", "16GB"),
        help="DuckDB memory limit.",
    )
    p.add_argument(
        "--duckdb-threads",
        type=int,
        default=int(os.environ.get("DUCKDB_THREADS", "4")),
        help="DuckDB threads.",
    )
    p.add_argument(
        "--duckdb-temp-dir",
        default=os.environ.get("DUCKDB_TEMP_DIR", "duckdb_tmp"),
        help="DuckDB temp_directory. Prefer node-local $TMPDIR on shared compute systems.",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return p


# ----------------------------- SQL builders ------------------------------


def build_monthly_sql(
    cfg: ActivityConfig,
    out_monthly: Path,
    submissions_source: str,
    comments_source: str,
    sub_cols: list[str],
    com_cols: list[str],
    sub_author_col: str,
    com_author_col: str,
    sub_subreddit_col: str,
    com_subreddit_col: str,
    thread_count_col: str | None,
) -> str:
    """Build monthly activity COPY SQL."""
    sub_month_expr = month_expr()
    com_month_expr = month_expr()

    sub_sr_expr = subreddit_expr(sub_subreddit_col)
    com_sr_expr = subreddit_expr(com_subreddit_col)
    ordinary_filter_sub = ordinary_subreddit_filter(sub_sr_expr, cfg.include_user_subreddits)
    ordinary_filter_com = ordinary_subreddit_filter(com_sr_expr, cfg.include_user_subreddits)

    sub_author_key = author_key_expr(
        sub_author_col,
        sub_cols,
        include_deleted=cfg.include_deleted_authors_in_user_count,
    )
    com_author_key = author_key_expr(
        com_author_col,
        com_cols,
        include_deleted=cfg.include_deleted_authors_in_user_count,
    )

    if thread_count_col:
        thread_metrics_select = f"""
            AVG(CAST({thread_count_col} AS DOUBLE)) AS mean_thread_size,
            median(CAST({thread_count_col} AS DOUBLE)) AS median_thread_size,
            AVG(CASE WHEN CAST({thread_count_col} AS DOUBLE) > 0 THEN 1.0 ELSE 0.0 END) AS pct_posts_with_comment
        """
    else:
        thread_metrics_select = """
            CAST(NULL AS DOUBLE) AS mean_thread_size,
            CAST(NULL AS DOUBLE) AS median_thread_size,
            CAST(NULL AS DOUBLE) AS pct_posts_with_comment
        """

    year_like = quote_sql_string(str(cfg.year) + "-%")
    selected_months_sql = sql_in_list_strings(cfg.selected_months)

    return f"""
    COPY (
        WITH submission_events AS (
            SELECT
                {sub_month_expr} AS month,
                {sub_sr_expr} AS subreddit,
                {sub_author_key} AS author_key,
                1::BIGINT AS is_submission,
                0::BIGINT AS is_comment
            FROM {submissions_source}
            WHERE {sub_subreddit_col} IS NOT NULL
              AND {ordinary_filter_sub}
              AND {sub_month_expr} LIKE {year_like}
              AND {sub_month_expr} IN {selected_months_sql}
        ),
        comment_events AS (
            SELECT
                {com_month_expr} AS month,
                {com_sr_expr} AS subreddit,
                {com_author_key} AS author_key,
                0::BIGINT AS is_submission,
                1::BIGINT AS is_comment
            FROM {comments_source}
            WHERE {com_subreddit_col} IS NOT NULL
              AND {ordinary_filter_com}
              AND {com_month_expr} LIKE {year_like}
              AND {com_month_expr} IN {selected_months_sql}
        ),
        all_events AS (
            SELECT * FROM submission_events
            UNION ALL
            SELECT * FROM comment_events
        ),
        monthly_activity AS (
            SELECT
                month,
                subreddit,
                CAST(SUM(is_submission) AS BIGINT) AS n_submissions,
                CAST(SUM(is_comment) AS BIGINT) AS n_comments,
                CAST(COUNT(*) AS BIGINT) AS n_total_activity,
                CAST(COUNT(DISTINCT author_key) AS BIGINT) AS n_users
            FROM all_events
            WHERE month IS NOT NULL AND month <> ''
              AND subreddit IS NOT NULL AND subreddit <> ''
            GROUP BY month, subreddit
        ),
        monthly_thread AS (
            SELECT
                {sub_month_expr} AS month,
                {sub_sr_expr} AS subreddit,
                {thread_metrics_select}
            FROM {submissions_source}
            WHERE {sub_subreddit_col} IS NOT NULL
              AND {ordinary_filter_sub}
              AND {sub_month_expr} LIKE {year_like}
              AND {sub_month_expr} IN {selected_months_sql}
            GROUP BY month, subreddit
        )
        SELECT
            a.month,
            a.subreddit,
            a.n_submissions,
            a.n_comments,
            a.n_total_activity,
            a.n_users,
            t.mean_thread_size,
            t.median_thread_size,
            t.pct_posts_with_comment
        FROM monthly_activity a
        LEFT JOIN monthly_thread t USING (month, subreddit)
        ORDER BY month, n_total_activity DESC, subreddit
    )
    TO {quote_sql_string(out_monthly)}
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """


CONSTRUCTION_STAGE_KEYS = (
    "raw_rows",
    "non_null_subreddit_rows",
    "ordinary_subreddit_rows",
    "filename_year_rows",
    "selected_filename_month_rows",
    "all_construction_filters_rows",
)


def _count_if(condition: str) -> str:
    """Return a BIGINT count expression for a SQL condition."""
    return f"CAST(SUM(CASE WHEN {condition} THEN 1 ELSE 0 END) AS BIGINT)"


def _fetch_distinct_values(
    con: duckdb.DuckDBPyConnection,
    source: str,
    value_expr: str,
    alias: str,
    limit: int = 20,
) -> list[str]:
    """Fetch a compact ordered sample of distinct non-null values."""
    rows = con.execute(
        f"""
        WITH values_seen AS (
            SELECT DISTINCT {value_expr} AS {alias}
            FROM {source}
        )
        SELECT CAST({alias} AS VARCHAR)
        FROM values_seen
        WHERE {alias} IS NOT NULL AND CAST({alias} AS VARCHAR) <> ''
        ORDER BY {alias}
        LIMIT {int(limit)}
        """
    ).fetchall()
    return [str(row[0]) for row in rows]


def _fetch_top_subreddits(
    con: duckdb.DuckDBPyConnection,
    source: str,
    subreddit_col: str,
    limit: int = 20,
) -> list[dict[str, object]]:
    """Fetch top subreddits by raw row count."""
    sr_expr = subreddit_expr(subreddit_col)
    rows = con.execute(
        f"""
        WITH raw_subreddits AS (
            SELECT {sr_expr} AS subreddit
            FROM {source}
        )
        SELECT CAST(subreddit AS VARCHAR) AS subreddit, COUNT(*) AS n_rows
        FROM raw_subreddits
        WHERE subreddit IS NOT NULL AND subreddit <> ''
        GROUP BY subreddit
        ORDER BY n_rows DESC, subreddit ASC
        LIMIT {int(limit)}
        """
    ).fetchall()
    return [{"subreddit": str(row[0]), "rows": int(row[1])} for row in rows]


def _fetch_created_utc_bounds(
    con: duckdb.DuckDBPyConnection,
    source: str,
    cols: Sequence[str],
) -> dict[str, str | None]:
    """Fetch compact min/max created_utc values when present."""
    if "created_utc" not in set(cols):
        return {"min": None, "max": None}
    col = quote_identifier("created_utc")
    row = con.execute(
        f"""
        SELECT
            CAST(MIN({col}) AS VARCHAR) AS min_created_utc,
            CAST(MAX({col}) AS VARCHAR) AS max_created_utc
        FROM {source}
        """
    ).fetchone()
    return {
        "min": None if row[0] is None else str(row[0]),
        "max": None if row[1] is None else str(row[1]),
    }


def relation_preflight_diagnostics(
    con: duckdb.DuckDBPyConnection,
    cfg: ActivityConfig,
    relation_label: str,
    source: str,
    cols: list[str],
    schema: Sequence[tuple[str, str]],
    author_col: str,
    subreddit_col: str,
) -> dict[str, object]:
    """Build staged filter diagnostics for one relation before writing output."""
    filename_month_sql = month_expr()
    sr_expr = subreddit_expr(subreddit_col)
    author_key_sql = author_key_expr(
        author_col,
        cols,
        include_deleted=cfg.include_deleted_authors_in_user_count,
    )
    ordinary_filter = ordinary_subreddit_filter("subreddit_norm", cfg.include_user_subreddits)
    year_like = quote_sql_string(str(cfg.year) + "-%")
    selected_months_sql = sql_in_list_strings(cfg.selected_months)

    created_type = schema_type(schema, "created_utc")
    if created_type is None:
        created_month_sql = "CAST(NULL AS VARCHAR)"
    else:
        created_month_sql = created_utc_month_expr("created_utc", created_type)

    non_null_subreddit = "subreddit_raw_present"
    ordinary_subreddit = f"{non_null_subreddit} AND {ordinary_filter}"
    filename_year = f"{ordinary_subreddit} AND filename_month LIKE {year_like}"
    selected_filename_month = f"{filename_year} AND filename_month IN {selected_months_sql}"
    all_construction_filters = (
        f"{selected_filename_month} "
        "AND filename_month IS NOT NULL AND filename_month <> '' "
        "AND subreddit_norm IS NOT NULL AND subreddit_norm <> ''"
    )
    valid_author = f"{ordinary_subreddit} AND author_key IS NOT NULL"
    created_utc_resolved = "created_utc_month IS NOT NULL AND created_utc_month <> ''"
    created_utc_year = f"{ordinary_subreddit} AND created_utc_month LIKE {year_like}"
    created_utc_selected_month = f"{ordinary_subreddit} AND created_utc_month IN {selected_months_sql}"

    fields = [
        "raw_rows",
        "non_null_subreddit_rows",
        "ordinary_subreddit_rows",
        "valid_author_rows",
        "created_utc_resolved_rows",
        "created_utc_year_rows",
        "created_utc_selected_month_rows",
        "filename_year_rows",
        "selected_filename_month_rows",
        "all_construction_filters_rows",
    ]
    row = con.execute(
        f"""
        WITH base AS (
            SELECT
                {subreddit_col} IS NOT NULL AS subreddit_raw_present,
                {sr_expr} AS subreddit_norm,
                {author_key_sql} AS author_key,
                {filename_month_sql} AS filename_month,
                {created_month_sql} AS created_utc_month
            FROM {source}
        )
        SELECT
            COUNT(*) AS raw_rows,
            {_count_if(non_null_subreddit)} AS non_null_subreddit_rows,
            {_count_if(ordinary_subreddit)} AS ordinary_subreddit_rows,
            {_count_if(valid_author)} AS valid_author_rows,
            {_count_if(created_utc_resolved)} AS created_utc_resolved_rows,
            {_count_if(created_utc_year)} AS created_utc_year_rows,
            {_count_if(created_utc_selected_month)} AS created_utc_selected_month_rows,
            {_count_if(filename_year)} AS filename_year_rows,
            {_count_if(selected_filename_month)} AS selected_filename_month_rows,
            {_count_if(all_construction_filters)} AS all_construction_filters_rows
        FROM base
        """
    ).fetchone()

    counts = {field: int(row[index] or 0) for index, field in enumerate(fields)}
    created_bounds = _fetch_created_utc_bounds(con, source, cols)
    filename_months = _fetch_distinct_values(con, source, filename_month_sql, "filename_month")

    if created_type is None:
        created_months: list[str] = []
    else:
        created_months = _fetch_distinct_values(con, source, created_month_sql, "created_utc_month")

    return {
        "relation": relation_label,
        "created_utc_type": created_type,
        "counts": counts,
        "created_utc_bounds": created_bounds,
        "created_utc_months_first20": created_months,
        "filename_months_first20": filename_months,
        "top_subreddits_raw_first20": _fetch_top_subreddits(con, source, subreddit_col),
    }


def preflight_monthly_activity_diagnostics(
    con: duckdb.DuckDBPyConnection,
    cfg: ActivityConfig,
    submissions_source: str,
    comments_source: str,
    sub_cols: list[str],
    com_cols: list[str],
    sub_schema: Sequence[tuple[str, str]],
    com_schema: Sequence[tuple[str, str]],
    sub_author_col: str,
    com_author_col: str,
    sub_subreddit_col: str,
    com_subreddit_col: str,
) -> dict[str, object]:
    """Run and log diagnostics before constructing subreddit_monthly_activity."""
    diagnostics = {
        "month_parsing_for_construction": "filename basename regex YYYY-MM",
        "selected_months": list(cfg.selected_months),
        "submissions": relation_preflight_diagnostics(
            con=con,
            cfg=cfg,
            relation_label="submissions",
            source=submissions_source,
            cols=sub_cols,
            schema=sub_schema,
            author_col=sub_author_col,
            subreddit_col=sub_subreddit_col,
        ),
        "comments": relation_preflight_diagnostics(
            con=con,
            cfg=cfg,
            relation_label="comments",
            source=comments_source,
            cols=com_cols,
            schema=com_schema,
            author_col=com_author_col,
            subreddit_col=com_subreddit_col,
        ),
    }
    log_preflight_diagnostics(diagnostics)
    return diagnostics


def _format_top_subreddits(items: object) -> str:
    """Format top-subreddit diagnostics compactly for logs."""
    if not isinstance(items, list) or not items:
        return "<none>"
    parts = []
    for item in items:
        if not isinstance(item, dict):
            continue
        parts.append(f"{item.get('subreddit')}={int(item.get('rows', 0)):,}")
    return ", ".join(parts) if parts else "<none>"


def log_preflight_diagnostics(diagnostics: dict[str, object]) -> None:
    """Log compact preflight diagnostics."""
    logger.info(
        "Preflight month parsing for monthly activity: %s",
        diagnostics.get("month_parsing_for_construction"),
    )
    for relation in ("submissions", "comments"):
        relation_diag = diagnostics.get(relation, {})
        if not isinstance(relation_diag, dict):
            continue
        counts = relation_diag.get("counts", {})
        if not isinstance(counts, dict):
            continue
        logger.info(
            "Preflight %s construction counts: raw=%s | non_null_subreddit=%s | ordinary_subreddit=%s | "
            "filename_year=%s | selected_filename_month=%s | all_construction_filters=%s",
            relation,
            f"{int(counts.get('raw_rows', 0)):,}",
            f"{int(counts.get('non_null_subreddit_rows', 0)):,}",
            f"{int(counts.get('ordinary_subreddit_rows', 0)):,}",
            f"{int(counts.get('filename_year_rows', 0)):,}",
            f"{int(counts.get('selected_filename_month_rows', 0)):,}",
            f"{int(counts.get('all_construction_filters_rows', 0)):,}",
        )
        logger.info(
            "Preflight %s author/created_utc counts: valid_author=%s | created_utc_resolved=%s | "
            "created_utc_year=%s | created_utc_selected_month=%s | created_utc_type=%s",
            relation,
            f"{int(counts.get('valid_author_rows', 0)):,}",
            f"{int(counts.get('created_utc_resolved_rows', 0)):,}",
            f"{int(counts.get('created_utc_year_rows', 0)):,}",
            f"{int(counts.get('created_utc_selected_month_rows', 0)):,}",
            relation_diag.get("created_utc_type"),
        )
        bounds = relation_diag.get("created_utc_bounds", {})
        if isinstance(bounds, dict):
            logger.info(
                "Preflight %s created_utc min/max: %s / %s",
                relation,
                bounds.get("min"),
                bounds.get("max"),
            )
        logger.info(
            "Preflight %s created_utc months first20: %s",
            relation,
            ", ".join(relation_diag.get("created_utc_months_first20", []) or ["<none>"]),
        )
        logger.info(
            "Preflight %s filename months first20: %s",
            relation,
            ", ".join(relation_diag.get("filename_months_first20", []) or ["<none>"]),
        )
        logger.info(
            "Preflight %s top subreddits raw first20: %s",
            relation,
            _format_top_subreddits(relation_diag.get("top_subreddits_raw_first20")),
        )


def diagnostic_total(diagnostics: dict | None, count_key: str) -> int | None:
    """Return submissions+comments count for a diagnostic count key."""
    if diagnostics is None:
        return None
    total = 0
    found = False
    for relation in ("submissions", "comments"):
        relation_diag = diagnostics.get(relation, {})
        if not isinstance(relation_diag, dict):
            continue
        counts = relation_diag.get("counts", {})
        if not isinstance(counts, dict) or count_key not in counts:
            continue
        total += int(counts[count_key])
        found = True
    return total if found else None


def first_zero_construction_stage(diagnostics: dict | None) -> str | None:
    """Return the first cumulative construction stage whose combined count is zero."""
    if diagnostics is None:
        return None
    for key in CONSTRUCTION_STAGE_KEYS:
        total = diagnostic_total(diagnostics, key)
        if total == 0:
            return key
    return None


def compact_diagnostic_counts_for_error(diagnostics: dict) -> str:
    """Return compact diagnostic counts suitable for a ValueError."""
    parts = []
    for relation in ("submissions", "comments"):
        relation_diag = diagnostics.get(relation, {})
        counts = relation_diag.get("counts", {}) if isinstance(relation_diag, dict) else {}
        if not isinstance(counts, dict):
            continue
        parts.append(
            f"{relation}("
            f"raw={int(counts.get('raw_rows', 0)):,}, "
            f"subreddit={int(counts.get('non_null_subreddit_rows', 0)):,}, "
            f"ordinary={int(counts.get('ordinary_subreddit_rows', 0)):,}, "
            f"filename_year={int(counts.get('filename_year_rows', 0)):,}, "
            f"selected_filename_month={int(counts.get('selected_filename_month_rows', 0)):,}, "
            f"all={int(counts.get('all_construction_filters_rows', 0)):,}, "
            f"created_utc_year={int(counts.get('created_utc_year_rows', 0)):,}, "
            f"created_utc_selected_month={int(counts.get('created_utc_selected_month_rows', 0)):,}"
            ")"
        )
    return "; ".join(parts) if parts else "<unavailable>"


def build_threshold_summary_sql(
    out_monthly: Path,
    out_threshold_summary: Path,
    min_users_grid: tuple[int, ...],
    min_activity_grid: tuple[int, ...],
    required_months: int,
) -> str:
    """Build annual threshold summary COPY SQL."""
    users_values_sql = sql_list_int(min_users_grid)
    activity_values_sql = sql_list_int(min_activity_grid)

    return f"""
    COPY (
        WITH monthly AS (
            SELECT * FROM read_parquet({quote_sql_string(out_monthly)})
        ),
        totals AS (
            SELECT
                COUNT(DISTINCT subreddit) AS total_subreddits,
                SUM(n_submissions) AS total_submissions,
                SUM(n_comments) AS total_comments,
                SUM(n_total_activity) AS total_activity
            FROM monthly
        ),
        thresholds AS (
            SELECT
                u.min_users_per_month,
                a.min_activity_per_month
            FROM (VALUES {users_values_sql}) AS u(min_users_per_month)
            CROSS JOIN (VALUES {activity_values_sql}) AS a(min_activity_per_month)
        ),
        qualifying_months AS (
            SELECT
                t.min_users_per_month,
                t.min_activity_per_month,
                m.month,
                m.subreddit,
                m.n_submissions,
                m.n_comments,
                m.n_total_activity,
                m.n_users,
                CASE
                    WHEN m.n_users >= t.min_users_per_month
                     AND m.n_total_activity >= t.min_activity_per_month
                    THEN 1 ELSE 0
                END AS qualifies_month
            FROM monthly m
            CROSS JOIN thresholds t
        ),
        selected_subreddits AS (
            SELECT
                min_users_per_month,
                min_activity_per_month,
                subreddit,
                CAST(SUM(qualifies_month) AS BIGINT) AS n_qualifying_months
            FROM qualifying_months
            GROUP BY min_users_per_month, min_activity_per_month, subreddit
            HAVING SUM(qualifies_month) >= {required_months}
        ),
        selected_keep_all_coverage AS (
            SELECT
                s.min_users_per_month,
                s.min_activity_per_month,
                COUNT(DISTINCT s.subreddit) AS n_subreddits_retained,
                SUM(m.n_submissions) AS retained_submissions_annual,
                SUM(m.n_comments) AS retained_comments_annual,
                SUM(m.n_total_activity) AS retained_activity_annual
            FROM selected_subreddits s
            JOIN monthly m USING (subreddit)
            GROUP BY s.min_users_per_month, s.min_activity_per_month
        ),
        qualifying_month_coverage AS (
            SELECT
                min_users_per_month,
                min_activity_per_month,
                COUNT(DISTINCT CASE WHEN qualifies_month = 1 THEN subreddit END) AS n_subreddits_with_any_qualifying_month,
                SUM(CASE WHEN qualifies_month = 1 THEN n_submissions ELSE 0 END) AS qualifying_month_submissions,
                SUM(CASE WHEN qualifies_month = 1 THEN n_comments ELSE 0 END) AS qualifying_month_comments,
                SUM(CASE WHEN qualifies_month = 1 THEN n_total_activity ELSE 0 END) AS qualifying_month_activity
            FROM qualifying_months
            GROUP BY min_users_per_month, min_activity_per_month
        )
        SELECT
            t.min_users_per_month,
            t.min_activity_per_month,
            {required_months} AS required_qualifying_months,

            totals.total_subreddits,
            totals.total_submissions,
            totals.total_comments,
            totals.total_activity,

            COALESCE(s.n_subreddits_retained, 0) AS n_subreddits_retained,
            COALESCE(s.n_subreddits_retained, 0)::DOUBLE / NULLIF(totals.total_subreddits, 0) AS pct_subreddits_retained,

            COALESCE(s.retained_submissions_annual, 0) AS retained_submissions_annual,
            COALESCE(s.retained_comments_annual, 0) AS retained_comments_annual,
            COALESCE(s.retained_activity_annual, 0) AS retained_activity_annual,
            COALESCE(s.retained_activity_annual, 0)::DOUBLE / NULLIF(totals.total_activity, 0) AS pct_activity_retained_annual,
            COALESCE(s.retained_submissions_annual, 0)::DOUBLE / NULLIF(totals.total_submissions, 0) AS pct_submissions_retained_annual,
            COALESCE(s.retained_comments_annual, 0)::DOUBLE / NULLIF(totals.total_comments, 0) AS pct_comments_retained_annual,

            COALESCE(q.n_subreddits_with_any_qualifying_month, 0) AS n_subreddits_with_any_qualifying_month,
            COALESCE(q.qualifying_month_submissions, 0) AS qualifying_month_submissions,
            COALESCE(q.qualifying_month_comments, 0) AS qualifying_month_comments,
            COALESCE(q.qualifying_month_activity, 0) AS qualifying_month_activity,
            COALESCE(q.qualifying_month_activity, 0)::DOUBLE / NULLIF(totals.total_activity, 0) AS pct_activity_in_qualifying_subreddit_months,
            COALESCE(q.qualifying_month_submissions, 0)::DOUBLE / NULLIF(totals.total_submissions, 0) AS pct_submissions_in_qualifying_subreddit_months,
            COALESCE(q.qualifying_month_comments, 0)::DOUBLE / NULLIF(totals.total_comments, 0) AS pct_comments_in_qualifying_subreddit_months
        FROM thresholds t
        CROSS JOIN totals
        LEFT JOIN selected_keep_all_coverage s
          ON t.min_users_per_month = s.min_users_per_month
         AND t.min_activity_per_month = s.min_activity_per_month
        LEFT JOIN qualifying_month_coverage q
          ON t.min_users_per_month = q.min_users_per_month
         AND t.min_activity_per_month = q.min_activity_per_month
        ORDER BY t.min_users_per_month, t.min_activity_per_month
    )
    TO {quote_sql_string(out_threshold_summary)}
    (HEADER, DELIMITER ',');
    """


def build_threshold_by_month_sql(
    out_monthly: Path,
    out_threshold_by_month: Path,
    min_users_grid: tuple[int, ...],
    min_activity_grid: tuple[int, ...],
    required_months: int,
) -> str:
    """Build month-level threshold summary COPY SQL."""
    users_values_sql = sql_list_int(min_users_grid)
    activity_values_sql = sql_list_int(min_activity_grid)

    return f"""
    COPY (
        WITH monthly AS (
            SELECT * FROM read_parquet({quote_sql_string(out_monthly)})
        ),
        month_totals AS (
            SELECT
                month,
                COUNT(DISTINCT subreddit) AS total_subreddits_month,
                SUM(n_submissions) AS total_submissions_month,
                SUM(n_comments) AS total_comments_month,
                SUM(n_total_activity) AS total_activity_month
            FROM monthly
            GROUP BY month
        ),
        thresholds AS (
            SELECT
                u.min_users_per_month,
                a.min_activity_per_month
            FROM (VALUES {users_values_sql}) AS u(min_users_per_month)
            CROSS JOIN (VALUES {activity_values_sql}) AS a(min_activity_per_month)
        ),
        qualifying AS (
            SELECT
                m.month,
                t.min_users_per_month,
                t.min_activity_per_month,
                m.subreddit,
                m.n_submissions,
                m.n_comments,
                m.n_total_activity,
                m.n_users,
                CASE
                    WHEN m.n_users >= t.min_users_per_month
                     AND m.n_total_activity >= t.min_activity_per_month
                    THEN 1 ELSE 0
                END AS qualifies
            FROM monthly m
            CROSS JOIN thresholds t
        ),
        selected_subreddits AS (
            SELECT
                min_users_per_month,
                min_activity_per_month,
                subreddit
            FROM qualifying
            GROUP BY min_users_per_month, min_activity_per_month, subreddit
            HAVING SUM(qualifies) >= {required_months}
        ),
        selected_by_month AS (
            SELECT
                m.month,
                s.min_users_per_month,
                s.min_activity_per_month,
                COUNT(DISTINCT s.subreddit) AS n_selected_subreddits_present,
                SUM(m.n_submissions) AS selected_submissions_month,
                SUM(m.n_comments) AS selected_comments_month,
                SUM(m.n_total_activity) AS selected_activity_month
            FROM selected_subreddits s
            JOIN monthly m USING (subreddit)
            GROUP BY m.month, s.min_users_per_month, s.min_activity_per_month
        ),
        qualifying_by_month AS (
            SELECT
                month,
                min_users_per_month,
                min_activity_per_month,
                SUM(qualifies) AS n_subreddits_qualifying,
                SUM(CASE WHEN qualifies = 1 THEN n_submissions ELSE 0 END) AS qualifying_submissions_month,
                SUM(CASE WHEN qualifies = 1 THEN n_comments ELSE 0 END) AS qualifying_comments_month,
                SUM(CASE WHEN qualifies = 1 THEN n_total_activity ELSE 0 END) AS qualifying_activity_month
            FROM qualifying
            GROUP BY month, min_users_per_month, min_activity_per_month
        )
        SELECT
            mt.month,
            t.min_users_per_month,
            t.min_activity_per_month,
            {required_months} AS required_qualifying_months,

            mt.total_subreddits_month,
            mt.total_submissions_month,
            mt.total_comments_month,
            mt.total_activity_month,

            COALESCE(s.n_selected_subreddits_present, 0) AS n_selected_subreddits_present,
            COALESCE(s.selected_submissions_month, 0) AS selected_submissions_month,
            COALESCE(s.selected_comments_month, 0) AS selected_comments_month,
            COALESCE(s.selected_activity_month, 0) AS selected_activity_month,
            COALESCE(s.selected_activity_month, 0)::DOUBLE / NULLIF(mt.total_activity_month, 0) AS pct_activity_retained_by_selected_subreddits_month,
            COALESCE(s.selected_submissions_month, 0)::DOUBLE / NULLIF(mt.total_submissions_month, 0) AS pct_submissions_retained_by_selected_subreddits_month,
            COALESCE(s.selected_comments_month, 0)::DOUBLE / NULLIF(mt.total_comments_month, 0) AS pct_comments_retained_by_selected_subreddits_month,

            COALESCE(q.n_subreddits_qualifying, 0) AS n_subreddits_qualifying_this_month,
            COALESCE(q.qualifying_submissions_month, 0) AS qualifying_submissions_month,
            COALESCE(q.qualifying_comments_month, 0) AS qualifying_comments_month,
            COALESCE(q.qualifying_activity_month, 0) AS qualifying_activity_month,
            COALESCE(q.qualifying_activity_month, 0)::DOUBLE / NULLIF(mt.total_activity_month, 0) AS pct_activity_qualifying_this_month,
            COALESCE(q.qualifying_submissions_month, 0)::DOUBLE / NULLIF(mt.total_submissions_month, 0) AS pct_submissions_qualifying_this_month,
            COALESCE(q.qualifying_comments_month, 0)::DOUBLE / NULLIF(mt.total_comments_month, 0) AS pct_comments_qualifying_this_month
        FROM month_totals mt
        CROSS JOIN thresholds t
        LEFT JOIN selected_by_month s
          ON mt.month = s.month
         AND t.min_users_per_month = s.min_users_per_month
         AND t.min_activity_per_month = s.min_activity_per_month
        LEFT JOIN qualifying_by_month q
          ON mt.month = q.month
         AND t.min_users_per_month = q.min_users_per_month
         AND t.min_activity_per_month = q.min_activity_per_month
        ORDER BY mt.month, t.min_users_per_month, t.min_activity_per_month
    )
    TO {quote_sql_string(out_threshold_by_month)}
    (HEADER, DELIMITER ',');
    """


# ----------------------------- Metadata ----------------------------------


def metadata_payload(
    cfg: ActivityConfig,
    duck: DuckDBConfig,
    duckdb_version: str,
    status: str,
    outputs: dict[str, str],
    diagnostics: dict,
) -> dict:
    """Construct run metadata payload."""
    return {
        "run_id": cfg.run_id,
        "timestamp_utc": utc_now_iso(),
        "status": status,
        "script": Path(__file__).name,
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "duckdb": {"version": duckdb_version, **asdict(duck)},
        "config": asdict(cfg),
        "outputs": outputs,
        "diagnostics": diagnostics,
    }


# ----------------------------- Main --------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    only_months = parse_only_months(args.only_months)
    if args.smoke and not only_months and args.limit_months == 0:
        args.limit_months = 1

    selected_months = build_selected_months(str(args.year), only_months, int(args.limit_months))
    submissions_input, comments_input = build_inputs(
        args.in_root,
        selected_months,
        args.submissions_glob,
        args.comments_glob,
    )

    min_users_grid = tuple(parse_int_grid(args.min_users_grid))
    min_activity_grid = tuple(parse_int_grid(args.min_activity_grid))

    cfg = ActivityConfig(
        in_root=args.in_root,
        out_root=args.out_root,
        year=str(args.year),
        selected_months=selected_months,
        only_months=only_months,
        limit_months=int(args.limit_months),
        submissions_input=submissions_input if isinstance(submissions_input, str) else list(submissions_input),
        comments_input=comments_input if isinstance(comments_input, str) else list(comments_input),
        author_col=str(args.author_col),
        include_deleted_authors_in_user_count=bool(args.include_deleted_authors_in_user_count),
        include_user_subreddits=bool(args.include_user_subreddits),
        min_users_grid=min_users_grid,
        min_activity_grid=min_activity_grid,
        required_months=str(args.required_months),
        overwrite=bool(args.overwrite),
        smoke=bool(args.smoke),
        smoke_max_rows=int(args.smoke_max_rows) if args.smoke else 0,
        run_id=utc_run_id("subreddit_monthly_activity", str(args.year)),
    )
    selected_month_count = len(selected_months)
    required_months = resolve_required_months(cfg.required_months, selected_month_count)

    duck = DuckDBConfig(
        memory_limit=args.duckdb_memory_limit,
        threads=int(args.duckdb_threads),
        temp_dir=args.duckdb_temp_dir,
    )

    out_root = Path(cfg.out_root)
    audit_dir = out_root / "audit"
    ensure_dir(out_root)
    ensure_dir(audit_dir)

    out_monthly = out_root / "subreddit_monthly_activity.parquet"
    out_threshold_summary = out_root / "threshold_retention_summary.csv"
    out_threshold_by_month = out_root / "threshold_retention_by_month.csv"
    out_metadata = out_root / "run_metadata.json"
    out_metadata_timestamped = audit_dir / f"subreddit_monthly_activity_run_{cfg.run_id}.json"
    out_log = audit_dir / f"subreddit_monthly_activity_{cfg.run_id}.log"

    output_paths = [out_monthly, out_threshold_summary, out_threshold_by_month, out_metadata]
    check_output_collisions(output_paths, cfg.overwrite)
    remove_outputs(output_paths, cfg.overwrite)

    setup_logging(args.log_level, out_log)

    input_file_count = validate_input_files(submissions_input, comments_input)

    logger.info("Run ID: %s", cfg.run_id)
    logger.info("Selected months: %s", ", ".join(selected_months))
    logger.info("Selected input month count: %s", selected_month_count)
    logger.info("Resolved required qualifying months: %s", required_months)
    logger.info("Submissions input: %s", submissions_input)
    logger.info("Comments input: %s", comments_input)
    logger.info("Matched input file count: %s", input_file_count)
    logger.info("Output root: %s", out_root)
    logger.info("Smoke mode: %s | smoke_max_rows=%s", cfg.smoke, cfg.smoke_max_rows)
    logger.info("Exclude user-profile subreddits: %s", not cfg.include_user_subreddits)
    logger.info("Exclude deleted/null authors from active-user counts: %s", not cfg.include_deleted_authors_in_user_count)

    con = duckdb.connect()
    configure_duckdb(con, duck)

    sub_schema = detect_schema(con, submissions_input)
    com_schema = detect_schema(con, comments_input)
    sub_cols = schema_column_names(sub_schema)
    com_cols = schema_column_names(com_schema)

    logger.info("Submission columns: %s", sub_cols)
    logger.info("Comment columns: %s", com_cols)
    logger.info("Submission schema: %s", sub_schema)
    logger.info("Comment schema: %s", com_schema)

    sub_author_col = choose_author_col(sub_cols, cfg.author_col)
    com_author_col = choose_author_col(com_cols, cfg.author_col)

    if sub_author_col != com_author_col:
        raise ValueError(
            "Detected different author identifier columns for submissions and comments "
            f"({sub_author_col!r} vs {com_author_col!r}). Exact cross-relation active-user "
            "counts require comparable identifiers. Pass --author-col explicitly after checking schema."
        )

    sub_subreddit_col = choose_subreddit_col(sub_cols)
    com_subreddit_col = choose_subreddit_col(com_cols)
    thread_count_col = choose_thread_count_col(sub_cols)

    logger.info("Author column: %s", sub_author_col)
    logger.info("Submission subreddit column: %s", sub_subreddit_col)
    logger.info("Comment subreddit column: %s", com_subreddit_col)
    logger.info("Month parsing: filename basename regex YYYY-MM")
    logger.info("Thread count column: %s", thread_count_col)

    submissions_source = parquet_source(submissions_input, cfg.smoke_max_rows)
    comments_source = parquet_source(comments_input, cfg.smoke_max_rows)

    preflight_diagnostics = preflight_monthly_activity_diagnostics(
        con=con,
        cfg=cfg,
        submissions_source=submissions_source,
        comments_source=comments_source,
        sub_cols=sub_cols,
        com_cols=com_cols,
        sub_schema=sub_schema,
        com_schema=com_schema,
        sub_author_col=sub_author_col,
        com_author_col=com_author_col,
        sub_subreddit_col=sub_subreddit_col,
        com_subreddit_col=com_subreddit_col,
    )

    monthly_sql = build_monthly_sql(
        cfg=cfg,
        out_monthly=out_monthly,
        submissions_source=submissions_source,
        comments_source=comments_source,
        sub_cols=sub_cols,
        com_cols=com_cols,
        sub_author_col=sub_author_col,
        com_author_col=com_author_col,
        sub_subreddit_col=sub_subreddit_col,
        com_subreddit_col=com_subreddit_col,
        thread_count_col=thread_count_col,
    )

    logger.info("Writing monthly subreddit activity: %s", out_monthly)
    con.execute(monthly_sql)

    row_count = int(
        con.execute(f"SELECT COUNT(*) FROM read_parquet({quote_sql_string(out_monthly)})").fetchone()[0]
    )
    if row_count == 0:
        filtered_event_count = diagnostic_total(preflight_diagnostics, "all_construction_filters_rows") or 0
        input_row_count = diagnostic_total(preflight_diagnostics, "raw_rows")
        validate_monthly_activity_nonempty(
            row_count=row_count,
            input_file_count=input_file_count,
            filtered_event_count=filtered_event_count,
            selected_month_count=selected_month_count,
            input_row_count=input_row_count,
            diagnostics=preflight_diagnostics,
        )

    n_months = int(
        con.execute(f"SELECT COUNT(DISTINCT month) FROM read_parquet({quote_sql_string(out_monthly)})").fetchone()[0]
    )
    observed_months = [
        row[0]
        for row in con.execute(
            f"SELECT DISTINCT month FROM read_parquet({quote_sql_string(out_monthly)}) ORDER BY month"
        ).fetchall()
    ]

    total_activity = int(
        con.execute(
            f"SELECT COALESCE(SUM(n_total_activity), 0) FROM read_parquet({quote_sql_string(out_monthly)})"
        ).fetchone()[0]
    )
    total_subreddits = int(
        con.execute(f"SELECT COUNT(DISTINCT subreddit) FROM read_parquet({quote_sql_string(out_monthly)})").fetchone()[0]
    )

    logger.info("Observed months: %s", ", ".join(observed_months))
    logger.info("Required qualifying months: %s", required_months)
    logger.info("Monthly activity rows: %s", row_count)
    logger.info("Ordinary subreddits represented: %s", total_subreddits)
    logger.info("Total activity represented: %s", total_activity)

    if n_months != len(selected_months):
        logger.warning(
            "Observed %s months in output but %s months were selected. This may be expected in smoke/custom runs, "
            "but should be checked for full-year analysis.",
            n_months,
            len(selected_months),
        )

    logger.info("Writing threshold summary: %s", out_threshold_summary)
    con.execute(
        build_threshold_summary_sql(
            out_monthly=out_monthly,
            out_threshold_summary=out_threshold_summary,
            min_users_grid=min_users_grid,
            min_activity_grid=min_activity_grid,
            required_months=required_months,
        )
    )

    logger.info("Writing monthly threshold summary: %s", out_threshold_by_month)
    con.execute(
        build_threshold_by_month_sql(
            out_monthly=out_monthly,
            out_threshold_by_month=out_threshold_by_month,
            min_users_grid=min_users_grid,
            min_activity_grid=min_activity_grid,
            required_months=required_months,
        )
    )

    outputs = {
        "subreddit_monthly_activity": str(out_monthly),
        "threshold_retention_summary": str(out_threshold_summary),
        "threshold_retention_by_month": str(out_threshold_by_month),
        "run_metadata": str(out_metadata),
        "timestamped_run_metadata": str(out_metadata_timestamped),
        "log": str(out_log),
    }
    diagnostics = {
        "selected_months": list(selected_months),
        "n_selected_months": selected_month_count,
        "observed_months": observed_months,
        "n_observed_months": n_months,
        "required_qualifying_months": required_months,
        "required_months_source": "selected_input_month_count",
        "monthly_activity_rows": row_count,
        "matched_input_file_count": input_file_count,
        "total_subreddits": total_subreddits,
        "total_activity": total_activity,
        "submission_columns": sub_cols,
        "comment_columns": com_cols,
        "author_column": sub_author_col,
        "submission_subreddit_column": sub_subreddit_col,
        "comment_subreddit_column": com_subreddit_col,
        "thread_count_column": thread_count_col,
        "month_parsing": "filename basename regex YYYY-MM",
        "preflight": preflight_diagnostics,
        "active_user_counting": "COUNT(DISTINCT author_key) after UNION ALL submissions+comments",
        "deleted_author_user_count_policy": (
            "included" if cfg.include_deleted_authors_in_user_count else "excluded"
        ),
        "user_profile_subreddit_policy": (
            "included" if cfg.include_user_subreddits else "excluded if lower(subreddit) starts with u_"
        ),
    }

    payload = metadata_payload(cfg, duck, duckdb.__version__, "finished", outputs, diagnostics)
    safe_json_dump(payload, out_metadata)
    safe_json_dump(payload, out_metadata_timestamped)

    con.close()

    logger.info("Done.")
    for label, path in outputs.items():
        logger.info("Wrote %s: %s", label, path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
