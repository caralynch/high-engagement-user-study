#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Cara Lynch
# See the LICENSE file for details.
"""
Preprocessing step 5: compute descriptive statistics for cleaned Reddit Parquet outputs.

This script computes publication-grade descriptive statistics for monthly cleaned
Reddit datasets produced by `03_0_cleaning.py` (can be used before or after pseudonymisation). It characterises dataset size, temporal coverage,
subreddit composition, user activity, and engagement / thread-size structure.

Design goals
------------
- Scalable: uses DuckDB SQL over Parquet and avoids loading full monthly tables
  into pandas except for final result tables.
- Reproducible: writes run metadata, resolved config, and machine-readable output
  tables.
- Pipeline-consistent: CLI, logging, and DuckDB tuning follow the style of the
  attached preprocessing scripts.
- Smoke-testable: supports an explicit month whitelist and a low-IO smoke mode.
- Thesis-friendly: outputs both a workbook and one-CSV-per-table for easy import
  into LaTeX / spreadsheets.

What this script computes
-------------------------
At a minimum, this script writes the following output tables:

- dataset_overview
    Overall counts for submissions, comments, users, subreddits, and date range.
- monthly_activity
    Per-month submissions, comments, users, and subreddits.
- subreddit_activity
    Per-subreddit submissions, comments, users, and mean comments per submission.
- subreddit_monthly_activity
    Per-subreddit per-month counts.
- user_contributions_summary
    Distribution summary of user contribution counts across the full period.
- user_contributions_by_month_summary
    Distribution summary of per-user monthly contribution counts.
- user_subreddit_participation_summary
    Distribution summary of number of distinct subreddits per user.
- thread_size_summary
    Distribution summary for thread size (comment count per submission; includes zeros).
- thread_size_by_subreddit
    Distribution summary for thread size by subreddit.
- thread_size_by_month
    Distribution summary for thread size by month.
- zero_inflation
    Proportions of zero-comment submissions and one-contribution users.
- audit_summary
    Summary derived from clean_audit_<year>.csv, when present.
- schema_summary
    Core schema / column presence checks and resolved ID / author columns.
- qc_checks
    Simple integrity checks relevant to descriptive analysis.
- user_activity
    Per-user submissions, comments, total activity, breadth across subreddits,
    and engagement received.
- user_engagement_received_summary
    Distribution summary of engagement received by user under the current
    operational definition: all comments in threads started by the user's
    submissions, plus direct replies received on the user's comments.
- top_users_top20_total_activity
    Top 20 users ranked by submissions + comments.
- top_users_top20_submissions
    Top 20 users ranked by number of submissions.
- top_users_top20_comments
    Top 20 users ranked by number of comments.
- top_users_top20_distinct_subreddits
    Top 20 users ranked by number of distinct subreddits participated in.
- top_users_top20_engagement_received
    Top 20 users ranked by engagement received under the current operational
    definition: all comments in threads started by their submissions, plus
    direct replies received on their comments.

Definitions
-----------
- Contribution:
    One submission or one comment authored by a user.
- Thread size:
    Number of comments attached to a submission, computed from comments_core.link_id.
    This script defines thread_size = 0 for submissions with no comments.
- Active user:
    For user-level summaries, an author with at least one submission or comment
    in the selected data after excluding rows flagged `is_deleted_author = TRUE`.
- Engagement received:
    Responses received on authored content in the selected data under the current
    operational definition:
    (i) for submissions, all comments in threads started by a user's submissions;
    (ii) for comments, direct replies received on a user's comments.

Current engagement operationalisation
-------------------------------------
The current implementation is asymmetric by design:
- for submissions, engagement received is counted as all comments in the
  submission thread;
- for comments, engagement received is counted as direct replies only.

This definition should be interpreted as a pragmatic descriptive measure rather
than a unified causal notion of influence or downstream cascade size.

Deleted-author handling
-----------------------
Content-level descriptive statistics retain rows authored by deleted users, so
that overall platform activity totals are preserved. User-level descriptive
statistics exclude rows flagged `is_deleted_author = TRUE`, because deleted
authors do not correspond to stable individual user entities for user-level
analysis. A separate summary table reports the contribution of deleted-author
rows to overall activity.

Inputs (expected layout)
------------------------
<in-root>/
  comments/comments_core_YYYY-MM.parquet
  comments/comments_body_YYYY-MM.parquet
  submissions/submissions_core_YYYY-MM.parquet
  submissions/submissions_selftext_YYYY-MM.parquet
  audit/clean_audit_<year>.csv          (optional, recommended)

This script reads only the *_core Parquet files for descriptive statistics.
Text sidecars are not required.

Outputs
-------
Writes descriptive outputs under:
<out-root>/
  audit/descriptive_stats_run_<run-id>.json
  audit/descriptive_stats_config_<run-id>.json
  audit/descriptive_stats_finished_<run-id>.json
  tables/<name>.csv                     (one CSV per output table)
  descriptive_stats_<year>.xlsx         (workbook, one sheet per output table)

Quick-run / smoke mode
----------------------
If --smoke is enabled, this script:
- limits processing to a small number of months unless --only-months is set
- can optionally limit scanned rows per input relation via --smoke-max-rows
- still writes the same output structure, but results are not suitable for analysis

Important note on row-limited smoke mode:
If --smoke-max-rows is used, counts and distribution summaries are approximate and
should only be used for plumbing / CI checks.

Command-line interface
----------------------
--in-root                 Input root containing cleaned Parquet outputs.
--out-root                Output root for descriptive tables.
--year                    Year to process (YYYY).
--only-months             Optional comma-separated list of YYYY-MM months to keep.
--limit-months            Optional limit on number of discovered months.
--author-col              Author identifier column to use: auto / author_pid / author.
--write-xlsx              Write Excel workbook (default: on).
--no-write-xlsx           Disable workbook writing.
--skip-existing           Skip work if final workbook already exists.
--overwrite               Overwrite existing tables/workbook.
--smoke                   Enable smoke mode.
--smoke-max-rows          Max rows read per relation in smoke mode (0 = no row limit).
--duckdb-memory-limit     DuckDB memory limit (e.g. '16GB').
--duckdb-threads          DuckDB threads.
--duckdb-temp-dir         DuckDB temp_directory.
--log-level               DEBUG/INFO/WARNING/ERROR/CRITICAL.

Examples
--------
Full 2021 descriptive tables:
  python 05_descriptive_stats.py \
    --in-root parquet_clean \
    --out-root descriptive_2021 \
    --year 2021 \
    --duckdb-memory-limit 32GB \
    --duckdb-threads 8 \
    --duckdb-temp-dir "$TMPDIR" \
    --log-level INFO

Single-month plumbing test:
  python 05_descriptive_stats.py \
    --in-root parquet_clean \
    --out-root descriptive_smoke \
    --year 2021 \
    --only-months 2021-01 \
    --smoke \
    --log-level INFO

Four-month smoke test with row limit:
  python 05_descriptive_stats.py \
    --in-root parquet_clean \
    --out-root descriptive_smoke \
    --year 2021 \
    --limit-months 4 \
    --smoke \
    --smoke-max-rows 200000 \
    --log-level INFO
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import duckdb
import pandas as pd


# ----------------------------- Logging ---------------------------------


def setup_logging(level: str) -> None:
    """Configure basic console logging."""
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


logger = logging.getLogger(__name__)


# ----------------------------- Constants ---------------------------------

MONTH_RE = re.compile(r"(?<!\d)(20\d{2})[-_](0[1-9]|1[0-2])(?!\d)")

COMMENTS_KIND = "comments_core"
SUBMISSIONS_KIND = "submissions_core"

FULL_TABLE_ORDER: Tuple[str, ...] = (
    "table_definitions",
    "dataset_overview",
    "schema_summary",
    "qc_checks",
    "audit_summary",
    "monthly_activity",
    "subreddit_activity",
    "subreddit_monthly_activity",
    "top_subreddits",
    "monthly_top_subreddits",
    "top_subreddits_top20_total_activity",
    "top_subreddits_top20_submissions",
    "top_subreddits_top20_comments",
    "top_subreddits_top20_unique_users",
    "monthly_top_subreddits_top20",
    "subreddit_threshold_summary",
    "subreddit_tail_summary",
    "subreddit_concentration_summary",
    "user_profile_community_summary",
    "nsfw_subreddit_summary",
    "subreddit_type_summary",
    "deleted_author_activity_summary",
    "user_activity",
    "user_engagement_received_summary",
    "top_users_top20_total_activity",
    "top_users_top20_submissions",
    "top_users_top20_comments",
    "top_users_top20_distinct_subreddits",
    "top_users_top20_engagement_received",
    "user_contributions_summary",
    "user_contributions_by_month_summary",
    "user_subreddit_participation_summary",
    "thread_size_summary",
    "thread_size_by_subreddit",
    "thread_size_by_month",
    "zero_inflation",
)

XLSX_TABLE_ORDER: Tuple[str, ...] = (
    "table_definitions",
    "dataset_overview",
    "schema_summary",
    "qc_checks",
    "audit_summary",
    "monthly_activity",
    "top_subreddits_top20_total_activity",
    "top_subreddits_top20_submissions",
    "top_subreddits_top20_comments",
    "top_subreddits_top20_unique_users",
    "monthly_top_subreddits_top20",
    "subreddit_threshold_summary",
    "subreddit_tail_summary",
    "subreddit_concentration_summary",
    "user_profile_community_summary",
    "nsfw_subreddit_summary",
    "subreddit_type_summary",
    "deleted_author_activity_summary",
    "user_engagement_received_summary",
    "top_users_top20_total_activity",
    "top_users_top20_submissions",
    "top_users_top20_comments",
    "top_users_top20_distinct_subreddits",
    "top_users_top20_engagement_received",
    "user_contributions_summary",
    "user_contributions_by_month_summary",
    "user_subreddit_participation_summary",
    "thread_size_summary",
    "thread_size_by_month",
    "zero_inflation",
)


# ----------------------------- Data classes ------------------------------


@dataclass(frozen=True)
class DuckDBConfig:
    """DuckDB runtime configuration."""

    memory_limit: str = "16GB"
    threads: int = 4
    temp_dir: str = "duckdb_tmp"


@dataclass(frozen=True)
class StatsConfig:
    """Configuration for descriptive statistics generation."""

    in_root: str
    out_root: str
    year: str
    only_months: Tuple[str, ...]
    limit_months: int
    author_col: str
    write_xlsx: bool
    skip_existing: bool
    overwrite: bool
    smoke: bool
    smoke_max_rows: int
    run_id: str


@dataclass(frozen=True)
class MonthFiles:
    """Paths to monthly comments/submissions core files."""

    comments_core: Optional[str] = None
    submissions_core: Optional[str] = None


Inventory = Dict[str, MonthFiles]  # month -> MonthFiles


# ----------------------------- Utilities ---------------------------------


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def safe_json_dump(obj: dict, path: str) -> None:
    """Write JSON with stable formatting."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def utc_now_iso() -> str:
    """Return UTC timestamp in stable ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_run_id(prefix: str, year: str) -> str:
    """Stable-ish run identifier string (UTC timestamp)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{year}_{ts}"


def parse_only_months(s: str) -> Tuple[str, ...]:
    """Parse comma-separated YYYY-MM list."""
    if not s.strip():
        return tuple()
    return tuple(m.strip() for m in s.split(",") if m.strip())


def month_of(path: str) -> str:
    """Extract YYYY-MM token from a file path."""
    m = MONTH_RE.search(path)
    return f"{m.group(1)}-{m.group(2)}" if m else "UNKNOWN"


def excel_sheet_name(name: str) -> str:
    """Convert a table name into a valid Excel sheet name (<= 31 chars)."""
    out = name[:31]
    return out.rstrip("'") or "sheet"


def quote_sql_string(s: str) -> str:
    """Safely quote a string literal for SQL."""
    return "'" + s.replace("'", "''") + "'"


# ----------------------------- Discovery ---------------------------------


def discover_core_files(in_root: str, year: str) -> Inventory:
    """
    Discover monthly comments/submissions core Parquet files under an input root.

    Returns mapping:
      month -> MonthFiles(comments_core=..., submissions_core=...)
    """
    inv: Dict[str, Dict[str, Optional[str]]] = {}

    patterns = [
        (
            COMMENTS_KIND,
            os.path.join(in_root, "comments", f"comments_core_{year}-*.parquet"),
        ),
        (
            SUBMISSIONS_KIND,
            os.path.join(in_root, "submissions", f"submissions_core_{year}-*.parquet"),
        ),
    ]

    for kind, pat in patterns:
        for path in sorted(Path().glob(pat) if "**" in pat else []):
            _ = path  # pragma: no cover (kept for future recursive support)

        for path in sorted(list(map(str, Path(in_root).glob(pat.replace(in_root + os.sep, ""))))):
            month = month_of(os.path.basename(path))
            if month == "UNKNOWN":
                continue
            inv.setdefault(month, {COMMENTS_KIND: None, SUBMISSIONS_KIND: None})
            inv[month][kind] = path

    # Fallback to glob.glob because pathlib with absolute-ish patterns is fiddly on HPC.
    if not inv:
        import glob

        for kind, pat in patterns:
            for path in sorted(glob.glob(pat)):
                month = month_of(os.path.basename(path))
                if month == "UNKNOWN":
                    continue
                inv.setdefault(month, {COMMENTS_KIND: None, SUBMISSIONS_KIND: None})
                inv[month][kind] = path

    out: Inventory = {}
    for month, d in sorted(inv.items()):
        out[month] = MonthFiles(
            comments_core=d.get(COMMENTS_KIND),
            submissions_core=d.get(SUBMISSIONS_KIND),
        )
    return out


def filter_month_inventory(
    inv: Inventory,
    only_months: Tuple[str, ...],
    limit_months: int,
) -> Inventory:
    """Apply optional month whitelist and limit."""
    months = list(sorted(inv.keys()))
    if only_months:
        keep = set(only_months)
        months = [m for m in months if m in keep]
    if limit_months > 0:
        months = months[:limit_months]
    return {m: inv[m] for m in months}


def require_complete_inventory(inv: Inventory) -> None:
    """Raise if any selected month is missing comments_core or submissions_core."""
    bad: List[str] = []
    for month, mf in inv.items():
        missing: List[str] = []
        if not mf.comments_core:
            missing.append(COMMENTS_KIND)
        if not mf.submissions_core:
            missing.append(SUBMISSIONS_KIND)
        if missing:
            bad.append(f"{month}: missing {', '.join(missing)}")
    if bad:
        raise RuntimeError("Incomplete monthly inventory:\n  - " + "\n  - ".join(bad))


# ----------------------------- DuckDB helpers ----------------------------


def configure_duckdb(con: duckdb.DuckDBPyConnection, cfg: DuckDBConfig) -> None:
    """Apply DuckDB runtime settings."""
    ensure_dir(cfg.temp_dir)
    con.execute(f"SET memory_limit={quote_sql_string(cfg.memory_limit)}")
    con.execute(f"SET threads={int(cfg.threads)}")
    con.execute(f"SET temp_directory={quote_sql_string(cfg.temp_dir)}")
    con.execute("SET preserve_insertion_order=false")
    con.execute("PRAGMA disable_object_cache")


def build_relation_union(paths: Sequence[str], smoke_max_rows: int = 0) -> str:
    """
    Build a SQL fragment that unions one or more Parquet files.

    If smoke_max_rows > 0, applies LIMIT per file to reduce IO for plumbing tests.
    """
    if not paths:
        raise ValueError("No paths supplied to build_relation_union().")

    parts: List[str] = []
    for path in paths:
        lit = quote_sql_string(path)
        if smoke_max_rows > 0:
            parts.append(
                f"SELECT * FROM (SELECT * FROM read_parquet({lit}) LIMIT {int(smoke_max_rows)})"
            )
        else:
            parts.append(f"SELECT * FROM read_parquet({lit})")
    return "\nUNION ALL\n".join(parts)


def drop_temp_tables(
    con: duckdb.DuckDBPyConnection,
    table_names: Sequence[str],
) -> None:
    """Drop temporary tables that are no longer needed."""
    for name in table_names:
        logger.info("Dropping temp table if exists: %s", name)
        con.execute(f"DROP TABLE IF EXISTS {name}")

# ----------------------------- Schema / QA -------------------------------


def get_table_columns(con: duckdb.DuckDBPyConnection, table_name: str) -> List[str]:
    """Return column names for a DuckDB table/view."""
    df = con.execute(f"DESCRIBE {table_name}").df()
    return df["column_name"].tolist()


def resolve_author_col(
    comments_cols: Sequence[str],
    submissions_cols: Sequence[str],
    requested: str,
) -> str:
    """
    Resolve the author identifier column.

    - auto: prefer author_pid if present in both, else author if present in both.
    - explicit: require presence in both comments and submissions.
    """
    comments_set = set(comments_cols)
    submissions_set = set(submissions_cols)

    if requested == "auto":
        if "author_pid" in comments_set and "author_pid" in submissions_set:
            return "author_pid"
        if "author" in comments_set and "author" in submissions_set:
            return "author"
        raise RuntimeError(
            "Could not resolve author column automatically. Neither 'author_pid' nor 'author' "
            "is present in both comments_core and submissions_core."
        )

    if requested not in comments_set or requested not in submissions_set:
        raise RuntimeError(
            f"Requested --author-col {requested!r} is not present in both comments_core and submissions_core."
        )
    return requested


def build_schema_summary(
    comments_cols: Sequence[str],
    submissions_cols: Sequence[str],
    author_col: str,
) -> pd.DataFrame:
    """Build a compact schema summary table."""
    required_comments = ["id", "created_utc", "link_id", "parent_id", "subreddit", author_col]
    required_submissions = ["id", "created_utc", "subreddit", author_col]

    rows: List[Dict[str, object]] = []
    cset = set(comments_cols)
    sset = set(submissions_cols)

    for col in sorted(set(required_comments + required_submissions)):
        rows.append(
            {
                "column_name": col,
                "comments_core_present": int(col in cset),
                "submissions_core_present": int(col in sset),
                "resolved_author_col": author_col,
            }
        )

    return pd.DataFrame(rows)


def build_qc_checks(
    con: duckdb.DuckDBPyConnection,
    comments_table: str = "comments_core_all",
    submissions_table: str = "submissions_core_all",
) -> pd.DataFrame:
    """Run lightweight integrity checks relevant to descriptive analysis."""
    checks: List[Dict[str, object]] = []

    q1 = con.execute(
        f"""
        SELECT
          COUNT(*) AS rows_total,
          SUM(CASE WHEN id IS NULL THEN 1 ELSE 0 END) AS id_null,
          SUM(CASE WHEN id IS NOT NULL AND NOT starts_with(id, 't1_') THEN 1 ELSE 0 END) AS id_bad_prefix,
          SUM(CASE WHEN link_id IS NOT NULL AND NOT starts_with(link_id, 't3_') THEN 1 ELSE 0 END) AS link_id_bad_prefix,
          SUM(CASE WHEN parent_id IS NOT NULL AND NOT (starts_with(parent_id, 't1_') OR starts_with(parent_id, 't3_')) THEN 1 ELSE 0 END) AS parent_id_bad_prefix
        FROM {comments_table} c
        """
    ).df().iloc[0].to_dict()
    for k, v in q1.items():
        checks.append({"check": f"comments_{k}", "value": int(v)})

    q2 = con.execute(
        f"""
        SELECT
          COUNT(*) AS rows_total,
          SUM(CASE WHEN id IS NULL THEN 1 ELSE 0 END) AS id_null,
          SUM(CASE WHEN id IS NOT NULL AND NOT starts_with(id, 't3_') THEN 1 ELSE 0 END) AS id_bad_prefix
        FROM {submissions_table} s
        """
    ).df().iloc[0].to_dict()
    for k, v in q2.items():
        checks.append({"check": f"submissions_{k}", "value": int(v)})

    q3 = con.execute(
        f"""
        SELECT COUNT(*) AS orphan_comment_links
        FROM {comments_table} c
        LEFT JOIN (
          SELECT DISTINCT id FROM {submissions_table}
        ) s
          ON c.link_id = s.id
        WHERE s.id IS NULL
        """
    ).df().iloc[0].to_dict()
    checks.append({"check": "orphan_comment_links", "value": int(q3["orphan_comment_links"])})

    return pd.DataFrame(checks)


# ----------------------------- Audit summary -----------------------------


def read_clean_audit_summary(audit_csv: str) -> pd.DataFrame:
    """Read and summarise clean_audit_<year>.csv if present."""
    if not os.path.exists(audit_csv):
        return pd.DataFrame(
            [{"metric": "audit_csv_present", "value": 0}, {"metric": "audit_csv_path", "value": audit_csv}]
        )

    df = pd.read_csv(audit_csv)
    rows: List[Dict[str, object]] = [
        {"metric": "audit_csv_present", "value": 1},
        {"metric": "audit_rows", "value": int(len(df))},
    ]

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for col in numeric_cols:
        rows.append({"metric": f"sum_{col}", "value": float(df[col].sum())})

    return pd.DataFrame(rows)

def build_table_definitions() -> pd.DataFrame:
    """Human-readable definitions for workbook summary sheets."""
    rows = [
        {
            "sheet_name": "dataset_overview",
            "meaning": "Headline counts across the selected data: submissions, comments, users, subreddits, and date ranges.",
        },
        {
            "sheet_name": "monthly_activity",
            "meaning": "Per-month counts of submissions, comments, active users, and active subreddits.",
        },
        {
            "sheet_name": "top_subreddits_top20_total_activity",
            "meaning": "Top 20 subreddits across the selected data, ranked by total activity, where total activity = submissions + comments.",
        },
        {
            "sheet_name": "top_subreddits_top20_submissions",
            "meaning": "Top 20 subreddits across the selected data, ranked by number of submissions.",
        },
        {
            "sheet_name": "top_subreddits_top20_comments",
            "meaning": "Top 20 subreddits across the selected data, ranked by number of comments.",
        },
        {
            "sheet_name": "top_subreddits_top20_unique_users",
            "meaning": "Top 20 subreddits across the selected data, ranked by number of distinct active users across submissions and comments.",
        },
        {
            "sheet_name": "monthly_top_subreddits_top20",
            "meaning": "Top 20 subreddits within each month, ranked by number of submissions.",
        },
        {
            "sheet_name": "subreddit_threshold_summary",
            "meaning": "For each activity metric and threshold, counts how many subreddits have activity greater than or equal to that threshold.",
        },
        {
            "sheet_name": "subreddit_tail_summary",
            "meaning": "Groups subreddits into total-activity bins, where total activity = submissions + comments.",
        },
        {
            "sheet_name": "subreddit_concentration_summary",
            "meaning": "For submissions/comments, shows the share of all activity contributed by the top-k subreddits.",
        },
        {
            "sheet_name": "user_profile_community_summary",
            "meaning": "Summarises subreddits with names beginning 'u_' and their contribution to total activity.",
        },
        {
            "sheet_name": "nsfw_subreddit_summary",
            "meaning": "Summarises subreddits flagged NSFW, where NSFW is derived from BOOL_OR(over_18) over submissions.",
        },
        {
            "sheet_name": "subreddit_type_summary",
            "meaning": "Summarises observed subreddit type status, including mixed-type subreddits where multiple types were observed.",
        },
        {
            "sheet_name": "deleted_author_activity_summary",
            "meaning": "Summary of rows flagged `is_deleted_author = TRUE`. Content-level tables retain these rows, but user-level tables exclude them.",
        },
        {
            "sheet_name": "user_activity",
            "meaning": "Per-user activity summary computed after excluding rows flagged `is_deleted_author = TRUE`.",
        },
        {
            "sheet_name": "user_engagement_received_summary",
            "meaning": "Distribution summary of engagement received by user after excluding deleted-author rows from user-level analysis. Engagement is defined as all comments in threads started by authored submissions, plus direct replies received on authored comments.",
        },
        {
            "sheet_name": "top_users_top20_total_activity",
            "meaning": "Top 20 non-deleted users ranked by total authored activity, where total activity = submissions + comments.",
        },
        {
            "sheet_name": "top_users_top20_submissions",
            "meaning": "Top 20 non-deleted users ranked by number of submissions.",
        },
        {
            "sheet_name": "top_users_top20_comments",
            "meaning": "Top 20 non-deleted users ranked by number of comments.",
        },
        {
            "sheet_name": "top_users_top20_distinct_subreddits",
            "meaning": "Top 20 non-deleted users ranked by number of distinct subreddits participated in.",
        },
        {
            "sheet_name": "top_users_top20_engagement_received",
            "meaning": "Top 20 non-deleted users ranked by engagement received under the current operational definition: all comments in threads started by their submissions, plus direct replies to their comments.",
        },
        {
            "sheet_name": "user_contributions_summary",
            "meaning": "Distribution summary of total per-user contributions after excluding deleted-author rows.",
        },
        {
            "sheet_name": "user_contributions_by_month_summary",
            "meaning": "Distribution summary of per-user monthly contributions after excluding deleted-author rows.",
        },
        {
            "sheet_name": "user_subreddit_participation_summary",
            "meaning": "Distribution summary of the number of distinct subreddits each user participated in after excluding deleted-author rows.",
        },
        {
            "sheet_name": "thread_size_summary",
            "meaning": "Distribution summary of thread size, where thread size = number of comments linked to a submission, including zeros.",
        },
        {
            "sheet_name": "thread_size_by_month",
            "meaning": "Per-month thread-size summary. Mean thread size is the average number of comments per submission, including zero-comment submissions.",
        },
        {
            "sheet_name": "zero_inflation",
            "meaning": "Simple sparsity summary, including the proportion of zero-comment submissions and one-contribution users.",
        },
    ]
    return pd.DataFrame(rows)

def add_reporting_summaries(
    tables: Dict[str, pd.DataFrame],
    subreddit_flags: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Add human-friendly summary tables and flag long raw tables."""
    out = dict(tables)
    flags = subreddit_flags.copy()
    flags["is_user_profile_subreddit"] = flags["subreddit"].astype(str).str.startswith("u_")

    # Attach flags to long-form subreddit tables.
    for name in (
        "subreddit_activity",
        "subreddit_monthly_activity",
        "top_subreddits",
        "monthly_top_subreddits",
        "thread_size_by_subreddit",
    ):
        if name in out and "subreddit" in out[name].columns:
            out[name] = out[name].merge(
                flags,
                on="subreddit",
                how="left",
            )

    if "top_subreddits" in out:
        top_df = out["top_subreddits"].copy()

        out["top_subreddits_top20_total_activity"] = (
            top_df.sort_values(
                ["n_total_activity", "n_all_users", "subreddit"],
                ascending=[False, False, True],
            )
            .head(20)
            .reset_index(drop=True)
        )

        out["top_subreddits_top20_submissions"] = (
            top_df.sort_values(
                ["n_submissions", "n_comments", "n_all_users", "subreddit"],
                ascending=[False, False, False, True],
            )
            .head(20)
            .reset_index(drop=True)
        )

        out["top_subreddits_top20_comments"] = (
            top_df.sort_values(
                ["n_comments", "n_submissions", "n_all_users", "subreddit"],
                ascending=[False, False, False, True],
            )
            .head(20)
            .reset_index(drop=True)
        )

        out["top_subreddits_top20_unique_users"] = (
            top_df.sort_values(
                ["n_all_users", "n_total_activity", "subreddit"],
                ascending=[False, False, True],
            )
            .head(20)
            .reset_index(drop=True)
        )

    if "monthly_top_subreddits" in out and "monthly_rank_by_submissions" in out["monthly_top_subreddits"].columns:
        out["monthly_top_subreddits_top20"] = (
            out["monthly_top_subreddits"]
            .loc[out["monthly_top_subreddits"]["monthly_rank_by_submissions"] <= 20]
            .copy()
        )

    if "subreddit_activity" in out:
        sr = out["subreddit_activity"].copy()

        sr["is_user_profile_subreddit"] = pd.Series(
            sr["is_user_profile_subreddit"], dtype="boolean"
        ).fillna(False).astype(bool)

        sr["is_nsfw_subreddit"] = pd.Series(
            sr["is_nsfw_subreddit"], dtype="boolean"
        ).fillna(False).astype(bool)

        sr["subreddit_type_status"] = sr["subreddit_type_status"].fillna("unknown").astype(str)
        sr["n_distinct_subreddit_types_observed"] = (
            pd.to_numeric(sr["n_distinct_subreddit_types_observed"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

        # Threshold summary
        thresholds = [1, 2, 5, 10, 50, 100, 500, 1000]
        thr_rows = []
        for metric in ("n_submissions", "n_comments"):
            for threshold in thresholds:
                thr_rows.append(
                    {
                        "metric": metric,
                        "at_least_threshold": threshold,
                        "n_subreddits_all": int((sr[metric] >= threshold).sum()),
                        "n_subreddits_excluding_user_profiles": int(
                            ((sr[metric] >= threshold) & (~sr["is_user_profile_subreddit"])).sum()
                        ),
                    }
                )
        out["subreddit_threshold_summary"] = (
            pd.DataFrame(thr_rows)
            .sort_values(["metric", "at_least_threshold"])
            .reset_index(drop=True)
        )

        # Tail summary on total activity
        sr["n_total_activity"] = sr["n_submissions"].fillna(0) + sr["n_comments"].fillna(0)

        def tail_bin(x: float) -> str:
            if x <= 1:
                return "1"
            if x <= 5:
                return "2-5"
            if x <= 10:
                return "6-10"
            if x <= 100:
                return "11-100"
            if x <= 1000:
                return "101-1000"
            return "1000+"

        tail_bin_order = ["1", "2-5", "6-10", "11-100", "101-1000", "1000+"]

        sr["activity_bin"] = pd.Categorical(
            sr["n_total_activity"].map(tail_bin),
            categories=tail_bin_order,
            ordered=True,
        )

        out["subreddit_tail_summary"] = (
            sr.groupby(
                ["activity_bin", "is_user_profile_subreddit"],
                dropna=False,
                observed=False,
            )
            .agg(
                n_subreddits=("subreddit", "count"),
                submissions_sum=("n_submissions", "sum"),
                comments_sum=("n_comments", "sum"),
            )
            .reset_index()
            .sort_values(["activity_bin", "is_user_profile_subreddit"])
        )

        # Concentration summary
        conc_rows = []
        for metric in ("n_submissions", "n_comments"):
            ranked = sr.sort_values(metric, ascending=False).reset_index(drop=True)
            total = float(ranked[metric].sum()) if len(ranked) else 0.0
            for k in (10, 20, 50, 100):
                share = float(ranked.head(k)[metric].sum()) / total if total > 0 else 0.0
                conc_rows.append(
                    {
                        "metric": metric,
                        "top_k": k,
                        "share_of_total_activity": share,
                    }
                )
        out["subreddit_concentration_summary"] = (
            pd.DataFrame(conc_rows)
            .sort_values(["metric", "top_k"])
            .reset_index(drop=True)
        )

        # u_-community summary
        u_df = sr.loc[sr["is_user_profile_subreddit"]].copy()
        totals = {
            "n_subreddits_all": int(len(sr)),
            "n_subreddits_user_profiles": int(len(u_df)),
            "submissions_all": int(sr["n_submissions"].sum()),
            "comments_all": int(sr["n_comments"].sum()),
            "submissions_user_profiles": int(u_df["n_submissions"].sum()),
            "comments_user_profiles": int(u_df["n_comments"].sum()),
        }

        out["user_profile_community_summary"] = pd.DataFrame(
            [
                {"metric": "n_subreddits_total", "value": totals["n_subreddits_all"]},
                {"metric": "n_user_profile_subreddits", "value": totals["n_subreddits_user_profiles"]},
                {"metric": "pct_subreddits_that_are_user_profile_subreddits", "value": totals["n_subreddits_user_profiles"] / totals["n_subreddits_all"] if totals["n_subreddits_all"] else 0.0},
                {"metric": "n_submissions_in_user_profile_subreddits", "value": totals["submissions_user_profiles"]},
                {"metric": "n_comments_in_user_profile_subreddits", "value": totals["comments_user_profiles"]},
                {"metric": "pct_submissions_in_user_profile_subreddits", "value": totals["submissions_user_profiles"] / totals["submissions_all"] if totals["submissions_all"] else 0.0},
                {"metric": "pct_comments_in_user_profile_subreddits", "value": totals["comments_user_profiles"] / totals["comments_all"] if totals["comments_all"] else 0.0},
            ]
        )

        # NSFW summary
        nsfw_df = sr.loc[sr["is_nsfw_subreddit"]].copy()
        out["nsfw_subreddit_summary"] = pd.DataFrame(
            [
                {"metric": "n_nsfw_subreddits", "value": int(len(nsfw_df))},
                {"metric": "pct_subreddits_that_are_nsfw", "value": float(len(nsfw_df) / len(sr)) if len(sr) else 0.0},
                {"metric": "n_submissions_in_nsfw_subreddits", "value": int(nsfw_df["n_submissions"].sum())},
                {"metric": "n_comments_in_nsfw_subreddits", "value": int(nsfw_df["n_comments"].sum())},
                {"metric": "pct_submissions_in_nsfw_subreddits", "value": float(nsfw_df["n_submissions"].sum() / sr["n_submissions"].sum()) if sr["n_submissions"].sum() else 0.0},
                {"metric": "pct_comments_in_nsfw_subreddits", "value": float(nsfw_df["n_comments"].sum() / sr["n_comments"].sum()) if sr["n_comments"].sum() else 0.0},
            ]
        )

        # Subreddit type summary
        out["subreddit_type_summary"] = (
            sr.groupby(
                [
                    "subreddit_type_status",
                    "observed_public_type",
                    "observed_restricted_type",
                    "observed_private_type",
                    "observed_user_type",
                    "observed_archived_type",
                    "observed_gold_restricted_type",
                ],
                dropna=False,
            )
            .agg(
                n_subreddits=("subreddit", "count"),
                submissions_sum=("n_submissions", "sum"),
                comments_sum=("n_comments", "sum"),
            )
            .reset_index()
            .sort_values(
                ["n_subreddits", "subreddit_type_status"],
                ascending=[False, True],
            )
        )

    return out


# ----------------------------- Query builders ----------------------------


def build_base_views(
    con: duckdb.DuckDBPyConnection,
    inv: Inventory,
    smoke_max_rows: int,
) -> None:
    """Create temp views for comments_core_all and submissions_core_all."""
    comments_paths = [mf.comments_core for mf in inv.values() if mf.comments_core]
    submissions_paths = [mf.submissions_core for mf in inv.values() if mf.submissions_core]

    comments_rel = build_relation_union(comments_paths, smoke_max_rows=smoke_max_rows)
    submissions_rel = build_relation_union(submissions_paths, smoke_max_rows=smoke_max_rows)

    logger.info("Creating comments_core_all view")
    con.execute(f"CREATE OR REPLACE TEMP VIEW comments_core_all AS {comments_rel}")

    logger.info("Creating submissions_core_all view")
    con.execute(f"CREATE OR REPLACE TEMP VIEW submissions_core_all AS {submissions_rel}")
    

def build_derived_tables(
    con: duckdb.DuckDBPyConnection,
    author_col: str,
) -> None:
    """Materialise only the highest-value reusable intermediates."""
    logger.info("Materialising comment_counts_by_link")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE comment_counts_by_link AS
        SELECT
            link_id,
            COUNT(*) AS n_comments
        FROM comments_core_all
        GROUP BY 1
        """
    )

    logger.info("Materialising parent_comment_author_map")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE parent_comment_author_map AS
        SELECT
            id AS comment_id,
            {author_col} AS parent_user_id
        FROM comments_core_userlevel
        """
    )

    logger.info("Materialising thread_sizes_all")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE thread_sizes_all AS
        SELECT
            s.id AS submission_id,
            s.subreddit,
            strftime(s.created_utc, '%Y-%m') AS month,
            COALESCE(c.n_comments, 0) AS thread_size
        FROM submissions_core_all s
        LEFT JOIN comment_counts_by_link c
          ON s.id = c.link_id
        """
    )

def build_user_level_views(
    con: duckdb.DuckDBPyConnection,
    author_col: str,
) -> None:
    """Create filtered views for user-level analysis excluding deleted authors."""
    logger.info("Creating submissions_core_userlevel view")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW submissions_core_userlevel AS
        SELECT *
        FROM submissions_core_all
        WHERE COALESCE(is_deleted_author, FALSE) = FALSE
          AND {author_col} IS NOT NULL
        """
    )

    logger.info("Creating comments_core_userlevel view")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW comments_core_userlevel AS
        SELECT *
        FROM comments_core_all
        WHERE COALESCE(is_deleted_author, FALSE) = FALSE
          AND {author_col} IS NOT NULL
        """
    )

def build_user_level_derived_tables(
    con: duckdb.DuckDBPyConnection,
    inv: Inventory,
    author_col: str,
    default_threads: int,
) -> None:
    """Materialise reusable user-level aggregates to keep later joins cheap.

    Counts and breadth tables are still built from the filtered user-level views.
    Engagement tables are built month-wise from the original monthly parquet files
    and then re-aggregated by user, to reduce peak memory pressure during the
    comment-level reply join.
    """
    logger.info("Materialising user_submission_counts")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE user_submission_counts AS
        SELECT
            {author_col} AS user_id,
            COUNT(*) AS n_submissions
        FROM submissions_core_userlevel
        GROUP BY 1
        """
    )

    logger.info("Materialising user_comment_counts")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE user_comment_counts AS
        SELECT
            {author_col} AS user_id,
            COUNT(*) AS n_comments
        FROM comments_core_userlevel
        GROUP BY 1
        """
    )

    logger.info("Materialising user_distinct_subreddits")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE user_distinct_subreddits AS
        SELECT
            user_id,
            COUNT(DISTINCT subreddit) AS n_distinct_subreddits
        FROM (
            SELECT {author_col} AS user_id, subreddit
            FROM submissions_core_userlevel
            UNION ALL
            SELECT {author_col} AS user_id, subreddit
            FROM comments_core_userlevel
        ) z
        GROUP BY 1
        """
    )

    submission_month_parts: List[str] = []
    comment_month_parts: List[str] = []

    for month in sorted(inv.keys()):
        mf = inv[month]

        if mf.submissions_core:
            sub_path = quote_sql_string(mf.submissions_core)
            submission_month_parts.append(
                f"""
                SELECT
                    s.{author_col} AS user_id,
                    SUM(COALESCE(cc.n_comments, 0)) AS n_submission_comments_received
                FROM read_parquet({sub_path}) s
                LEFT JOIN comment_counts_by_link cc
                  ON s.id = cc.link_id
                WHERE COALESCE(s.is_deleted_author, FALSE) = FALSE
                  AND s.{author_col} IS NOT NULL
                GROUP BY 1
                """
            )

        if mf.comments_core:
            com_path = quote_sql_string(mf.comments_core)
            comment_month_parts.append(
                f"""
                SELECT
                    p.parent_user_id AS user_id,
                    COUNT(*) AS n_comment_replies_received
                FROM read_parquet({com_path}) child
                INNER JOIN parent_comment_author_map p
                    ON child.parent_id = p.comment_id
                WHERE child.parent_id IS NOT NULL
                  AND starts_with(child.parent_id, 't1_')
                GROUP BY 1
                """
            )

    if not submission_month_parts:
        raise RuntimeError("No monthly submissions_core files available for user_submission_engagement.")
    if not comment_month_parts:
        raise RuntimeError("No monthly comments_core files available for user_comment_engagement.")

    logger.info("Materialising user_submission_engagement (month-wise)")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE user_submission_engagement AS
        SELECT
            user_id,
            SUM(n_submission_comments_received) AS n_submission_comments_received
        FROM (
            {' UNION ALL '.join(submission_month_parts)}
        ) t
        GROUP BY 1
        """
    )

    logger.info("Temporarily setting DuckDB threads to 1 for user_comment_engagement")
    con.execute("SET threads=1")
    try:
        logger.info("Materialising user_comment_engagement (month-wise)")
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE user_comment_engagement AS
            SELECT
                user_id,
                SUM(n_comment_replies_received) AS n_comment_replies_received
            FROM (
                {' UNION ALL '.join(comment_month_parts)}
            ) t
            GROUP BY 1
            """
        )
    finally:
        logger.info("Restoring DuckDB threads to %d", int(default_threads))
        con.execute(f"SET threads={int(default_threads)}")

    logger.info("Materialising user_universe")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE user_universe AS
        SELECT user_id FROM user_submission_counts
        UNION
        SELECT user_id FROM user_comment_counts
        """
    )

def build_deleted_author_activity_summary(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Summarise deleted-author rows retained in content-level descriptives."""
    return con.execute(
        """
        WITH s AS (
          SELECT
            COUNT(*) AS n_submissions_all,
            SUM(CASE WHEN COALESCE(is_deleted_author, FALSE) THEN 1 ELSE 0 END) AS n_submissions_deleted_author
          FROM submissions_core_all
        ),
        c AS (
          SELECT
            COUNT(*) AS n_comments_all,
            SUM(CASE WHEN COALESCE(is_deleted_author, FALSE) THEN 1 ELSE 0 END) AS n_comments_deleted_author
          FROM comments_core_all
        )
        SELECT * FROM (
          SELECT
            'n_submissions_deleted_author' AS metric,
            CAST(n_submissions_deleted_author AS DOUBLE) AS value
          FROM s
          UNION ALL
          SELECT
            'pct_submissions_deleted_author',
            CASE
              WHEN n_submissions_all > 0
              THEN CAST(n_submissions_deleted_author AS DOUBLE) / CAST(n_submissions_all AS DOUBLE)
              ELSE 0.0
            END
          FROM s
          UNION ALL
          SELECT
            'n_comments_deleted_author',
            CAST(n_comments_deleted_author AS DOUBLE)
          FROM c
          UNION ALL
          SELECT
            'pct_comments_deleted_author',
            CASE
              WHEN n_comments_all > 0
              THEN CAST(n_comments_deleted_author AS DOUBLE) / CAST(n_comments_all AS DOUBLE)
              ELSE 0.0
            END
          FROM c
        ) t
        """
    ).df()

def build_subreddit_flags(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Build per-subreddit flag/status summaries.

    Notes
    -----
    - `is_nsfw_subreddit` is defined as BOOL_OR(over_18) over submissions.
    - Because many subreddits have multiple observed subreddit_type values,
      we do not force a single type unless only one distinct type is observed.
      Otherwise we label the subreddit_type_status as 'mixed'.
    """
    logger.info("Building subreddit_flags")

    return con.execute(
        """
        WITH all_types AS (
          SELECT subreddit, subreddit_type
          FROM submissions_core_all
          UNION ALL
          SELECT subreddit, subreddit_type
          FROM comments_core_all
        ),
        type_flags AS (
          SELECT
            subreddit,
            COUNT(DISTINCT subreddit_type) AS n_distinct_subreddit_types_observed,
            STRING_AGG(
              DISTINCT COALESCE(subreddit_type, 'NULL'),
              ', ' ORDER BY COALESCE(subreddit_type, 'NULL')
            ) AS observed_subreddit_types,
            BOOL_OR(subreddit_type = 'public') AS observed_public_type,
            BOOL_OR(subreddit_type = 'restricted') AS observed_restricted_type,
            BOOL_OR(subreddit_type = 'private') AS observed_private_type,
            BOOL_OR(subreddit_type = 'user') AS observed_user_type,
            BOOL_OR(subreddit_type = 'archived') AS observed_archived_type,
            BOOL_OR(subreddit_type = 'gold_restricted') AS observed_gold_restricted_type,
            CASE
              WHEN COUNT(DISTINCT subreddit_type) = 1 THEN MAX(subreddit_type)
              WHEN COUNT(DISTINCT subreddit_type) = 0 THEN 'unknown'
              ELSE 'mixed'
            END AS subreddit_type_status
          FROM all_types
          GROUP BY 1
        ),
        nsfw_flags AS (
          SELECT
            subreddit,
            BOOL_OR(COALESCE(over_18, FALSE)) AS is_nsfw_subreddit,
            AVG(CASE WHEN COALESCE(over_18, FALSE) THEN 1.0 ELSE 0.0 END) AS pct_nsfw_submissions
          FROM submissions_core_all
          GROUP BY 1
        )
        SELECT
          COALESCE(t.subreddit, n.subreddit) AS subreddit,
          COALESCE(n.is_nsfw_subreddit, FALSE) AS is_nsfw_subreddit,
          COALESCE(n.pct_nsfw_submissions, 0.0) AS pct_nsfw_submissions,
          COALESCE(t.n_distinct_subreddit_types_observed, 0) AS n_distinct_subreddit_types_observed,
          COALESCE(t.observed_subreddit_types, 'unknown') AS observed_subreddit_types,
          COALESCE(t.observed_public_type, FALSE) AS observed_public_type,
          COALESCE(t.observed_restricted_type, FALSE) AS observed_restricted_type,
          COALESCE(t.observed_private_type, FALSE) AS observed_private_type,
          COALESCE(t.observed_user_type, FALSE) AS observed_user_type,
          COALESCE(t.observed_archived_type, FALSE) AS observed_archived_type,
          COALESCE(t.observed_gold_restricted_type, FALSE) AS observed_gold_restricted_type,
          COALESCE(t.subreddit_type_status, 'unknown') AS subreddit_type_status
        FROM type_flags t
        FULL OUTER JOIN nsfw_flags n
          ON t.subreddit = n.subreddit
        ORDER BY subreddit
        """
    ).df()

def build_user_activity_tables(
    con: duckdb.DuckDBPyConnection,
) -> Dict[str, pd.DataFrame]:
    """Build user-level activity and engagement-received tables from materialised aggregates."""
    out: Dict[str, pd.DataFrame] = {}

    logger.info("Building user_activity")
    out["user_activity"] = con.execute(
        """
        SELECT
          uu.user_id,
          COALESCE(us.n_submissions, 0) AS n_submissions,
          COALESCE(uc.n_comments, 0) AS n_comments,
          COALESCE(us.n_submissions, 0) + COALESCE(uc.n_comments, 0) AS n_total_activity,
          COALESCE(ud.n_distinct_subreddits, 0) AS n_distinct_subreddits,
          COALESCE(useg.n_submission_comments_received, 0) AS n_submission_comments_received,
          COALESCE(uceg.n_comment_replies_received, 0) AS n_comment_replies_received,
          COALESCE(useg.n_submission_comments_received, 0) + COALESCE(uceg.n_comment_replies_received, 0) AS n_total_engagement_received,
          ROW_NUMBER() OVER (
            ORDER BY
              COALESCE(us.n_submissions, 0) + COALESCE(uc.n_comments, 0) DESC,
              COALESCE(ud.n_distinct_subreddits, 0) DESC,
              uu.user_id
          ) AS overall_rank_by_total_activity,
          ROW_NUMBER() OVER (
            ORDER BY
              COALESCE(us.n_submissions, 0) DESC,
              COALESCE(uc.n_comments, 0) DESC,
              uu.user_id
          ) AS overall_rank_by_submissions,
          ROW_NUMBER() OVER (
            ORDER BY
              COALESCE(uc.n_comments, 0) DESC,
              COALESCE(us.n_submissions, 0) DESC,
              uu.user_id
          ) AS overall_rank_by_comments,
          ROW_NUMBER() OVER (
            ORDER BY
              COALESCE(ud.n_distinct_subreddits, 0) DESC,
              COALESCE(us.n_submissions, 0) + COALESCE(uc.n_comments, 0) DESC,
              uu.user_id
          ) AS overall_rank_by_distinct_subreddits,
          ROW_NUMBER() OVER (
            ORDER BY
              COALESCE(useg.n_submission_comments_received, 0) + COALESCE(uceg.n_comment_replies_received, 0) DESC,
              COALESCE(us.n_submissions, 0) + COALESCE(uc.n_comments, 0) DESC,
              uu.user_id
          ) AS overall_rank_by_engagement_received
        FROM user_universe uu
        LEFT JOIN user_submission_counts us
          ON uu.user_id = us.user_id
        LEFT JOIN user_comment_counts uc
          ON uu.user_id = uc.user_id
        LEFT JOIN user_distinct_subreddits ud
          ON uu.user_id = ud.user_id
        LEFT JOIN user_submission_engagement useg
          ON uu.user_id = useg.user_id
        LEFT JOIN user_comment_engagement uceg
          ON uu.user_id = uceg.user_id
        ORDER BY overall_rank_by_total_activity
        """
    ).df()

    logger.info("Building user_engagement_received_summary")
    ua = out["user_activity"].copy()

    out["user_engagement_received_summary"] = pd.DataFrame(
        [
            {
                "n_users": int(len(ua)),
                "mean_submission_comments_received": float(ua["n_submission_comments_received"].mean()),
                "median_submission_comments_received": float(ua["n_submission_comments_received"].median()),
                "p90_submission_comments_received": float(ua["n_submission_comments_received"].quantile(0.90)),
                "p95_submission_comments_received": float(ua["n_submission_comments_received"].quantile(0.95)),
                "p99_submission_comments_received": float(ua["n_submission_comments_received"].quantile(0.99)),
                "mean_comment_replies_received": float(ua["n_comment_replies_received"].mean()),
                "median_comment_replies_received": float(ua["n_comment_replies_received"].median()),
                "p90_comment_replies_received": float(ua["n_comment_replies_received"].quantile(0.90)),
                "p95_comment_replies_received": float(ua["n_comment_replies_received"].quantile(0.95)),
                "p99_comment_replies_received": float(ua["n_comment_replies_received"].quantile(0.99)),
                "mean_total_engagement_received": float(ua["n_total_engagement_received"].mean()),
                "median_total_engagement_received": float(ua["n_total_engagement_received"].median()),
                "sd_total_engagement_received": float(ua["n_total_engagement_received"].std(ddof=1)),
                "min_total_engagement_received": float(ua["n_total_engagement_received"].min()),
                "max_total_engagement_received": float(ua["n_total_engagement_received"].max()),
                "p75_total_engagement_received": float(ua["n_total_engagement_received"].quantile(0.75)),
                "p90_total_engagement_received": float(ua["n_total_engagement_received"].quantile(0.90)),
                "p95_total_engagement_received": float(ua["n_total_engagement_received"].quantile(0.95)),
                "p99_total_engagement_received": float(ua["n_total_engagement_received"].quantile(0.99)),
            }
        ]
    )

    user_activity = out["user_activity"].copy()

    out["top_users_top20_total_activity"] = (
        user_activity
        .sort_values(
            ["n_total_activity", "n_distinct_subreddits", "user_id"],
            ascending=[False, False, True],
        )
        .head(20)
        .reset_index(drop=True)
    )

    out["top_users_top20_submissions"] = (
        user_activity
        .sort_values(
            ["n_submissions", "n_comments", "user_id"],
            ascending=[False, False, True],
        )
        .head(20)
        .reset_index(drop=True)
    )

    out["top_users_top20_comments"] = (
        user_activity
        .sort_values(
            ["n_comments", "n_submissions", "user_id"],
            ascending=[False, False, True],
        )
        .head(20)
        .reset_index(drop=True)
    )

    out["top_users_top20_distinct_subreddits"] = (
        user_activity
        .sort_values(
            ["n_distinct_subreddits", "n_total_activity", "user_id"],
            ascending=[False, False, True],
        )
        .head(20)
        .reset_index(drop=True)
    )

    out["top_users_top20_engagement_received"] = (
        user_activity
        .sort_values(
            ["n_total_engagement_received", "n_total_activity", "user_id"],
            ascending=[False, False, True],
        )
        .head(20)
        .reset_index(drop=True)
    )

    return out

def make_tables(
    con: duckdb.DuckDBPyConnection,
    author_col: str,
) -> Dict[str, pd.DataFrame]:
    """Run all descriptive queries and return output tables."""
    tables: Dict[str, pd.DataFrame] = {}

    # Dataset overview
    logger.info("Building dataset_overview")
    tables["dataset_overview"] = con.execute(
        f"""
        WITH user_union AS (
          SELECT {author_col} AS user_id FROM submissions_core_all
          UNION ALL
          SELECT {author_col} AS user_id FROM comments_core_all
        )
        SELECT * FROM (
          SELECT 'year' AS metric, CAST(MIN(EXTRACT(year FROM created_utc)) AS VARCHAR) AS value FROM submissions_core_all
          UNION ALL
          SELECT 'submissions_n_rows', CAST(COUNT(*) AS VARCHAR) FROM submissions_core_all
          UNION ALL
          SELECT 'comments_n_rows', CAST(COUNT(*) AS VARCHAR) FROM comments_core_all
          UNION ALL
          SELECT 'submissions_n_unique_users', CAST(COUNT(DISTINCT {author_col}) AS VARCHAR) FROM submissions_core_all
          UNION ALL
          SELECT 'comments_n_unique_users', CAST(COUNT(DISTINCT {author_col}) AS VARCHAR) FROM comments_core_all
          UNION ALL
          SELECT 'all_n_unique_users', CAST(COUNT(DISTINCT user_id) AS VARCHAR) FROM user_union
          UNION ALL
          SELECT 'submissions_n_unique_subreddits', CAST(COUNT(DISTINCT subreddit) AS VARCHAR) FROM submissions_core_all
          UNION ALL
          SELECT 'comments_n_unique_subreddits', CAST(COUNT(DISTINCT subreddit) AS VARCHAR) FROM comments_core_all
          UNION ALL
          SELECT 'submissions_min_created_utc', CAST(MIN(created_utc) AS VARCHAR) FROM submissions_core_all
          UNION ALL
          SELECT 'submissions_max_created_utc', CAST(MAX(created_utc) AS VARCHAR) FROM submissions_core_all
          UNION ALL
          SELECT 'comments_min_created_utc', CAST(MIN(created_utc) AS VARCHAR) FROM comments_core_all
          UNION ALL
          SELECT 'comments_max_created_utc', CAST(MAX(created_utc) AS VARCHAR) FROM comments_core_all
        ) t
        """
    ).df()

    # Monthly activity
    logger.info("Building monthly_activity")
    tables["monthly_activity"] = con.execute(
        f"""
        WITH s AS (
          SELECT
            strftime(created_utc, '%Y-%m') AS month,
            COUNT(*) AS n_submissions,
            COUNT(DISTINCT {author_col}) AS n_submission_users,
            COUNT(DISTINCT subreddit) AS n_submission_subreddits
          FROM submissions_core_all
          GROUP BY 1
        ),
        c AS (
          SELECT
            strftime(created_utc, '%Y-%m') AS month,
            COUNT(*) AS n_comments,
            COUNT(DISTINCT {author_col}) AS n_comment_users,
            COUNT(DISTINCT subreddit) AS n_comment_subreddits
          FROM comments_core_all
          GROUP BY 1
        ),
        u AS (
          SELECT month, COUNT(DISTINCT user_id) AS n_all_users
          FROM (
            SELECT strftime(created_utc, '%Y-%m') AS month, {author_col} AS user_id
            FROM submissions_core_all
            UNION ALL
            SELECT strftime(created_utc, '%Y-%m') AS month, {author_col} AS user_id
            FROM comments_core_all
          ) z
          GROUP BY 1
        )
        SELECT
          COALESCE(s.month, c.month) AS month,
          COALESCE(n_submissions, 0) AS n_submissions,
          COALESCE(n_comments, 0) AS n_comments,
          COALESCE(n_submission_users, 0) AS n_submission_users,
          COALESCE(n_comment_users, 0) AS n_comment_users,
          COALESCE(n_all_users, 0) AS n_all_users,
          COALESCE(n_submission_subreddits, 0) AS n_submission_subreddits,
          COALESCE(n_comment_subreddits, 0) AS n_comment_subreddits
        FROM s
        FULL OUTER JOIN c ON s.month = c.month
        LEFT JOIN u ON COALESCE(s.month, c.month) = u.month
        ORDER BY month
        """
    ).df()

    # Subreddit activity
    logger.info("Building subreddit_activity")
    tables["subreddit_activity"] = con.execute(
        f"""
        WITH s AS (
          SELECT
            subreddit,
            COUNT(*) AS n_submissions,
            COUNT(DISTINCT {author_col}) AS n_submission_users
          FROM submissions_core_all
          GROUP BY 1
        ),
        c AS (
          SELECT
            subreddit,
            COUNT(*) AS n_comments,
            COUNT(DISTINCT {author_col}) AS n_comment_users
          FROM comments_core_all
          GROUP BY 1
        ),
        tc AS (
          SELECT
            s.subreddit,
            AVG(COALESCE(cc.n_comments, 0)) AS mean_comments_per_submission,
            MEDIAN(COALESCE(cc.n_comments, 0)) AS median_comments_per_submission
          FROM submissions_core_all s
          LEFT JOIN comment_counts_by_link cc
            ON s.id = cc.link_id
          GROUP BY 1
        ),
        u AS (
          SELECT subreddit, COUNT(DISTINCT user_id) AS n_all_users
          FROM (
            SELECT subreddit, {author_col} AS user_id FROM submissions_core_all
            UNION ALL
            SELECT subreddit, {author_col} AS user_id FROM comments_core_all
          ) z
          GROUP BY 1
        )
        SELECT
          COALESCE(s.subreddit, c.subreddit) AS subreddit,
          COALESCE(n_submissions, 0) AS n_submissions,
          COALESCE(n_comments, 0) AS n_comments,
          COALESCE(n_submission_users, 0) AS n_submission_users,
          COALESCE(n_comment_users, 0) AS n_comment_users,
          COALESCE(n_all_users, 0) AS n_all_users,
          mean_comments_per_submission,
          median_comments_per_submission
        FROM s
        FULL OUTER JOIN c ON s.subreddit = c.subreddit
        LEFT JOIN tc ON COALESCE(s.subreddit, c.subreddit) = tc.subreddit
        LEFT JOIN u ON COALESCE(s.subreddit, c.subreddit) = u.subreddit
        ORDER BY n_submissions DESC, n_comments DESC, subreddit
        """
    ).df()

    # Subreddit monthly activity
    logger.info("Building subreddit_monthly_activity")
    tables["subreddit_monthly_activity"] = con.execute(
        f"""
        WITH s AS (
          SELECT
            strftime(created_utc, '%Y-%m') AS month,
            subreddit,
            COUNT(*) AS n_submissions,
            COUNT(DISTINCT {author_col}) AS n_submission_users
          FROM submissions_core_all
          GROUP BY 1, 2
        ),
        c AS (
          SELECT
            strftime(created_utc, '%Y-%m') AS month,
            subreddit,
            COUNT(*) AS n_comments,
            COUNT(DISTINCT {author_col}) AS n_comment_users
          FROM comments_core_all
          GROUP BY 1, 2
        )
        SELECT
          COALESCE(s.month, c.month) AS month,
          COALESCE(s.subreddit, c.subreddit) AS subreddit,
          COALESCE(n_submissions, 0) AS n_submissions,
          COALESCE(n_comments, 0) AS n_comments,
          COALESCE(n_submission_users, 0) AS n_submission_users,
          COALESCE(n_comment_users, 0) AS n_comment_users
        FROM s
        FULL OUTER JOIN c
          ON s.month = c.month AND s.subreddit = c.subreddit
        ORDER BY month, n_submissions DESC, n_comments DESC, subreddit
        """
    ).df()

    # Top subreddits overall
    logger.info("Building top_subreddits")
    tables["top_subreddits"] = con.execute(
        f"""
        WITH s AS (
        SELECT
            subreddit,
            COUNT(*) AS n_submissions,
            COUNT(DISTINCT {author_col}) AS n_submission_users
        FROM submissions_core_all
        GROUP BY 1
        ),
        c AS (
        SELECT
            subreddit,
            COUNT(*) AS n_comments,
            COUNT(DISTINCT {author_col}) AS n_comment_users
        FROM comments_core_all
        GROUP BY 1
        ),
        u AS (
        SELECT subreddit, COUNT(DISTINCT user_id) AS n_all_users
        FROM (
            SELECT subreddit, {author_col} AS user_id FROM submissions_core_all
            UNION ALL
            SELECT subreddit, {author_col} AS user_id FROM comments_core_all
        ) z
        GROUP BY 1
        ),
        thread_sizes AS (
        SELECT subreddit, thread_size
        FROM thread_sizes_all
        )
        SELECT
        COALESCE(s.subreddit, c.subreddit) AS subreddit,
        COALESCE(n_submissions, 0) AS n_submissions,
        COALESCE(n_comments, 0) AS n_comments,
        COALESCE(n_submission_users, 0) AS n_submission_users,
        COALESCE(n_comment_users, 0) AS n_comment_users,
        COALESCE(n_all_users, 0) AS n_all_users,
        COALESCE(n_submissions, 0) + COALESCE(n_comments, 0) AS n_total_activity,
        AVG(thread_size) AS mean_thread_size,
        MEDIAN(thread_size) AS median_thread_size,
        AVG(CASE WHEN thread_size = 0 THEN 1.0 ELSE 0.0 END) AS pct_zero_comment_submissions,
        ROW_NUMBER() OVER (
            ORDER BY
            COALESCE(n_submissions, 0) + COALESCE(n_comments, 0) DESC,
            COALESCE(n_all_users, 0) DESC,
            COALESCE(s.subreddit, c.subreddit)
        ) AS overall_rank_by_total_activity,
        ROW_NUMBER() OVER (
            ORDER BY
            COALESCE(n_submissions, 0) DESC,
            COALESCE(n_comments, 0) DESC,
            COALESCE(n_all_users, 0) DESC,
            COALESCE(s.subreddit, c.subreddit)
        ) AS overall_rank_by_submissions,
        ROW_NUMBER() OVER (
            ORDER BY
            COALESCE(n_comments, 0) DESC,
            COALESCE(n_submissions, 0) DESC,
            COALESCE(n_all_users, 0) DESC,
            COALESCE(s.subreddit, c.subreddit)
        ) AS overall_rank_by_comments,
        ROW_NUMBER() OVER (
            ORDER BY
            COALESCE(n_all_users, 0) DESC,
            COALESCE(n_submissions, 0) + COALESCE(n_comments, 0) DESC,
            COALESCE(s.subreddit, c.subreddit)
        ) AS overall_rank_by_unique_users
        FROM s
        FULL OUTER JOIN c ON s.subreddit = c.subreddit
        LEFT JOIN u ON COALESCE(s.subreddit, c.subreddit) = u.subreddit
        LEFT JOIN thread_sizes ts ON COALESCE(s.subreddit, c.subreddit) = ts.subreddit
        GROUP BY 1,2,3,4,5,6
        ORDER BY overall_rank_by_total_activity
        """
    ).df()

    # Top subreddits within each month
    logger.info("Building monthly_top_subreddits")
    tables["monthly_top_subreddits"] = con.execute(
        f"""
        WITH s AS (
          SELECT
            strftime(created_utc, '%Y-%m') AS month,
            subreddit,
            COUNT(*) AS n_submissions,
            COUNT(DISTINCT {author_col}) AS n_submission_users
          FROM submissions_core_all
          GROUP BY 1, 2
        ),
        c AS (
          SELECT
            strftime(created_utc, '%Y-%m') AS month,
            subreddit,
            COUNT(*) AS n_comments,
            COUNT(DISTINCT {author_col}) AS n_comment_users
          FROM comments_core_all
          GROUP BY 1, 2
        ),
        u AS (
          SELECT month, subreddit, COUNT(DISTINCT user_id) AS n_all_users
          FROM (
            SELECT strftime(created_utc, '%Y-%m') AS month, subreddit, {author_col} AS user_id
            FROM submissions_core_all
            UNION ALL
            SELECT strftime(created_utc, '%Y-%m') AS month, subreddit, {author_col} AS user_id
            FROM comments_core_all
          ) z
          GROUP BY 1, 2
        ),
        thread_sizes AS (
          SELECT month, subreddit, thread_size
          FROM thread_sizes_all
        )
        SELECT
          COALESCE(s.month, c.month) AS month,
          COALESCE(s.subreddit, c.subreddit) AS subreddit,
          COALESCE(n_submissions, 0) AS n_submissions,
          COALESCE(n_comments, 0) AS n_comments,
          COALESCE(n_submission_users, 0) AS n_submission_users,
          COALESCE(n_comment_users, 0) AS n_comment_users,
          COALESCE(n_all_users, 0) AS n_all_users,
          AVG(thread_size) AS mean_thread_size,
          MEDIAN(thread_size) AS median_thread_size,
          AVG(CASE WHEN thread_size = 0 THEN 1.0 ELSE 0.0 END) AS pct_zero_comment_submissions,
          ROW_NUMBER() OVER (
            PARTITION BY COALESCE(s.month, c.month)
            ORDER BY COALESCE(n_submissions, 0) DESC,
                     COALESCE(n_comments, 0) DESC,
                     COALESCE(s.subreddit, c.subreddit)
          ) AS monthly_rank_by_submissions
        FROM s
        FULL OUTER JOIN c
          ON s.month = c.month AND s.subreddit = c.subreddit
        LEFT JOIN u
          ON COALESCE(s.month, c.month) = u.month
         AND COALESCE(s.subreddit, c.subreddit) = u.subreddit
        LEFT JOIN thread_sizes ts
          ON COALESCE(s.month, c.month) = ts.month
         AND COALESCE(s.subreddit, c.subreddit) = ts.subreddit
        GROUP BY 1,2,3,4,5,6,7
        ORDER BY month, monthly_rank_by_submissions
        """
    ).df()

        # User contribution distribution (whole period; excluding deleted authors)
    logger.info("Building user_contributions_summary")
    tables["user_contributions_summary"] = con.execute(
        f"""
        WITH user_contribs AS (
          SELECT user_id, COUNT(*) AS n_contributions
          FROM (
            SELECT {author_col} AS user_id FROM submissions_core_userlevel
            UNION ALL
            SELECT {author_col} AS user_id FROM comments_core_userlevel
          ) z
          GROUP BY 1
        )
        SELECT
          COUNT(*) AS n_users,
          AVG(n_contributions) AS mean_contributions,
          MEDIAN(n_contributions) AS median_contributions,
          STDDEV_SAMP(n_contributions) AS sd_contributions,
          MIN(n_contributions) AS min_contributions,
          MAX(n_contributions) AS max_contributions,
          QUANTILE_CONT(n_contributions, 0.75) AS p75_contributions,
          QUANTILE_CONT(n_contributions, 0.90) AS p90_contributions,
          QUANTILE_CONT(n_contributions, 0.95) AS p95_contributions,
          QUANTILE_CONT(n_contributions, 0.99) AS p99_contributions
        FROM user_contribs
        """
    ).df()

    # User contribution distribution by month (excluding deleted authors)
    logger.info("Building user_contributions_by_month_summary")
    tables["user_contributions_by_month_summary"] = con.execute(
        f"""
        WITH user_month_contribs AS (
          SELECT month, user_id, COUNT(*) AS n_contributions
          FROM (
            SELECT strftime(created_utc, '%Y-%m') AS month, {author_col} AS user_id
            FROM submissions_core_userlevel
            UNION ALL
            SELECT strftime(created_utc, '%Y-%m') AS month, {author_col} AS user_id
            FROM comments_core_userlevel
          ) z
          GROUP BY 1, 2
        )
        SELECT
          month,
          COUNT(*) AS n_user_months,
          AVG(n_contributions) AS mean_contributions,
          MEDIAN(n_contributions) AS median_contributions,
          STDDEV_SAMP(n_contributions) AS sd_contributions,
          MIN(n_contributions) AS min_contributions,
          MAX(n_contributions) AS max_contributions,
          QUANTILE_CONT(n_contributions, 0.75) AS p75_contributions,
          QUANTILE_CONT(n_contributions, 0.90) AS p90_contributions,
          QUANTILE_CONT(n_contributions, 0.95) AS p95_contributions,
          QUANTILE_CONT(n_contributions, 0.99) AS p99_contributions
        FROM user_month_contribs
        GROUP BY 1
        ORDER BY month
        """
    ).df()

    # Distinct subreddit participation per user (excluding deleted authors)
    logger.info("Building user_subreddit_participation_summary")
    tables["user_subreddit_participation_summary"] = con.execute(
        f"""
        WITH user_sr AS (
          SELECT user_id, COUNT(DISTINCT subreddit) AS n_subreddits
          FROM (
            SELECT {author_col} AS user_id, subreddit FROM submissions_core_userlevel
            UNION ALL
            SELECT {author_col} AS user_id, subreddit FROM comments_core_userlevel
          ) z
          GROUP BY 1
        )
        SELECT
          COUNT(*) AS n_users,
          AVG(n_subreddits) AS mean_subreddits,
          MEDIAN(n_subreddits) AS median_subreddits,
          STDDEV_SAMP(n_subreddits) AS sd_subreddits,
          MIN(n_subreddits) AS min_subreddits,
          MAX(n_subreddits) AS max_subreddits,
          QUANTILE_CONT(n_subreddits, 0.75) AS p75_subreddits,
          QUANTILE_CONT(n_subreddits, 0.90) AS p90_subreddits,
          QUANTILE_CONT(n_subreddits, 0.95) AS p95_subreddits,
          QUANTILE_CONT(n_subreddits, 0.99) AS p99_subreddits
        FROM user_sr
        """
    ).df()

    # Thread size summary (including zero-comment submissions)
    logger.info("Building thread_size_summary")
    tables["thread_size_summary"] = con.execute(
        """
        SELECT
          COUNT(*) AS n_submissions,
          AVG(thread_size) AS mean_thread_size,
          MEDIAN(thread_size) AS median_thread_size,
          STDDEV_SAMP(thread_size) AS sd_thread_size,
          MIN(thread_size) AS min_thread_size,
          MAX(thread_size) AS max_thread_size,
          QUANTILE_CONT(thread_size, 0.75) AS p75_thread_size,
          QUANTILE_CONT(thread_size, 0.90) AS p90_thread_size,
          QUANTILE_CONT(thread_size, 0.95) AS p95_thread_size,
          QUANTILE_CONT(thread_size, 0.99) AS p99_thread_size,
          SUM(CASE WHEN thread_size = 0 THEN 1 ELSE 0 END) AS n_zero_comment_submissions,
          AVG(CASE WHEN thread_size = 0 THEN 1.0 ELSE 0.0 END) AS pct_zero_comment_submissions,
          SUM(CASE WHEN thread_size = 1 THEN 1 ELSE 0 END) AS n_one_comment_submissions
        FROM thread_sizes_all
        """
    ).df()

    # Thread size by subreddit
    logger.info("Building thread_size_by_subreddit")
    tables["thread_size_by_subreddit"] = con.execute(
        """
        SELECT
          subreddit,
          COUNT(*) AS n_submissions,
          AVG(thread_size) AS mean_thread_size,
          MEDIAN(thread_size) AS median_thread_size,
          STDDEV_SAMP(thread_size) AS sd_thread_size,
          MIN(thread_size) AS min_thread_size,
          MAX(thread_size) AS max_thread_size,
          QUANTILE_CONT(thread_size, 0.75) AS p75_thread_size,
          QUANTILE_CONT(thread_size, 0.90) AS p90_thread_size,
          QUANTILE_CONT(thread_size, 0.95) AS p95_thread_size,
          QUANTILE_CONT(thread_size, 0.99) AS p99_thread_size,
          AVG(CASE WHEN thread_size = 0 THEN 1.0 ELSE 0.0 END) AS pct_zero_comment_submissions
        FROM thread_sizes_all
        GROUP BY 1
        ORDER BY n_submissions DESC, subreddit
        """
    ).df()

    # Thread size by month
    logger.info("Building thread_size_by_month")
    tables["thread_size_by_month"] = con.execute(
        """
        SELECT
          month,
          COUNT(*) AS n_submissions,
          AVG(thread_size) AS mean_thread_size,
          MEDIAN(thread_size) AS median_thread_size,
          STDDEV_SAMP(thread_size) AS sd_thread_size,
          MIN(thread_size) AS min_thread_size,
          MAX(thread_size) AS max_thread_size,
          QUANTILE_CONT(thread_size, 0.75) AS p75_thread_size,
          QUANTILE_CONT(thread_size, 0.90) AS p90_thread_size,
          QUANTILE_CONT(thread_size, 0.95) AS p95_thread_size,
          QUANTILE_CONT(thread_size, 0.99) AS p99_thread_size,
          AVG(CASE WHEN thread_size = 0 THEN 1.0 ELSE 0.0 END) AS pct_zero_comment_submissions
        FROM thread_sizes_all
        GROUP BY 1
        ORDER BY month
        """
    ).df()

    # Zero-inflation / sparsity summaries
    logger.info("Building zero_inflation")
    tables["zero_inflation"] = con.execute(
        f"""
        WITH user_contribs AS (
          SELECT user_id, COUNT(*) AS n_contributions
          FROM (
            SELECT {author_col} AS user_id FROM submissions_core_userlevel
            UNION ALL
            SELECT {author_col} AS user_id FROM comments_core_userlevel
          ) z
          GROUP BY 1
        ),
        thread_sizes AS (
          SELECT submission_id, thread_size
          FROM thread_sizes_all
        )
        SELECT * FROM (
          SELECT 'pct_zero_comment_submissions' AS metric,
                 AVG(CASE WHEN thread_size = 0 THEN 1.0 ELSE 0.0 END) AS value
          FROM thread_sizes
          UNION ALL
          SELECT 'pct_one_contribution_users',
                 AVG(CASE WHEN n_contributions = 1 THEN 1.0 ELSE 0.0 END)
          FROM user_contribs
          UNION ALL
          SELECT 'pct_multi_contribution_users',
                 AVG(CASE WHEN n_contributions > 1 THEN 1.0 ELSE 0.0 END)
          FROM user_contribs
        ) t
        """
    ).df()

    return tables


# ----------------------------- Output writers ----------------------------


def write_csv_tables(out_root: str, tables: Dict[str, pd.DataFrame], overwrite: bool) -> None:
    """Write one CSV per output table under out_root/tables/."""
    tables_dir = os.path.join(out_root, "tables")
    ensure_dir(tables_dir)

    for name in FULL_TABLE_ORDER:
        if name not in tables:
            continue
        out_path = os.path.join(tables_dir, f"{name}.csv")
        if os.path.exists(out_path) and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing table: {out_path}")
        tables[name].to_csv(out_path, index=False)
    
def write_parquet_tables(out_root: str, tables: Dict[str, pd.DataFrame], overwrite: bool) -> None:
    """Write one Parquet file per output table under out_root/tables_parquet/."""
    tables_dir = os.path.join(out_root, "tables_parquet")
    ensure_dir(tables_dir)

    for name in FULL_TABLE_ORDER:
        if name not in tables:
            continue
        out_path = os.path.join(tables_dir, f"{name}.parquet")
        if os.path.exists(out_path) and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing Parquet table: {out_path}")
        tables[name].to_parquet(out_path, index=False)


def write_xlsx_workbook(
    out_root: str,
    year: str,
    tables: Dict[str, pd.DataFrame],
    overwrite: bool,
) -> str:
    """Write a single workbook with one sheet per output table."""
    out_path = os.path.join(out_root, f"descriptive_stats_{year}.xlsx")
    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing workbook: {out_path}")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        used_sheet_names: set[str] = set()
        for name in XLSX_TABLE_ORDER:
            if name not in tables:
                continue
            sheet = excel_sheet_name(name)
            # avoid collisions after truncation
            if sheet in used_sheet_names:
                i = 2
                base = sheet[:28]
                while f"{base}_{i}" in used_sheet_names:
                    i += 1
                sheet = f"{base}_{i}"
            used_sheet_names.add(sheet)
            tables[name].to_excel(writer, sheet_name=sheet, index=False)
    return out_path


def write_run_metadata(out_root: str, cfg: StatsConfig, duck: DuckDBConfig) -> str:
    """Write JSON run metadata file and return its path."""
    ensure_dir(os.path.join(out_root, "audit"))
    out_path = os.path.join(out_root, "audit", f"descriptive_stats_run_{cfg.run_id}.json")
    payload = {
        "run_id": cfg.run_id,
        "timestamp_utc": utc_now_iso(),
        "script": os.path.basename(__file__),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "duckdb": {"version": duckdb.__version__, **asdict(duck)},
        "config": asdict(cfg),
    }
    safe_json_dump(payload, out_path)
    return out_path


# ----------------------------- CLI --------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI parser for this script."""
    p = argparse.ArgumentParser(
        description="Compute descriptive statistics for cleaned Reddit monthly Parquet files (DuckDB).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--in-root", required=True, help="Input root containing cleaned Parquet outputs.")
    p.add_argument("--out-root", required=True, help="Output root for descriptive tables.")
    p.add_argument("--year", default="2021", help="Year to process (YYYY).")
    p.add_argument(
        "--only-months",
        default="",
        help="Comma-separated subset like 2021-01,2021-06.",
    )
    p.add_argument(
        "--limit-months",
        type=int,
        default=0,
        help="Limit number of discovered months after any whitelist is applied (0 = no limit).",
    )

    p.add_argument(
        "--author-col",
        default="auto",
        choices=["auto", "author_pid", "author"],
        help="Author identifier column to use for user-level summaries.",
    )

    p.add_argument(
        "--write-xlsx",
        dest="write_xlsx",
        action="store_true",
        help="Write Excel workbook with one sheet per output table.",
    )
    p.add_argument(
        "--no-write-xlsx",
        dest="write_xlsx",
        action="store_false",
        help="Disable Excel workbook writing.",
    )
    p.set_defaults(write_xlsx=True)

    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip the run if the final workbook already exists.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV tables / workbook if present.",
    )

    p.add_argument("--smoke", action="store_true", help="Enable smoke mode (quick plumbing run).")
    p.add_argument(
        "--smoke-max-rows",
        type=int,
        default=0,
        help="If > 0 and --smoke, limit rows read per file/relation. Use only for plumbing checks.",
    )

    p.add_argument(
        "--duckdb-memory-limit",
        default=os.environ.get("DUCKDB_MEMORY_LIMIT", "16GB"),
        help="DuckDB memory limit (e.g. 4GB, 16GB, 32GB).",
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
        help="DuckDB temp_directory (prefer node-local scratch on HPC).",
    )

    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )

    return p


# ----------------------------- Main -------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    setup_logging(args.log_level)

    if args.smoke and not args.only_months and args.limit_months == 0:
        # Conservative default for fast smoke runs.
        args.limit_months = 2

    cfg = StatsConfig(
        in_root=args.in_root,
        out_root=args.out_root,
        year=str(args.year),
        only_months=parse_only_months(args.only_months),
        limit_months=int(args.limit_months),
        author_col=str(args.author_col),
        write_xlsx=bool(args.write_xlsx),
        skip_existing=bool(args.skip_existing),
        overwrite=bool(args.overwrite),
        smoke=bool(args.smoke),
        smoke_max_rows=int(args.smoke_max_rows) if args.smoke else 0,
        run_id=utc_run_id("descriptive_stats", str(args.year)),
    )
    duck = DuckDBConfig(
        memory_limit=args.duckdb_memory_limit,
        threads=int(args.duckdb_threads),
        temp_dir=args.duckdb_temp_dir,
    )

    ensure_dir(cfg.out_root)
    ensure_dir(os.path.join(cfg.out_root, "audit"))
    ensure_dir(os.path.join(cfg.out_root, "tables"))

    final_workbook = os.path.join(cfg.out_root, f"descriptive_stats_{cfg.year}.xlsx")
    if cfg.skip_existing and cfg.write_xlsx and os.path.exists(final_workbook):
        logger.info("Workbook already exists and --skip-existing set: %s", final_workbook)
        return 0

    # Persist resolved config early.
    config_json = os.path.join(cfg.out_root, "audit", f"descriptive_stats_config_{cfg.run_id}.json")
    safe_json_dump({"config": asdict(cfg), "duckdb": asdict(duck)}, config_json)

    inv = discover_core_files(cfg.in_root, cfg.year)
    inv = filter_month_inventory(inv, cfg.only_months, cfg.limit_months)
    if not inv:
        raise RuntimeError("No monthly core files found after applying filters.")
    require_complete_inventory(inv)

    months = list(sorted(inv.keys()))
    logger.info("Selected months (%d): %s", len(months), ", ".join(months))
    logger.info("Smoke mode: %s | smoke_max_rows=%s", cfg.smoke, cfg.smoke_max_rows)

    con = duckdb.connect()
    configure_duckdb(con, duck)

    try:
        build_base_views(con, inv, cfg.smoke_max_rows)
        comments_cols = get_table_columns(con, "comments_core_all")
        submissions_cols = get_table_columns(con, "submissions_core_all")
        author_col = resolve_author_col(comments_cols, submissions_cols, cfg.author_col)
        logger.info("Resolved author column: %s", author_col)

        build_user_level_views(con, author_col=author_col)
        build_derived_tables(con, author_col=author_col)
        build_user_level_derived_tables(
            con,
            inv=inv,
            author_col=author_col,
            default_threads=duck.threads,
        )
        drop_temp_tables(
            con,
            [
                "parent_comment_author_map",
            ],
        )
        subreddit_flags = build_subreddit_flags(con)

        tables = make_tables(con, author_col=author_col)
        tables["deleted_author_activity_summary"] = build_deleted_author_activity_summary(con)
        tables.update(build_user_activity_tables(con))

        drop_temp_tables(
            con,
            [
                "user_submission_counts",
                "user_comment_counts",
                "user_distinct_subreddits",
                "user_submission_engagement",
                "user_comment_engagement",
                "user_universe",
                "comment_counts_by_link",
            ],
        )

        tables["schema_summary"] = build_schema_summary(comments_cols, submissions_cols, author_col)
        tables["qc_checks"] = build_qc_checks(con)
        audit_csv = os.path.join(cfg.in_root, "audit", f"clean_audit_{cfg.year}.csv")
        tables["audit_summary"] = read_clean_audit_summary(audit_csv)

        tables = add_reporting_summaries(tables, subreddit_flags)
        tables["table_definitions"] = build_table_definitions()

        write_csv_tables(cfg.out_root, tables, overwrite=cfg.overwrite)
        write_parquet_tables(cfg.out_root, tables, overwrite=cfg.overwrite)

        workbook_path: Optional[str] = None
        if cfg.write_xlsx:
            workbook_path = write_xlsx_workbook(cfg.out_root, cfg.year, tables, overwrite=cfg.overwrite)

        run_meta = write_run_metadata(cfg.out_root, cfg, duck)
        finished_json = os.path.join(
            cfg.out_root,
            "audit",
            f"descriptive_stats_finished_{cfg.run_id}.json",
        )
        safe_json_dump(
            {
                "run_id": cfg.run_id,
                "timestamp_utc": utc_now_iso(),
                "months": months,
                "author_col": author_col,
                "tables_written": [name for name in FULL_TABLE_ORDER if name in tables],
                "workbook_path": workbook_path,
                "config_json": config_json,
                "run_metadata_json": run_meta,
            },
            finished_json,
        )

        logger.info("Wrote %d CSV tables under %s", len([k for k in FULL_TABLE_ORDER if k in tables]), os.path.join(cfg.out_root, "tables"))
        if workbook_path:
            logger.info("Wrote workbook: %s", workbook_path)
        logger.info("Done.")
        return 0

    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
