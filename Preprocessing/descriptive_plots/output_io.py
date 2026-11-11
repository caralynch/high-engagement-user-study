from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import os
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Tuple

import duckdb
import pandas as pd

from .config import (
    CSV_TABLE_NAMES,
    DuckDBConfig,
    LARGE_INTERMEDIATE_TABLE_NAMES,
    LAG_ROW_LEVEL_LARGE_INTERMEDIATE_NAMES,
    MONTH_RE,
    PlotConfig,
    TAIL_AUDIT_TABLE_NAMES,
    csv_table_names,
    figure_names,
)


logger = logging.getLogger(__name__)


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

def quote_sql_string(s: str) -> str:
    """Safely quote a string literal for SQL."""
    return "'" + s.replace("'", "''") + "'"

def normalize_metadata_path(path: str) -> str:
    """Normalise a filesystem path for checkpoint metadata comparisons."""
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))

def month_add_one(month_str: str) -> str:
    """Add one month to a YYYY-MM string."""
    dt = pd.Period(month_str, freq="M") + 1
    return str(dt)

@contextmanager
def timed_stage(label: str) -> Iterator[None]:
    """Log elapsed wall-clock time for a processing stage."""
    start = time.perf_counter()
    logger.info("Starting %s", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("Finished %s in %.1f seconds", label, elapsed)

@contextmanager
def temporary_duckdb_threads(
    con: duckdb.DuckDBPyConnection,
    threads: int,
    restore_threads: int,
) -> Iterator[None]:
    """Temporarily set DuckDB threads for memory-sensitive sort/window stages."""
    con.execute(f"SET threads={int(threads)}")
    try:
        yield
    finally:
        con.execute(f"SET threads={int(restore_threads)}")

def log_table_count(con: duckdb.DuckDBPyConnection, table_name: str) -> int:
    """Log and return a DuckDB table row count."""
    n_rows = int(con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
    logger.info("Table %s row count: %s", table_name, f"{n_rows:,}")
    return n_rows

def expected_output_paths(
    out_root: str,
    write_large_intermediates: bool = False,
    skip_subreddit_subset_diagnostics: bool = False,
) -> List[str]:
    """Return the deterministic table and figure paths this script writes."""
    tables_dir = os.path.join(out_root, "tables")
    figures_dir = os.path.join(out_root, "figures")

    paths = [
        os.path.join(tables_dir, f"{name}.csv")
        for name in csv_table_names(skip_subreddit_subset_diagnostics)
    ]
    paths.extend(os.path.join(tables_dir, f"{name}.csv.gz") for name in TAIL_AUDIT_TABLE_NAMES)
    if write_large_intermediates:
        paths.extend(
            os.path.join(tables_dir, f"{name}.parquet")
            for name in LARGE_INTERMEDIATE_TABLE_NAMES
            if name not in LAG_ROW_LEVEL_LARGE_INTERMEDIATE_NAMES
        )
    for name in figure_names(skip_subreddit_subset_diagnostics):
        paths.append(os.path.join(figures_dir, f"{name}.png"))
        paths.append(os.path.join(figures_dir, f"{name}.svg"))
        paths.append(os.path.join(figures_dir, f"{name}.pdf"))
    return paths

def check_output_collisions(
    out_root: str,
    overwrite: bool,
    write_large_intermediates: bool = False,
    skip_subreddit_subset_diagnostics: bool = False,
) -> None:
    """Fail fast if deterministic outputs already exist and overwrite is disabled."""
    if overwrite:
        return
    collisions = [
        path
        for path in expected_output_paths(
            out_root,
            write_large_intermediates=write_large_intermediates,
            skip_subreddit_subset_diagnostics=skip_subreddit_subset_diagnostics,
        )
        if os.path.exists(path)
    ]
    if collisions:
        raise FileExistsError(
            "Refusing to overwrite existing outputs without --overwrite. Existing files:\n  - "
            + "\n  - ".join(collisions)
        )

def large_intermediate_export_queries() -> Dict[str, str]:
    """Return SQL queries for optional large intermediate Parquet exports."""
    return {
        "per_user_activity": "SELECT * FROM per_user_activity",
        "per_user_active_months": "SELECT * FROM per_user_active_months",
        "user_subreddit_hhi": "SELECT * FROM user_subreddit_hhi",
        "thread_size_distribution": "SELECT thread_size FROM thread_sizes_all",
        "user_activity_lorenz_curve": """
            WITH user_totals AS (
              SELECT
                user_id,
                SUM(COALESCE(n_total_activity, 0)) AS activity_value
              FROM user_month_panel
              GROUP BY 1
            ),
            ranked AS (
              SELECT
                ROW_NUMBER() OVER (ORDER BY activity_value ASC, user_id ASC) AS user_rank,
                COUNT(*) OVER () AS n_users_total,
                activity_value,
                SUM(activity_value) OVER (
                  ORDER BY activity_value ASC, user_id ASC
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS cum_activity
              FROM user_totals
            )
            SELECT
              user_rank,
              n_users_total,
              user_rank AS cum_users,
              CASE
                WHEN n_users_total > 0 THEN CAST(user_rank AS DOUBLE) / CAST(n_users_total AS DOUBLE)
                ELSE NULL
              END AS cum_user_share,
              activity_value,
              cum_activity,
              CASE
                WHEN MAX(cum_activity) OVER () > 0 THEN CAST(cum_activity AS DOUBLE) / CAST(MAX(cum_activity) OVER () AS DOUBLE)
                ELSE 0.0
              END AS cum_activity_share
            FROM ranked
            ORDER BY user_rank
        """,
        "user_engagement_lorenz_curve": """
            WITH user_totals AS (
              SELECT
                user_id,
                SUM(COALESCE(n_total_engagement_received, 0)) AS engagement_value
              FROM user_month_panel
              GROUP BY 1
            ),
            ranked AS (
              SELECT
                ROW_NUMBER() OVER (ORDER BY engagement_value ASC, user_id ASC) AS user_rank,
                COUNT(*) OVER () AS n_users_total,
                engagement_value,
                SUM(engagement_value) OVER (
                  ORDER BY engagement_value ASC, user_id ASC
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS cum_engagement
              FROM user_totals
            )
            SELECT
              user_rank,
              n_users_total,
              user_rank AS cum_users,
              CASE
                WHEN n_users_total > 0 THEN CAST(user_rank AS DOUBLE) / CAST(n_users_total AS DOUBLE)
                ELSE NULL
              END AS cum_user_share,
              engagement_value,
              cum_engagement,
              CASE
                WHEN MAX(cum_engagement) OVER () > 0 THEN CAST(cum_engagement AS DOUBLE) / CAST(MAX(cum_engagement) OVER () AS DOUBLE)
                ELSE 0.0
              END AS cum_engagement_share
            FROM ranked
            ORDER BY user_rank
        """,
        "active_subreddits_per_user_distribution": """
            SELECT
              user_id,
              n_active_subreddits
            FROM user_subreddit_hhi
            ORDER BY n_active_subreddits ASC, user_id ASC
        """,
    }

def write_large_intermediate_tables_from_duckdb(
    con: duckdb.DuckDBPyConnection,
    out_root: str,
    overwrite: bool,
) -> List[str]:
    """Export optional large intermediate tables directly from DuckDB to Parquet."""
    tables_dir = os.path.join(out_root, "tables")
    ensure_dir(tables_dir)
    queries = large_intermediate_export_queries()
    written: List[str] = []
    safe_names = [name for name in LARGE_INTERMEDIATE_TABLE_NAMES if name not in LAG_ROW_LEVEL_LARGE_INTERMEDIATE_NAMES]
    skipped_names = [name for name in LARGE_INTERMEDIATE_TABLE_NAMES if name in LAG_ROW_LEVEL_LARGE_INTERMEDIATE_NAMES]
    if skipped_names:
        logger.warning(
            "Skipping row-level lag large exports in Stage 1 because full-year lag materialisation is disabled: %s",
            ", ".join(skipped_names),
        )
    logger.info("Exporting large intermediates as zstd Parquet: %s", ", ".join(safe_names))
    for name in safe_names:
        out_path = os.path.join(tables_dir, f"{name}.parquet")
        if os.path.exists(out_path):
            if not overwrite:
                raise FileExistsError(f"Refusing to overwrite existing table: {out_path}")
            os.remove(out_path)
        query = queries[name]
        con.execute(
            f"""
            COPY ({query})
            TO {quote_sql_string(out_path)}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        written.append(name)
    return written

def write_tables(
    out_root: str,
    tables: Dict[str, pd.DataFrame],
    overwrite: bool,
    write_large_intermediates: bool,
) -> Dict[str, List[str]]:
    """Write tables according to the CSV/gzip/Parquet output policy."""
    tables_dir = os.path.join(out_root, "tables")
    ensure_dir(tables_dir)
    written = {
        "tables_written_csv": [],
        "tables_written_csv_gz": [],
        "tables_written_parquet": [],
    }

    for name in CSV_TABLE_NAMES:
        if name not in tables:
            continue
        out_path = os.path.join(tables_dir, f"{name}.csv")
        if os.path.exists(out_path) and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing table: {out_path}")
        tables[name].to_csv(out_path, index=False)
        written["tables_written_csv"].append(name)

    for name in TAIL_AUDIT_TABLE_NAMES:
        if name not in tables:
            continue
        out_path = os.path.join(tables_dir, f"{name}.csv.gz")
        if os.path.exists(out_path) and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing table: {out_path}")
        tables[name].to_csv(out_path, index=False, compression="gzip")
        written["tables_written_csv_gz"].append(name)

    if write_large_intermediates:
        logger.debug("Large intermediate Parquet files are exported directly from DuckDB, not pandas.")

    return written

def write_run_metadata(out_root: str, cfg: PlotConfig, duck: DuckDBConfig, script_name: str = "06_0_descriptive_plots.py") -> str:
    """Write JSON run metadata file and return its path."""
    ensure_dir(os.path.join(out_root, "audit"))
    out_path = os.path.join(out_root, "audit", f"descriptive_plots_run_{cfg.run_id}.json")
    payload = {
        "run_id": cfg.run_id,
        "timestamp_utc": utc_now_iso(),
        "script": script_name,
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "duckdb": {"version": duckdb.__version__, **asdict(duck)},
        "config": asdict(cfg),
    }
    safe_json_dump(payload, out_path)
    return out_path
