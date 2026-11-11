#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Cara Lynch
# See the LICENSE file for details.
"""
Preprocessing step 7.3: describe the selected subreddit modelling subset.

This script reads the selected-subset monthly Parquet files produced by
`07_2_make_selected_subreddit_parquets.py` and writes thesis/report-friendly
descriptive tables, figures, and audit metadata for the actual dataset that will
enter the modelling study.

It deliberately does not perform subreddit subset selection, recompute candidate
scores, apply threshold rules, or filter subreddits beyond validating the input
dataset against an explicit selected-subreddit list when one is supplied.

Expected input layout
---------------------
<in-root>/comments/comments_core_YYYY-MM.parquet
<in-root>/submissions/submissions_core_YYYY-MM.parquet

Preferred selected-list input
-----------------------------
Use the audit list written by `07_2_make_selected_subreddit_parquets.py`:

<selected-subset-root>/audit/selected_subreddits.csv

When `--selected-subreddits-csv` is provided, this script verifies that all
subreddits present in the selected-subset Parquets are in that list, and that
all listed subreddits are represented in the data. Missing listed subreddits are
fatal except in smoke mode, where they are recorded as warnings.

Denominator policy
------------------
- Content-level counts include all rows in the selected-subset Parquets.
- User-level counts exclude rows where `is_deleted_author` is true and exclude
  rows where the resolved author identifier column is NULL.
- Thread-size denominators include all selected-subset submissions, including
  zero-comment submissions. Survival plots are drawn over positive support when
  a log-scaled axis is used, but the table denominator still includes
  zero-comment submissions.
- Received-engagement summaries are not computed here. The aligned descriptive
  received-engagement outputs live in `06_0_descriptive_plots.py`/stage 07.4
  and use the modelling contract: event-month unique other-user child comments
  received by authored content, with self-engagement excluded, submission-thread
  and direct-reply components retained, and overlapping component attribution
  de-duplicated in the main total.

Outputs
-------
Tables under <out-root>/tables/:

  selected_subreddits_overview.csv
  selected_subset_monthly_activity.csv
  selected_subreddit_monthly_activity.csv
  selected_subset_thread_size_distribution_plot.csv
  selected_subset_user_activity_distribution_plot.csv
  selected_subset_validation_summary.csv
  selected_subset_summary.json

Figures under <out-root>/figures/ are written as PNG, SVG, and PDF.

Audit files under <out-root>/audit/ include the exact selected subreddit list
used for the run and JSON metadata with denominator policy, validation results,
runtime configuration, and output paths.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from Preprocessing.descriptive_plots.config import COLORBLIND_SAFE_PALETTE, DuckDBConfig, PLOT_SIZES
from Preprocessing.descriptive_plots.derived_tables import discover_core_files, require_complete_inventory
from Preprocessing.descriptive_plots.duckdb_helpers import (
    build_base_views,
    build_user_level_views,
    configure_duckdb,
    get_table_columns,
    resolve_author_col,
)
from Preprocessing.descriptive_plots.output_io import ensure_dir, quote_sql_string, safe_json_dump, utc_now_iso, utc_run_id
from Preprocessing.descriptive_plots.plotting import save_figure


logger = logging.getLogger(__name__)


TABLE_NAMES = (
    "selected_subreddits_overview",
    "selected_subset_monthly_activity",
    "selected_subreddit_monthly_activity",
    "selected_subset_thread_size_distribution_plot",
    "selected_subset_user_activity_distribution_plot",
    "selected_subset_validation_summary",
)

FIGURE_NAMES = (
    "selected_subset_monthly_activity",
    "selected_subreddit_activity_comparison",
    "selected_subreddit_active_users_comparison",
    "selected_subreddit_mean_thread_size_comparison",
    "selected_subreddit_repeat_user_share_comparison",
    "selected_subset_thread_size_distribution",
    "selected_subset_user_activity_distribution",
)

DENOMINATOR_POLICY = {
    "content_level_counts": "include all rows in the selected-subset Parquets",
    "user_level_counts": (
        "exclude rows where is_deleted_author is true and rows where the resolved author identifier is NULL"
    ),
    "thread_size": (
        "include all selected-subset submissions in denominators, including zero-comment submissions; "
        "log-scale figures plot positive support only"
    ),
}

RECEIVED_ENGAGEMENT_OMISSION = {
    "computed": False,
    "definition_if_computed": (
        "event-month unique other-user child comments received by authored content; "
        "self-engagement excluded; submission-thread and direct-reply components retained; "
        "overlapping component attribution de-duplicated in the main total"
    ),
    "reason": (
        "Received-engagement summaries are omitted from 07_3 to keep selected-subset descriptives focused. "
        "Use the aligned 06_0/07_4 descriptive plots pipeline for engagement-derived outputs; older "
        "engagement-derived outputs are superseded until rerun under the revised modelling contract."
    ),
}


@dataclass(frozen=True)
class SelectedDescriptivesConfig:
    """Resolved runtime configuration."""

    in_root: str
    out_root: str
    selected_subreddits_csv: Optional[str]
    year: str
    author_col: str
    overwrite: bool
    publication_style: bool
    smoke: bool
    smoke_max_rows: int
    run_id: str


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


def quote_identifier(name: str) -> str:
    """Return a double-quoted SQL identifier."""
    return '"' + name.replace('"', '""') + '"'


def read_csv_expr(path: str | Path) -> str:
    """Return a DuckDB CSV scan expression using all-varchar inference."""
    return f"read_csv_auto({quote_sql_string(str(path))}, all_varchar=true)"


def detect_csv_columns(con: duckdb.DuckDBPyConnection, path: str | Path) -> list[str]:
    """Return CSV column names via DuckDB."""
    try:
        rows = con.execute(f"DESCRIBE SELECT * FROM {read_csv_expr(path)} LIMIT 0").fetchall()
    except duckdb.Error as exc:
        raise RuntimeError(f"Could not read CSV schema for {path}: {exc}") from exc
    return [str(row[0]) for row in rows]


def check_output_collisions(out_root: Path, overwrite: bool) -> None:
    """Fail fast if deterministic outputs already exist and overwrite is disabled."""
    if overwrite:
        return

    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"
    audit_dir = out_root / "audit"
    paths: list[Path] = [tables_dir / f"{name}.csv" for name in TABLE_NAMES]
    paths.append(tables_dir / "selected_subset_summary.json")
    paths.append(audit_dir / "selected_subreddits.csv")
    paths.append(audit_dir / "selected_subset_descriptives_metadata.json")
    for figure_name in FIGURE_NAMES:
        paths.extend(figures_dir / f"{figure_name}.{ext}" for ext in ("png", "svg", "pdf"))

    collisions = [path for path in paths if path.exists()]
    if collisions:
        raise FileExistsError(
            "Refusing to overwrite existing outputs without --overwrite. Existing files:\n  - "
            + "\n  - ".join(str(path) for path in collisions)
        )


def add_validation(
    rows: list[dict[str, str]],
    check_name: str,
    status: str,
    severity: str,
    observed_value: Any,
    expected_value: Any,
    message: str,
) -> None:
    """Append one validation row using stable string fields."""
    effective_severity = "info" if status == "pass" and severity in {"warning", "error"} else severity
    rows.append(
        {
            "check_name": check_name,
            "status": status,
            "severity": effective_severity,
            "observed_value": "" if observed_value is None else str(observed_value),
            "expected_value": "" if expected_value is None else str(expected_value),
            "message": message,
        }
    )


def validation_has_fatal(rows: Sequence[dict[str, str]]) -> bool:
    """Return True when any validation row failed with error severity."""
    return any(row["status"] == "fail" and row["severity"] == "error" for row in rows)


def make_month_dim(con: duckdb.DuckDBPyConnection, months: Sequence[str]) -> None:
    """Create a temp month dimension table from discovered inventory months."""
    con.execute("CREATE OR REPLACE TEMP TABLE month_dim(month VARCHAR)")
    con.executemany("INSERT INTO month_dim VALUES (?)", [(month,) for month in months])


def create_selected_subreddits_table(
    con: duckdb.DuckDBPyConnection,
    selected_csv: Optional[str],
) -> tuple[str, list[str]]:
    """Create temp selected_subreddits and return the source kind plus names."""
    if selected_csv:
        source = Path(selected_csv)
        if not source.exists():
            raise FileNotFoundError(f"Selected subreddits CSV not found: {source}")
        columns = detect_csv_columns(con, source)
        if "subreddit" not in set(columns):
            raise ValueError(
                f"{source} is missing required column 'subreddit'. Available columns: {', '.join(columns)}"
            )

        selection_order_expr = (
            "TRY_CAST(NULLIF(trim(CAST(selection_order AS VARCHAR)), '') AS BIGINT)"
            if "selection_order" in set(columns)
            else "CAST(NULL AS BIGINT)"
        )
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE selected_subreddits AS
            WITH raw AS (
              SELECT
                ROW_NUMBER() OVER () AS input_order,
                {selection_order_expr} AS supplied_selection_order,
                NULLIF(trim(CAST(subreddit AS VARCHAR)), '') AS subreddit
              FROM {read_csv_expr(source)}
            ),
            deduped AS (
              SELECT
                subreddit,
                MIN(COALESCE(supplied_selection_order, input_order)) AS order_key
              FROM raw
              WHERE subreddit IS NOT NULL
              GROUP BY 1
            )
            SELECT
              ROW_NUMBER() OVER (ORDER BY order_key ASC, lower(subreddit) ASC, subreddit ASC) AS selection_order,
              subreddit
            FROM deduped
            ORDER BY selection_order
            """
        )
        return "csv", selected_subreddit_names(con)

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE selected_subreddits AS
        WITH observed AS (
          SELECT CAST(subreddit AS VARCHAR) AS subreddit
          FROM submissions_core_all
          WHERE subreddit IS NOT NULL
          UNION
          SELECT CAST(subreddit AS VARCHAR) AS subreddit
          FROM comments_core_all
          WHERE subreddit IS NOT NULL
        )
        SELECT
          ROW_NUMBER() OVER (ORDER BY lower(subreddit) ASC, subreddit ASC) AS selection_order,
          subreddit
        FROM observed
        ORDER BY selection_order
        """
    )
    return "inferred_from_data", selected_subreddit_names(con)


def selected_subreddit_names(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Return selected subreddit names in run order."""
    return [
        str(row[0])
        for row in con.execute("SELECT subreddit FROM selected_subreddits ORDER BY selection_order").fetchall()
    ]


def write_selected_subreddits(con: duckdb.DuckDBPyConnection, out_path: Path) -> None:
    """Write the exact selected subreddit list used by this run."""
    con.execute(
        f"""
        COPY (
          SELECT selection_order, subreddit
          FROM selected_subreddits
          ORDER BY selection_order
        )
        TO {quote_sql_string(str(out_path))}
        (HEADER, DELIMITER ',')
        """
    )


def validate_selected_subreddits(
    con: duckdb.DuckDBPyConnection,
    validation_rows: list[dict[str, str]],
    selected_source: str,
    smoke: bool,
) -> dict[str, Any]:
    """Validate selected-list integrity against subreddits observed in content rows."""
    selected_count = int(con.execute("SELECT COUNT(*) FROM selected_subreddits").fetchone()[0])
    add_validation(
        validation_rows,
        "selected_subreddit_count_positive",
        "pass" if selected_count > 0 else "fail",
        "error",
        selected_count,
        "> 0",
        "Selected subreddit count must be positive.",
    )

    duplicate_input_count = 0
    if selected_source == "csv":
        duplicate_input_count = int(
            con.execute(
                """
                WITH raw AS (
                  SELECT NULLIF(trim(CAST(subreddit AS VARCHAR)), '') AS subreddit
                  FROM selected_subreddits_input
                )
                SELECT COALESCE(SUM(n_rows - 1), 0)
                FROM (
                  SELECT subreddit, COUNT(*) AS n_rows
                  FROM raw
                  WHERE subreddit IS NOT NULL
                  GROUP BY 1
                  HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
        ) if table_exists(con, "selected_subreddits_input") else 0
        add_validation(
            validation_rows,
            "selected_subreddit_input_duplicates",
            "warn" if duplicate_input_count > 0 else "pass",
            "warning" if duplicate_input_count > 0 else "info",
            duplicate_input_count,
            "0",
            "Duplicate input subreddit rows are de-duplicated for the run list.",
        )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE data_subreddits AS
        SELECT DISTINCT subreddit
        FROM (
          SELECT CAST(subreddit AS VARCHAR) AS subreddit
          FROM submissions_core_all
          WHERE subreddit IS NOT NULL
          UNION
          SELECT CAST(subreddit AS VARCHAR) AS subreddit
          FROM comments_core_all
          WHERE subreddit IS NOT NULL
        )
        """
    )

    unexpected = [
        str(row[0])
        for row in con.execute(
            """
            SELECT d.subreddit
            FROM data_subreddits d
            LEFT JOIN selected_subreddits s
              ON d.subreddit = s.subreddit
            WHERE s.subreddit IS NULL
            ORDER BY lower(d.subreddit), d.subreddit
            """
        ).fetchall()
    ]
    missing = [
        str(row[0])
        for row in con.execute(
            """
            SELECT s.subreddit
            FROM selected_subreddits s
            LEFT JOIN data_subreddits d
              ON s.subreddit = d.subreddit
            WHERE d.subreddit IS NULL
            ORDER BY s.selection_order
            """
        ).fetchall()
    ]

    add_validation(
        validation_rows,
        "unexpected_subreddits_absent",
        "pass" if not unexpected else "fail",
        "error",
        len(unexpected),
        "0",
        (
            "All data subreddits are present in the selected list."
            if not unexpected
            else "Unexpected data subreddits: " + ", ".join(unexpected[:20])
        ),
    )
    missing_status = "pass"
    missing_severity = "info"
    if missing and smoke:
        missing_status = "warn"
        missing_severity = "warning"
    elif missing:
        missing_status = "fail"
        missing_severity = "error"
    add_validation(
        validation_rows,
        "selected_subreddits_present_in_data",
        missing_status,
        missing_severity,
        len(missing),
        "0",
        (
            "All selected subreddits are represented in the data."
            if not missing
            else "Selected subreddits missing from data: " + ", ".join(missing[:20])
        ),
    )

    return {
        "selected_subreddit_count": selected_count,
        "unexpected_subreddits": unexpected,
        "missing_selected_subreddits": missing,
        "duplicate_input_rows": duplicate_input_count,
    }


def table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    """Return whether a temp/main table exists."""
    return bool(
        con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = ?
            """,
            [table_name],
        ).fetchone()[0]
    )


def materialise_selected_subreddit_summaries(
    con: duckdb.DuckDBPyConnection,
    author_col: str,
) -> None:
    """Materialise compact selected-subset summary tables in DuckDB."""
    q_author = quote_identifier(author_col)

    logger.info("Materialising comment_counts_by_link")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE comment_counts_by_link AS
        SELECT link_id, COUNT(*) AS n_comments
        FROM comments_core_all
        GROUP BY 1
        """
    )

    logger.info("Materialising thread_sizes_all")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE thread_sizes_all AS
        SELECT
          s.id AS submission_id,
          CAST(s.subreddit AS VARCHAR) AS subreddit,
          strftime(s.created_utc, '%Y-%m') AS month,
          CAST(COALESCE(c.n_comments, 0) AS BIGINT) AS comments_per_submission
        FROM submissions_core_all s
        LEFT JOIN comment_counts_by_link c
          ON s.id = c.link_id
        """
    )

    logger.info("Materialising selected content counts")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE subreddit_month_content_counts AS
        SELECT
          month,
          subreddit,
          SUM(n_submissions) AS n_submissions,
          SUM(n_comments) AS n_comments,
          SUM(n_total_activity) AS n_total_activity
        FROM (
          SELECT
            strftime(created_utc, '%Y-%m') AS month,
            CAST(subreddit AS VARCHAR) AS subreddit,
            COUNT(*) AS n_submissions,
            0::BIGINT AS n_comments,
            COUNT(*) AS n_total_activity
          FROM submissions_core_all
          WHERE subreddit IS NOT NULL
          GROUP BY 1, 2
          UNION ALL
          SELECT
            strftime(created_utc, '%Y-%m') AS month,
            CAST(subreddit AS VARCHAR) AS subreddit,
            0::BIGINT AS n_submissions,
            COUNT(*) AS n_comments,
            COUNT(*) AS n_total_activity
          FROM comments_core_all
          WHERE subreddit IS NOT NULL
          GROUP BY 1, 2
        ) z
        GROUP BY 1, 2
        """
    )

    logger.info("Materialising selected user activity")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE subreddit_month_user_activity AS
        SELECT
          month,
          subreddit,
          user_id,
          SUM(n_activity) AS n_activity
        FROM (
          SELECT
            strftime(created_utc, '%Y-%m') AS month,
            CAST(subreddit AS VARCHAR) AS subreddit,
            {q_author} AS user_id,
            COUNT(*) AS n_activity
          FROM submissions_core_userlevel
          WHERE subreddit IS NOT NULL
          GROUP BY 1, 2, 3
          UNION ALL
          SELECT
            strftime(created_utc, '%Y-%m') AS month,
            CAST(subreddit AS VARCHAR) AS subreddit,
            {q_author} AS user_id,
            COUNT(*) AS n_activity
          FROM comments_core_userlevel
          WHERE subreddit IS NOT NULL
          GROUP BY 1, 2, 3
        ) z
        GROUP BY 1, 2, 3
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE subreddit_month_user_summary AS
        SELECT
          month,
          subreddit,
          COUNT(*) AS n_active_users,
          AVG(CASE WHEN n_activity >= 2 THEN 1.0 ELSE 0.0 END) AS repeat_user_share_2plus,
          AVG(CASE WHEN n_activity >= 5 THEN 1.0 ELSE 0.0 END) AS repeat_user_share_5plus
        FROM subreddit_month_user_activity
        GROUP BY 1, 2
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE per_user_activity AS
        WITH us AS (
          SELECT {q_author} AS user_id, COUNT(*) AS n_submissions
          FROM submissions_core_userlevel
          GROUP BY 1
        ),
        uc AS (
          SELECT {q_author} AS user_id, COUNT(*) AS n_comments
          FROM comments_core_userlevel
          GROUP BY 1
        ),
        uu AS (
          SELECT user_id FROM us
          UNION
          SELECT user_id FROM uc
        )
        SELECT
          uu.user_id,
          COALESCE(us.n_submissions, 0) AS n_submissions,
          COALESCE(uc.n_comments, 0) AS n_comments,
          COALESCE(us.n_submissions, 0) + COALESCE(uc.n_comments, 0) AS n_total_activity
        FROM uu
        LEFT JOIN us ON uu.user_id = us.user_id
        LEFT JOIN uc ON uu.user_id = uc.user_id
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE subreddit_full_period_users AS
        SELECT subreddit, COUNT(DISTINCT user_id) AS n_active_users
        FROM (
          SELECT subreddit, user_id FROM subreddit_month_user_activity
        ) z
        GROUP BY 1
        """
    )

    logger.info("Materialising comments-per-submission summaries")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE subreddit_month_thread_summary AS
        SELECT
          month,
          subreddit,
          AVG(comments_per_submission) AS mean_comments_per_submission,
          MEDIAN(comments_per_submission) AS median_comments_per_submission
        FROM thread_sizes_all
        GROUP BY 1, 2
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE subreddit_thread_summary AS
        SELECT
          subreddit,
          AVG(comments_per_submission) AS mean_comments_per_submission,
          MEDIAN(comments_per_submission) AS median_comments_per_submission,
          QUANTILE_CONT(comments_per_submission, 0.75) AS p75_comments_per_submission,
          QUANTILE_CONT(comments_per_submission, 0.90) AS p90_comments_per_submission
        FROM thread_sizes_all
        GROUP BY 1
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE subset_month_thread_summary AS
        SELECT
          month,
          AVG(comments_per_submission) AS mean_comments_per_submission,
          MEDIAN(comments_per_submission) AS median_comments_per_submission
        FROM thread_sizes_all
        GROUP BY 1
        """
    )

    logger.info("Materialising selected_subreddit_monthly_activity")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE selected_subreddit_monthly_activity AS
        SELECT
          md.month,
          ss.selection_order,
          ss.subreddit,
          CAST(COALESCE(cc.n_submissions, 0) AS BIGINT) AS n_submissions,
          CAST(COALESCE(cc.n_comments, 0) AS BIGINT) AS n_comments,
          CAST(COALESCE(cc.n_total_activity, 0) AS BIGINT) AS n_total_activity,
          CAST(COALESCE(us.n_active_users, 0) AS BIGINT) AS n_active_users,
          ts.mean_comments_per_submission,
          ts.median_comments_per_submission,
          us.repeat_user_share_2plus,
          us.repeat_user_share_5plus
        FROM month_dim md
        CROSS JOIN selected_subreddits ss
        LEFT JOIN subreddit_month_content_counts cc
          ON md.month = cc.month
         AND ss.subreddit = cc.subreddit
        LEFT JOIN subreddit_month_user_summary us
          ON md.month = us.month
         AND ss.subreddit = us.subreddit
        LEFT JOIN subreddit_month_thread_summary ts
          ON md.month = ts.month
         AND ss.subreddit = ts.subreddit
        ORDER BY md.month, ss.selection_order
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE subset_month_users AS
        SELECT month, COUNT(DISTINCT user_id) AS n_active_users
        FROM (
          SELECT month, user_id FROM subreddit_month_user_activity
        ) z
        GROUP BY 1
        """
    )

    logger.info("Materialising selected_subset_monthly_activity")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE selected_subset_monthly_activity AS
        SELECT
          md.month,
          CAST(COALESCE(SUM(sm.n_submissions), 0) AS BIGINT) AS n_submissions,
          CAST(COALESCE(SUM(sm.n_comments), 0) AS BIGINT) AS n_comments,
          CAST(COALESCE(SUM(sm.n_total_activity), 0) AS BIGINT) AS n_total_activity,
          CAST(COALESCE(mu.n_active_users, 0) AS BIGINT) AS n_active_users,
          CAST(COUNT(DISTINCT CASE WHEN sm.n_total_activity > 0 THEN sm.subreddit ELSE NULL END) AS BIGINT)
            AS n_active_subreddits,
          mts.mean_comments_per_submission,
          mts.median_comments_per_submission
        FROM month_dim md
        LEFT JOIN selected_subreddit_monthly_activity sm
          ON md.month = sm.month
        LEFT JOIN subset_month_users mu
          ON md.month = mu.month
        LEFT JOIN subset_month_thread_summary mts
          ON md.month = mts.month
        GROUP BY 1, mu.n_active_users, mts.mean_comments_per_submission, mts.median_comments_per_submission
        ORDER BY md.month
        """
    )

    logger.info("Materialising selected_subreddits_overview")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE selected_subreddits_overview AS
        SELECT
          ss.selection_order,
          ss.subreddit,
          CAST(COALESCE(SUM(sm.n_submissions), 0) AS BIGINT) AS n_submissions,
          CAST(COALESCE(SUM(sm.n_comments), 0) AS BIGINT) AS n_comments,
          CAST(COALESCE(SUM(sm.n_total_activity), 0) AS BIGINT) AS n_total_activity,
          CAST(COALESCE(MAX(fpu.n_active_users), 0) AS BIGINT) AS n_active_users,
          CAST(SUM(CASE WHEN sm.n_total_activity > 0 THEN 1 ELSE 0 END) AS BIGINT) AS n_active_months,
          MIN(CASE WHEN sm.n_total_activity > 0 THEN sm.month ELSE NULL END) AS first_month,
          MAX(CASE WHEN sm.n_total_activity > 0 THEN sm.month ELSE NULL END) AS last_month,
          AVG(CAST(sm.n_active_users AS DOUBLE)) AS mean_monthly_active_users,
          MIN(CAST(sm.n_active_users AS BIGINT)) AS min_monthly_active_users,
          MEDIAN(CAST(sm.n_active_users AS DOUBLE)) AS median_monthly_active_users,
          MAX(sts.mean_comments_per_submission) AS mean_comments_per_submission,
          MAX(sts.median_comments_per_submission) AS median_comments_per_submission,
          MAX(sts.p75_comments_per_submission) AS p75_comments_per_submission,
          MAX(sts.p90_comments_per_submission) AS p90_comments_per_submission,
          AVG(sm.repeat_user_share_2plus) AS mean_repeat_user_share_2plus,
          AVG(sm.repeat_user_share_5plus) AS mean_repeat_user_share_5plus
        FROM selected_subreddits ss
        LEFT JOIN selected_subreddit_monthly_activity sm
          ON ss.subreddit = sm.subreddit
        LEFT JOIN subreddit_full_period_users fpu
          ON ss.subreddit = fpu.subreddit
        LEFT JOIN subreddit_thread_summary sts
          ON ss.subreddit = sts.subreddit
        GROUP BY ss.selection_order, ss.subreddit
        ORDER BY ss.selection_order
        """
    )

    logger.info("Materialising survival distribution tables")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE selected_subset_thread_size_distribution_plot AS
        WITH counts AS (
          SELECT
            comments_per_submission,
            COUNT(*) AS n_submissions
          FROM thread_sizes_all
          GROUP BY 1
        ),
        totals AS (
          SELECT
            COALESCE(SUM(n_submissions), 0) AS total_n_submissions,
            COALESCE(SUM(CASE WHEN comments_per_submission = 0 THEN n_submissions ELSE 0 END), 0)
              AS zero_comment_submissions
          FROM counts
        ),
        survival AS (
          SELECT
            c.comments_per_submission,
            c.n_submissions,
            SUM(c.n_submissions) OVER (
              ORDER BY c.comments_per_submission DESC
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS survival_n_submissions,
            t.total_n_submissions,
            t.zero_comment_submissions
          FROM counts c
          CROSS JOIN totals t
        )
        SELECT
          comments_per_submission,
          n_submissions,
          survival_n_submissions,
          CASE
            WHEN total_n_submissions > 0
            THEN CAST(survival_n_submissions AS DOUBLE) / CAST(total_n_submissions AS DOUBLE)
            ELSE NULL
          END AS survival_share,
          total_n_submissions,
          zero_comment_submissions,
          CASE
            WHEN total_n_submissions > 0
            THEN CAST(zero_comment_submissions AS DOUBLE) / CAST(total_n_submissions AS DOUBLE)
            ELSE NULL
          END AS zero_comment_submission_share
        FROM survival
        ORDER BY comments_per_submission
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE selected_subset_user_activity_distribution_plot AS
        WITH counts AS (
          SELECT
            n_total_activity,
            COUNT(*) AS n_users
          FROM per_user_activity
          GROUP BY 1
        ),
        totals AS (
          SELECT COALESCE(SUM(n_users), 0) AS total_n_users
          FROM counts
        ),
        survival AS (
          SELECT
            c.n_total_activity,
            c.n_users,
            SUM(c.n_users) OVER (
              ORDER BY c.n_total_activity DESC
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS survival_n_users,
            t.total_n_users
          FROM counts c
          CROSS JOIN totals t
        )
        SELECT
          n_total_activity,
          n_users,
          survival_n_users,
          CASE
            WHEN total_n_users > 0
            THEN CAST(survival_n_users AS DOUBLE) / CAST(total_n_users AS DOUBLE)
            ELSE NULL
          END AS survival_share,
          total_n_users
        FROM survival
        ORDER BY n_total_activity
        """
    )


def fetch_output_tables(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    """Fetch compact output tables into pandas."""
    return {
        "selected_subreddits_overview": con.execute("SELECT * FROM selected_subreddits_overview").df(),
        "selected_subset_monthly_activity": con.execute("SELECT * FROM selected_subset_monthly_activity").df(),
        "selected_subreddit_monthly_activity": con.execute("SELECT * FROM selected_subreddit_monthly_activity").df(),
        "selected_subset_thread_size_distribution_plot": con.execute(
            "SELECT * FROM selected_subset_thread_size_distribution_plot"
        ).df(),
        "selected_subset_user_activity_distribution_plot": con.execute(
            "SELECT * FROM selected_subset_user_activity_distribution_plot"
        ).df(),
    }


def build_summary(
    con: duckdb.DuckDBPyConnection,
    cfg: SelectedDescriptivesConfig,
    author_col: str,
    selected_source: str,
    selected_names: Sequence[str],
) -> dict[str, Any]:
    """Build global summary JSON payload."""
    row = con.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM selected_subreddits) AS selected_subreddit_count,
          (SELECT COALESCE(SUM(n_submissions), 0) FROM selected_subset_monthly_activity) AS total_submissions,
          (SELECT COALESCE(SUM(n_comments), 0) FROM selected_subset_monthly_activity) AS total_comments,
          (SELECT COALESCE(SUM(n_total_activity), 0) FROM selected_subset_monthly_activity) AS total_activity,
          (SELECT COUNT(*) FROM per_user_activity) AS total_active_users,
          (SELECT COUNT(*) FROM month_dim) AS selected_month_count,
          (SELECT MIN(month) FROM month_dim) AS first_month,
          (SELECT MAX(month) FROM month_dim) AS last_month
        """
    ).fetchone()
    months = [str(r[0]) for r in con.execute("SELECT month FROM month_dim ORDER BY month").fetchall()]
    return {
        "selected_subreddit_count": int(row[0]),
        "total_submissions": int(row[1]),
        "total_comments": int(row[2]),
        "total_activity": int(row[3]),
        "total_active_users": int(row[4]),
        "selected_month_count": int(row[5]),
        "selected_months": months,
        "first_month": row[6],
        "last_month": row[7],
        "timestamp_utc": utc_now_iso(),
        "input_root": cfg.in_root,
        "selected_subreddits_csv": cfg.selected_subreddits_csv,
        "selected_subreddit_source": selected_source,
        "selected_subreddit_names": list(selected_names),
        "author_column_used": author_col,
        "smoke": cfg.smoke,
        "approximate": bool(cfg.smoke),
        "denominator_policy": DENOMINATOR_POLICY,
        "received_engagement": RECEIVED_ENGAGEMENT_OMISSION,
    }


def write_tables(
    out_root: Path,
    tables: dict[str, pd.DataFrame],
    validation_rows: Sequence[dict[str, str]],
    summary: dict[str, Any],
    overwrite: bool,
) -> dict[str, str]:
    """Write compact tables and return output paths."""
    tables_dir = out_root / "tables"
    ensure_dir(str(tables_dir))
    output_paths: dict[str, str] = {}
    for name, df in tables.items():
        out_path = tables_dir / f"{name}.csv"
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing table: {out_path}")
        df.to_csv(out_path, index=False)
        output_paths[name] = str(out_path)

    validation_df = pd.DataFrame(
        validation_rows,
        columns=["check_name", "status", "severity", "observed_value", "expected_value", "message"],
    )
    validation_path = tables_dir / "selected_subset_validation_summary.csv"
    if validation_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing table: {validation_path}")
    validation_df.to_csv(validation_path, index=False)
    output_paths["selected_subset_validation_summary"] = str(validation_path)

    summary_path = tables_dir / "selected_subset_summary.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing table: {summary_path}")
    safe_json_dump(summary, str(summary_path))
    output_paths["selected_subset_summary"] = str(summary_path)
    return output_paths


def add_late_validation_checks(
    tables: dict[str, pd.DataFrame],
    validation_rows: list[dict[str, str]],
    expected_months: Sequence[str],
    author_col: str,
    smoke: bool,
) -> None:
    """Add validation checks that depend on materialised output tables."""
    monthly = tables["selected_subset_monthly_activity"]
    observed_months = set(str(month) for month in monthly["month"].tolist())
    expected_month_set = set(expected_months)
    add_validation(
        validation_rows,
        "monthly_rows_for_expected_months",
        "pass" if observed_months == expected_month_set else "fail",
        "error",
        ",".join(sorted(observed_months)),
        ",".join(sorted(expected_month_set)),
        "Monthly subset activity has one row for each discovered input month.",
    )

    total_activity = int(pd.to_numeric(monthly["n_total_activity"], errors="coerce").fillna(0).sum())
    add_validation(
        validation_rows,
        "selected_subset_total_activity_positive",
        "pass" if total_activity > 0 else "fail",
        "error",
        total_activity,
        "> 0",
        "Selected subset contains at least one content row.",
    )

    add_validation(
        validation_rows,
        "user_level_resolved_author_column_used",
        "pass",
        "info",
        author_col,
        author_col,
        "User-level summaries are built from deleted/null-author filtered views using the resolved author column.",
    )

    for name, df in tables.items():
        is_nonempty = len(df) > 0
        empty_allowed_in_smoke = smoke and name == "selected_subset_thread_size_distribution_plot"
        status = "pass" if is_nonempty or empty_allowed_in_smoke else "fail"
        severity = "info" if status == "pass" else "error"
        if is_nonempty:
            message = f"{name}.csv is non-empty."
        elif empty_allowed_in_smoke:
            message = f"{name}.csv has no rows; this is allowed only in smoke mode."
        else:
            message = f"{name}.csv has no rows."
        add_validation(
            validation_rows,
            f"table_nonempty_{name}",
            status,
            severity,
            len(df),
            "> 0",
            message,
        )

    add_validation(
        validation_rows,
        "smoke_outputs_marked_approximate",
        "pass",
        "info",
        str(bool(smoke)).lower(),
        str(bool(smoke)).lower(),
        "Smoke mode is recorded in summary and metadata; smoke outputs are approximate.",
    )

    add_validation(
        validation_rows,
        "received_engagement_omitted",
        "pass",
        "info",
        "omitted",
        "omitted",
        RECEIVED_ENGAGEMENT_OMISSION["reason"],
    )


def _format_readable_number(value: float, _pos: int) -> str:
    """Format large counts without scientific notation."""
    if pd.isna(value):
        return ""
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:g}"


def _prepare_months(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a datetime month column for plotting."""
    out = df.copy()
    out["month_dt"] = pd.to_datetime(out["month"], format="%Y-%m", errors="raise")
    return out.sort_values("month_dt").reset_index(drop=True)


def _style_axes(ax: plt.Axes) -> None:
    """Apply the same quiet axis style used by descriptive_plots."""
    ax.grid(False)
    ax.grid(False, which="minor")
    ax.minorticks_off()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", labelsize=PLOT_SIZES["tick"])


def _format_month_axis(ax: plt.Axes) -> None:
    """Keep month labels readable."""
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")


def _barh_comparison_plot(
    df: pd.DataFrame,
    value_col: str,
    xlabel: str,
    publication_style: bool,
    percent: bool = False,
) -> plt.Figure:
    """Build a horizontal selected-subreddit comparison plot."""
    plot_df = df[["subreddit", value_col]].copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce").fillna(0)
    plot_df = plot_df.sort_values(value_col, ascending=True)
    height = max(4.8, min(16.0, 1.2 + 0.30 * len(plot_df)))
    fig, ax = plt.subplots(figsize=(8.2, height))
    ax.barh(plot_df["subreddit"], plot_df[value_col], color=COLORBLIND_SAFE_PALETTE[0])
    ax.set_xlabel(xlabel)
    if percent:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:.0%}"))
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(_format_readable_number))
    _style_axes(ax)
    return fig


def _repeat_user_share_plot(df: pd.DataFrame, publication_style: bool) -> plt.Figure:
    """Build grouped horizontal bars for repeat-user shares."""
    cols = ["subreddit", "mean_repeat_user_share_2plus", "mean_repeat_user_share_5plus"]
    plot_df = df[cols].copy()
    for col in cols[1:]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce").fillna(0)
    plot_df = plot_df.sort_values("mean_repeat_user_share_2plus", ascending=True)
    height = max(4.8, min(16.0, 1.2 + 0.34 * len(plot_df)))
    y = list(range(len(plot_df)))
    fig, ax = plt.subplots(figsize=(8.2, height))
    ax.barh(
        [idx - 0.18 for idx in y],
        plot_df["mean_repeat_user_share_2plus"],
        height=0.34,
        label="2+ activities",
        color=COLORBLIND_SAFE_PALETTE[0],
    )
    ax.barh(
        [idx + 0.18 for idx in y],
        plot_df["mean_repeat_user_share_5plus"],
        height=0.34,
        label="5+ activities",
        color=COLORBLIND_SAFE_PALETTE[1],
    )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["subreddit"])
    ax.set_xlabel("Mean monthly repeat-user share")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:.0%}"))
    if not publication_style:
        ax.legend(frameon=False)
    _style_axes(ax)
    return fig


def _monthly_activity_plot(df: pd.DataFrame, publication_style: bool) -> plt.Figure:
    """Build selected subset monthly activity plot."""
    monthly = _prepare_months(df)
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 7.8), sharex=True, gridspec_kw={"hspace": 0.18})
    series = [
        ("n_comments", "Comments", COLORBLIND_SAFE_PALETTE[0]),
        ("n_submissions", "Submissions", COLORBLIND_SAFE_PALETTE[1]),
        ("n_active_users", "Active users", COLORBLIND_SAFE_PALETTE[2]),
    ]
    for ax, (col, label, color) in zip(axes, series):
        ax.plot(monthly["month_dt"], monthly[col], marker="o", linewidth=1.8, color=color)
        ax.set_ylabel(label)
        ax.yaxis.set_major_formatter(FuncFormatter(_format_readable_number))
        _style_axes(ax)
    axes[-1].set_xlabel("Month")
    _format_month_axis(axes[-1])
    return fig


def _survival_plot(
    df: pd.DataFrame,
    value_col: str,
    survival_col: str,
    xlabel: str,
    ylabel: str,
    publication_style: bool,
    annotation: Optional[str] = None,
) -> plt.Figure:
    """Build a positive-support log-log survival plot."""
    plot_df = df.copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df[survival_col] = pd.to_numeric(plot_df[survival_col], errors="coerce")
    plot_df = plot_df.loc[
        plot_df[value_col].notna()
        & plot_df[survival_col].notna()
        & (plot_df[value_col] > 0)
        & (plot_df[survival_col] > 0)
    ].sort_values(value_col)

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    if len(plot_df):
        ax.step(plot_df[value_col], plot_df[survival_col], where="post", linewidth=1.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if annotation and not publication_style:
        ax.text(
            0.98,
            0.98,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=PLOT_SIZES["annotation"],
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
    _style_axes(ax)
    return fig


def generate_figures(out_root: Path, tables: dict[str, pd.DataFrame], publication_style: bool) -> None:
    """Generate and save all selected-subset figures."""
    overview = tables["selected_subreddits_overview"].copy()
    monthly = tables["selected_subset_monthly_activity"].copy()
    thread = tables["selected_subset_thread_size_distribution_plot"].copy()
    user_activity = tables["selected_subset_user_activity_distribution_plot"].copy()

    save_figure(
        _monthly_activity_plot(monthly, publication_style),
        str(out_root),
        "selected_subset_monthly_activity",
    )
    save_figure(
        _barh_comparison_plot(
            overview,
            "n_total_activity",
            "Total activity",
            publication_style,
        ),
        str(out_root),
        "selected_subreddit_activity_comparison",
    )
    save_figure(
        _barh_comparison_plot(
            overview,
            "n_active_users",
            "Active users",
            publication_style,
        ),
        str(out_root),
        "selected_subreddit_active_users_comparison",
    )
    save_figure(
        _barh_comparison_plot(
            overview,
            "mean_comments_per_submission",
            "Mean comments per submission",
            publication_style,
        ),
        str(out_root),
        "selected_subreddit_mean_thread_size_comparison",
    )
    save_figure(
        _repeat_user_share_plot(overview, publication_style),
        str(out_root),
        "selected_subreddit_repeat_user_share_comparison",
    )

    zero_annotation = None
    if len(thread) and "zero_comment_submission_share" in thread.columns:
        zero_share = pd.to_numeric(thread["zero_comment_submission_share"], errors="coerce").dropna()
        if len(zero_share):
            zero_annotation = f"Zero-comment submissions: {zero_share.iloc[0]:.1%}"
    save_figure(
        _survival_plot(
            thread,
            "comments_per_submission",
            "survival_n_submissions",
            "Comments per submission",
            "Submissions with at least x comments",
            publication_style,
            annotation=zero_annotation,
        ),
        str(out_root),
        "selected_subset_thread_size_distribution",
    )
    save_figure(
        _survival_plot(
            user_activity,
            "n_total_activity",
            "survival_n_users",
            "Total selected-subset activity",
            "Users with at least x activities",
            publication_style,
        ),
        str(out_root),
        "selected_subset_user_activity_distribution",
    )


def build_metadata(
    cfg: SelectedDescriptivesConfig,
    duck: DuckDBConfig,
    author_col: str,
    selected_source: str,
    selected_names: Sequence[str],
    selected_validation: dict[str, Any],
    validation_rows: Sequence[dict[str, str]],
    output_paths: dict[str, str],
    status: str,
) -> dict[str, Any]:
    """Build run metadata payload."""
    return {
        "run_id": cfg.run_id,
        "timestamp_utc": utc_now_iso(),
        "status": status,
        "script": Path(__file__).name,
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "duckdb": {"version": duckdb.__version__, **asdict(duck)},
        "config": asdict(cfg),
        "resolved_author_col": author_col,
        "selected_subreddit_source": selected_source,
        "selected_subreddit_names": list(selected_names),
        "selected_validation": selected_validation,
        "denominator_policy": DENOMINATOR_POLICY,
        "received_engagement": RECEIVED_ENGAGEMENT_OMISSION,
        "smoke": cfg.smoke,
        "approximate": bool(cfg.smoke),
        "validation_results": list(validation_rows),
        "outputs": output_paths,
    }


def write_metadata(out_root: Path, cfg: SelectedDescriptivesConfig, metadata: dict[str, Any]) -> dict[str, str]:
    """Write deterministic and timestamped metadata JSON files."""
    audit_dir = out_root / "audit"
    ensure_dir(str(audit_dir))
    latest = audit_dir / "selected_subset_descriptives_metadata.json"
    timestamped = audit_dir / f"selected_subset_descriptives_run_{cfg.run_id}.json"
    safe_json_dump(metadata, str(latest))
    safe_json_dump(metadata, str(timestamped))
    return {
        "metadata": str(latest),
        "timestamped_metadata": str(timestamped),
    }


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    p = argparse.ArgumentParser(
        description="Generate selected-subreddit modelling-subset descriptive tables and figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in-root", required=True, help="Input root containing selected-subset monthly Parquet outputs.")
    p.add_argument("--out-root", required=True, help="Output root for selected-subset descriptives.")
    p.add_argument(
        "--selected-subreddits-csv",
        default=None,
        help="Optional CSV with subreddit column, ideally <selected-subset-root>/audit/selected_subreddits.csv.",
    )
    p.add_argument("--year", default="2021", help="Year to process (YYYY).")
    p.add_argument(
        "--author-col",
        default="auto",
        choices=["auto", "author_pid", "author"],
        help="Author identifier column to use for user-level summaries.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs if present.")
    p.add_argument(
        "--publication-style",
        action="store_true",
        help="Use thesis/manuscript-oriented styling for annotations and legends; plot titles are omitted by default.",
    )
    p.add_argument("--smoke", action="store_true", help="Enable smoke mode for quick plumbing checks.")
    p.add_argument(
        "--smoke-max-rows",
        type=int,
        default=10000,
        help="Maximum rows read per input file in smoke mode.",
    )
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
        help="DuckDB temp_directory. Prefer node-local scratch on HPC.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""
    parser = build_argparser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    if args.smoke and int(args.smoke_max_rows) <= 0:
        parser.error("--smoke-max-rows must be positive when --smoke is set.")
    if not args.smoke and int(args.smoke_max_rows) < 0:
        parser.error("--smoke-max-rows must be non-negative.")
    if not math.isfinite(float(args.duckdb_threads)) or int(args.duckdb_threads) <= 0:
        parser.error("--duckdb-threads must be positive.")

    cfg = SelectedDescriptivesConfig(
        in_root=str(args.in_root),
        out_root=str(args.out_root),
        selected_subreddits_csv=str(args.selected_subreddits_csv) if args.selected_subreddits_csv else None,
        year=str(args.year),
        author_col=str(args.author_col),
        overwrite=bool(args.overwrite),
        publication_style=bool(args.publication_style),
        smoke=bool(args.smoke),
        smoke_max_rows=int(args.smoke_max_rows) if args.smoke else 0,
        run_id=utc_run_id("selected_subset_descriptives", str(args.year)),
    )
    duck = DuckDBConfig(
        memory_limit=str(args.duckdb_memory_limit),
        threads=int(args.duckdb_threads),
        temp_dir=str(args.duckdb_temp_dir),
    )

    out_root = Path(cfg.out_root)
    ensure_dir(str(out_root / "tables"))
    ensure_dir(str(out_root / "figures"))
    ensure_dir(str(out_root / "audit"))
    check_output_collisions(out_root, cfg.overwrite)

    logger.info("Run ID: %s", cfg.run_id)
    logger.info("Input root: %s", cfg.in_root)
    logger.info("Output root: %s", cfg.out_root)
    logger.info("Selected subreddits CSV: %s", cfg.selected_subreddits_csv or "(infer from data)")
    logger.info("Smoke mode: %s | smoke_max_rows=%s", cfg.smoke, cfg.smoke_max_rows)
    logger.info("Denominator policy: %s", DENOMINATOR_POLICY)
    logger.info("Received engagement: omitted (%s)", RECEIVED_ENGAGEMENT_OMISSION["reason"])

    validation_rows: list[dict[str, str]] = []
    selected_validation: dict[str, Any] = {}
    selected_names: list[str] = []
    selected_source = "unresolved"
    author_col = "unresolved"
    output_paths: dict[str, str] = {}

    con = duckdb.connect()
    configure_duckdb(con, duck)
    try:
        inv = discover_core_files(cfg.in_root, cfg.year)
        if not inv:
            raise RuntimeError("No monthly core files found under --in-root.")
        require_complete_inventory(inv)
        months = sorted(inv.keys())
        make_month_dim(con, months)
        logger.info("Discovered months (%d): %s", len(months), ", ".join(months))

        build_base_views(con, inv, cfg.smoke_max_rows)
        comments_cols = get_table_columns(con, "comments_core_all")
        submissions_cols = get_table_columns(con, "submissions_core_all")
        author_col = resolve_author_col(comments_cols, submissions_cols, cfg.author_col)
        logger.info("Resolved author column: %s", author_col)
        build_user_level_views(con, author_col)

        if cfg.selected_subreddits_csv:
            con.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE selected_subreddits_input AS
                SELECT *
                FROM {read_csv_expr(cfg.selected_subreddits_csv)}
                """
            )
        selected_source, selected_names = create_selected_subreddits_table(con, cfg.selected_subreddits_csv)
        audit_selected_path = out_root / "audit" / "selected_subreddits.csv"
        write_selected_subreddits(con, audit_selected_path)
        output_paths["selected_subreddits"] = str(audit_selected_path)
        logger.info("Selected subreddit count: %s", f"{len(selected_names):,}")
        logger.info("Wrote run selected subreddit list: %s", audit_selected_path)

        selected_validation = validate_selected_subreddits(
            con,
            validation_rows,
            selected_source=selected_source,
            smoke=cfg.smoke,
        )

        if validation_has_fatal(validation_rows):
            validation_df = pd.DataFrame(
                validation_rows,
                columns=["check_name", "status", "severity", "observed_value", "expected_value", "message"],
            )
            validation_path = out_root / "tables" / "selected_subset_validation_summary.csv"
            validation_df.to_csv(validation_path, index=False)
            output_paths["selected_subset_validation_summary"] = str(validation_path)
            metadata = build_metadata(
                cfg,
                duck,
                author_col,
                selected_source,
                selected_names,
                selected_validation,
                validation_rows,
                output_paths,
                status="failed_validation",
            )
            output_paths.update(write_metadata(out_root, cfg, metadata))
            raise RuntimeError("Selected subreddit validation failed; see selected_subset_validation_summary.csv.")

        materialise_selected_subreddit_summaries(con, author_col)
        tables = fetch_output_tables(con)
        add_late_validation_checks(tables, validation_rows, months, author_col, cfg.smoke)
        summary = build_summary(con, cfg, author_col, selected_source, selected_names)
        output_paths.update(write_tables(out_root, tables, validation_rows, summary, cfg.overwrite))
        generate_figures(out_root, tables, cfg.publication_style)
        for figure_name in FIGURE_NAMES:
            for ext in ("png", "svg", "pdf"):
                output_paths[f"{figure_name}.{ext}"] = str(out_root / "figures" / f"{figure_name}.{ext}")

        final_status = "finished" if not validation_has_fatal(validation_rows) else "failed_validation"
        metadata = build_metadata(
            cfg,
            duck,
            author_col,
            selected_source,
            selected_names,
            selected_validation,
            validation_rows,
            output_paths,
            status=final_status,
        )
        output_paths.update(write_metadata(out_root, cfg, metadata))

        if validation_has_fatal(validation_rows):
            raise RuntimeError("Validation failed; see selected_subset_validation_summary.csv.")

        logger.info("Wrote %d tables and %d figures.", len(TABLE_NAMES) + 1, len(FIGURE_NAMES))
        logger.info("Done.")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
