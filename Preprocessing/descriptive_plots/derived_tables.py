from __future__ import annotations

import glob
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import duckdb

from .checkpoints import (
    build_monthly_received_engagement_metadata,
    checkpoint_file_exists,
    load_checkpoint_table,
    mark_monthly_received_engagement_month_complete,
    monthly_received_engagement_checkpoint_status,
    monthly_received_engagement_part_path,
    validate_monthly_received_engagement_checkpoint_parts,
    write_checkpoint_table,
    write_monthly_received_engagement_manifest,
)
from .config import COMMENTS_KIND, Inventory, MonthFiles, ORDINARY_SUBREDDIT_SQL, SUBMISSIONS_KIND
from .output_io import (
    ensure_dir,
    log_table_count,
    month_of,
    quote_sql_string,
    temporary_duckdb_threads,
    timed_stage,
)


logger = logging.getLogger(__name__)


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


def discover_core_files(in_root: str, year: str) -> Inventory:
    """Discover monthly comments/submissions core Parquet files under an input root."""
    inv: Dict[str, Dict[str, Optional[str]]] = {}
    patterns = [
        (COMMENTS_KIND, os.path.join(in_root, "comments", f"comments_core_{year}-*.parquet")),
        (SUBMISSIONS_KIND, os.path.join(in_root, "submissions", f"submissions_core_{year}-*.parquet")),
    ]

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

def filter_month_inventory(inv: Inventory, only_months: Tuple[str, ...], limit_months: int) -> Inventory:
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

def build_single_file_relation(path: str, smoke_max_rows: int = 0) -> str:
    """Build a SQL relation fragment for one Parquet file with smoke limiting."""
    lit = quote_sql_string(path)
    if smoke_max_rows > 0:
        return f"(SELECT * FROM read_parquet({lit}) LIMIT {int(smoke_max_rows)})"
    return f"read_parquet({lit})"

def build_subreddit_month_summary(
    con: duckdb.DuckDBPyConnection,
    inv: Inventory,
    author_col: str,
    default_threads: int,
    checkpoint_dir: str,
    reuse_checkpoints: bool,
    smoke_max_rows: int = 0,
) -> None:
    """Build compact subreddit-month active-user summaries month by month."""
    if reuse_checkpoints and checkpoint_file_exists(checkpoint_dir, "subreddit_month_summary"):
        logger.info("Loading subreddit_month_summary from checkpoint: %s", os.path.join(checkpoint_dir, "subreddit_month_summary.parquet"))
        with timed_stage("loading subreddit_month_summary checkpoint"):
            load_checkpoint_table(con, checkpoint_dir, "subreddit_month_summary")
        return
    if reuse_checkpoints:
        logger.info("subreddit_month_summary checkpoint not available for compatible metadata; rebuilding compact summary.")
    else:
        logger.info("Rebuilding subreddit_month_summary because checkpoint reuse is disabled or metadata-incompatible.")

    ordinary_src_subreddit_sql = ORDINARY_SUBREDDIT_SQL.replace("subreddit", "src.subreddit")

    with timed_stage("building subreddit_month_summary"):
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE subreddit_month_summary (
              month VARCHAR,
              subreddit VARCHAR,
              n_active_users BIGINT,
              n_total_activity BIGINT,
              pct_users_exactly_1_activity DOUBLE,
              pct_users_2plus_activity DOUBLE,
              pct_users_5plus_activity DOUBLE,
              mean_activity_per_active_user DOUBLE
            )
            """
        )
        for month in sorted(inv.keys()):
            mf = inv[month]
            if not mf.submissions_core or not mf.comments_core:
                raise RuntimeError(f"Missing monthly core files for subreddit_month_summary: {month}")

            month_sql = quote_sql_string(month)
            submissions_relation = build_single_file_relation(mf.submissions_core, smoke_max_rows)
            comments_relation = build_single_file_relation(mf.comments_core, smoke_max_rows)
            with timed_stage(f"subreddit_month_summary month {month}"):
                con.execute(
                    f"""
                    CREATE OR REPLACE TEMP TABLE subreddit_month_user_activity_current AS
                    WITH source_activity AS (
                      SELECT
                        strftime(src.created_utc, '%Y-%m') AS month,
                        src.subreddit,
                        src.{author_col} AS user_id,
                        COUNT(*) AS n_activity
                      FROM {submissions_relation} AS src
                      WHERE strftime(src.created_utc, '%Y-%m') = {month_sql}
                        AND COALESCE(src.is_deleted_author, FALSE) = FALSE
                        AND src.{author_col} IS NOT NULL
                        AND {ordinary_src_subreddit_sql}
                      GROUP BY 1, 2, 3
                      UNION ALL
                      SELECT
                        strftime(src.created_utc, '%Y-%m') AS month,
                        src.subreddit,
                        src.{author_col} AS user_id,
                        COUNT(*) AS n_activity
                      FROM {comments_relation} AS src
                      WHERE strftime(src.created_utc, '%Y-%m') = {month_sql}
                        AND COALESCE(src.is_deleted_author, FALSE) = FALSE
                        AND src.{author_col} IS NOT NULL
                        AND {ordinary_src_subreddit_sql}
                      GROUP BY 1, 2, 3
                    )
                    SELECT
                      month,
                      subreddit,
                      user_id,
                      SUM(n_activity) AS n_activity
                    FROM source_activity
                    GROUP BY 1, 2, 3
                    """
                )
                n_user_subreddit_month_rows = log_table_count(con, "subreddit_month_user_activity_current")
                con.execute(
                    """
                    INSERT INTO subreddit_month_summary
                    SELECT
                      month,
                      subreddit,
                      COUNT(*) AS n_active_users,
                      CAST(SUM(n_activity) AS BIGINT) AS n_total_activity,
                      AVG(CASE WHEN n_activity = 1 THEN 1.0 ELSE 0.0 END) AS pct_users_exactly_1_activity,
                      AVG(CASE WHEN n_activity >= 2 THEN 1.0 ELSE 0.0 END) AS pct_users_2plus_activity,
                      AVG(CASE WHEN n_activity >= 5 THEN 1.0 ELSE 0.0 END) AS pct_users_5plus_activity,
                      AVG(n_activity) AS mean_activity_per_active_user
                    FROM subreddit_month_user_activity_current
                    GROUP BY 1, 2
                    """
                )
                n_summary_rows = int(
                    con.execute(
                        "SELECT COUNT(*) FROM subreddit_month_summary WHERE month = ?",
                        [month],
                    ).fetchone()[0]
                )
                logger.info(
                    "Inserted subreddit_month_summary rows for %s: %s summary rows from %s user-subreddit rows",
                    month,
                    f"{n_summary_rows:,}",
                    f"{n_user_subreddit_month_rows:,}",
                )
                con.execute("DROP TABLE IF EXISTS subreddit_month_user_activity_current")

        log_table_count(con, "subreddit_month_summary")
        write_checkpoint_table(con, checkpoint_dir, "subreddit_month_summary")


def _parquet_path_list_sql(paths: Sequence[str]) -> str:
    """Return a DuckDB SQL list literal for Parquet paths."""
    return "[" + ", ".join(quote_sql_string(path) for path in paths) + "]"


def _load_monthwise_user_month_engagement_tables(
    con: duckdb.DuckDBPyConnection,
    checkpoint_dir: str,
    months: Sequence[str],
) -> None:
    """Expose month-wise checkpoint parts as the public temp tables used downstream."""
    user_month_paths = [
        monthly_received_engagement_part_path(checkpoint_dir, "user_month_panel", month)
        for month in months
    ]
    audit_paths = [
        monthly_received_engagement_part_path(checkpoint_dir, "received_engagement_definition_audit", month)
        for month in months
    ]
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE user_month_panel AS
        SELECT *
        FROM read_parquet({_parquet_path_list_sql(user_month_paths)})
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE received_engagement_definition_audit AS
        SELECT *
        FROM read_parquet({_parquet_path_list_sql(audit_paths)})
        """
    )
    log_table_count(con, "user_month_panel")
    log_table_count(con, "received_engagement_definition_audit")


def _copy_monthly_checkpoint_part(
    con: duckdb.DuckDBPyConnection,
    checkpoint_dir: str,
    table_name: str,
    source_table: str,
    month: str,
) -> None:
    """Write one month-wise received-engagement checkpoint part."""
    out_path = monthly_received_engagement_part_path(checkpoint_dir, table_name, month)
    ensure_dir(os.path.dirname(out_path))
    if os.path.exists(out_path):
        os.remove(out_path)
    con.execute(
        f"""
        COPY (SELECT * FROM {source_table})
        TO {quote_sql_string(out_path)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    logger.info("Wrote month-wise checkpoint part %s %s -> %s", table_name, month, out_path)


def _clear_monthly_received_engagement_parts(checkpoint_dir: str) -> None:
    """Remove stale internal monthly received-engagement parts after metadata invalidation."""
    for table_name in ("user_month_panel", "received_engagement_definition_audit"):
        table_dir = os.path.dirname(monthly_received_engagement_part_path(checkpoint_dir, table_name, "2000-01"))
        if not os.path.isdir(table_dir):
            continue
        for path in glob.glob(os.path.join(table_dir, f"{table_name}_*.parquet")):
            os.remove(path)


def _materialise_received_engagement_owner_maps(
    con: duckdb.DuckDBPyConnection,
    author_col: str,
) -> None:
    """Materialise full-period owner maps used by month-wise child-comment attribution."""
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


def _build_monthwise_user_month_engagement(
    con: duckdb.DuckDBPyConnection,
    inv: Inventory,
    author_col: str,
    default_threads: int,
    smoke_max_rows: int,
    checkpoint_dir: str,
    reuse_checkpoints: bool,
    checkpoint_metadata: Optional[Dict[str, object]],
) -> None:
    """Build or load user_month_panel and engagement audit via month-wise checkpoint parts."""
    months = list(sorted(inv))
    metadata_source = checkpoint_metadata or {
        "schema_version": "ad_hoc",
        "selected_months": months,
        "year": months[0][:4] if months else "",
        "author_col": author_col,
        "smoke": bool(smoke_max_rows),
        "smoke_max_rows": int(smoke_max_rows),
        "in_root": "unknown",
    }
    monthly_metadata = build_monthly_received_engagement_metadata(metadata_source, months, author_col)
    allow_empty_parts = int(smoke_max_rows) > 0

    status = monthly_received_engagement_checkpoint_status(checkpoint_dir, monthly_metadata) if reuse_checkpoints else {
        "allowed": False,
        "completed_months": [],
        "reason": "--reuse-checkpoints was not supplied or metadata was incompatible",
    }
    completed_months = set(status["completed_months"]) if status["allowed"] else set()
    if status["allowed"]:
        logger.info(
            "Month-wise received-engagement checkpoint metadata matched; completed months: %s",
            ", ".join(sorted(completed_months)) if completed_months else "none",
        )
    else:
        logger.info("Month-wise received-engagement checkpoint reuse not used: %s", status["reason"])
        completed_months = set()
        _clear_monthly_received_engagement_parts(checkpoint_dir)
        write_monthly_received_engagement_manifest(
            checkpoint_dir,
            monthly_metadata,
            completed_months=[],
            run_status="partial",
        )

    months_to_build = [month for month in months if month not in completed_months]

    if months_to_build:
        logger.info(
            "Materialising received-engagement attribution maps once for month-wise checkpointing"
        )
        _materialise_received_engagement_owner_maps(con, author_col)

    with timed_stage("building month-wise user_month_panel checkpoints"):
        for month in months_to_build:
            mf = inv[month]
            if not mf.submissions_core or not mf.comments_core:
                raise RuntimeError(f"Cannot build month-wise user_month_panel for incomplete month: {month}")
            logger.info("Materialising month-wise user_month_panel checkpoint for %s", month)
            submissions_relation = build_single_file_relation(mf.submissions_core, smoke_max_rows)
            comments_relation = build_single_file_relation(mf.comments_core, smoke_max_rows)
            month_sql = quote_sql_string(month)

            con.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE user_month_activity_current AS
                SELECT
                  month,
                  user_id,
                  SUM(n_submissions) AS n_submissions,
                  SUM(n_comments) AS n_comments,
                  SUM(n_total_activity) AS n_total_activity
                FROM (
                  SELECT
                    {month_sql} AS month,
                    CAST({quote_identifier(author_col)} AS VARCHAR) AS user_id,
                    COUNT(*) AS n_submissions,
                    0 AS n_comments,
                    COUNT(*) AS n_total_activity
                  FROM {submissions_relation}
                  WHERE COALESCE(is_deleted_author, FALSE) = FALSE
                    AND {quote_identifier(author_col)} IS NOT NULL
                  GROUP BY 1, 2
                  UNION ALL
                  SELECT
                    {month_sql} AS month,
                    CAST({quote_identifier(author_col)} AS VARCHAR) AS user_id,
                    0 AS n_submissions,
                    COUNT(*) AS n_comments,
                    COUNT(*) AS n_total_activity
                  FROM {comments_relation}
                  WHERE COALESCE(is_deleted_author, FALSE) = FALSE
                    AND {quote_identifier(author_col)} IS NOT NULL
                  GROUP BY 1, 2
                ) z
                GROUP BY 1, 2
                """
            )
            con.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE comments_engagement_events_current AS
                SELECT
                  {month_sql} AS month,
                  {clean_varchar_sql(qcol("src", "id"))} AS comment_id_raw,
                  {strip_reddit_prefix_sql(qcol("src", "id"), "t1_")} AS comment_id_norm,
                  {clean_varchar_sql(qcol("src", "link_id"))} AS link_id_raw,
                  {strip_reddit_prefix_sql(qcol("src", "link_id"), "t3_")} AS link_submission_id_norm,
                  {clean_varchar_sql(qcol("src", "parent_id"))} AS parent_id_raw,
                  {strip_reddit_prefix_sql(qcol("src", "parent_id"), "t1_")} AS parent_comment_id_norm,
                  CASE
                    WHEN COALESCE({qcol("src", "is_deleted_author")}, FALSE) = FALSE
                     AND {qcol("src", author_col)} IS NOT NULL
                    THEN CAST({qcol("src", author_col)} AS VARCHAR)
                    ELSE NULL
                  END AS engager_user_id
                FROM {comments_relation} AS src
                """
            )
            con.execute(
                """
                CREATE OR REPLACE TEMP TABLE submission_engagement_events_normalized_all_current AS
                SELECT DISTINCT
                  child.month,
                  som.user_id,
                  child.comment_id_norm AS child_comment_id,
                  child.engager_user_id,
                  CAST(child.engager_user_id IS NOT NULL AND child.engager_user_id = som.user_id AS BOOLEAN)
                    AS is_self_engagement
                FROM comments_engagement_events_current AS child
                INNER JOIN submission_owner_map som
                  ON child.link_submission_id_norm = som.submission_id_norm
                WHERE child.comment_id_norm IS NOT NULL
                  AND child.link_submission_id_norm IS NOT NULL
                """
            )
            con.execute(
                """
                CREATE OR REPLACE TEMP TABLE submission_engagement_events_current AS
                SELECT DISTINCT
                  month,
                  user_id,
                  child_comment_id,
                  engager_user_id
                FROM submission_engagement_events_normalized_all_current
                WHERE NOT is_self_engagement
                """
            )
            con.execute(
                """
                CREATE OR REPLACE TEMP TABLE comment_reply_engagement_events_normalized_all_current AS
                SELECT DISTINCT
                  child.month,
                  pcom.user_id,
                  child.comment_id_norm AS child_comment_id,
                  child.engager_user_id,
                  CAST(child.engager_user_id IS NOT NULL AND child.engager_user_id = pcom.user_id AS BOOLEAN)
                    AS is_self_engagement
                FROM comments_engagement_events_current AS child
                INNER JOIN parent_comment_owner_map pcom
                  ON child.parent_comment_id_norm = pcom.comment_id_norm
                WHERE child.comment_id_norm IS NOT NULL
                  AND child.parent_comment_id_norm IS NOT NULL
                  AND NOT starts_with(child.parent_id_raw, 't3_')
                """
            )
            con.execute(
                """
                CREATE OR REPLACE TEMP TABLE comment_reply_engagement_events_current AS
                SELECT DISTINCT
                  month,
                  user_id,
                  child_comment_id,
                  engager_user_id
                FROM comment_reply_engagement_events_normalized_all_current
                WHERE NOT is_self_engagement
                """
            )

            with temporary_duckdb_threads(con, 1, default_threads):
                con.execute(
                    """
                    CREATE OR REPLACE TEMP TABLE user_month_engagement_current AS
                    WITH route_events AS (
                      SELECT
                        month,
                        user_id,
                        child_comment_id,
                        engager_user_id,
                        1::BIGINT AS is_submission_thread_route,
                        0::BIGINT AS is_comment_reply_route
                      FROM submission_engagement_events_current
                      UNION ALL
                      SELECT
                        month,
                        user_id,
                        child_comment_id,
                        engager_user_id,
                        0::BIGINT AS is_submission_thread_route,
                        1::BIGINT AS is_comment_reply_route
                      FROM comment_reply_engagement_events_current
                    ),
                    component_counts AS (
                      SELECT
                        month,
                        user_id,
                        CAST(SUM(is_submission_thread_route) AS BIGINT)
                          AS n_submission_thread_comments_received,
                        CAST(SUM(is_comment_reply_route) AS BIGINT)
                          AS n_comment_replies_received,
                        CAST(COUNT(DISTINCT CASE
                          WHEN is_submission_thread_route = 1 THEN engager_user_id
                          ELSE NULL
                        END) AS BIGINT) AS n_distinct_commenters_on_submissions,
                        CAST(COUNT(DISTINCT CASE
                          WHEN is_comment_reply_route = 1 THEN engager_user_id
                          ELSE NULL
                        END) AS BIGINT) AS n_distinct_repliers_to_comments,
                        CAST(COUNT(DISTINCT engager_user_id) AS BIGINT) AS n_distinct_received_engagers
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
                        CAST(COUNT(*) AS BIGINT) AS n_total_engagement_received,
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
                      c.n_submission_thread_comments_received,
                      c.n_comment_replies_received,
                      u.n_total_engagement_received,
                      u.n_overlap_submission_and_comment_reply_events,
                      c.n_distinct_commenters_on_submissions,
                      c.n_distinct_repliers_to_comments,
                      c.n_distinct_received_engagers
                    FROM component_counts c
                    INNER JOIN unique_counts u
                      ON c.month = u.month AND c.user_id = u.user_id
                    """
                )

            con.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE received_engagement_definition_audit_current AS
                WITH months AS (
                  SELECT {month_sql} AS month
                ),
                submission_before AS (
                  SELECT
                    month,
                    COUNT(*) AS all_submission_thread_comments_before_self_exclusion,
                    SUM(CASE WHEN is_self_engagement THEN 1 ELSE 0 END)
                      AS self_comments_on_own_submissions_excluded
                  FROM submission_engagement_events_normalized_all_current
                  GROUP BY 1
                ),
                submission_after AS (
                  SELECT
                    month,
                    COUNT(*) AS other_user_submission_thread_comments_after_self_exclusion,
                    COUNT(DISTINCT engager_user_id) AS distinct_commenters_on_submissions
                  FROM submission_engagement_events_current
                  GROUP BY 1
                ),
                direct_before AS (
                  SELECT
                    month,
                    COUNT(*) AS all_direct_comment_replies_before_self_exclusion,
                    SUM(CASE WHEN is_self_engagement THEN 1 ELSE 0 END)
                      AS self_replies_to_own_comments_excluded
                  FROM comment_reply_engagement_events_normalized_all_current
                  GROUP BY 1
                ),
                direct_after AS (
                  SELECT
                    month,
                    COUNT(*) AS other_user_direct_comment_replies_after_self_exclusion,
                    COUNT(DISTINCT engager_user_id) AS distinct_repliers_to_comments
                  FROM comment_reply_engagement_events_current
                  GROUP BY 1
                ),
                distinct_after AS (
                  SELECT
                    month,
                    COUNT(DISTINCT engager_user_id) AS distinct_received_engagers
                  FROM (
                    SELECT month, engager_user_id FROM submission_engagement_events_current
                    UNION ALL
                    SELECT month, engager_user_id FROM comment_reply_engagement_events_current
                  )
                  GROUP BY 1
                ),
                unique_totals AS (
                  SELECT
                    month,
                    SUM(n_total_engagement_received) AS unique_total_after_deduplication,
                    SUM(n_overlap_submission_and_comment_reply_events)
                      AS overlap_submission_and_comment_reply_events
                  FROM user_month_engagement_current
                  GROUP BY 1
                )
                SELECT
                  m.month,
                  COALESCE(sb.all_submission_thread_comments_before_self_exclusion, 0)
                    AS all_submission_thread_comments_before_self_exclusion,
                  COALESCE(sa.other_user_submission_thread_comments_after_self_exclusion, 0)
                    AS other_user_submission_thread_comments_after_self_exclusion,
                  COALESCE(sb.self_comments_on_own_submissions_excluded, 0)
                    AS self_comments_on_own_submissions_excluded,
                  COALESCE(db.all_direct_comment_replies_before_self_exclusion, 0)
                    AS all_direct_comment_replies_before_self_exclusion,
                  COALESCE(da.other_user_direct_comment_replies_after_self_exclusion, 0)
                    AS other_user_direct_comment_replies_after_self_exclusion,
                  COALESCE(db.self_replies_to_own_comments_excluded, 0)
                    AS self_replies_to_own_comments_excluded,
                  COALESCE(sa.other_user_submission_thread_comments_after_self_exclusion, 0)
                    + COALESCE(da.other_user_direct_comment_replies_after_self_exclusion, 0)
                    AS component_total_after_self_exclusion,
                  COALESCE(ut.unique_total_after_deduplication, 0)
                    AS unique_total_after_deduplication,
                  COALESCE(ut.overlap_submission_and_comment_reply_events, 0)
                    AS overlap_submission_and_comment_reply_events,
                  CASE
                    WHEN COALESCE(sa.other_user_submission_thread_comments_after_self_exclusion, 0)
                         + COALESCE(da.other_user_direct_comment_replies_after_self_exclusion, 0) > 0
                    THEN CAST(COALESCE(ut.overlap_submission_and_comment_reply_events, 0) AS DOUBLE)
                         / CAST(
                             COALESCE(sa.other_user_submission_thread_comments_after_self_exclusion, 0)
                             + COALESCE(da.other_user_direct_comment_replies_after_self_exclusion, 0)
                             AS DOUBLE
                           )
                    ELSE NULL
                  END AS overlap_share_of_component_total,
                  COALESCE(sb.self_comments_on_own_submissions_excluded, 0)
                    + COALESCE(db.self_replies_to_own_comments_excluded, 0)
                    AS total_self_engagement_excluded,
                  CASE
                    WHEN COALESCE(sb.all_submission_thread_comments_before_self_exclusion, 0)
                         + COALESCE(db.all_direct_comment_replies_before_self_exclusion, 0) > 0
                    THEN CAST(
                           COALESCE(sb.self_comments_on_own_submissions_excluded, 0)
                           + COALESCE(db.self_replies_to_own_comments_excluded, 0)
                           AS DOUBLE
                         )
                         / CAST(
                             COALESCE(sb.all_submission_thread_comments_before_self_exclusion, 0)
                             + COALESCE(db.all_direct_comment_replies_before_self_exclusion, 0)
                             AS DOUBLE
                           )
                    ELSE NULL
                  END AS self_engagement_excluded_share,
                  COALESCE(sa.distinct_commenters_on_submissions, 0)
                    AS distinct_commenters_on_submissions,
                  COALESCE(da.distinct_repliers_to_comments, 0)
                    AS distinct_repliers_to_comments,
                  COALESCE(distinct_after.distinct_received_engagers, 0)
                    AS distinct_received_engagers
                FROM months m
                LEFT JOIN submission_before sb
                  ON sb.month = m.month
                LEFT JOIN submission_after sa
                  ON sa.month = m.month
                LEFT JOIN direct_before db
                  ON db.month = m.month
                LEFT JOIN direct_after da
                  ON da.month = m.month
                LEFT JOIN distinct_after
                  ON distinct_after.month = m.month
                LEFT JOIN unique_totals ut
                  ON ut.month = m.month
                """
            )
            con.execute(
                """
                CREATE OR REPLACE TEMP TABLE user_month_panel_current AS
                WITH panel AS (
                  SELECT
                    COALESCE(a.month, e.month) AS month,
                    COALESCE(a.user_id, e.user_id) AS user_id,
                    COALESCE(a.n_submissions, 0) AS n_submissions,
                    COALESCE(a.n_comments, 0) AS n_comments,
                    COALESCE(a.n_total_activity, 0) AS n_total_activity,
                    COALESCE(e.n_submission_thread_comments_received, 0) AS n_submission_thread_comments_received,
                    COALESCE(e.n_comment_replies_received, 0) AS n_comment_replies_received,
                    COALESCE(e.n_total_engagement_received, 0) AS n_total_engagement_received,
                    COALESCE(e.n_overlap_submission_and_comment_reply_events, 0)
                      AS n_overlap_submission_and_comment_reply_events,
                    COALESCE(e.n_distinct_commenters_on_submissions, 0) AS n_distinct_commenters_on_submissions,
                    COALESCE(e.n_distinct_repliers_to_comments, 0) AS n_distinct_repliers_to_comments,
                    COALESCE(e.n_distinct_received_engagers, 0) AS n_distinct_received_engagers
                  FROM user_month_activity_current a
                  FULL OUTER JOIN user_month_engagement_current e
                    ON a.month = e.month AND a.user_id = e.user_id
                )
                SELECT
                  *,
                  CASE
                    WHEN n_total_activity > 0
                    THEN CAST(n_total_engagement_received AS DOUBLE) / CAST(n_total_activity AS DOUBLE)
                    ELSE NULL
                  END AS received_engagement_per_activity,
                  CASE
                    WHEN n_submissions > 0
                    THEN CAST(n_submission_thread_comments_received AS DOUBLE) / CAST(n_submissions AS DOUBLE)
                    ELSE NULL
                  END AS received_comments_per_submission,
                  CASE
                    WHEN n_comments > 0
                    THEN CAST(n_comment_replies_received AS DOUBLE) / CAST(n_comments AS DOUBLE)
                    ELSE NULL
                  END AS received_replies_per_comment
                FROM panel
                """
            )
            log_table_count(con, "user_month_panel_current")
            log_table_count(con, "received_engagement_definition_audit_current")
            _copy_monthly_checkpoint_part(con, checkpoint_dir, "user_month_panel", "user_month_panel_current", month)
            _copy_monthly_checkpoint_part(
                con,
                checkpoint_dir,
                "received_engagement_definition_audit",
                "received_engagement_definition_audit_current",
                month,
            )
            mark_monthly_received_engagement_month_complete(checkpoint_dir, monthly_metadata, month)

            for table_name in (
                "user_month_activity_current",
                "comments_engagement_events_current",
                "submission_engagement_events_normalized_all_current",
                "submission_engagement_events_current",
                "comment_reply_engagement_events_normalized_all_current",
                "comment_reply_engagement_events_current",
                "user_month_engagement_current",
                "received_engagement_definition_audit_current",
                "user_month_panel_current",
            ):
                con.execute(f"DROP TABLE IF EXISTS {table_name}")

    validate_monthly_received_engagement_checkpoint_parts(
        con,
        checkpoint_dir,
        months,
        allow_empty=allow_empty_parts,
    )
    write_monthly_received_engagement_manifest(
        checkpoint_dir,
        monthly_metadata,
        completed_months=months,
        run_status="complete",
    )
    _load_monthwise_user_month_engagement_tables(con, checkpoint_dir, months)


def build_derived_tables(
    con: duckdb.DuckDBPyConnection,
    inv: Inventory,
    author_col: str,
    default_threads: int,
    submissions_cols: Sequence[str],
    smoke_max_rows: int,
    min_monthly_active_users_for_subset: int,
    nsfw_submission_share_threshold_for_subset: float,
    checkpoint_dir: str,
    reuse_checkpoints: bool,
    checkpoint_metadata: Optional[Dict[str, object]] = None,
    skip_subreddit_subset_diagnostics: bool = False,
) -> None:
    """Materialise a small set of reusable intermediates."""
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
            s.subreddit,
            strftime(s.created_utc, '%Y-%m') AS submission_month,
            COALESCE(c.n_comments, 0) AS thread_size
        FROM submissions_core_all s
        LEFT JOIN comment_counts_by_link c
          ON s.id = c.link_id
        """
    )

    logger.info("Materialising per_user_activity")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE per_user_activity AS
        WITH us AS (
          SELECT {author_col} AS user_id, COUNT(*) AS n_submissions
          FROM submissions_core_userlevel
          GROUP BY 1
        ),
        uc AS (
          SELECT {author_col} AS user_id, COUNT(*) AS n_comments
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
    log_table_count(con, "per_user_activity")

    logger.info("Materialising user_subreddit_hhi")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE user_subreddit_hhi AS
        WITH user_subreddit_activity AS (
          SELECT user_id, subreddit, COUNT(*) AS n_activity
          FROM (
            SELECT {author_col} AS user_id, subreddit FROM submissions_core_userlevel
            UNION ALL
            SELECT {author_col} AS user_id, subreddit FROM comments_core_userlevel
          ) z
          GROUP BY 1, 2
        ),
        totals AS (
          SELECT user_id, SUM(n_activity) AS total_activity
          FROM user_subreddit_activity
          GROUP BY 1
        )
        SELECT
          usa.user_id,
          SUM(POWER(CAST(usa.n_activity AS DOUBLE) / CAST(t.total_activity AS DOUBLE), 2)) AS subreddit_hhi,
          COUNT(*) AS n_active_subreddits,
          MAX(t.total_activity) AS total_activity
        FROM user_subreddit_activity usa
        INNER JOIN totals t
          ON usa.user_id = t.user_id
        GROUP BY 1
        """
    )
    log_table_count(con, "user_subreddit_hhi")

    _build_monthwise_user_month_engagement(
        con,
        inv,
        author_col,
        default_threads,
        smoke_max_rows,
        checkpoint_dir,
        reuse_checkpoints,
        checkpoint_metadata,
    )

    logger.info("Materialising per_user_active_months")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE per_user_active_months AS
        SELECT
          user_id,
          -- Count distinct active months over the selected period; this is not
          -- an adjacent-month or consecutive-month persistence measure.
          COUNT(DISTINCT month) AS n_active_months,
          SUM(n_total_activity) AS total_activity_across_active_months,
          SUM(n_total_engagement_received) AS total_engagement_across_months
        FROM user_month_panel
        WHERE n_total_activity > 0
        GROUP BY 1
        """
    )
    log_table_count(con, "per_user_active_months")

    build_subreddit_month_summary(
        con,
        inv,
        author_col,
        default_threads,
        checkpoint_dir,
        reuse_checkpoints,
        smoke_max_rows=smoke_max_rows,
    )

    with timed_stage("materialising subreddit_month_active_user_summary"):
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE subreddit_month_active_user_summary AS
            SELECT
              month,
              subreddit,
              n_active_users,
              n_total_activity
            FROM subreddit_month_summary
            """
        )
        log_table_count(con, "subreddit_month_active_user_summary")

    with timed_stage("materialising subreddit_repeat_user_share"):
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE subreddit_repeat_user_share AS
            SELECT
              subreddit,
              COUNT(*) AS n_months_observed,
              AVG(n_active_users) AS mean_monthly_active_users,
              AVG(pct_users_exactly_1_activity) AS mean_pct_users_exactly_1_activity,
              AVG(pct_users_2plus_activity) AS mean_pct_users_2plus_activity,
              AVG(pct_users_5plus_activity) AS mean_pct_users_5plus_activity,
              AVG(mean_activity_per_active_user) AS mean_activity_per_active_user
            FROM subreddit_month_summary
            GROUP BY 1
            """
        )
        log_table_count(con, "subreddit_repeat_user_share")

    logger.info("Materialising subreddit_monthly_totals")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE subreddit_monthly_totals AS
        SELECT
          month,
          subreddit,
          SUM(n_submissions) AS n_submissions,
          SUM(n_comments) AS n_comments,
          SUM(n_total_activity) AS n_total_activity
        FROM (
          SELECT strftime(created_utc, '%Y-%m') AS month, subreddit, COUNT(*) AS n_submissions, 0 AS n_comments, COUNT(*) AS n_total_activity
          FROM submissions_core_all
          WHERE {ORDINARY_SUBREDDIT_SQL}
          GROUP BY 1, 2
          UNION ALL
          SELECT strftime(created_utc, '%Y-%m') AS month, subreddit, 0 AS n_submissions, COUNT(*) AS n_comments, COUNT(*) AS n_total_activity
          FROM comments_core_all
          WHERE {ORDINARY_SUBREDDIT_SQL}
          GROUP BY 1, 2
        ) z
        GROUP BY 1, 2
        """
    )

    logger.info(
        "Materialising subreddit_activity from monthly totals and subreddit_month_summary; "
        "annual distinct user counts are not computed."
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE subreddit_activity AS
        WITH totals AS (
          SELECT
            subreddit,
            SUM(n_submissions) AS n_submissions,
            SUM(n_comments) AS n_comments,
            SUM(n_total_activity) AS n_total_activity,
            COUNT(DISTINCT month) AS n_months_observed
          FROM subreddit_monthly_totals
          GROUP BY 1
        ),
        monthly_users AS (
          SELECT
            mt.subreddit,
            AVG(CAST(COALESCE(sms.n_active_users, 0) AS DOUBLE)) AS mean_monthly_active_users,
            MIN(CAST(COALESCE(sms.n_active_users, 0) AS BIGINT)) AS min_monthly_active_users,
            MAX(CAST(COALESCE(sms.n_active_users, 0) AS BIGINT)) AS max_monthly_active_users
          FROM subreddit_monthly_totals mt
          LEFT JOIN subreddit_month_summary sms
            ON mt.month = sms.month
           AND mt.subreddit = sms.subreddit
          GROUP BY 1
        ),
        ts AS (
          SELECT
            s.subreddit,
            AVG(COALESCE(cc.n_comments, 0)) AS mean_thread_size,
            MEDIAN(COALESCE(cc.n_comments, 0)) AS median_thread_size,
            AVG(CASE WHEN COALESCE(cc.n_comments, 0) > 0 THEN 1.0 ELSE 0.0 END) AS pct_posts_with_comment
          FROM submissions_core_all s
          LEFT JOIN comment_counts_by_link cc
            ON s.id = cc.link_id
          WHERE {ORDINARY_SUBREDDIT_SQL}
          GROUP BY 1
        )
        SELECT
          t.subreddit,
          COALESCE(t.n_submissions, 0) AS n_submissions,
          COALESCE(t.n_comments, 0) AS n_comments,
          COALESCE(t.n_total_activity, 0) AS n_total_activity,
          COALESCE(t.n_months_observed, 0) AS n_months_observed,
          COALESCE(mu.mean_monthly_active_users, 0.0) AS mean_monthly_active_users,
          COALESCE(mu.min_monthly_active_users, 0) AS min_monthly_active_users,
          COALESCE(mu.max_monthly_active_users, 0) AS max_monthly_active_users,
          ts.mean_thread_size,
          ts.median_thread_size,
          ts.pct_posts_with_comment
        FROM totals t
        LEFT JOIN monthly_users mu
          ON t.subreddit = mu.subreddit
        LEFT JOIN ts
          ON t.subreddit = ts.subreddit
        """
    )
    log_table_count(con, "subreddit_activity")

    if skip_subreddit_subset_diagnostics:
        logger.info("Skipping subreddit subset-selection diagnostics materialisation.")
        return

    ordinary_sa_subreddit_sql = ORDINARY_SUBREDDIT_SQL.replace("subreddit", "sa.subreddit")
    n_months_expected = len(inv)
    has_over_18 = "over_18" in set(submissions_cols)
    nsfw_threshold_sql = f"{float(nsfw_submission_share_threshold_for_subset):.17g}"

    logger.info("Materialising subreddit_month_coverage")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE subreddit_month_coverage AS
        SELECT
          subreddit,
          COUNT(DISTINCT month) AS n_months_observed
        FROM subreddit_monthly_totals
        GROUP BY 1
        """
    )

    if has_over_18:
        logger.info("Materialising subreddit_nsfw_summary from submissions over_18 metadata")
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE subreddit_nsfw_summary AS
            WITH nsfw_base AS (
              SELECT
                subreddit,
                BOOL_OR(COALESCE(over_18, FALSE)) AS has_any_nsfw_submission,
                AVG(CASE WHEN COALESCE(over_18, FALSE) THEN 1.0 ELSE 0.0 END) AS nsfw_submission_share
              FROM submissions_core_all
              WHERE {ORDINARY_SUBREDDIT_SQL}
              GROUP BY 1
            )
            SELECT
              subreddit,
              has_any_nsfw_submission,
              nsfw_submission_share,
              nsfw_submission_share >= {nsfw_threshold_sql} AS nsfw_threshold_rule_failed
            FROM nsfw_base
            """
        )
    else:
        logger.info("Submission over_18 metadata unavailable; NSFW exclusion will not be applied in subset selection.")
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE subreddit_nsfw_summary AS
            SELECT
              CAST(NULL AS VARCHAR) AS subreddit,
              CAST(NULL AS BOOLEAN) AS has_any_nsfw_submission,
              CAST(NULL AS DOUBLE) AS nsfw_submission_share,
              CAST(NULL AS BOOLEAN) AS nsfw_threshold_rule_failed
            WHERE FALSE
            """
        )

    logger.info("Materialising subreddit_subset_diagnostics")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE subreddit_subset_diagnostics AS
        WITH stability AS (
          SELECT
            subreddit,
            AVG(n_total_activity) AS mean_monthly_total_activity,
            STDDEV_SAMP(n_total_activity) AS sd_monthly_total_activity,
            CASE
              WHEN AVG(n_total_activity) > 0 THEN STDDEV_SAMP(n_total_activity) / AVG(n_total_activity)
              ELSE NULL
            END AS cv_monthly_total_activity
          FROM subreddit_monthly_totals
          GROUP BY 1
        ),
        active_threshold_counts AS (
          SELECT
            mt.subreddit,
            COUNT(DISTINCT mt.month) AS n_raw_months_observed,
            MIN(CAST(COALESCE(aus.n_active_users, 0) AS BIGINT)) AS min_observed_monthly_active_users,
            AVG(CAST(COALESCE(aus.n_active_users, 0) AS DOUBLE)) AS mean_monthly_active_users_observed,
            MAX(CAST(COALESCE(aus.n_active_users, 0) AS BIGINT)) AS max_monthly_active_users,
            CAST(SUM(CASE WHEN COALESCE(aus.n_active_users, 0) >= {int(min_monthly_active_users_for_subset)} THEN 1 ELSE 0 END) AS BIGINT)
              AS n_observed_months_meeting_active_user_threshold
          FROM subreddit_monthly_totals mt
          LEFT JOIN subreddit_month_active_user_summary aus
            ON mt.month = aus.month
           AND mt.subreddit = aus.subreddit
          GROUP BY 1
        ),
        active_threshold_summary AS (
          SELECT
            subreddit,
            CASE
              WHEN n_raw_months_observed < {n_months_expected} THEN 0
              ELSE COALESCE(min_observed_monthly_active_users, 0)
            END AS min_monthly_active_users,
            mean_monthly_active_users_observed,
            max_monthly_active_users,
            CAST(
              n_observed_months_meeting_active_user_threshold
              + CASE
                  WHEN {int(min_monthly_active_users_for_subset)} <= 0 THEN {n_months_expected} - n_raw_months_observed
                  ELSE 0
                END
              AS BIGINT
            ) AS n_months_meeting_active_user_threshold,
            CASE
              WHEN (
                n_observed_months_meeting_active_user_threshold
                + CASE
                    WHEN {int(min_monthly_active_users_for_subset)} <= 0 THEN {n_months_expected} - n_raw_months_observed
                    ELSE 0
                  END
              ) = {n_months_expected} THEN TRUE
              ELSE FALSE
            END AS active_user_threshold_rule_passed
          FROM active_threshold_counts
        ),
        eligibility_base AS (
          SELECT
            sa.subreddit,
            sa.n_submissions,
            sa.n_comments,
            sa.n_total_activity,
            sa.mean_thread_size,
            sa.median_thread_size,
            sa.pct_posts_with_comment,
            rs.mean_pct_users_2plus_activity,
            rs.mean_pct_users_5plus_activity,
            rs.mean_monthly_active_users,
            rs.mean_activity_per_active_user,
            st.mean_monthly_total_activity,
            st.sd_monthly_total_activity,
            st.cv_monthly_total_activity,
            {n_months_expected} AS n_months_expected,
            COALESCE(mc.n_months_observed, 0) AS n_months_observed,
            CASE
              WHEN COALESCE(mc.n_months_observed, 0) = {n_months_expected} THEN TRUE
              ELSE FALSE
            END AS is_full_period,
            COALESCE(ats.min_monthly_active_users, 0) AS min_monthly_active_users,
            COALESCE(ats.mean_monthly_active_users_observed, 0.0) AS mean_monthly_active_users_observed,
            COALESCE(ats.max_monthly_active_users, 0) AS max_monthly_active_users_observed,
            COALESCE(ats.n_months_meeting_active_user_threshold, 0) AS n_months_meeting_active_user_threshold,
            COALESCE(ats.active_user_threshold_rule_passed, FALSE) AS active_user_threshold_rule_passed,
            CASE
              WHEN COALESCE(mc.n_months_observed, 0) = {n_months_expected}
               AND COALESCE(ats.active_user_threshold_rule_passed, FALSE)
              THEN TRUE
              ELSE FALSE
            END AS full_period_active_user_threshold_rule_passed,
            {int(min_monthly_active_users_for_subset)} AS min_monthly_active_users_required,
            nsfw.has_any_nsfw_submission,
            nsfw.nsfw_submission_share,
            {nsfw_threshold_sql} AS nsfw_submission_share_threshold_for_subset,
            nsfw.nsfw_threshold_rule_failed,
            regexp_matches(lower(sa.subreddit), '(^|[^a-z0-9])(live|livestream|stream)([^a-z0-9]|$)|(^live)|(^stream)|(stream$)|(live$)') AS likely_live_pattern,
            regexp_matches(lower(sa.subreddit), '(^|[^a-z0-9])(contest|giveaway)([^a-z0-9]|$)|(^contest)|(^giveaway)|(contest$)|(giveaway$)') AS likely_contest_pattern,
            regexp_matches(lower(sa.subreddit), '(^|[^a-z0-9])(bot|automod|auto_moderator)([^a-z0-9]|$)|(^bot)|(bot$)') AS likely_bot_pattern
          FROM subreddit_activity sa
          LEFT JOIN subreddit_repeat_user_share rs
            ON sa.subreddit = rs.subreddit
          LEFT JOIN stability st
            ON sa.subreddit = st.subreddit
          LEFT JOIN subreddit_month_coverage mc
            ON sa.subreddit = mc.subreddit
          LEFT JOIN active_threshold_summary ats
            ON sa.subreddit = ats.subreddit
          LEFT JOIN subreddit_nsfw_summary nsfw
            ON sa.subreddit = nsfw.subreddit
          WHERE {ordinary_sa_subreddit_sql}
        ),
        eligibility_scored AS (
          SELECT
            *,
            CASE
              WHEN full_period_active_user_threshold_rule_passed
               AND NOT COALESCE(nsfw_threshold_rule_failed, FALSE)
              THEN TRUE
              ELSE FALSE
            END AS is_eligible_for_subset_selection
          FROM eligibility_base
        )
        SELECT
          *,
          CASE
            WHEN is_eligible_for_subset_selection THEN 'eligible'
            ELSE CONCAT_WS(
              '|',
              CASE WHEN NOT is_full_period THEN 'not_full_period' ELSE NULL END,
              CASE WHEN NOT active_user_threshold_rule_passed THEN 'active_user_threshold_failed' ELSE NULL END,
              CASE WHEN COALESCE(nsfw_threshold_rule_failed, FALSE) THEN 'nsfw_submission_share_threshold_failed' ELSE NULL END
            )
          END AS subset_ineligibility_reason
        FROM eligibility_scored
        """
    )
    log_table_count(con, "subreddit_subset_diagnostics")

    # Lag and transition summaries are materialised by
    # build_compact_lag_and_transition_outputs(). They deliberately avoid the
    # former full-year row-level lag_relationships tables.
