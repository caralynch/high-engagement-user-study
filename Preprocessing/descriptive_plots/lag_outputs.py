from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import duckdb

from .checkpoints import checkpoint_file_exists, load_checkpoint_table, write_checkpoint_table
from .config import COMPACT_LAG_CHECKPOINT_TABLES
from .output_io import (
    log_table_count,
    month_add_one,
    quote_sql_string,
    temporary_duckdb_threads,
    timed_stage,
)


logger = logging.getLogger(__name__)


def valid_adjacent_month_pairs(months: Sequence[str]) -> List[Tuple[str, str]]:
    """Return selected adjacent month pairs where t and t+1 both exist."""
    month_set = set(months)
    pairs: List[Tuple[str, str]] = []
    for month in sorted(months):
        month_t1 = month_add_one(month)
        if month_t1 in month_set:
            pairs.append((month, month_t1))
    return pairs

def create_compact_lag_work_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create empty work tables used to aggregate lag outputs compactly."""
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE lag_activity_future_activity_outcome_counts (
          activity_t_decile INTEGER,
          outcome_value BIGINT,
          n_user_months BIGINT
        )
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE lag_engagement_future_engagement_outcome_counts (
          engagement_t_decile INTEGER,
          outcome_value BIGINT,
          n_user_months BIGINT
        )
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE lag_activity_future_engagement_outcome_counts (
          activity_t_decile INTEGER,
          outcome_value BIGINT,
          n_user_months BIGINT
        )
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE activity_transition_counts (
          month_t VARCHAR,
          month_t1 VARCHAR,
          activity_t_decile INTEGER,
          activity_t1_state VARCHAR,
          n_user_months BIGINT
        )
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE engagement_transition_counts (
          month_t VARCHAR,
          month_t1 VARCHAR,
          engagement_t_decile INTEGER,
          engagement_t1_state VARCHAR,
          n_user_months BIGINT
        )
        """
    )
    for name in (
        "lag_activity_activity_decile10_tail_candidates",
        "lag_activity_engagement_decile10_tail_candidates",
        "lag_engagement_engagement_decile10_tail_candidates",
    ):
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {name} (
              month VARCHAR,
              user_id VARCHAR,
              activity_t BIGINT,
              activity_t1 BIGINT,
              engagement_t BIGINT,
              engagement_t1 BIGINT,
              activity_t_decile INTEGER,
              engagement_t_decile INTEGER,
              likely_automoderator BOOLEAN,
              likely_bot BOOLEAN
            )
            """
        )
    for name in ("lag_activity_decile10_user_rollup_parts", "lag_engagement_decile10_user_rollup_parts"):
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {name} (
              user_id VARCHAR,
              n_decile10_source_months BIGINT,
              first_source_month VARCHAR,
              last_source_month VARCHAR,
              sum_activity_t DOUBLE,
              max_activity_t BIGINT,
              sum_activity_t1 DOUBLE,
              max_activity_t1 BIGINT,
              sum_engagement_t DOUBLE,
              max_engagement_t BIGINT,
              sum_engagement_t1 DOUBLE,
              max_engagement_t1 BIGINT,
              likely_automoderator BOOLEAN,
              likely_bot BOOLEAN
            )
            """
        )

def materialise_lag_summary_from_counts(
    con: duckdb.DuckDBPyConnection,
    count_table: str,
    decile_col: str,
    summary_table: str,
    ribbon_table: str,
    mean_col: str,
    median_col: str,
    p25_col: str,
    p75_col: str,
    iqr_col: str,
) -> None:
    """Materialise exact pooled mean and continuous quantiles from value counts."""
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {summary_table} AS
        WITH totals AS (
          SELECT
            {decile_col},
            SUM(n_user_months) AS total_n,
            SUM(CAST(outcome_value AS DOUBLE) * CAST(n_user_months AS DOUBLE)) AS sum_value
          FROM {count_table}
          GROUP BY 1
        ),
        positions AS (
          SELECT
            {decile_col},
            total_n,
            sum_value,
            1.0 + 0.25 * (CAST(total_n AS DOUBLE) - 1.0) AS p25_pos,
            CAST(FLOOR(1.0 + 0.25 * (CAST(total_n AS DOUBLE) - 1.0)) AS BIGINT) AS p25_lo_rank,
            CAST(CEIL(1.0 + 0.25 * (CAST(total_n AS DOUBLE) - 1.0)) AS BIGINT) AS p25_hi_rank,
            1.0 + 0.50 * (CAST(total_n AS DOUBLE) - 1.0) AS median_pos,
            CAST(FLOOR(1.0 + 0.50 * (CAST(total_n AS DOUBLE) - 1.0)) AS BIGINT) AS median_lo_rank,
            CAST(CEIL(1.0 + 0.50 * (CAST(total_n AS DOUBLE) - 1.0)) AS BIGINT) AS median_hi_rank,
            1.0 + 0.75 * (CAST(total_n AS DOUBLE) - 1.0) AS p75_pos,
            CAST(FLOOR(1.0 + 0.75 * (CAST(total_n AS DOUBLE) - 1.0)) AS BIGINT) AS p75_lo_rank,
            CAST(CEIL(1.0 + 0.75 * (CAST(total_n AS DOUBLE) - 1.0)) AS BIGINT) AS p75_hi_rank
          FROM totals
        ),
        ordered AS (
          SELECT
            {decile_col},
            outcome_value,
            n_user_months,
            SUM(n_user_months) OVER (
              PARTITION BY {decile_col}
              ORDER BY outcome_value
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS cum_n
          FROM {count_table}
        ),
        quantile_values AS (
          SELECT
            p.{decile_col},
            p.total_n,
            p.sum_value,
            p.p25_pos,
            p.p25_lo_rank,
            p.median_pos,
            p.median_lo_rank,
            p.p75_pos,
            p.p75_lo_rank,
            MIN(o.outcome_value) FILTER (WHERE o.cum_n >= p.p25_lo_rank) AS p25_lo_value,
            MIN(o.outcome_value) FILTER (WHERE o.cum_n >= p.p25_hi_rank) AS p25_hi_value,
            MIN(o.outcome_value) FILTER (WHERE o.cum_n >= p.median_lo_rank) AS median_lo_value,
            MIN(o.outcome_value) FILTER (WHERE o.cum_n >= p.median_hi_rank) AS median_hi_value,
            MIN(o.outcome_value) FILTER (WHERE o.cum_n >= p.p75_lo_rank) AS p75_lo_value,
            MIN(o.outcome_value) FILTER (WHERE o.cum_n >= p.p75_hi_rank) AS p75_hi_value
          FROM positions p
          INNER JOIN ordered o
            ON p.{decile_col} = o.{decile_col}
          GROUP BY
            p.{decile_col},
            p.total_n,
            p.sum_value,
            p.p25_pos,
            p.p25_lo_rank,
            p.median_pos,
            p.median_lo_rank,
            p.p75_pos,
            p.p75_lo_rank
        ),
        final AS (
          SELECT
            {decile_col},
            CAST(total_n AS BIGINT) AS n_user_months,
            sum_value / CAST(total_n AS DOUBLE) AS {mean_col},
            CAST(median_lo_value AS DOUBLE)
              + (median_pos - CAST(median_lo_rank AS DOUBLE))
              * (CAST(median_hi_value AS DOUBLE) - CAST(median_lo_value AS DOUBLE)) AS {median_col},
            CAST(p25_lo_value AS DOUBLE)
              + (p25_pos - CAST(p25_lo_rank AS DOUBLE))
              * (CAST(p25_hi_value AS DOUBLE) - CAST(p25_lo_value AS DOUBLE)) AS {p25_col},
            CAST(p75_lo_value AS DOUBLE)
              + (p75_pos - CAST(p75_lo_rank AS DOUBLE))
              * (CAST(p75_hi_value AS DOUBLE) - CAST(p75_lo_value AS DOUBLE)) AS {p75_col}
          FROM quantile_values
        )
        SELECT
          {decile_col},
          n_user_months,
          {mean_col},
          {median_col},
          {p25_col},
          {p75_col},
          {p75_col} - {p25_col} AS {iqr_col}
        FROM final
        ORDER BY {decile_col}
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {ribbon_table} AS
        SELECT
          {decile_col},
          {median_col},
          {p25_col},
          {p75_col}
        FROM {summary_table}
        ORDER BY {decile_col}
        """
    )

def materialise_decile10_tail_outputs(con: duckdb.DuckDBPyConnection) -> None:
    """Materialise capped decile-10 audit tables from compact candidate tables."""
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE lag_activity_decile10_user_rollup AS
        SELECT
          user_id,
          SUM(n_decile10_source_months) AS n_decile10_source_months,
          MIN(first_source_month) AS first_source_month,
          MAX(last_source_month) AS last_source_month,
          SUM(sum_activity_t) AS sum_activity_t,
          MAX(max_activity_t) AS max_activity_t,
          SUM(sum_activity_t1) AS sum_activity_t1,
          MAX(max_activity_t1) AS max_activity_t1,
          SUM(sum_engagement_t) AS sum_engagement_t,
          MAX(max_engagement_t) AS max_engagement_t,
          SUM(sum_engagement_t1) AS sum_engagement_t1,
          MAX(max_engagement_t1) AS max_engagement_t1,
          BOOL_OR(likely_automoderator) AS likely_automoderator,
          BOOL_OR(likely_bot) AS likely_bot
        FROM lag_activity_decile10_user_rollup_parts
        GROUP BY 1
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE lag_engagement_decile10_user_rollup AS
        SELECT
          user_id,
          SUM(n_decile10_source_months) AS n_decile10_source_months,
          MIN(first_source_month) AS first_source_month,
          MAX(last_source_month) AS last_source_month,
          SUM(sum_activity_t) AS sum_activity_t,
          MAX(max_activity_t) AS max_activity_t,
          SUM(sum_activity_t1) AS sum_activity_t1,
          MAX(max_activity_t1) AS max_activity_t1,
          SUM(sum_engagement_t) AS sum_engagement_t,
          MAX(max_engagement_t) AS max_engagement_t,
          SUM(sum_engagement_t1) AS sum_engagement_t1,
          MAX(max_engagement_t1) AS max_engagement_t1,
          BOOL_OR(likely_automoderator) AS likely_automoderator,
          BOOL_OR(likely_bot) AS likely_bot
        FROM lag_engagement_decile10_user_rollup_parts
        GROUP BY 1
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE lag_past_activity_future_activity_decile10_tail_usermonths AS
        SELECT *
        FROM lag_activity_activity_decile10_tail_candidates
        ORDER BY activity_t1 DESC, engagement_t1 DESC, activity_t DESC, engagement_t DESC, month ASC, user_id ASC
        LIMIT 500
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE lag_past_activity_future_engagement_decile10_tail_usermonths AS
        SELECT *
        FROM lag_activity_engagement_decile10_tail_candidates
        ORDER BY engagement_t1 DESC, activity_t1 DESC, activity_t DESC, engagement_t DESC, month ASC, user_id ASC
        LIMIT 500
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE lag_past_engagement_future_engagement_decile10_tail_usermonths AS
        SELECT *
        FROM lag_engagement_engagement_decile10_tail_candidates
        ORDER BY engagement_t1 DESC, activity_t1 DESC, engagement_t DESC, activity_t DESC, month ASC, user_id ASC
        LIMIT 500
        """
    )
    activity_user_select = """
        SELECT
          user_id,
          n_decile10_source_months,
          first_source_month,
          last_source_month,
          sum_activity_t / CAST(n_decile10_source_months AS DOUBLE) AS mean_activity_t,
          max_activity_t,
          sum_activity_t1 / CAST(n_decile10_source_months AS DOUBLE) AS mean_activity_t1,
          max_activity_t1,
          sum_engagement_t / CAST(n_decile10_source_months AS DOUBLE) AS mean_engagement_t,
          max_engagement_t,
          sum_engagement_t1 / CAST(n_decile10_source_months AS DOUBLE) AS mean_engagement_t1,
          max_engagement_t1,
          likely_automoderator,
          likely_bot
        FROM lag_activity_decile10_user_rollup
    """
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE lag_past_activity_future_activity_decile10_tail_users_aggregated AS
        {activity_user_select}
        ORDER BY max_activity_t1 DESC, max_engagement_t1 DESC, n_decile10_source_months DESC, user_id ASC
        LIMIT 5000
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE lag_past_activity_future_engagement_decile10_tail_users_aggregated AS
        {activity_user_select}
        ORDER BY max_engagement_t1 DESC, max_activity_t1 DESC, n_decile10_source_months DESC, user_id ASC
        LIMIT 5000
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE lag_past_engagement_future_engagement_decile10_tail_users_aggregated AS
        SELECT
          user_id,
          n_decile10_source_months,
          first_source_month,
          last_source_month,
          sum_activity_t / CAST(n_decile10_source_months AS DOUBLE) AS mean_activity_t,
          max_activity_t,
          sum_activity_t1 / CAST(n_decile10_source_months AS DOUBLE) AS mean_activity_t1,
          max_activity_t1,
          sum_engagement_t / CAST(n_decile10_source_months AS DOUBLE) AS mean_engagement_t,
          max_engagement_t,
          sum_engagement_t1 / CAST(n_decile10_source_months AS DOUBLE) AS mean_engagement_t1,
          max_engagement_t1,
          likely_automoderator,
          likely_bot
        FROM lag_engagement_decile10_user_rollup
        ORDER BY max_engagement_t1 DESC, max_activity_t1 DESC, n_decile10_source_months DESC, user_id ASC
        LIMIT 5000
        """
    )

def materialise_transition_outputs_from_counts(
    con: duckdb.DuckDBPyConnection,
    prefix: str,
    source_decile_col: str,
) -> None:
    """Build transition output tables from compact count tables."""
    counts_table = f"{prefix}_transition_counts"
    state_col = f"{prefix}_t1_state"
    matrix_table = f"{prefix}_transition_matrix"
    row_totals_table = f"{prefix}_transition_row_totals"
    column_totals_table = f"{prefix}_transition_column_totals"
    by_monthpair_table = f"{prefix}_transition_by_monthpair"
    summary_table = f"{prefix}_transition_summary"
    monthpair_share_col = "row_share_within_monthpair_source_decile"
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {matrix_table} AS
        WITH collapsed AS (
          SELECT
            {source_decile_col},
            {state_col},
            SUM(n_user_months) AS n_user_months
          FROM {counts_table}
          GROUP BY 1, 2
        ),
        row_totals AS (
          SELECT
            {source_decile_col},
            SUM(n_user_months) AS row_total_n_user_months
          FROM collapsed
          GROUP BY 1
        ),
        overall_total AS (
          SELECT SUM(n_user_months) AS total_n_user_months
          FROM collapsed
        )
        SELECT
          c.{source_decile_col},
          c.{state_col},
          c.n_user_months,
          CASE
            WHEN rt.row_total_n_user_months > 0 THEN CAST(c.n_user_months AS DOUBLE) / CAST(rt.row_total_n_user_months AS DOUBLE)
            ELSE NULL
          END AS row_share,
          CASE
            WHEN ot.total_n_user_months > 0 THEN CAST(c.n_user_months AS DOUBLE) / CAST(ot.total_n_user_months AS DOUBLE)
            ELSE NULL
          END AS overall_share
        FROM collapsed c
        INNER JOIN row_totals rt
          ON c.{source_decile_col} = rt.{source_decile_col}
        CROSS JOIN overall_total ot
        ORDER BY
          c.{source_decile_col},
          CASE
            WHEN c.{state_col} = 'Inactive' THEN 0
            ELSE CAST(c.{state_col} AS INTEGER)
          END
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {row_totals_table} AS
        SELECT
          {source_decile_col},
          SUM(n_user_months) AS row_total_n_user_months
        FROM {counts_table}
        GROUP BY 1
        ORDER BY 1
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {column_totals_table} AS
        WITH collapsed AS (
          SELECT
            {state_col},
            SUM(n_user_months) AS column_total_n_user_months
          FROM {counts_table}
          GROUP BY 1
        ),
        overall_total AS (
          SELECT SUM(column_total_n_user_months) AS total_n_user_months
          FROM collapsed
        )
        SELECT
          c.{state_col},
          c.column_total_n_user_months,
          CASE
            WHEN ot.total_n_user_months > 0 THEN CAST(c.column_total_n_user_months AS DOUBLE) / CAST(ot.total_n_user_months AS DOUBLE)
            ELSE NULL
          END AS column_total_share
        FROM collapsed c
        CROSS JOIN overall_total ot
        ORDER BY
          CASE
            WHEN c.{state_col} = 'Inactive' THEN 0
            ELSE CAST(c.{state_col} AS INTEGER)
          END
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {by_monthpair_table} AS
        WITH row_totals AS (
          SELECT
            month_t,
            month_t1,
            {source_decile_col},
            SUM(n_user_months) AS row_total_n_user_months
          FROM {counts_table}
          GROUP BY 1, 2, 3
        )
        SELECT
          c.month_t,
          c.month_t1,
          c.{source_decile_col},
          c.{state_col},
          c.n_user_months,
          CASE
            WHEN rt.row_total_n_user_months > 0 THEN CAST(c.n_user_months AS DOUBLE) / CAST(rt.row_total_n_user_months AS DOUBLE)
            ELSE NULL
          END AS {monthpair_share_col}
        FROM {counts_table} c
        INNER JOIN row_totals rt
          ON c.month_t = rt.month_t
         AND c.month_t1 = rt.month_t1
         AND c.{source_decile_col} = rt.{source_decile_col}
        ORDER BY
          c.month_t,
          c.month_t1,
          c.{source_decile_col},
          CASE
            WHEN c.{state_col} = 'Inactive' THEN 0
            ELSE CAST(c.{state_col} AS INTEGER)
          END
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {summary_table} AS
        WITH summary AS (
          SELECT
            COALESCE(SUM(n_user_months), 0) AS n_source_user_months,
            COALESCE(SUM(CASE WHEN {state_col} = 'Inactive' THEN n_user_months ELSE 0 END), 0) AS n_inactive_t1,
            COUNT(DISTINCT month_t || '->' || month_t1) AS n_month_pairs
          FROM {counts_table}
        )
        SELECT metric, value
        FROM (
          SELECT 'n_source_user_months' AS metric, CAST(n_source_user_months AS DOUBLE) AS value, 1 AS metric_order FROM summary
          UNION ALL
          SELECT 'n_inactive_t1' AS metric, CAST(n_inactive_t1 AS DOUBLE) AS value, 2 AS metric_order FROM summary
          UNION ALL
          SELECT
            'inactive_t1_share' AS metric,
            CASE
              WHEN n_source_user_months > 0 THEN CAST(n_inactive_t1 AS DOUBLE) / CAST(n_source_user_months AS DOUBLE)
              ELSE NULL
            END AS value,
            3 AS metric_order
          FROM summary
          UNION ALL
          SELECT
            'n_matched_t1' AS metric,
            CAST(n_source_user_months - n_inactive_t1 AS DOUBLE) AS value,
            4 AS metric_order
          FROM summary
          UNION ALL
          SELECT
            'matched_t1_share' AS metric,
            CASE
              WHEN n_source_user_months > 0 THEN CAST(n_source_user_months - n_inactive_t1 AS DOUBLE) / CAST(n_source_user_months AS DOUBLE)
              ELSE NULL
            END AS value,
            5 AS metric_order
          FROM summary
          UNION ALL
          SELECT 'n_month_pairs' AS metric, CAST(n_month_pairs AS DOUBLE) AS value, 6 AS metric_order FROM summary
        ) ordered_metrics
        ORDER BY metric_order
        """
    )

def validate_transition_outputs(con: duckdb.DuckDBPyConnection, prefix: str, source_decile_col: str) -> None:
    """Log share-sum checks for transition outputs."""
    matrix_table = f"{prefix}_transition_matrix"
    n_rows = int(con.execute(f"SELECT COUNT(*) FROM {matrix_table}").fetchone()[0])
    if n_rows == 0:
        logger.warning("%s has no rows; transition share checks skipped.", matrix_table)
        return
    overall_share = con.execute(f"SELECT SUM(overall_share) FROM {matrix_table}").fetchone()[0]
    max_row_error = con.execute(
        f"""
        WITH row_sums AS (
          SELECT {source_decile_col}, SUM(row_share) AS row_share_sum
          FROM {matrix_table}
          GROUP BY 1
        )
        SELECT MAX(ABS(row_share_sum - 1.0))
        FROM row_sums
        """
    ).fetchone()[0]
    logger.info(
        "%s validation: overall_share_sum=%.12f max_row_share_error=%.3g",
        matrix_table,
        float(overall_share or 0.0),
        float(max_row_error or 0.0),
    )
    if overall_share is not None and abs(float(overall_share) - 1.0) > 1e-6:
        logger.warning("%s overall_share sums to %.12f, expected approximately 1.", matrix_table, float(overall_share))
    if max_row_error is not None and float(max_row_error) > 1e-6:
        logger.warning("%s row shares have max error %.3g, expected approximately 0.", matrix_table, float(max_row_error))

def log_transition_output_table_counts(con: duckdb.DuckDBPyConnection, prefix: str) -> None:
    """Log row counts for transition outputs derived from compact count checkpoints."""
    for table_name in (
        f"{prefix}_transition_matrix",
        f"{prefix}_transition_summary",
        f"{prefix}_transition_row_totals",
        f"{prefix}_transition_column_totals",
        f"{prefix}_transition_by_monthpair",
    ):
        log_table_count(con, table_name)

def validate_lag_summary_deciles(con: duckdb.DuckDBPyConnection, table_name: str, decile_col: str) -> None:
    """Log decile coverage for a lag summary table."""
    row = con.execute(
        f"""
        SELECT
          COUNT(*) AS n_rows,
          MIN({decile_col}) AS min_decile,
          MAX({decile_col}) AS max_decile,
          COUNT(DISTINCT {decile_col}) AS n_deciles
        FROM {table_name}
        """
    ).fetchone()
    n_rows, min_decile, max_decile, n_deciles = row
    logger.info(
        "%s decile coverage: rows=%s distinct_deciles=%s min=%s max=%s",
        table_name,
        n_rows,
        n_deciles,
        min_decile,
        max_decile,
    )
    if n_rows and (int(min_decile) < 1 or int(max_decile) > 10):
        logger.warning("%s has deciles outside the expected 1-10 range.", table_name)

def materialise_all_compact_lag_outputs(con: duckdb.DuckDBPyConnection) -> None:
    """Materialise all compact lag and transition output tables from work counts."""
    materialise_lag_summary_from_counts(
        con,
        "lag_activity_future_activity_outcome_counts",
        "activity_t_decile",
        "lag_past_activity_future_activity_summary",
        "lag_past_activity_future_activity_quantile_ribbon_summary",
        "mean_activity_t1",
        "median_activity_t1",
        "p25_activity_t1",
        "p75_activity_t1",
        "iqr_activity_t1",
    )
    materialise_lag_summary_from_counts(
        con,
        "lag_engagement_future_engagement_outcome_counts",
        "engagement_t_decile",
        "lag_past_engagement_future_engagement_summary",
        "lag_past_engagement_future_engagement_quantile_ribbon_summary",
        "mean_engagement_t1",
        "median_engagement_t1",
        "p25_engagement_t1",
        "p75_engagement_t1",
        "iqr_engagement_t1",
    )
    materialise_lag_summary_from_counts(
        con,
        "lag_activity_future_engagement_outcome_counts",
        "activity_t_decile",
        "lag_past_activity_future_engagement_summary",
        "lag_past_activity_future_engagement_quantile_ribbon_summary",
        "mean_engagement_t1",
        "median_engagement_t1",
        "p25_engagement_t1",
        "p75_engagement_t1",
        "iqr_engagement_t1",
    )
    materialise_decile10_tail_outputs(con)
    materialise_transition_outputs_from_counts(con, "activity", "activity_t_decile")
    materialise_transition_outputs_from_counts(con, "engagement", "engagement_t_decile")
    validate_lag_summary_deciles(con, "lag_past_activity_future_activity_summary", "activity_t_decile")
    validate_lag_summary_deciles(con, "lag_past_engagement_future_engagement_summary", "engagement_t_decile")
    validate_lag_summary_deciles(con, "lag_past_activity_future_engagement_summary", "activity_t_decile")
    validate_transition_outputs(con, "activity", "activity_t_decile")
    validate_transition_outputs(con, "engagement", "engagement_t_decile")

def compact_lag_checkpoints_available(checkpoint_dir: str) -> bool:
    """Return whether all compact lag checkpoint tables are present."""
    missing = [name for name in COMPACT_LAG_CHECKPOINT_TABLES if not checkpoint_file_exists(checkpoint_dir, name)]
    if missing:
        logger.info("Compact lag checkpoint reuse unavailable; missing: %s", ", ".join(missing))
        return False
    return True

def build_compact_lag_and_transition_outputs(
    con: duckdb.DuckDBPyConnection,
    months: Sequence[str],
    default_threads: int,
    checkpoint_dir: str,
    reuse_checkpoints: bool,
) -> None:
    """Build compact lag summaries and transition outputs without full-year lag rows."""
    if reuse_checkpoints and compact_lag_checkpoints_available(checkpoint_dir):
        with timed_stage("loading compact lag checkpoints"):
            for table_name in COMPACT_LAG_CHECKPOINT_TABLES:
                load_checkpoint_table(con, checkpoint_dir, table_name)
            materialise_transition_outputs_from_counts(con, "activity", "activity_t_decile")
            materialise_transition_outputs_from_counts(con, "engagement", "engagement_t_decile")
            log_transition_output_table_counts(con, "activity")
            log_transition_output_table_counts(con, "engagement")
            validate_lag_summary_deciles(con, "lag_past_activity_future_activity_summary", "activity_t_decile")
            validate_lag_summary_deciles(con, "lag_past_engagement_future_engagement_summary", "engagement_t_decile")
            validate_lag_summary_deciles(con, "lag_past_activity_future_engagement_summary", "activity_t_decile")
            validate_transition_outputs(con, "activity", "activity_t_decile")
            validate_transition_outputs(con, "engagement", "engagement_t_decile")
        return

    pairs = valid_adjacent_month_pairs(months)
    logger.info("Building compact lag outputs from %d adjacent month pairs.", len(pairs))
    create_compact_lag_work_tables(con)
    for month_t, month_t1 in pairs:
        month_t_sql = quote_sql_string(month_t)
        month_t1_sql = quote_sql_string(month_t1)
        with timed_stage(f"compact lag month pair {month_t}->{month_t1}"):
            # These NTILE windows are the memory-sensitive part of the lag path.
            # They are scoped to one month-pair at a time rather than the full year.
            with temporary_duckdb_threads(con, 1, default_threads):
                con.execute(
                    f"""
                    CREATE OR REPLACE TEMP TABLE lag_source_deciles_current AS
                    SELECT
                      month,
                      {month_t1_sql} AS month_t1,
                      user_id,
                      CAST(n_total_activity AS BIGINT) AS activity_t,
                      CAST(n_total_engagement_received AS BIGINT) AS engagement_t,
                      NTILE(10) OVER (ORDER BY n_total_activity, user_id) AS activity_t_decile,
                      NTILE(10) OVER (ORDER BY n_total_engagement_received, user_id) AS engagement_t_decile
                    FROM user_month_panel
                    WHERE month = {month_t_sql}
                      AND n_total_activity > 0
                    """
                )
                con.execute(
                    f"""
                    CREATE OR REPLACE TEMP TABLE lag_destination_deciles_current AS
                    SELECT
                      user_id,
                      CAST(NTILE(10) OVER (ORDER BY n_total_activity, user_id) AS VARCHAR) AS activity_t1_state,
                      CAST(NTILE(10) OVER (ORDER BY n_total_engagement_received, user_id) AS VARCHAR) AS engagement_t1_state
                    FROM user_month_panel
                    WHERE month = {month_t1_sql}
                    """
                )
            con.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE lag_pair_current AS
                SELECT
                  s.month,
                  s.month_t1,
                  s.user_id,
                  s.activity_t,
                  s.engagement_t,
                  CAST(COALESCE(f.n_total_activity, 0) AS BIGINT) AS activity_t1,
                  CAST(COALESCE(f.n_total_engagement_received, 0) AS BIGINT) AS engagement_t1,
                  s.activity_t_decile,
                  s.engagement_t_decile,
                  COALESCE(d.activity_t1_state, 'Inactive') AS activity_t1_state,
                  COALESCE(d.engagement_t1_state, 'Inactive') AS engagement_t1_state
                FROM lag_source_deciles_current s
                LEFT JOIN user_month_panel f
                  ON s.user_id = f.user_id
                 AND f.month = {month_t1_sql}
                LEFT JOIN lag_destination_deciles_current d
                  ON s.user_id = d.user_id
                """
            )
            log_table_count(con, "lag_pair_current")
            con.execute(
                """
                INSERT INTO lag_activity_future_activity_outcome_counts
                SELECT activity_t_decile, activity_t1 AS outcome_value, COUNT(*) AS n_user_months
                FROM lag_pair_current
                GROUP BY 1, 2
                """
            )
            con.execute(
                """
                INSERT INTO lag_engagement_future_engagement_outcome_counts
                SELECT engagement_t_decile, engagement_t1 AS outcome_value, COUNT(*) AS n_user_months
                FROM lag_pair_current
                GROUP BY 1, 2
                """
            )
            con.execute(
                """
                INSERT INTO lag_activity_future_engagement_outcome_counts
                SELECT activity_t_decile, engagement_t1 AS outcome_value, COUNT(*) AS n_user_months
                FROM lag_pair_current
                GROUP BY 1, 2
                """
            )
            con.execute(
                """
                INSERT INTO activity_transition_counts
                SELECT
                  month AS month_t,
                  month_t1,
                  activity_t_decile,
                  activity_t1_state,
                  COUNT(*) AS n_user_months
                FROM lag_pair_current
                GROUP BY 1, 2, 3, 4
                """
            )
            con.execute(
                """
                INSERT INTO engagement_transition_counts
                SELECT
                  month AS month_t,
                  month_t1,
                  engagement_t_decile,
                  engagement_t1_state,
                  COUNT(*) AS n_user_months
                FROM lag_pair_current
                GROUP BY 1, 2, 3, 4
                """
            )
            for table_name, where_clause, order_clause in (
                (
                    "lag_activity_activity_decile10_tail_candidates",
                    "activity_t_decile = 10",
                    "activity_t1 DESC, engagement_t1 DESC, activity_t DESC, engagement_t DESC, month ASC, user_id ASC",
                ),
                (
                    "lag_activity_engagement_decile10_tail_candidates",
                    "activity_t_decile = 10",
                    "engagement_t1 DESC, activity_t1 DESC, activity_t DESC, engagement_t DESC, month ASC, user_id ASC",
                ),
                (
                    "lag_engagement_engagement_decile10_tail_candidates",
                    "engagement_t_decile = 10",
                    "engagement_t1 DESC, activity_t1 DESC, engagement_t DESC, activity_t DESC, month ASC, user_id ASC",
                ),
            ):
                con.execute(
                    f"""
                    INSERT INTO {table_name}
                    SELECT
                      month,
                      CAST(user_id AS VARCHAR) AS user_id,
                      activity_t,
                      activity_t1,
                      engagement_t,
                      engagement_t1,
                      activity_t_decile,
                      engagement_t_decile,
                      CASE WHEN lower(CAST(user_id AS VARCHAR)) LIKE '%automoderator%' THEN TRUE ELSE FALSE END AS likely_automoderator,
                      CASE WHEN lower(CAST(user_id AS VARCHAR)) LIKE '%bot%' THEN TRUE ELSE FALSE END AS likely_bot
                    FROM lag_pair_current
                    WHERE {where_clause}
                    ORDER BY {order_clause}
                    LIMIT 500
                    """
                )
            for table_name, where_clause in (
                ("lag_activity_decile10_user_rollup_parts", "activity_t_decile = 10"),
                ("lag_engagement_decile10_user_rollup_parts", "engagement_t_decile = 10"),
            ):
                con.execute(
                    f"""
                    INSERT INTO {table_name}
                    SELECT
                      CAST(user_id AS VARCHAR) AS user_id,
                      COUNT(*) AS n_decile10_source_months,
                      MIN(month) AS first_source_month,
                      MAX(month) AS last_source_month,
                      SUM(CAST(activity_t AS DOUBLE)) AS sum_activity_t,
                      MAX(activity_t) AS max_activity_t,
                      SUM(CAST(activity_t1 AS DOUBLE)) AS sum_activity_t1,
                      MAX(activity_t1) AS max_activity_t1,
                      SUM(CAST(engagement_t AS DOUBLE)) AS sum_engagement_t,
                      MAX(engagement_t) AS max_engagement_t,
                      SUM(CAST(engagement_t1 AS DOUBLE)) AS sum_engagement_t1,
                      MAX(engagement_t1) AS max_engagement_t1,
                      BOOL_OR(CASE WHEN lower(CAST(user_id AS VARCHAR)) LIKE '%automoderator%' THEN TRUE ELSE FALSE END) AS likely_automoderator,
                      BOOL_OR(CASE WHEN lower(CAST(user_id AS VARCHAR)) LIKE '%bot%' THEN TRUE ELSE FALSE END) AS likely_bot
                    FROM lag_pair_current
                    WHERE {where_clause}
                    GROUP BY 1
                    """
                )

    with timed_stage("materialising compact lag and transition outputs"):
        materialise_all_compact_lag_outputs(con)
        for table_name in COMPACT_LAG_CHECKPOINT_TABLES:
            write_checkpoint_table(con, checkpoint_dir, table_name)
