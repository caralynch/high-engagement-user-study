from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import duckdb
import numpy as np
import pandas as pd

from .config import SUBSET_DIAG_TOP_LABEL_COUNT, SUBSET_THRESHOLD_COLUMNS


logger = logging.getLogger(__name__)


def fetch_tables(
    con: duckdb.DuckDBPyConnection,
    write_large_intermediates: bool = False,
    skip_subreddit_subset_diagnostics: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Fetch machine-readable tables used for plotting and diagnostics."""
    if write_large_intermediates:
        logger.info("Large intermediates will be exported directly from DuckDB; compact tables only are fetched into pandas.")
    else:
        logger.info("Large intermediates are not fetched into pandas and will not be written.")

    tables: Dict[str, pd.DataFrame] = {}

    tables["monthly_activity"] = con.execute(
        """
        WITH s AS (
          SELECT
            strftime(created_utc, '%Y-%m') AS month,
            COUNT(*) AS n_submissions,
            COUNT(DISTINCT subreddit) AS n_submission_subreddits
          FROM submissions_core_all
          GROUP BY 1
        ),
        c AS (
          SELECT
            strftime(created_utc, '%Y-%m') AS month,
            COUNT(*) AS n_comments,
            COUNT(DISTINCT subreddit) AS n_comment_subreddits
          FROM comments_core_all
          GROUP BY 1
        ),
        u AS (
          SELECT month, COUNT(DISTINCT user_id) AS n_active_users
          FROM user_month_panel
          WHERE n_total_activity > 0
          GROUP BY 1
        )
        SELECT
          COALESCE(s.month, c.month) AS month,
          COALESCE(s.n_submissions, 0) AS n_submissions,
          COALESCE(c.n_comments, 0) AS n_comments,
          COALESCE(u.n_active_users, 0) AS n_active_users,
          GREATEST(COALESCE(s.n_submission_subreddits, 0), COALESCE(c.n_comment_subreddits, 0)) AS n_active_subreddits
        FROM s
        FULL OUTER JOIN c ON s.month = c.month
        LEFT JOIN u ON COALESCE(s.month, c.month) = u.month
        ORDER BY month
        """
    ).df()

    subreddit_tables = ["subreddit_activity", "subreddit_repeat_user_share"]
    if not skip_subreddit_subset_diagnostics:
        subreddit_tables.append("subreddit_subset_diagnostics")
    for name in subreddit_tables:
        tables[name] = con.execute(f"SELECT * FROM {name}").df()

    for name in (
        "lag_past_activity_future_activity_summary",
        "lag_past_activity_future_activity_quantile_ribbon_summary",
        "lag_past_engagement_future_engagement_summary",
        "lag_past_engagement_future_engagement_quantile_ribbon_summary",
        "lag_past_activity_future_engagement_summary",
        "lag_past_activity_future_engagement_quantile_ribbon_summary",
        "lag_past_activity_future_activity_decile10_tail_usermonths",
        "lag_past_activity_future_activity_decile10_tail_users_aggregated",
        "lag_past_activity_future_engagement_decile10_tail_usermonths",
        "lag_past_activity_future_engagement_decile10_tail_users_aggregated",
        "lag_past_engagement_future_engagement_decile10_tail_usermonths",
        "lag_past_engagement_future_engagement_decile10_tail_users_aggregated",
        "activity_transition_matrix",
        "activity_transition_summary",
        "activity_transition_row_totals",
        "activity_transition_column_totals",
        "activity_transition_by_monthpair",
        "engagement_transition_matrix",
        "engagement_transition_summary",
        "engagement_transition_row_totals",
        "engagement_transition_column_totals",
        "engagement_transition_by_monthpair",
    ):
        tables[name] = con.execute(f"SELECT * FROM {name}").df()

    # Collapse the unchanged user-level table to a machine-readable distribution
    # with exact, cumulative ("at most k"), and survival ("at least k") shares.
    tables["active_months_per_user_summary"] = con.execute(
        """
        WITH counts AS (
          SELECT
            n_active_months,
            COUNT(*) AS n_users
          FROM per_user_active_months
          GROUP BY 1
        ),
        totals AS (
          SELECT SUM(n_users) AS total_users
          FROM counts
        ),
        augmented AS (
          SELECT
            c.n_active_months,
            c.n_users,
            CASE
              WHEN t.total_users > 0 THEN CAST(c.n_users AS DOUBLE) / CAST(t.total_users AS DOUBLE)
              ELSE NULL
            END AS user_share,
            SUM(c.n_users) OVER (
              ORDER BY c.n_active_months ASC
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS cumulative_n_users,
            SUM(c.n_users) OVER (
              ORDER BY c.n_active_months DESC
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS survival_n_users,
            t.total_users
          FROM counts c
          CROSS JOIN totals t
        )
        SELECT
          n_active_months,
          n_users,
          user_share,
          cumulative_n_users,
          CASE
            WHEN total_users > 0 THEN CAST(cumulative_n_users AS DOUBLE) / CAST(total_users AS DOUBLE)
            ELSE NULL
          END AS cumulative_user_share,
          survival_n_users,
          CASE
            WHEN total_users > 0 THEN CAST(survival_n_users AS DOUBLE) / CAST(total_users AS DOUBLE)
            ELSE NULL
          END AS survival_user_share
        FROM augmented
        ORDER BY n_active_months
        """
    ).df()

    def _survival_table_from_relation(
        relation_name: str,
        value_col: str,
        output_value_col: str,
        count_col: str,
        survival_col: str,
        share_col: str,
    ) -> pd.DataFrame:
        return con.execute(
            f"""
            WITH counts AS (
              SELECT
                {value_col} AS {output_value_col},
                COUNT(*) AS {count_col}
              FROM {relation_name}
              WHERE {value_col} > 0
              GROUP BY 1
            ),
            totals AS (
              SELECT
                COUNT(*) AS total_n,
                SUM(CASE WHEN {value_col} = 0 THEN 1 ELSE 0 END) AS zero_n
              FROM {relation_name}
              WHERE {value_col} IS NOT NULL
            ),
            survival AS (
              SELECT
                c.{output_value_col},
                c.{count_col},
                SUM(c.{count_col}) OVER (
                  ORDER BY c.{output_value_col} DESC
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS {survival_col},
                t.total_n,
                t.zero_n
              FROM counts c
              CROSS JOIN totals t
            )
            SELECT
              {output_value_col},
              {count_col},
              {survival_col},
              CASE
                WHEN total_n > 0 THEN CAST({survival_col} AS DOUBLE) / CAST(total_n AS DOUBLE)
                ELSE NULL
              END AS {share_col},
              total_n,
              zero_n,
              CASE
                WHEN total_n > 0 THEN CAST(zero_n AS DOUBLE) / CAST(total_n AS DOUBLE)
                ELSE NULL
              END AS zero_share
            FROM survival
            ORDER BY {output_value_col}
            """
        ).df()

    tables["user_activity_distribution_total_plot"] = _survival_table_from_relation(
        "per_user_activity",
        "n_total_activity",
        "activity_value",
        "n_users",
        "survival_n_users",
        "survival_share",
    )
    tables["user_activity_distribution_submissions_plot"] = _survival_table_from_relation(
        "per_user_activity",
        "n_submissions",
        "submission_count",
        "n_users",
        "survival_n_users",
        "survival_share",
    )
    tables["user_activity_distribution_comments_plot"] = _survival_table_from_relation(
        "per_user_activity",
        "n_comments",
        "comment_count",
        "n_users",
        "survival_n_users",
        "survival_share",
    )

    tables["thread_size_distribution_plot"] = con.execute(
        """
        WITH counts AS (
          SELECT
            thread_size,
            COUNT(*) AS n_submissions
          FROM thread_sizes_all
          GROUP BY 1
        ),
        totals AS (
          SELECT SUM(n_submissions) AS total_n_submissions
          FROM counts
        ),
        survival AS (
          SELECT
            c.thread_size,
            c.n_submissions,
            SUM(c.n_submissions) OVER (
              ORDER BY c.thread_size DESC
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS survival_n_submissions,
            t.total_n_submissions
          FROM counts c
          CROSS JOIN totals t
        )
        SELECT
          thread_size,
          n_submissions,
          survival_n_submissions,
          CASE
            WHEN total_n_submissions > 0
            THEN CAST(survival_n_submissions AS DOUBLE) / CAST(total_n_submissions AS DOUBLE)
            ELSE NULL
          END AS survival_share
        FROM survival
        ORDER BY thread_size
        """
    ).df()
    tables["monthly_thread_size"] = con.execute(
        """
        SELECT
          submission_month AS month,
          AVG(thread_size) AS mean_thread_size,
          MEDIAN(thread_size) AS median_thread_size
        FROM thread_sizes_all
        GROUP BY 1
        ORDER BY month
        """
    ).df()
    tables["monthly_thread_size_quantile_ribbon_summary"] = con.execute(
        """
        SELECT
          submission_month AS month,
          COUNT(*) AS n_submissions,
          AVG(thread_size) AS mean_thread_size,
          MEDIAN(thread_size) AS median_thread_size,
          QUANTILE_CONT(thread_size, 0.25) AS p25_thread_size,
          QUANTILE_CONT(thread_size, 0.75) AS p75_thread_size
        FROM thread_sizes_all
        GROUP BY 1
        ORDER BY month
        """
    ).df()

    tables["user_activity_lorenz_curve_plot"] = con.execute(
        """
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
            SUM(activity_value) OVER () AS total_activity,
            SUM(activity_value) OVER (
              ORDER BY activity_value ASC, user_id ASC
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS cum_activity
          FROM user_totals
        ),
        totals AS (
          SELECT MAX(n_users_total) AS total_users
          FROM ranked
        ),
        targets AS (
          SELECT DISTINCT
            CAST(
              ROUND(
                1 + i * (CAST(total_users AS DOUBLE) - 1) / 4999.0
              ) AS BIGINT
            ) AS target_rank
          FROM totals
          CROSS JOIN range(0, 5000) AS sample_idx(i)
        )
        SELECT
          r.user_rank,
          r.n_users_total,
          r.user_rank AS cum_users,
          CASE
            WHEN r.n_users_total > 0 THEN CAST(r.user_rank AS DOUBLE) / CAST(r.n_users_total AS DOUBLE)
            ELSE NULL
          END AS cum_user_share,
          r.activity_value,
          r.cum_activity,
          CASE
            WHEN r.total_activity > 0 THEN CAST(r.cum_activity AS DOUBLE) / CAST(r.total_activity AS DOUBLE)
            ELSE 0.0
          END AS cum_activity_share
        FROM ranked r
        WHERE r.n_users_total <= 5000
           OR r.user_rank IN (SELECT target_rank FROM targets)
        ORDER BY r.user_rank
        """
    ).df()
    tables["user_activity_lorenz_summary"] = con.execute(
        """
        WITH user_totals AS (
          SELECT
            user_id,
            SUM(COALESCE(n_total_activity, 0)) AS activity_value
          FROM user_month_panel
          GROUP BY 1
        ),
        ranked_asc AS (
          SELECT
            user_id,
            activity_value,
            ROW_NUMBER() OVER (ORDER BY activity_value ASC, user_id ASC) AS rank_asc,
            COUNT(*) OVER () AS n_users,
            SUM(activity_value) OVER () AS total_value
          FROM user_totals
        ),
        ranked_desc AS (
          SELECT
            user_id,
            activity_value,
            ROW_NUMBER() OVER (ORDER BY activity_value DESC, user_id ASC) AS rank_desc,
            COUNT(*) OVER () AS n_users,
            SUM(activity_value) OVER () AS total_value
          FROM user_totals
        )
        SELECT
          CASE
            WHEN MAX(ranked_asc.total_value) > 0 AND MAX(ranked_asc.n_users) > 0 THEN
              (2.0 * SUM(CAST(ranked_asc.rank_asc AS DOUBLE) * ranked_asc.activity_value) / (MAX(ranked_asc.n_users) * MAX(ranked_asc.total_value)))
              - ((MAX(ranked_asc.n_users) + 1.0) / MAX(ranked_asc.n_users))
            ELSE NULL
          END AS gini,
          CASE
            WHEN MAX(ranked_desc.total_value) > 0 THEN SUM(CASE WHEN ranked_desc.rank_desc <= CEIL(0.10 * ranked_desc.n_users) THEN ranked_desc.activity_value ELSE 0 END) / MAX(ranked_desc.total_value)
            ELSE NULL
          END AS top_10pct_share,
          CASE
            WHEN MAX(ranked_desc.total_value) > 0 THEN SUM(CASE WHEN ranked_desc.rank_desc <= CEIL(0.05 * ranked_desc.n_users) THEN ranked_desc.activity_value ELSE 0 END) / MAX(ranked_desc.total_value)
            ELSE NULL
          END AS top_5pct_share,
          CASE
            WHEN MAX(ranked_desc.total_value) > 0 THEN SUM(CASE WHEN ranked_desc.rank_desc <= CEIL(0.01 * ranked_desc.n_users) THEN ranked_desc.activity_value ELSE 0 END) / MAX(ranked_desc.total_value)
            ELSE NULL
          END AS top_1pct_share
        FROM ranked_asc
        INNER JOIN ranked_desc
          ON ranked_asc.user_id = ranked_desc.user_id
        """
    ).df()

    tables["user_engagement_lorenz_curve_plot"] = con.execute(
        """
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
            SUM(engagement_value) OVER () AS total_engagement,
            SUM(engagement_value) OVER (
              ORDER BY engagement_value ASC, user_id ASC
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS cum_engagement
          FROM user_totals
        ),
        totals AS (
          SELECT MAX(n_users_total) AS total_users
          FROM ranked
        ),
        targets AS (
          SELECT DISTINCT
            CAST(
              ROUND(
                1 + i * (CAST(total_users AS DOUBLE) - 1) / 4999.0
              ) AS BIGINT
            ) AS target_rank
          FROM totals
          CROSS JOIN range(0, 5000) AS sample_idx(i)
        )
        SELECT
          r.user_rank,
          r.n_users_total,
          r.user_rank AS cum_users,
          CASE
            WHEN r.n_users_total > 0 THEN CAST(r.user_rank AS DOUBLE) / CAST(r.n_users_total AS DOUBLE)
            ELSE NULL
          END AS cum_user_share,
          r.engagement_value,
          r.cum_engagement,
          CASE
            WHEN r.total_engagement > 0 THEN CAST(r.cum_engagement AS DOUBLE) / CAST(r.total_engagement AS DOUBLE)
            ELSE 0.0
          END AS cum_engagement_share
        FROM ranked r
        WHERE r.n_users_total <= 5000
           OR r.user_rank IN (SELECT target_rank FROM targets)
        ORDER BY r.user_rank
        """
    ).df()
    tables["user_engagement_received_distribution_plot"] = con.execute(
        """
        WITH user_totals AS (
          SELECT
            user_id,
            SUM(COALESCE(n_total_engagement_received, 0)) AS engagement_value
          FROM user_month_panel
          GROUP BY 1
        ),
        counts AS (
          SELECT
            engagement_value,
            COUNT(*) AS n_users
          FROM user_totals
          WHERE engagement_value > 0
          GROUP BY 1
        ),
        totals AS (
          SELECT
            COUNT(*) AS total_n_users,
            SUM(CASE WHEN engagement_value = 0 THEN 1 ELSE 0 END) AS zero_n_users
          FROM user_totals
          WHERE engagement_value IS NOT NULL
        ),
        survival AS (
          SELECT
            c.engagement_value,
            c.n_users,
            SUM(c.n_users) OVER (
              ORDER BY c.engagement_value DESC
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS survival_n_users,
            t.total_n_users,
            t.zero_n_users
          FROM counts c
          CROSS JOIN totals t
        )
        SELECT
          engagement_value,
          n_users,
          survival_n_users,
          CASE
            WHEN total_n_users > 0 THEN CAST(survival_n_users AS DOUBLE) / CAST(total_n_users AS DOUBLE)
            ELSE NULL
          END AS survival_share,
          total_n_users,
          zero_n_users,
          CASE
            WHEN total_n_users > 0 THEN CAST(zero_n_users AS DOUBLE) / CAST(total_n_users AS DOUBLE)
            ELSE NULL
          END AS zero_share
        FROM survival
        ORDER BY engagement_value
        """
    ).df()
    tables["received_engagement_definition_audit"] = con.execute(
        """
        SELECT *
        FROM received_engagement_definition_audit
        ORDER BY month
        """
    ).df()
    tables["monthly_engagement_rate_summary"] = con.execute(
        """
        WITH activity_thresholds AS (
          SELECT *
          FROM (
            VALUES
              (1, 'all_active_users'),
              (2, 'activity_ge_2'),
              (5, 'activity_ge_5'),
              (10, 'activity_ge_10')
          ) AS t(min_total_activity, activity_threshold_label)
        ),
        base AS (
          SELECT
            month,
            n_submissions,
            n_comments,
            n_total_activity,
            n_submission_thread_comments_received,
            n_comment_replies_received,
            n_total_engagement_received,
            n_overlap_submission_and_comment_reply_events,
            received_engagement_per_activity,
            received_comments_per_submission,
            received_replies_per_comment
          FROM user_month_panel
          WHERE n_total_activity > 0
        ),
        cohorts AS (
          SELECT
            b.*,
            a.min_total_activity,
            a.activity_threshold_label
          FROM base b
          INNER JOIN activity_thresholds a
            ON b.n_total_activity >= a.min_total_activity
        )
        SELECT
          month,
          activity_threshold_label,
          min_total_activity,
          COUNT(*) AS n_user_months,
          SUM(n_total_activity) AS total_authored_activity,
          SUM(n_submissions) AS total_authored_submissions,
          SUM(n_comments) AS total_authored_comments,
          SUM(n_total_engagement_received) AS total_unique_received_engagement,
          SUM(n_submission_thread_comments_received) AS total_submission_thread_comments_received,
          SUM(n_comment_replies_received) AS total_comment_replies_received,
          SUM(n_overlap_submission_and_comment_reply_events)
            AS total_overlap_submission_and_comment_reply_events,
          SUM(CASE WHEN n_total_engagement_received > 0 THEN 1 ELSE 0 END)
            AS n_user_months_with_received_engagement,
          AVG(CASE WHEN n_total_engagement_received > 0 THEN 1.0 ELSE 0.0 END)
            AS share_user_months_with_received_engagement,
          COUNT(received_engagement_per_activity) AS n_rate_denominator_activity,
          AVG(received_engagement_per_activity) AS mean_received_engagement_per_activity,
          MEDIAN(received_engagement_per_activity) AS median_received_engagement_per_activity,
          QUANTILE_CONT(received_engagement_per_activity, 0.25)
            AS p25_received_engagement_per_activity,
          QUANTILE_CONT(received_engagement_per_activity, 0.75)
            AS p75_received_engagement_per_activity,
          QUANTILE_CONT(received_engagement_per_activity, 0.90)
            AS p90_received_engagement_per_activity,
          COUNT(received_comments_per_submission) AS n_rate_denominator_submissions,
          AVG(received_comments_per_submission) AS mean_received_comments_per_submission,
          MEDIAN(received_comments_per_submission) AS median_received_comments_per_submission,
          QUANTILE_CONT(received_comments_per_submission, 0.25)
            AS p25_received_comments_per_submission,
          QUANTILE_CONT(received_comments_per_submission, 0.75)
            AS p75_received_comments_per_submission,
          COUNT(received_replies_per_comment) AS n_rate_denominator_comments,
          AVG(received_replies_per_comment) AS mean_received_replies_per_comment,
          MEDIAN(received_replies_per_comment) AS median_received_replies_per_comment,
          QUANTILE_CONT(received_replies_per_comment, 0.25)
            AS p25_received_replies_per_comment,
          QUANTILE_CONT(received_replies_per_comment, 0.75)
            AS p75_received_replies_per_comment
        FROM cohorts
        GROUP BY 1, 2, 3
        ORDER BY month, min_total_activity
        """
    ).df()
    tables["user_engagement_rate_distribution_plot"] = con.execute(
        """
        WITH activity_thresholds AS (
          SELECT *
          FROM (
            VALUES
              (1, 'all_active_users'),
              (2, 'activity_ge_2'),
              (5, 'activity_ge_5'),
              (10, 'activity_ge_10')
          ) AS t(min_total_activity, activity_threshold_label)
        ),
        rate_thresholds AS (
          SELECT *
          FROM (
            VALUES
              (0.001::DOUBLE),
              (0.005::DOUBLE),
              (0.01::DOUBLE),
              (0.025::DOUBLE),
              (0.05::DOUBLE),
              (0.1::DOUBLE),
              (0.25::DOUBLE),
              (0.5::DOUBLE),
              (1.0::DOUBLE),
              (2.0::DOUBLE),
              (5.0::DOUBLE),
              (10.0::DOUBLE),
              (20.0::DOUBLE),
              (50.0::DOUBLE),
              (100.0::DOUBLE)
          ) AS t(rate_threshold)
        ),
        base AS (
          SELECT
            month,
            user_id,
            n_total_activity,
            received_engagement_per_activity
          FROM user_month_panel
          WHERE n_total_activity > 0
            AND received_engagement_per_activity IS NOT NULL
        ),
        cohorts AS (
          SELECT
            b.*,
            a.min_total_activity,
            a.activity_threshold_label
          FROM base b
          INNER JOIN activity_thresholds a
            ON b.n_total_activity >= a.min_total_activity
        ),
        cohort_totals AS (
          SELECT
            min_total_activity,
            activity_threshold_label,
            COUNT(*) AS total_n_user_months,
            SUM(CASE WHEN received_engagement_per_activity = 0 THEN 1 ELSE 0 END)
              AS zero_rate_n_user_months
          FROM cohorts
          GROUP BY 1, 2
        ),
        survival AS (
          SELECT
            c.min_total_activity,
            c.activity_threshold_label,
            rt.rate_threshold,
            SUM(CASE WHEN c.received_engagement_per_activity >= rt.rate_threshold THEN 1 ELSE 0 END)
              AS n_user_months_at_or_above_threshold
          FROM cohorts c
          CROSS JOIN rate_thresholds rt
          GROUP BY 1, 2, 3
        )
        SELECT
          s.activity_threshold_label,
          s.min_total_activity,
          s.rate_threshold,
          ct.total_n_user_months,
          s.n_user_months_at_or_above_threshold,
          CASE
            WHEN ct.total_n_user_months > 0
            THEN CAST(s.n_user_months_at_or_above_threshold AS DOUBLE)
                 / CAST(ct.total_n_user_months AS DOUBLE)
            ELSE NULL
          END AS survival_share,
          ct.zero_rate_n_user_months,
          CASE
            WHEN ct.total_n_user_months > 0
            THEN CAST(ct.zero_rate_n_user_months AS DOUBLE) / CAST(ct.total_n_user_months AS DOUBLE)
            ELSE NULL
          END AS zero_rate_share
        FROM survival s
        INNER JOIN cohort_totals ct
          ON s.min_total_activity = ct.min_total_activity
         AND s.activity_threshold_label = ct.activity_threshold_label
        ORDER BY s.min_total_activity, s.rate_threshold
        """
    ).df()
    tables["user_engagement_lorenz_summary"] = con.execute(
        """
        WITH user_totals AS (
          SELECT
            user_id,
            SUM(COALESCE(n_total_engagement_received, 0)) AS engagement_value
          FROM user_month_panel
          GROUP BY 1
        ),
        ranked_asc AS (
          SELECT
            user_id,
            engagement_value,
            ROW_NUMBER() OVER (ORDER BY engagement_value ASC, user_id ASC) AS rank_asc,
            COUNT(*) OVER () AS n_users,
            SUM(engagement_value) OVER () AS total_value
          FROM user_totals
        ),
        ranked_desc AS (
          SELECT
            user_id,
            engagement_value,
            ROW_NUMBER() OVER (ORDER BY engagement_value DESC, user_id ASC) AS rank_desc,
            COUNT(*) OVER () AS n_users,
            SUM(engagement_value) OVER () AS total_value
          FROM user_totals
        )
        SELECT
          CASE
            WHEN MAX(ranked_asc.total_value) > 0 AND MAX(ranked_asc.n_users) > 0 THEN
              (2.0 * SUM(CAST(ranked_asc.rank_asc AS DOUBLE) * ranked_asc.engagement_value) / (MAX(ranked_asc.n_users) * MAX(ranked_asc.total_value)))
              - ((MAX(ranked_asc.n_users) + 1.0) / MAX(ranked_asc.n_users))
            ELSE NULL
          END AS gini,
          CASE
            WHEN MAX(ranked_desc.total_value) > 0 THEN SUM(CASE WHEN ranked_desc.rank_desc <= CEIL(0.10 * ranked_desc.n_users) THEN ranked_desc.engagement_value ELSE 0 END) / MAX(ranked_desc.total_value)
            ELSE NULL
          END AS top_10pct_share,
          CASE
            WHEN MAX(ranked_desc.total_value) > 0 THEN SUM(CASE WHEN ranked_desc.rank_desc <= CEIL(0.05 * ranked_desc.n_users) THEN ranked_desc.engagement_value ELSE 0 END) / MAX(ranked_desc.total_value)
            ELSE NULL
          END AS top_5pct_share,
          CASE
            WHEN MAX(ranked_desc.total_value) > 0 THEN SUM(CASE WHEN ranked_desc.rank_desc <= CEIL(0.01 * ranked_desc.n_users) THEN ranked_desc.engagement_value ELSE 0 END) / MAX(ranked_desc.total_value)
            ELSE NULL
          END AS top_1pct_share
        FROM ranked_asc
        INNER JOIN ranked_desc
          ON ranked_asc.user_id = ranked_desc.user_id
        """
    ).df()

    # Transition tables are read from compact temp tables materialised before
    # fetch_tables(); the repeated row-level CTEs formerly used here have been
    # removed to avoid recomputing destination deciles for each output.

    tables["user_subreddit_hhi_summary"] = con.execute(
        """
        WITH stats AS (
          SELECT
            COUNT(*) AS n_users,
            SUM(CASE WHEN subreddit_hhi = 1 THEN 1 ELSE 0 END) AS n_hhi_eq_1,
            SUM(CASE WHEN subreddit_hhi < 1 THEN 1 ELSE 0 END) AS n_hhi_lt_1,
            AVG(subreddit_hhi) AS mean_hhi,
            MEDIAN(subreddit_hhi) AS median_hhi,
            QUANTILE_CONT(subreddit_hhi, 0.25) AS p25_hhi,
            QUANTILE_CONT(subreddit_hhi, 0.75) AS p75_hhi,
            QUANTILE_CONT(subreddit_hhi, 0.90) AS p90_hhi,
            QUANTILE_CONT(subreddit_hhi, 0.95) AS p95_hhi,
            QUANTILE_CONT(subreddit_hhi, 0.99) AS p99_hhi
          FROM user_subreddit_hhi
        )
        SELECT
          n_users,
          n_hhi_eq_1,
          n_hhi_lt_1,
          CASE WHEN n_users > 0 THEN CAST(n_hhi_eq_1 AS DOUBLE) / CAST(n_users AS DOUBLE) ELSE NULL END AS share_hhi_eq_1,
          CASE WHEN n_users > 0 THEN CAST(n_hhi_lt_1 AS DOUBLE) / CAST(n_users AS DOUBLE) ELSE NULL END AS share_hhi_lt_1,
          mean_hhi,
          median_hhi,
          p25_hhi,
          p75_hhi,
          p90_hhi,
          p95_hhi,
          p99_hhi
        FROM stats
        """
    ).df()
    tables["user_subreddit_hhi_plot"] = con.execute(
        """
        WITH binned AS (
          SELECT
            CASE
              WHEN subreddit_hhi >= 0 AND subreddit_hhi < 1
              THEN LEAST(CAST(FLOOR(subreddit_hhi * 40) AS INTEGER), 39)
              ELSE NULL
            END AS bin_id,
            COUNT(*) AS n_users
          FROM user_subreddit_hhi
          WHERE subreddit_hhi >= 0 AND subreddit_hhi < 1
          GROUP BY 1
        ),
        totals AS (
          SELECT SUM(n_users) AS total_lt_1
          FROM binned
        )
        SELECT
          CAST(bin_id AS DOUBLE) / 40.0 AS bin_left,
          CAST(bin_id + 1 AS DOUBLE) / 40.0 AS bin_right,
          n_users,
          CASE
            WHEN total_lt_1 > 0 THEN CAST(n_users AS DOUBLE) / CAST(total_lt_1 AS DOUBLE)
            ELSE NULL
          END AS user_share_lt_1
        FROM binned
        CROSS JOIN totals
        WHERE bin_id IS NOT NULL
        ORDER BY bin_id
        """
    ).df()

    tables["active_subreddits_per_user_summary"] = con.execute(
        """
        WITH counts AS (
          SELECT
            n_active_subreddits,
            COUNT(*) AS n_users
          FROM user_subreddit_hhi
          GROUP BY 1
        ),
        totals AS (
          SELECT SUM(n_users) AS total_users
          FROM counts
        ),
        augmented AS (
          SELECT
            c.n_active_subreddits,
            c.n_users,
            CASE
              WHEN t.total_users > 0 THEN CAST(c.n_users AS DOUBLE) / CAST(t.total_users AS DOUBLE)
              ELSE NULL
            END AS user_share,
            SUM(c.n_users) OVER (
              ORDER BY c.n_active_subreddits ASC
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS cumulative_n_users,
            SUM(c.n_users) OVER (
              ORDER BY c.n_active_subreddits DESC
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS survival_n_users,
            t.total_users
          FROM counts c
          CROSS JOIN totals t
        )
        SELECT
          n_active_subreddits,
          n_users,
          user_share,
          cumulative_n_users,
          CASE
            WHEN total_users > 0 THEN CAST(cumulative_n_users AS DOUBLE) / CAST(total_users AS DOUBLE)
            ELSE NULL
          END AS cumulative_user_share,
          survival_n_users,
          CASE
            WHEN total_users > 0 THEN CAST(survival_n_users AS DOUBLE) / CAST(total_users AS DOUBLE)
            ELSE NULL
          END AS survival_user_share
        FROM augmented
        ORDER BY n_active_subreddits
        """
    ).df()

    tables["active_subreddits_percentile_summary"] = con.execute(
        """
        WITH stats AS (
          SELECT
            COUNT(*) AS n_users,
            AVG(CAST(n_active_subreddits AS DOUBLE)) AS mean,
            MEDIAN(n_active_subreddits) AS median,
            QUANTILE_CONT(n_active_subreddits, 0.75) AS p75,
            QUANTILE_CONT(n_active_subreddits, 0.90) AS p90,
            QUANTILE_CONT(n_active_subreddits, 0.95) AS p95,
            QUANTILE_CONT(n_active_subreddits, 0.99) AS p99,
            MAX(n_active_subreddits) AS max
          FROM user_subreddit_hhi
        )
        SELECT metric, value
        FROM (
          SELECT 'n_users' AS metric, CAST(n_users AS DOUBLE) AS value, 1 AS metric_order FROM stats
          UNION ALL
          SELECT 'mean' AS metric, mean AS value, 2 AS metric_order FROM stats
          UNION ALL
          SELECT 'median' AS metric, CAST(median AS DOUBLE) AS value, 3 AS metric_order FROM stats
          UNION ALL
          SELECT 'p75' AS metric, p75 AS value, 4 AS metric_order FROM stats
          UNION ALL
          SELECT 'p90' AS metric, p90 AS value, 5 AS metric_order FROM stats
          UNION ALL
          SELECT 'p95' AS metric, p95 AS value, 6 AS metric_order FROM stats
          UNION ALL
          SELECT 'p99' AS metric, p99 AS value, 7 AS metric_order FROM stats
          UNION ALL
          SELECT 'max' AS metric, CAST(max AS DOUBLE) AS value, 8 AS metric_order FROM stats
        ) ordered_metrics
        ORDER BY metric_order
        """
    ).df()

    tables["active_subreddits_tail_users"] = con.execute(
        """
        WITH tail AS (
          SELECT
            user_id,
            n_active_subreddits,
            total_activity,
            subreddit_hhi
          FROM user_subreddit_hhi
          ORDER BY n_active_subreddits DESC, total_activity DESC, user_id ASC
          LIMIT 100
        )
        SELECT
          t.user_id,
          t.n_active_subreddits,
          t.total_activity,
          pua.n_submissions,
          pua.n_comments,
          pam.n_active_months,
          t.subreddit_hhi
        FROM tail t
        LEFT JOIN per_user_activity pua
          ON t.user_id = pua.user_id
        LEFT JOIN per_user_active_months pam
          ON t.user_id = pam.user_id
        ORDER BY t.n_active_subreddits DESC, t.total_activity DESC, t.user_id ASC
        """
    ).df()

    tables["monthly_engagement_bin_prevalence"] = con.execute(
        """
        WITH base AS (
          SELECT
            month,
            CASE
              WHEN n_total_engagement_received = 0 THEN '0'
              WHEN n_total_engagement_received = 1 THEN '1'
              WHEN n_total_engagement_received BETWEEN 2 AND 5 THEN '2-5'
              WHEN n_total_engagement_received BETWEEN 6 AND 20 THEN '6-20'
              ELSE '21+'
            END AS engagement_bin
          FROM user_month_panel
          WHERE n_total_activity > 0
        ),
        counts AS (
          SELECT
            month,
            engagement_bin,
            COUNT(*) AS n_users
          FROM base
          GROUP BY 1, 2
        ),
        month_totals AS (
          SELECT month, SUM(n_users) AS n_active_users_total
          FROM counts
          GROUP BY 1
        )
        SELECT
          c.month,
          c.engagement_bin,
          c.n_users,
          CASE
            WHEN mt.n_active_users_total > 0 THEN CAST(c.n_users AS DOUBLE) / CAST(mt.n_active_users_total AS DOUBLE)
            ELSE NULL
          END AS user_share
        FROM counts c
        INNER JOIN month_totals mt
          ON c.month = mt.month
        ORDER BY c.month,
          CASE c.engagement_bin
            WHEN '0' THEN 1
            WHEN '1' THEN 2
            WHEN '2-5' THEN 3
            WHEN '6-20' THEN 4
            ELSE 5
          END
        """
    ).df()

    tables["monthly_engagement_bin_prevalence_audit"] = con.execute(
        """
        WITH base AS (
          SELECT
            month,
            n_total_activity,
            n_total_engagement_received
          FROM user_month_panel
          WHERE n_total_activity > 0
        )
        SELECT
          month,
          COUNT(*) AS n_active_users,
          SUM(CASE WHEN n_total_engagement_received = 0 THEN 1 ELSE 0 END) AS n_zero_engagement_users,
          AVG(CASE WHEN n_total_engagement_received = 0 THEN 1.0 ELSE 0.0 END) AS share_zero_engagement_users,
          AVG(n_total_engagement_received) AS mean_engagement_received,
          MEDIAN(n_total_engagement_received) AS median_engagement_received,
          QUANTILE_CONT(n_total_engagement_received, 0.25) AS p25_engagement_received,
          QUANTILE_CONT(n_total_engagement_received, 0.75) AS p75_engagement_received,
          AVG(n_total_activity) AS mean_activity,
          MEDIAN(n_total_activity) AS median_activity,
          QUANTILE_CONT(n_total_activity, 0.25) AS p25_activity,
          QUANTILE_CONT(n_total_activity, 0.75) AS p75_activity,
          SUM(CASE WHEN n_total_engagement_received = 0 THEN 1 ELSE 0 END) AS n_bin_0,
          SUM(CASE WHEN n_total_engagement_received = 1 THEN 1 ELSE 0 END) AS n_bin_1,
          SUM(CASE WHEN n_total_engagement_received BETWEEN 2 AND 5 THEN 1 ELSE 0 END) AS n_bin_2_5,
          SUM(CASE WHEN n_total_engagement_received BETWEEN 6 AND 20 THEN 1 ELSE 0 END) AS n_bin_6_20,
          SUM(CASE WHEN n_total_engagement_received >= 21 THEN 1 ELSE 0 END) AS n_bin_21_plus,
          AVG(CASE WHEN n_total_engagement_received = 0 THEN 1.0 ELSE 0.0 END) AS share_bin_0,
          AVG(CASE WHEN n_total_engagement_received = 1 THEN 1.0 ELSE 0.0 END) AS share_bin_1,
          AVG(CASE WHEN n_total_engagement_received BETWEEN 2 AND 5 THEN 1.0 ELSE 0.0 END) AS share_bin_2_5,
          AVG(CASE WHEN n_total_engagement_received BETWEEN 6 AND 20 THEN 1.0 ELSE 0.0 END) AS share_bin_6_20,
          AVG(CASE WHEN n_total_engagement_received >= 21 THEN 1.0 ELSE 0.0 END) AS share_bin_21_plus
        FROM base
        GROUP BY 1
        ORDER BY 1
        """
    ).df()

    if skip_subreddit_subset_diagnostics:
        logger.info("Skipping subreddit subset-selection diagnostics fetches.")
        return tables

    subset_summary = con.execute("SELECT * FROM subreddit_subset_diagnostics").df()
    if len(subset_summary):
        subset_summary = subset_summary.copy()
        eligible_mask = subset_summary["is_eligible_for_subset_selection"].fillna(False).astype(bool)
        subset_summary["activity_score"] = np.nan
        subset_summary["engagement_score"] = np.nan
        subset_summary["repeat_user_score"] = np.nan
        subset_summary["stability_score"] = np.nan
        subset_summary["candidate_score"] = np.nan
        subset_summary["candidate_flag"] = False
        subset_summary["candidate_rank"] = pd.Series(np.nan, index=subset_summary.index, dtype=float)

        if eligible_mask.any():
            eligible = subset_summary.loc[eligible_mask].copy()
            eligible["activity_score"] = eligible["n_total_activity"].rank(method="average", pct=True)
            eligible["engagement_score"] = eligible["mean_thread_size"].rank(method="average", pct=True)
            eligible["repeat_user_score"] = eligible["mean_pct_users_2plus_activity"].rank(method="average", pct=True)
            eligible["stability_score"] = eligible["cv_monthly_total_activity"].rank(
                method="average",
                pct=True,
                ascending=False,
                na_option="bottom",
            )
            eligible["candidate_score"] = eligible[
                ["activity_score", "engagement_score", "repeat_user_score", "stability_score"]
            ].mean(axis=1)
            eligible["candidate_flag"] = eligible["candidate_score"] >= 0.75
            eligible = eligible.sort_values(
                ["candidate_score", "n_total_activity", "subreddit"],
                ascending=[False, False, True],
            ).reset_index(drop=True)
            eligible["candidate_rank"] = np.arange(1, len(eligible) + 1, dtype=int)

            eligible_by_subreddit = eligible.set_index("subreddit")
            for col in (
                "activity_score",
                "engagement_score",
                "repeat_user_score",
                "stability_score",
                "candidate_score",
                "candidate_flag",
                "candidate_rank",
            ):
                subset_summary.loc[eligible_mask, col] = subset_summary.loc[eligible_mask, "subreddit"].map(eligible_by_subreddit[col])

        subset_summary = subset_summary.sort_values(
            ["is_eligible_for_subset_selection", "candidate_score", "n_total_activity", "subreddit"],
            ascending=[False, False, False, True],
            na_position="last",
        ).reset_index(drop=True)
    else:
        subset_summary["activity_score"] = pd.Series(dtype=float)
        subset_summary["engagement_score"] = pd.Series(dtype=float)
        subset_summary["repeat_user_score"] = pd.Series(dtype=float)
        subset_summary["stability_score"] = pd.Series(dtype=float)
        subset_summary["candidate_score"] = pd.Series(dtype=float)
        subset_summary["candidate_flag"] = pd.Series(dtype=bool)
        subset_summary["candidate_rank"] = pd.Series(dtype=float)
    tables["subreddit_subset_selection_summary"] = subset_summary
    labelled_candidates = subset_summary.loc[
        subset_summary["candidate_flag"].fillna(False).astype(bool)
        & subset_summary["candidate_rank"].notna()
    ].copy()
    if len(labelled_candidates):
        labelled_candidates["candidate_rank"] = pd.to_numeric(labelled_candidates["candidate_rank"], errors="coerce")
        labelled_candidates = labelled_candidates.sort_values("candidate_rank").head(SUBSET_DIAG_TOP_LABEL_COUNT)
        labelled_candidates = labelled_candidates[
            [
                "subreddit",
                "candidate_rank",
                "candidate_score",
                "candidate_flag",
                "is_eligible_for_subset_selection",
                "n_total_activity",
                "mean_thread_size",
                "mean_pct_users_2plus_activity",
            ]
        ].reset_index(drop=True)
    else:
        labelled_candidates = pd.DataFrame(
            columns=[
                "subreddit",
                "candidate_rank",
                "candidate_score",
                "candidate_flag",
                "is_eligible_for_subset_selection",
                "n_total_activity",
                "mean_thread_size",
                "mean_pct_users_2plus_activity",
            ]
        )
    tables["subset_diag_labelled_candidates"] = labelled_candidates
    return tables

def _positive_support_survival_from_series(
    values: pd.Series,
    value_col: str,
    count_col: str,
    survival_col: str,
    share_col: str,
    total_col: str = "total_n",
    zero_col: str = "zero_n",
) -> pd.DataFrame:
    """Build a compact positive-support survival table from a row-level vector."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    total_n = int(len(numeric))
    zero_n = int((numeric == 0).sum())
    counts = (
        numeric.loc[numeric > 0]
        .value_counts()
        .rename_axis(value_col)
        .reset_index(name=count_col)
        .sort_values(value_col, ascending=False)
        .reset_index(drop=True)
    )
    if len(counts):
        counts[survival_col] = counts[count_col].cumsum()
        counts = counts.sort_values(value_col).reset_index(drop=True)
    else:
        counts[survival_col] = pd.Series(dtype="int64")
    counts[share_col] = counts[survival_col] / total_n if total_n > 0 else np.nan
    counts[total_col] = total_n
    counts[zero_col] = zero_n
    counts["zero_share"] = zero_n / total_n if total_n > 0 else np.nan
    return counts[[value_col, count_col, survival_col, share_col, total_col, zero_col, "zero_share"]]

def _thread_size_survival_from_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the compact thread-size survival table, retaining zero-comment submissions."""
    if "thread_size" not in df.columns:
        raise ValueError("thread_size_distribution.parquet does not contain required column 'thread_size'.")
    thread_size = pd.to_numeric(df["thread_size"], errors="coerce").dropna()
    counts = (
        thread_size.value_counts()
        .rename_axis("thread_size")
        .reset_index(name="n_submissions")
        .sort_values("thread_size", ascending=False)
        .reset_index(drop=True)
    )
    total_n = int(counts["n_submissions"].sum()) if len(counts) else 0
    if len(counts):
        counts["survival_n_submissions"] = counts["n_submissions"].cumsum()
        counts = counts.sort_values("thread_size").reset_index(drop=True)
    else:
        counts["survival_n_submissions"] = pd.Series(dtype="int64")
    counts["survival_share"] = counts["survival_n_submissions"] / total_n if total_n > 0 else np.nan
    return counts[["thread_size", "n_submissions", "survival_n_submissions", "survival_share"]]

def downsample_lorenz_for_plot(df: pd.DataFrame, max_points: int = 5000) -> pd.DataFrame:
    """Downsample a full Lorenz table to the same compact plotting scale as full-run mode."""
    if len(df) <= max_points:
        return df.sort_values("user_rank").reset_index(drop=True) if "user_rank" in df.columns else df.reset_index(drop=True)
    if "user_rank" in df.columns:
        ordered = df.sort_values("user_rank").reset_index(drop=True)
    else:
        ordered = df.reset_index(drop=True)
    target_idx = np.unique(np.rint(np.linspace(0, len(ordered) - 1, max_points)).astype(int))
    return ordered.iloc[target_idx].reset_index(drop=True)

def _hhi_plot_from_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the full-run 40-bin HHI plotting table from the large intermediate."""
    if "subreddit_hhi" not in df.columns:
        raise ValueError("user_subreddit_hhi.parquet does not contain required column 'subreddit_hhi'.")
    hhi = pd.to_numeric(df["subreddit_hhi"], errors="coerce")
    hhi = hhi.loc[(hhi >= 0) & (hhi < 1)].dropna()
    bin_id = np.floor(hhi * 40).astype(int).clip(upper=39)
    counts = (
        bin_id.value_counts()
        .rename_axis("bin_id")
        .reset_index(name="n_users")
        .sort_values("bin_id")
        .reset_index(drop=True)
    )
    total_lt_1 = int(counts["n_users"].sum()) if len(counts) else 0
    counts["bin_left"] = counts["bin_id"].astype(float) / 40.0
    counts["bin_right"] = (counts["bin_id"].astype(float) + 1.0) / 40.0
    counts["user_share_lt_1"] = counts["n_users"] / total_lt_1 if total_lt_1 > 0 else np.nan
    return counts[["bin_left", "bin_right", "n_users", "user_share_lt_1"]]

def _read_parquet_intermediate(
    tables_dir: str,
    filename: str,
    parquet_cache: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Read one optional large intermediate for replot-only fallback derivations."""
    if parquet_cache is not None and filename in parquet_cache:
        return parquet_cache[filename]
    path = os.path.join(tables_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f"Could not read Parquet intermediate {path}: {exc}") from exc
    if parquet_cache is not None:
        parquet_cache[filename] = df
    return df

def _derive_missing_replot_table(
    name: str,
    tables_dir: str,
    parquet_cache: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Derive one missing compact replot CSV from converted large Parquet intermediates."""
    if name in {
        "user_activity_distribution_total_plot",
        "user_activity_distribution_submissions_plot",
        "user_activity_distribution_comments_plot",
    }:
        per_user = _read_parquet_intermediate(tables_dir, "per_user_activity.parquet", parquet_cache)
        specs = {
            "user_activity_distribution_total_plot": ("n_total_activity", "activity_value"),
            "user_activity_distribution_submissions_plot": ("n_submissions", "submission_count"),
            "user_activity_distribution_comments_plot": ("n_comments", "comment_count"),
        }
        source_col, output_col = specs[name]
        if source_col not in per_user.columns:
            raise ValueError(f"per_user_activity.parquet does not contain required column '{source_col}'.")
        return _positive_support_survival_from_series(
            per_user[source_col],
            output_col,
            "n_users",
            "survival_n_users",
            "survival_share",
        )

    if name == "thread_size_distribution_plot":
        thread_sizes = _read_parquet_intermediate(tables_dir, "thread_size_distribution.parquet", parquet_cache)
        return _thread_size_survival_from_frame(thread_sizes)

    if name == "user_activity_lorenz_curve_plot":
        lorenz = _read_parquet_intermediate(tables_dir, "user_activity_lorenz_curve.parquet", parquet_cache)
        return downsample_lorenz_for_plot(lorenz)

    if name == "user_engagement_lorenz_curve_plot":
        lorenz = _read_parquet_intermediate(tables_dir, "user_engagement_lorenz_curve.parquet", parquet_cache)
        return downsample_lorenz_for_plot(lorenz)

    if name == "user_engagement_received_distribution_plot":
        lorenz = _read_parquet_intermediate(tables_dir, "user_engagement_lorenz_curve.parquet", parquet_cache)
        if "engagement_value" not in lorenz.columns:
            raise ValueError(
                "Cannot derive user_engagement_received_distribution_plot.csv: "
                "user_engagement_lorenz_curve.parquet is present but lacks required column 'engagement_value'."
            )
        return _positive_support_survival_from_series(
            lorenz["engagement_value"],
            "engagement_value",
            "n_users",
            "survival_n_users",
            "survival_share",
            total_col="total_n_users",
            zero_col="zero_n_users",
        )

    if name == "user_subreddit_hhi_plot":
        hhi = _read_parquet_intermediate(tables_dir, "user_subreddit_hhi.parquet", parquet_cache)
        return _hhi_plot_from_frame(hhi)

    raise KeyError(f"No replot-only Parquet fallback is defined for {name}.")

def load_csv_tables(
    out_root: str,
    skip_subreddit_subset_diagnostics: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Load previously materialised plotting tables for replot-only mode."""
    required = [
        "monthly_activity",
        "subreddit_activity",
        "active_months_per_user_summary",
        "user_activity_distribution_total_plot",
        "user_activity_distribution_submissions_plot",
        "user_activity_distribution_comments_plot",
        "user_engagement_received_distribution_plot",
        "user_engagement_rate_distribution_plot",
        "user_subreddit_hhi_plot",
        "user_subreddit_hhi_summary",
        "subreddit_repeat_user_share",
        "lag_past_activity_future_activity_summary",
        "lag_past_activity_future_activity_quantile_ribbon_summary",
        "lag_past_engagement_future_engagement_summary",
        "lag_past_engagement_future_engagement_quantile_ribbon_summary",
        "lag_past_activity_future_engagement_summary",
        "lag_past_activity_future_engagement_quantile_ribbon_summary",
        "thread_size_distribution_plot",
        "monthly_thread_size",
        "monthly_thread_size_quantile_ribbon_summary",
        "user_activity_lorenz_curve_plot",
        "user_activity_lorenz_summary",
        "user_engagement_lorenz_curve_plot",
        "user_engagement_lorenz_summary",
        "received_engagement_definition_audit",
        "monthly_engagement_rate_summary",
        "activity_transition_matrix",
        "engagement_transition_matrix",
        "active_subreddits_per_user_summary",
        "monthly_engagement_bin_prevalence",
    ]
    if not skip_subreddit_subset_diagnostics:
        required.append("subreddit_subset_selection_summary")
    tables_dir = os.path.join(out_root, "tables")
    tables: Dict[str, pd.DataFrame] = {}
    parquet_cache: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []
    derivation_failures: List[str] = []
    derivable = {
        "user_activity_distribution_total_plot",
        "user_activity_distribution_submissions_plot",
        "user_activity_distribution_comments_plot",
        "user_engagement_received_distribution_plot",
        "user_subreddit_hhi_plot",
        "thread_size_distribution_plot",
        "user_activity_lorenz_curve_plot",
        "user_engagement_lorenz_curve_plot",
    }
    for name in required:
        path = os.path.join(tables_dir, f"{name}.csv")
        if not os.path.exists(path):
            if name in derivable:
                try:
                    tables[name] = _derive_missing_replot_table(name, tables_dir, parquet_cache)
                    tables[name].to_csv(path, index=False)
                    logger.info("Derived and cached missing replot table: %s", path)
                    continue
                except Exception as exc:
                    derivation_failures.append(f"{path} :: {exc}")
                    continue
            missing.append(path)
            continue
        tables[name] = pd.read_csv(path)
    if missing or derivation_failures:
        parts = [
            "Replot-only mode requires previously saved compact CSV tables under <out-root>/tables/.",
        ]
        if missing:
            parts.append("Genuinely missing required CSVs with no Parquet fallback attempted:\n  - " + "\n  - ".join(missing))
        if derivation_failures:
            parts.append(
                "Compact CSVs that were missing and could not be derived from available Parquet intermediates:\n  - "
                + "\n  - ".join(derivation_failures)
            )
        raise FileNotFoundError("\n".join(parts))
    if not skip_subreddit_subset_diagnostics:
        subset_cols = set(tables["subreddit_subset_selection_summary"].columns)
        missing_threshold_cols = [col for col in SUBSET_THRESHOLD_COLUMNS if col not in subset_cols]
        if missing_threshold_cols:
            logger.warning(
                "Replot-only loaded subreddit_subset_selection_summary.csv without current subset-threshold audit columns "
                "(missing: %s). Figures can be regenerated, but old outputs should be regenerated before using "
                "threshold-based NSFW subset eligibility.",
                ", ".join(missing_threshold_cols),
            )
    return tables
