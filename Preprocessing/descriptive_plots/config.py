from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# ----------------------------- Constants ---------------------------------

MONTH_RE = re.compile(r"(?<!\d)(20\d{2})[-_](0[1-9]|1[0-2])(?!\d)")
COMMENTS_KIND = "comments_core"
SUBMISSIONS_KIND = "submissions_core"
ORDINARY_SUBREDDIT_SQL = "subreddit IS NOT NULL AND lower(subreddit) NOT LIKE 'u\\_%' ESCAPE '\\'"
FIGURE_ORDER: Tuple[str, ...] = (
    "monthly_platform_activity",
    "subreddit_rank_size_submissions",
    "subreddit_rank_size_comments",
    "user_activity_distribution_total",
    "user_activity_distribution_submissions",
    "user_activity_distribution_comments",
    "user_activity_lorenz_curve",
    "user_engagement_received_distribution",
    "received_engagement_per_activity_distribution",
    "monthly_received_engagement_per_activity_iqr",
    "thread_size_distribution",
    "monthly_thread_size",
    "monthly_thread_size_quantile_ribbon",
    "active_months_per_user_distribution",
    "active_subreddits_per_user_distribution",
    "user_subreddit_hhi_distribution",
    "repeat_user_share_by_subreddit",
    "lag_past_activity_future_activity",
    "lag_past_activity_future_activity_quantile_ribbon",
    "activity_transition_heatmap_t_to_t1",
    "lag_past_engagement_future_engagement",
    "lag_past_engagement_future_engagement_quantile_ribbon",
    "lag_past_activity_future_engagement",
    "lag_past_activity_future_engagement_quantile_ribbon",
    "user_engagement_lorenz_curve",
    "engagement_transition_heatmap_t_to_t1",
    "monthly_engagement_bin_prevalence",
    "subreddit_subset_selection_ranked",
    "subset_diag_size_vs_engagement",
    "subset_diag_size_vs_repeat_share",
    "subset_diag_engagement_vs_repeat_share",
)
SUBREDDIT_SUBSET_DIAGNOSTIC_FIGURE_NAMES: Tuple[str, ...] = (
    "subreddit_subset_selection_ranked",
    "subset_diag_size_vs_engagement",
    "subset_diag_size_vs_repeat_share",
    "subset_diag_engagement_vs_repeat_share",
)
LARGE_INTERMEDIATE_TABLE_NAMES: Tuple[str, ...] = (
    "per_user_activity",
    "per_user_active_months",
    "user_subreddit_hhi",
    "lag_relationships",
    "lag_relationships_valid_pairs",
    "thread_size_distribution",
    "user_activity_lorenz_curve",
    "user_engagement_lorenz_curve",
    "active_subreddits_per_user_distribution",
)
TAIL_AUDIT_TABLE_NAMES: Tuple[str, ...] = (
    "lag_past_activity_future_activity_decile10_tail_usermonths",
    "lag_past_activity_future_activity_decile10_tail_users_aggregated",
    "lag_past_activity_future_engagement_decile10_tail_usermonths",
    "lag_past_activity_future_engagement_decile10_tail_users_aggregated",
    "lag_past_engagement_future_engagement_decile10_tail_usermonths",
    "lag_past_engagement_future_engagement_decile10_tail_users_aggregated",
    "active_subreddits_tail_users",
)
SUBSET_THRESHOLD_COLUMNS: Tuple[str, ...] = (
    "min_monthly_active_users",
    "mean_monthly_active_users_observed",
    "max_monthly_active_users_observed",
    "n_months_meeting_active_user_threshold",
    "active_user_threshold_rule_passed",
    "full_period_active_user_threshold_rule_passed",
    "min_monthly_active_users_required",
    "has_any_nsfw_submission",
    "nsfw_submission_share",
    "nsfw_submission_share_threshold_for_subset",
    "nsfw_threshold_rule_failed",
    "subset_ineligibility_reason",
)
CSV_TABLE_NAMES: Tuple[str, ...] = (
    "monthly_activity",
    "subreddit_activity",
    "active_months_per_user_summary",
    "active_subreddits_per_user_summary",
    "active_subreddits_percentile_summary",
    "monthly_thread_size",
    "monthly_thread_size_quantile_ribbon_summary",
    "received_engagement_definition_audit",
    "monthly_engagement_rate_summary",
    "monthly_engagement_bin_prevalence",
    "monthly_engagement_bin_prevalence_audit",
    "lag_past_activity_future_activity_summary",
    "lag_past_activity_future_activity_quantile_ribbon_summary",
    "lag_past_engagement_future_engagement_summary",
    "lag_past_engagement_future_engagement_quantile_ribbon_summary",
    "lag_past_activity_future_engagement_summary",
    "lag_past_activity_future_engagement_quantile_ribbon_summary",
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
    "subreddit_repeat_user_share",
    "subreddit_subset_diagnostics",
    "subreddit_subset_selection_summary",
    "user_activity_lorenz_curve_plot",
    "user_engagement_lorenz_curve_plot",
    "thread_size_distribution_plot",
    "user_activity_distribution_total_plot",
    "user_activity_distribution_submissions_plot",
    "user_activity_distribution_comments_plot",
    "user_engagement_received_distribution_plot",
    "user_engagement_rate_distribution_plot",
    "user_subreddit_hhi_plot",
    "user_subreddit_hhi_summary",
    "subset_diag_labelled_candidates",
    "user_activity_lorenz_summary",
    "user_engagement_lorenz_summary",
)
SUBREDDIT_SUBSET_DIAGNOSTIC_TABLE_NAMES: Tuple[str, ...] = (
    "subreddit_subset_diagnostics",
    "subreddit_subset_selection_summary",
    "subset_diag_labelled_candidates",
)
EXPECTED_TABLE_NAMES: Tuple[str, ...] = tuple(
    dict.fromkeys(CSV_TABLE_NAMES + LARGE_INTERMEDIATE_TABLE_NAMES + TAIL_AUDIT_TABLE_NAMES)
)
COLORBLIND_SAFE_PALETTE: Tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)
SUBSET_DIAG_TOP_LABEL_COUNT = 10
CHECKPOINT_SCHEMA_VERSION = "06_0_descriptive_plots_stage2_received_engagement_v3_monthwise"
RECEIVED_ENGAGEMENT_DEFINITION_VERSION = "unique_other_user_child_comments_v1"
RECEIVED_ENGAGEMENT_SELF_ENGAGEMENT_SETTING = "exclude_self_engagement"
RECEIVED_ENGAGEMENT_NORMALISATION_RULES = {
    "comment_id": "trim varchar; empty to NULL; strip one leading t1_ where present",
    "submission_id": "trim varchar; empty to NULL; strip one leading t3_ where present",
    "link_id": "raw t3_/bare submission IDs and normalised t3-stripped IDs are both matched",
    "parent_id": "raw t1_/bare comment IDs and normalised t1-stripped IDs are both matched; t3_ parents are excluded from direct-comment replies",
}
MONTHLY_RECEIVED_ENGAGEMENT_CHECKPOINT_MANIFEST_NAME = "user_month_engagement_checkpoint_manifest.json"
MONTHLY_USER_MONTH_PANEL_CHECKPOINT_DIR = "user_month_panel_by_month"
MONTHLY_RECEIVED_ENGAGEMENT_AUDIT_CHECKPOINT_DIR = "received_engagement_definition_audit_by_month"
CHECKPOINT_MANIFEST_NAME = "checkpoint_manifest.json"
CHECKPOINT_PARTIAL_MANIFEST_NAME = "checkpoint_partial_manifest.json"
LAG_ROW_LEVEL_LARGE_INTERMEDIATE_NAMES: Tuple[str, ...] = (
    "lag_relationships",
    "lag_relationships_valid_pairs",
)
CORE_CHECKPOINT_TABLES: Tuple[str, ...] = (
    "user_month_panel",
    "received_engagement_definition_audit",
    "subreddit_month_summary",
)
COMPACT_LAG_CHECKPOINT_TABLES: Tuple[str, ...] = (
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
    "activity_transition_counts",
    "engagement_transition_counts",
)
CHECKPOINT_TABLES: Tuple[str, ...] = CORE_CHECKPOINT_TABLES + COMPACT_LAG_CHECKPOINT_TABLES
PLOT_SIZES = {
    # Report/thesis-friendly defaults: sized for inclusion in LaTeX PDFs,
    # where figures are usually scaled down from their standalone dimensions.
    "title": 14,
    "axis_label": 15,
    "tick": 13,
    "legend": 13,
    "annotation": 12,
    "direct_label": 13,
}


def csv_table_names(skip_subreddit_subset_diagnostics: bool = False) -> Tuple[str, ...]:
    """Return CSV table names for a run configuration."""
    if not skip_subreddit_subset_diagnostics:
        return CSV_TABLE_NAMES
    skipped = set(SUBREDDIT_SUBSET_DIAGNOSTIC_TABLE_NAMES)
    return tuple(name for name in CSV_TABLE_NAMES if name not in skipped)


def figure_names(skip_subreddit_subset_diagnostics: bool = False) -> Tuple[str, ...]:
    """Return figure names for a run configuration."""
    if not skip_subreddit_subset_diagnostics:
        return FIGURE_ORDER
    skipped = set(SUBREDDIT_SUBSET_DIAGNOSTIC_FIGURE_NAMES)
    return tuple(name for name in FIGURE_ORDER if name not in skipped)


# ----------------------------- Data classes ------------------------------


@dataclass(frozen=True)
class DuckDBConfig:
    """DuckDB runtime configuration."""

    memory_limit: str = "16GB"
    threads: int = 4
    temp_dir: str = "duckdb_tmp"
    max_temp_directory_size: str = ""


@dataclass(frozen=True)
class PlotConfig:
    """Configuration for descriptive plot generation."""

    in_root: str
    out_root: str
    year: str
    only_months: Tuple[str, ...]
    limit_months: int
    author_col: str
    overwrite: bool
    replot_only: bool
    smoke: bool
    smoke_max_rows: int
    publication_style: bool
    write_large_intermediates: bool
    checkpoint_dir: str
    reuse_checkpoints: bool
    top_n_subreddits_subset_diag: int
    min_monthly_active_users_for_subset: int
    nsfw_submission_share_threshold_for_subset: float
    run_id: str
    skip_subreddit_subset_diagnostics: bool = False


@dataclass(frozen=True)
class MonthFiles:
    """Paths to monthly comments/submissions core files."""

    comments_core: Optional[str] = None
    submissions_core: Optional[str] = None


Inventory = Dict[str, MonthFiles]
