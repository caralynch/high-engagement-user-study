#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Cara Lynch
# See the LICENSE file for details.
"""
Preprocessing step 6.0: generate key descriptive plots for cleaned Reddit Parquet outputs.

This script produces a curated set of publication-grade figures for the Reddit 2021
user-engagement study. It is designed to complement `05_descriptive_stats.py` and
follows the same general conventions: DuckDB over Parquet for scalability,
reproducible outputs, explicit logging, and smoke-test support.

Implementation layout
---------------------
The executable CLI/orchestrator remains in this file; implementation helpers
live in the `Preprocessing/descriptive_plots/` package.

Design goals
------------
- Scalable: uses DuckDB SQL over Parquet and materialises only a small number of
  reusable temporary tables. The default lag path materialises compact
  aggregate/count intermediates rather than full-year row-level lag tables, and
  the default subreddit-month path retains compact aggregate summaries rather
  than row-level user-subreddit-month data.
- Reproducible: writes run metadata, resolved config, machine-readable summary
  tables, and deterministic figure names.
- Pipeline-consistent: CLI, logging, and DuckDB tuning follow the style of the
  descriptive-statistics script.
- Thesis-friendly: writes both PNG and SVG versions of each figure, plus key
  summary CSV files used to generate them.
- Focused: produces a small, defensible figure set rather than an exhaustive
  dashboard.

What this script produces
-------------------------
Core figures:
- monthly_platform_activity
    Two-panel monthly summary: posts/comments on the top panel and active
    users/active subreddits on the bottom panel.
- subreddit_rank_size_submissions
    Log-log rank-size plot for ordinary subreddits by submissions, ordered from
    largest to smallest and plotted as activity count against subreddit rank.
- subreddit_rank_size_comments
    Log-log rank-size plot for ordinary subreddits by comments, ordered from
    largest to smallest and plotted as activity count against subreddit rank.
- user_activity_distribution_total
    Survival distribution plot for per-user total activity, shown on the
    positive support because log-x excludes zero values.
- user_activity_distribution_submissions
    Survival distribution plot for per-user submissions, shown on the positive
    support because log-x excludes zero values.
- user_activity_distribution_comments
    Survival distribution plot for per-user comments, shown on the positive
    support because log-x excludes zero values.
- user_activity_lorenz_curve
    Lorenz curve for the cumulative share of users versus the cumulative share of
    full-period total activity, annotated with compact concentration summaries.
- user_engagement_received_distribution
    Survival distribution plot for per-user received engagement, shown on the
    positive support because log-x excludes zero values.
- received_engagement_per_activity_distribution
    Compact survival distribution for received engagement per authored activity,
    stratified by minimum user-month activity thresholds.
- monthly_received_engagement_per_activity_iqr
    Monthly median/IQR ribbon for received engagement per authored activity.
- thread_size_distribution
    Thread-size survival distribution. The underlying table includes
    zero-comment submissions, but the plotted survival curve is shown on the
    positive support because log-x excludes zero values.
- monthly_thread_size
    Monthly mean/median thread size with direct endpoint labels.
- monthly_thread_size_quantile_ribbon
    Monthly median thread size with IQR ribbon.
- active_months_per_user_distribution
    Distribution of number of active months per user across the year.
- active_subreddits_per_user_distribution
    Distribution of the number of distinct subreddits each user was active in
    over the selected period.
- user_subreddit_hhi_distribution
    Two-panel distribution of user-level subreddit concentration (HHI), with
    the point mass at `HHI = 1` separated from the distribution over `HHI < 1`
    for interpretability.
- repeat_user_share_by_subreddit
    Subreddit-level density summary of mean monthly active users versus the mean
    monthly share of active users with >=2 activities; each hex/bin corresponds
    to ordinary subreddits only and both axes are annual means over
    subreddit-month summaries.
- lag_past_activity_future_activity
    Decile plot: activity at t versus mean and median activity at t+1.
- lag_past_activity_future_activity_quantile_ribbon
    Decile plot: activity at t versus median activity at t+1 with IQR ribbon.
- activity_transition_heatmap_t_to_t1
    Heatmap of activity-decile transitions from month t to month t+1.
- lag_past_engagement_future_engagement
    Decile plot: engagement received at t versus mean and median engagement
    received at t+1.
- lag_past_engagement_future_engagement_quantile_ribbon
    Decile plot: engagement received at t versus median engagement received at
    t+1 with IQR ribbon.
- lag_past_activity_future_engagement
    Decile plot: activity at t versus mean and median engagement received at
    t+1.
- lag_past_activity_future_engagement_quantile_ribbon
    Decile plot: activity at t versus median engagement received at t+1 with
    IQR ribbon.
- user_engagement_lorenz_curve
    Lorenz curve for the cumulative share of users versus the cumulative share of
    total received engagement, annotated with compact concentration summaries.
- engagement_transition_heatmap_t_to_t1
    Heatmap of received-engagement decile transitions from month t to month t+1.
- monthly_engagement_bin_prevalence
    Monthly prevalence of received-engagement bins among active users, with the
    legend placed outside the plotting area on the right.
- subreddit_subset_selection_ranked
    Ranked subreddit subset-selection summary based on transparent candidate
    scores computed only within the eligible candidate pool.

Subset-selection diagnostics:
- subset_diag_size_vs_engagement
- subset_diag_size_vs_repeat_share
- subset_diag_engagement_vs_repeat_share
    Selection-oriented scatter summaries with one point per eligible subreddit,
    shortlisted candidates highlighted, and a descriptive trend line for visual
    guidance only.

For subreddit-level diagnostics and candidate selection, the script excludes
user-profile subreddits whose names begin with `u_`, because they are not
comparable to ordinary topic-based communities.
The subreddit rank-size plots and their underlying `subreddit_activity` table
also exclude these `u_` subreddits so the figures represent ordinary topic-based
communities only. `subreddit_activity` no longer reports an annual distinct
user count; exact annual subreddit-user distincts are not available from the
compact checkpointed summaries without the memory-heavy full-year distinct
aggregation. It instead reports clearly named monthly active-user summaries:
`mean_monthly_active_users`, `min_monthly_active_users`, and
`max_monthly_active_users`.
For subset selection, candidate scoring is further restricted to ordinary
subreddits that are observed in every selected month, meet the configured
minimum monthly active-user threshold in every selected subreddit-month, and,
when submission `over_18` metadata are available, do not meet the configured
subreddit-level NSFW submission-share exclusion threshold. The NSFW criterion is
based on the subreddit-level share of submissions marked `over_18`, not on the
presence of any single NSFW-marked submission. `has_any_nsfw_submission` is
retained as an audit field only. The default exclusion threshold is
`nsfw_submission_share >= 0.3`. This rule applies only to subset-selection
eligibility and diagnostics; global descriptive plots are not filtered by it.
Heuristic live/contest/bot name indicators are retained as audit flags only and
are not automatic exclusion rules.

Table outputs:
- Small summary and plot-input tables are written as human-readable CSV files.
  This includes compact plot tables for Lorenz curves, thread-size survival,
  per-user activity survival, received-engagement survival, received-engagement
  rate summaries, and the HHI histogram.
- Tail audit tables are written as compressed `.csv.gz` files.
- The default full run uses compact aggregate subreddit-month summaries plus
  compact aggregate lag and transition intermediates. Row-level
  user-subreddit-month data are not retained by default.
- Large row-level or intermediate tables are not written by default. Passing
  `--write-large-intermediates` writes only the large artefacts that can be
  exported safely without reconstructing full-year row-level lag tables.
- Full Lorenz tables and the full thread-size vector are treated as optional
  large intermediates; compact plot-input CSVs are written by default and are
  sufficient for figure regeneration.
- Internal checkpoint Parquet files may be written under the checkpoint
  directory to support resume/reuse. `checkpoint_manifest.json` is written at
  run start with `run_status="partial"` and updated after each checkpoint table
  is written, so a rerun can safely reuse completed checkpoints after a failed
  run. These are not publication outputs.

Definitions
-----------
- Activity:
    One authored submission or one authored comment.
- Active user-month:
    A user with at least one authored submission/comment in a month.
- Subreddit-month active users for subset selection:
    The number of distinct valid author identifiers that appear in
    `submissions_core_userlevel` or `comments_core_userlevel` for a subreddit
    in a month. This uses the resolved author identifier column and therefore
    excludes deleted authors and NULL author IDs consistently with the
    user-level pipeline. `active_user_threshold_rule_passed` is the active-user
    threshold component only; `full_period_active_user_threshold_rule_passed`
    also requires observation in every selected month. A threshold of 0
    reproduces the previous full-period plus NSFW-threshold subset-eligibility
    rule.
- Subreddit-level NSFW exclusion for subset selection:
    When submission `over_18` metadata are available, the subset-selection NSFW
    rule excludes subreddits with `nsfw_submission_share >= 0.3` by default, or
    the value supplied through
    `--nsfw-submission-share-threshold-for-subset`. This criterion is based on
    the subreddit-level share of submissions marked `over_18`, not on the
    presence of any single NSFW-marked submission. `has_any_nsfw_submission` is
    retained as an audit field only. This rule applies only to
    subset-selection eligibility and diagnostics.
- Active months per user:
    The number of distinct months in which a non-deleted user authored at least
    one submission or comment during the selected period. This is not a
    consecutiveness or adjacent-month measure. User-level summaries already
    exclude deleted authors and NULL author IDs via the user-level pipeline.
- Received engagement (event-month definition):
    In month t, the unique count of other-user child comments authored in month
    t and received by content authored by the user:
      (i) other-user comments in the user's submission threads;
      (ii) other-user direct replies to the user's comments.
    Self-engagement is excluded: a submission author's comments on their own
    submission and a comment author's replies to their own comment do not count.
    Submission-thread and direct-reply component counts are retained separately,
    and the main `n_total_engagement_received` total de-duplicates child
    comments that qualify through both routes for the same recipient user-month.
    This is an event-month measure of engagement receipt and is therefore suitable
    for month-to-month lag summaries.
- User subreddit concentration (HHI):
    Sum over subreddits of squared activity shares for each user over the selected
    period. Higher values indicate more concentrated participation. HHI is
    bounded on [0, 1], and `HHI = 1` means the user was active in exactly one
    subreddit over the selected period. Because the distribution contains a
    large point mass at 1, the figure separates that degenerate mass from the
    remainder of the distribution to reveal variation among multi-subreddit
    users.
- Repeat-user share by subreddit:
    For each subreddit-month, the share of active users with activity count >= 2
    or >= 5 in that subreddit-month. The script reports annual means of these
    monthly shares by subreddit. The `repeat_user_share_by_subreddit` figure is
    a subreddit-level density plot built from those annual means, with `u_`
    subreddits excluded via the ordinary-subreddit filters already used for the
    underlying table.
- Subreddit rank-size plots:
    The subreddit rank-size figures use the ordinary-subreddit
    `subreddit_activity` table, which excludes user-profile subreddits whose
    names begin with `u_`. For each plotted quantity, subreddits are sorted from
    largest to smallest by the relevant activity count, assigned rank 1 for the
    largest subreddit, and then plotted as activity count against rank on log-log
    axes. The table's previous `n_users` annual distinct-user column is not
    approximated from monthly summaries; use `mean_monthly_active_users`,
    `min_monthly_active_users`, and `max_monthly_active_users` for the compact
    monthly active-user definition.
- Lorenz curve:
    Cumulative share of users versus cumulative share of total received engagement,
    after ranking users by full-period total received engagement.
- Activity Lorenz curve:
    Cumulative share of users versus cumulative share of full-period total
    activity, after ranking users by summed activity over the selected period.
- Lorenz concentration summaries:
    The activity and engagement Lorenz figures are annotated with compact
    concentration summaries derived from the same user-total distributions used
    to build the curves, including the Gini coefficient and top-share measures.
- Activity transition matrix:
    For active user-months at t, pooled across all usable month pairs where both
    t and t+1 are present in the selected data window, transitions from
    month-specific activity deciles at t to destination states at t+1. Source
    months are therefore excluded when their adjacent next month is absent from
    the selected data. Destination states comprise an explicit `Inactive` state
    for users with no matched t+1 row plus deciles 1-10 computed from the full
    month-t+1 `user_month_panel` population for that month rather than only the
    carried-over cohort from t.
- Engagement transition matrix:
    For active user-months at t, transitions from month-specific received-
    engagement deciles at t to pooled destination states at t+1 across all
    usable month pairs where both t and t+1 are present in the selected data
    window. Source months are therefore excluded when their adjacent next month
    is absent from the selected data. Destination states comprise an explicit
    `Inactive` state for users with no matched t+1 row plus deciles 1-10
    computed from the full month-t+1 `user_month_panel` population for that
    month rather than only the carried-over cohort from t.
- Active subreddits per user:
    The number of distinct subreddits in which a user authored at least one
    submission or comment over the selected period. The
    `active_subreddits_per_user_distribution` figure is plotted from the
    machine-readable `active_subreddits_per_user_summary.csv` table rather than
    recomputing raw counts inside the plotting block.
- Active-subreddits summary tables:
    `active_subreddits_per_user_summary.csv` provides exact, cumulative, and
    survival shares over distinct subreddits per user;
    `active_subreddits_percentile_summary.csv` provides a compact descriptive
    summary of that same distribution; and `active_subreddits_tail_users.csv.gz`
    is a capped audit table for extreme multi-subreddit users. These
    user-level active-subreddit outputs are based on `user_subreddit_hhi`,
    which is built from `submissions_core_userlevel` and
    `comments_core_userlevel`, so deleted authors and NULL author IDs are
    already excluded.
- Lag summary plots:
    The `t -> t+1` lag-summary plots use compact decile summary CSVs derived
    from month-pair compact aggregate counts over valid adjacent month pairs.
    This excludes right-censored source months for which the actual adjacent
    next month is absent from the selected data window. The former full-year
    row-level `lag_relationships` and `lag_relationships_valid_pairs` tables
    are no longer materialised by default; they are optional debug/large
    intermediates only if implemented through a safe chunked path. Stage 1
    skips those two large exports with a warning rather than recreating them.
    Users active at t but absent at t+1 are retained in these default
    summaries and contribute zero outcome at t+1 via the existing left-join
    construction, so the lag plots represent an unconditional source-cohort
    view rather than a conditional-on-return view. The activity- and
    engagement-persistence summaries, plus the activity-to-future-engagement
    summaries, are materialised as dedicated CSV tables so the mean/median line
    plots and quantile-ribbon figures can be reproduced directly in replot-only
    mode. Additional capped decile-10 tail audit summaries identify extreme
    user-months and users driving the upper tail of these lag relationships,
    including simple string-based account flags for likely AutoModerator and bot
    accounts; they are audit samples rather than exhaustive population tables.
    The main lag-summary line plots connect ordered deciles with lighter dashed
    lines and point markers and intentionally omit error bars because they are
    descriptive rather than inferential; distributional spread is shown via the
    companion quantile-ribbon figures and summary tables.
- Monthly thread-size quantile-ribbon summary:
    `monthly_thread_size_quantile_ribbon_summary.csv` provides one row per month
    with median and interquartile-range summaries for thread size, so the
    companion quantile-ribbon figure can be reproduced directly in replot-only
    mode.
- Positive-support survival plots:
    The thread-size, per-user total-activity, per-user submission-count,
    per-user comment-count, per-user received-engagement, and received-
    engagement-rate survival plots are drawn on the positive support of the
    plotted quantity because the log-scaled x-axis cannot display zeros. For
    user-level survival plots, the y-axis
    denominator still uses the full non-null user population, including
    zero-valued users, and zero-value shares are reported as figure annotations
    where applicable. For thread size, the underlying
    `thread_size_distribution_plot.csv` table still counts zero-comment
    submissions in the denominator, and the figure annotates their share
    directly.
- Monthly engagement-bin prevalence audit:
    `monthly_engagement_bin_prevalence_audit.csv` provides month-level active-
    user counts, engagement-bin counts/shares, and distribution summaries for
    received engagement and activity. It is intended as an audit table for
    inspecting month-level composition changes, including the apparent May 2021
    anomaly.
- Received-engagement definition and rate audits:
    `received_engagement_definition_audit.csv` compares monthly pre/post
    self-exclusion component counts, unique totals after de-duplication, overlap,
    self-engagement excluded, and distinct-engager audit counts.
    `monthly_engagement_rate_summary.csv` reports denominator-safe received-
    engagement rates stratified by minimum activity thresholds. Per-activity
    uses the unique other-user received-engagement total divided by total
    authored activity; per-submission uses other-user comments received on
    authored submissions divided by authored submissions; per-comment uses
    other-user direct replies received to authored comments divided by authored
    comments.
    Older engagement-derived descriptive outputs produced before this
    other-user/de-duplicated contract are superseded until rerun.
- Subreddit subset-selection summary:
    Ranked subreddit summary combining activity, engagement richness, repeat-user
    share, and stability into a transparent candidate score and threshold flag.
    User-profile subreddits whose names begin with `u_` are excluded from these
    subreddit-level diagnostics and selection outputs because they are not
    comparable to ordinary topic-based communities. Candidate scores are
    computed only within the eligible candidate pool: ordinary subreddits that
    are observed in every selected month, meet the configured minimum monthly
    active-user threshold in every selected subreddit-month, and, when
    submission `over_18` metadata are available, have an NSFW submission share
    below the configured subset-selection exclusion threshold. The active-user
    threshold uses distinct valid authors from the user-level cleaned views, not
    raw content rows. `has_any_nsfw_submission` is retained as an audit-only
    field and is not an exclusion rule.
    Heuristic live/contest/bot name indicators are retained as audit flags only
    and are not automatic exclusion rules.
    If `over_18` metadata are unavailable in the cleaned submissions input, the
    script retains NULL NSFW audit fields, logs that NSFW exclusion could not be
    applied, and does not exclude subreddits on that basis.
- Subset-selection diagnostic plots:
    The three subset-selection diagnostic plots use one point per eligible
    subreddit from `subreddit_subset_selection_summary`. Eligible here means:
    ordinary subreddit, observed in every selected month, meeting the configured
    minimum monthly active-user threshold in every selected month, and below the
    configured NSFW submission-share threshold when submission `over_18`
    metadata are available. Shortlisted
    `candidate_flag = TRUE` subreddits are highlighted, only a small number of
    top-ranked shortlisted candidates are directly labelled, and any trend line
    is descriptive only rather than inferential.
    `subset_diag_labelled_candidates.csv` provides the machine-readable audit
    list for those directly labelled shortlisted candidates.

Deleted-author handling
-----------------------
Content-level counts retain all rows. User-level analysis excludes rows flagged
`is_deleted_author = TRUE` and rows where the resolved author column is NULL.
This matches the descriptive-statistics script.

Inputs (expected layout)
------------------------
<in-root>/
  comments/comments_core_YYYY-MM.parquet
  submissions/submissions_core_YYYY-MM.parquet

Outputs
-------
Writes outputs under:
<out-root>/
  audit/descriptive_plots_run_<run-id>.json
  audit/descriptive_plots_config_<run-id>.json
  audit/descriptive_plots_finished_<run-id>.json
  tables/<name>.csv
  tables/<tail-audit-name>.csv.gz
  tables/<large-intermediate-name>.parquet   # safe optional exports only
  figures/<name>.png
  figures/<name>.svg
  figures/<name>.pdf

Internal resume artifacts are written under:
<checkpoint-dir>/
  checkpoint_manifest.json
  user_month_engagement_checkpoint_manifest.json
  user_month_panel_by_month/user_month_panel_YYYY-MM.parquet
  received_engagement_definition_audit_by_month/received_engagement_definition_audit_YYYY-MM.parquet
  subreddit_month_summary.parquet
  <compact-lag-or-transition-checkpoint>.parquet

Small summary and compact plot-input tables are written as CSV. Tail audit
tables are written as compressed `.csv.gz`. Large row-level/intermediate tables
are not written by default; pass `--write-large-intermediates` to write them as
compressed Parquet exported directly from DuckDB, without first materialising
them as pandas DataFrames. In the compact lag implementation, the historical
row-level `lag_relationships` and `lag_relationships_valid_pairs` exports are
disabled unless a safe chunked export path is implemented. Checkpoint Parquet
files are internal resume artifacts and should not be treated as publication
tables. All figures are written as PNG, SVG, and PDF. If --publication-style is
enabled, the script writes the same outputs but suppresses standalone plot
titles to better fit manuscript-ready layouts.

Replot-only mode
----------------
If --replot-only is enabled, the script skips DuckDB and raw Parquet processing,
loads previously written compact CSV tables from <out-root>/tables/, and
regenerates the figures only. It does not require large row-level intermediates,
optional large Parquet files, or tail audit `.csv.gz` files. The full Lorenz
tables and full thread-size vector are therefore unnecessary for replot-only;
the compact Lorenz and thread-size plot-input CSVs are used instead.
If older `subreddit_subset_selection_summary.csv` outputs lack current active-
user or NSFW threshold audit columns, replot-only mode warns but still
regenerates figures; those older subset-selection outputs should be regenerated
before using threshold-based NSFW subset eligibility. If
`--skip-subreddit-subset-diagnostics` is also supplied, replot-only mode does
not require or regenerate the subset-selection CSV tables or figures.

Important note on smoke mode
----------------------------
If --smoke-max-rows is used, summaries and figures are approximate and suitable
only for plumbing / CI checks.

Command-line arguments
----------------------
Required:
- --out-root PATH
    Output root for audit files, summary tables, optional large intermediates,
    and figures.

Input and temporal scope:
- --in-root PATH
    Input root containing cleaned monthly Parquet outputs. Expected subfolders
    are `comments/` and `submissions/`. Not required with `--replot-only`.
- --year YYYY
    Year to process. Defaults to `2021`.
- --only-months YYYY-MM[,YYYY-MM,...]
    Optional comma-separated whitelist of months to process, for example
    `2021-01,2021-06`.
- --limit-months N
    Optional cap on the number of discovered months after applying
    `--only-months`. Use `0` for no cap.

User identifier handling:
- --author-col {auto,author_pid,author}
    Author identifier column used for user-level summaries. `auto` prefers
    `author_pid` when available and otherwise falls back to `author`.

Execution modes and output control:
- --overwrite
    Overwrite existing outputs where applicable.
- --replot-only
    Skip DuckDB/raw Parquet processing and regenerate figures from previously
    written compact CSV tables in `<out-root>/tables/`.
- --write-large-intermediates
    Write large row-level/intermediate artefacts as compressed Parquet. Disabled
    by default to reduce disk usage. Full-year row-level lag exports are
    skipped in the Stage 1 compact-lag implementation unless a safe chunked path
    is implemented.
- --checkpoint-dir PATH
    Directory for internal resume checkpoints. Defaults to
    `<out-root>/checkpoints`.
- --reuse-checkpoints
    Reuse metadata-compatible checkpoint Parquet files when present. Checkpoint
    metadata include schema version, selected months, year, resolved author
    column, smoke settings, input root, and subset-selection thresholds. The
    script can reuse tables marked complete in a `checkpoint_manifest.json`
    whose `run_status` is `partial`, `started`, or `complete`. It can also
    validate older failed runs from adjacent start-of-run audit config JSON when
    the final manifest was never written. If metadata do not match, the script
    logs the mismatch and rebuilds.
- --smoke
    Enable smoke mode for quick plumbing checks.
- --smoke-max-rows N
    If greater than zero and `--smoke` is enabled, cap rows read per file or
    relation. Smoke-mode outputs are approximate and not suitable for reporting.
- --publication-style
    Suppress standalone plot titles and use thesis/manuscript-oriented plot
    styling.
- --skip-subreddit-subset-diagnostics
    Skip subset-selection diagnostic/ranking tables and figures. Core
    descriptive, received-engagement, lag, transition, thread-size, and general
    subreddit outputs are still written.

Plot-selection diagnostics:
- --top-n-subreddits-subset-diag N
    Number of top-ranked subreddits to show in subset-selection ranked
    diagnostics.
- --min-monthly-active-users-for-subset N
    Minimum number of distinct valid active users required in each selected
    subreddit-month for subset-selection eligibility. This affects only the
    subset-selection candidate pool and diagnostics. Use 0 to reproduce the
    previous full-period plus NSFW-threshold eligibility rule.
- --nsfw-submission-share-threshold-for-subset VALUE
    Subreddit-level NSFW exclusion threshold for subset-selection eligibility.
    Subreddits are excluded when `nsfw_submission_share >= VALUE`; the default
    is `0.3`. This affects only the subset-selection candidate pool and
    diagnostics.

DuckDB and runtime configuration:
- --duckdb-memory-limit VALUE
    DuckDB memory limit, for example `4GB`, `16GB`, or `32GB`. Defaults to the
    `DUCKDB_MEMORY_LIMIT` environment variable if set, otherwise `16GB`.
- --duckdb-threads N
    Number of DuckDB threads. Defaults to the `DUCKDB_THREADS` environment
    variable if set, otherwise `4`.
- --duckdb-temp-dir PATH
    DuckDB temporary directory. Prefer node-local scratch on HPC. Defaults to
    the `DUCKDB_TEMP_DIR` environment variable if set, otherwise `duckdb_tmp`.
- --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
    Console logging verbosity. Defaults to `INFO`.

Examples
--------
Full 2021 figure generation:
  python 06_0_descriptive_plots.py \
    --in-root parquet_clean \
    --out-root descriptive_plots_2021 \
    --year 2021 \
    --nsfw-submission-share-threshold-for-subset 0.3 \
    --duckdb-memory-limit 32GB \
    --duckdb-threads 1 \
    --duckdb-temp-dir "$TMPDIR" \
    --checkpoint-dir descriptive_plots_2021/checkpoints \
    --reuse-checkpoints \
    --log-level INFO

Two-month smoke test:
  python 06_0_descriptive_plots.py \
    --in-root parquet_clean \
    --out-root descriptive_plots_smoke \
    --year 2021 \
    --limit-months 2 \
    --smoke \
    --smoke-max-rows 200000 \
    --checkpoint-dir descriptive_plots_smoke/checkpoints \
    --log-level INFO
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from dataclasses import asdict
from typing import Optional, Sequence

import duckdb
import pandas as pd

from Preprocessing.descriptive_plots.checkpoints import (
    build_checkpoint_metadata,
    checkpoint_reuse_status,
    write_checkpoint_manifest,
    write_checkpoint_started_manifest,
)
from Preprocessing.descriptive_plots.config import DuckDBConfig, PlotConfig, figure_names
from Preprocessing.descriptive_plots.derived_tables import (
    build_derived_tables,
    discover_core_files,
    filter_month_inventory,
    require_complete_inventory,
)
from Preprocessing.descriptive_plots.duckdb_helpers import (
    build_base_views,
    build_user_level_views,
    configure_duckdb,
    get_table_columns,
    resolve_author_col,
)
from Preprocessing.descriptive_plots.fetch_tables import fetch_tables, load_csv_tables
from Preprocessing.descriptive_plots.lag_outputs import build_compact_lag_and_transition_outputs
from Preprocessing.descriptive_plots.output_io import (
    check_output_collisions,
    ensure_dir,
    parse_only_months,
    safe_json_dump,
    utc_now_iso,
    utc_run_id,
    write_large_intermediate_tables_from_duckdb,
    write_run_metadata,
    write_tables,
)
from Preprocessing.descriptive_plots.plotting import generate_figures


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


logger = logging.getLogger(__name__)


# ----------------------------- CLI --------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI parser for this script."""
    p = argparse.ArgumentParser(
        description="Generate key descriptive plots for cleaned Reddit monthly Parquet files (DuckDB).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in-root", default="", help="Input root containing cleaned Parquet outputs. Not required with --replot-only.")
    p.add_argument("--out-root", required=True, help="Output root for figures and summary tables.")
    p.add_argument("--year", default="2021", help="Year to process (YYYY).")
    p.add_argument("--only-months", default="", help="Comma-separated subset like 2021-01,2021-06.")
    p.add_argument("--limit-months", type=int, default=0, help="Limit number of discovered months after any whitelist is applied (0 = no limit).")
    p.add_argument("--author-col", default="auto", choices=["auto", "author_pid", "author"], help="Author identifier column to use for user-level summaries.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs if present.")
    p.add_argument("--replot-only", action="store_true", help="Skip DuckDB/raw Parquet processing and rebuild figures from previously saved CSV tables in <out-root>/tables/.")
    p.add_argument(
        "--write-large-intermediates",
        action="store_true",
        help="Write large row-level/intermediate artefacts as compressed Parquet. Disabled by default to reduce disk usage.",
    )
    p.add_argument(
        "--checkpoint-dir",
        default="",
        help="Directory for internal resume checkpoints. Defaults to <out-root>/checkpoints.",
    )
    p.add_argument(
        "--reuse-checkpoints",
        action="store_true",
        help="Reuse metadata-compatible checkpoint Parquet files when present; mismatched checkpoints are rebuilt.",
    )
    p.add_argument("--smoke", action="store_true", help="Enable smoke mode (quick plumbing run).")
    p.add_argument("--smoke-max-rows", type=int, default=0, help="If > 0 and --smoke, limit rows read per file/relation. Use only for plumbing checks.")
    p.add_argument(
        "--publication-style",
        action="store_true",
        help="Use publication-style plots: remove titles from standalone plots; reserve short left-aligned labels for future multi-panel figures.",
    )
    p.add_argument("--top-n-subreddits-subset-diag", type=int, default=50, help="Number of top-ranked subreddits to show in subset-selection ranked diagnostics.")
    p.add_argument(
        "--skip-subreddit-subset-diagnostics",
        action="store_true",
        help="Skip subreddit subset-selection diagnostic tables and figures while keeping general subreddit descriptive outputs.",
    )
    p.add_argument(
        "--min-monthly-active-users-for-subset",
        type=int,
        default=100,
        help=(
            "Minimum number of distinct valid active users required in each selected subreddit-month "
            "for subset-selection eligibility. Applies only to subset-selection candidate scoring and diagnostics."
        ),
    )
    p.add_argument(
        "--nsfw-submission-share-threshold-for-subset",
        type=float,
        default=0.3,
        help=(
            "Exclude a subreddit from subset-selection eligibility when the share of submissions marked "
            "over_18 is greater than or equal to this threshold. Applies only to subset-selection eligibility "
            "and diagnostics."
        ),
    )
    p.add_argument("--duckdb-memory-limit", default=os.environ.get("DUCKDB_MEMORY_LIMIT", "16GB"), help="DuckDB memory limit (e.g. 4GB, 16GB, 32GB).")
    p.add_argument("--duckdb-threads", type=int, default=int(os.environ.get("DUCKDB_THREADS", "4")), help="DuckDB threads.")
    p.add_argument("--duckdb-temp-dir", default=os.environ.get("DUCKDB_TEMP_DIR", "duckdb_tmp"), help="DuckDB temp_directory (prefer node-local scratch on HPC).")
    p.add_argument(
        "--duckdb-max-temp-directory-size",
        default=os.environ.get("DUCKDB_MAX_TEMP_DIRECTORY_SIZE", ""),
        help="Optional DuckDB max_temp_directory_size (e.g. 500GB). Leave unset to use DuckDB/default filesystem limits.",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging verbosity.")
    return p


# ----------------------------- Main -------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    if int(args.min_monthly_active_users_for_subset) < 0:
        parser.error("--min-monthly-active-users-for-subset must be >= 0.")
    nsfw_threshold = float(args.nsfw_submission_share_threshold_for_subset)
    if not math.isfinite(nsfw_threshold) or nsfw_threshold < 0.0 or nsfw_threshold > 1.0:
        parser.error("--nsfw-submission-share-threshold-for-subset must be between 0 and 1 inclusive.")

    if args.smoke and not args.only_months and args.limit_months == 0:
        args.limit_months = 2

    checkpoint_dir = args.checkpoint_dir or os.path.join(args.out_root, "checkpoints")

    cfg = PlotConfig(
        in_root=args.in_root,
        out_root=args.out_root,
        year=str(args.year),
        only_months=parse_only_months(args.only_months),
        limit_months=int(args.limit_months),
        author_col=str(args.author_col),
        overwrite=bool(args.overwrite),
        replot_only=bool(args.replot_only),
        smoke=bool(args.smoke),
        smoke_max_rows=int(args.smoke_max_rows) if args.smoke else 0,
        publication_style=bool(args.publication_style),
        write_large_intermediates=bool(args.write_large_intermediates),
        checkpoint_dir=checkpoint_dir,
        reuse_checkpoints=bool(args.reuse_checkpoints),
        top_n_subreddits_subset_diag=int(args.top_n_subreddits_subset_diag),
        min_monthly_active_users_for_subset=int(args.min_monthly_active_users_for_subset),
        nsfw_submission_share_threshold_for_subset=nsfw_threshold,
        run_id=utc_run_id("descriptive_plots", str(args.year)),
        skip_subreddit_subset_diagnostics=bool(args.skip_subreddit_subset_diagnostics),
    )
    duck = DuckDBConfig(
        memory_limit=args.duckdb_memory_limit,
        threads=int(args.duckdb_threads),
        temp_dir=args.duckdb_temp_dir,
        max_temp_directory_size=str(args.duckdb_max_temp_directory_size or ""),
    )

    ensure_dir(cfg.out_root)
    ensure_dir(os.path.join(cfg.out_root, "audit"))
    ensure_dir(os.path.join(cfg.out_root, "tables"))
    ensure_dir(os.path.join(cfg.out_root, "figures"))
    ensure_dir(cfg.checkpoint_dir)

    config_json = os.path.join(cfg.out_root, "audit", f"descriptive_plots_config_{cfg.run_id}.json")
    safe_json_dump({"config": asdict(cfg), "duckdb": asdict(duck)}, config_json)
    logger.info(
        "Subset-selection monthly active-user threshold: %s valid active users per selected subreddit-month",
        cfg.min_monthly_active_users_for_subset,
    )
    logger.info(
        "Subset-selection NSFW submission-share exclusion threshold: %.3f",
        cfg.nsfw_submission_share_threshold_for_subset,
    )
    logger.info("Checkpoint directory: %s", cfg.checkpoint_dir)
    logger.info("Reuse checkpoints: %s", cfg.reuse_checkpoints)
    logger.info("Skip subreddit subset-selection diagnostics: %s", cfg.skip_subreddit_subset_diagnostics)
    logger.info(
        "DuckDB max temp directory size: %s",
        duck.max_temp_directory_size if duck.max_temp_directory_size else "DuckDB default",
    )

    planned_figures = list(figure_names(cfg.skip_subreddit_subset_diagnostics))

    if cfg.replot_only:
        logger.info("Replot-only mode enabled: loading previously saved CSV tables.")
        if not cfg.out_root:
            raise RuntimeError("--out-root is required for --replot-only mode.")
        tables = load_csv_tables(
            cfg.out_root,
            skip_subreddit_subset_diagnostics=cfg.skip_subreddit_subset_diagnostics,
        )
        months = sorted(pd.to_datetime(tables["monthly_activity"]["month"] + "-01").dt.strftime("%Y-%m").tolist()) if len(tables.get("monthly_activity", [])) else []
        author_col = "unresolved_replot_only"
        generate_figures(
            cfg.out_root,
            tables,
            cfg.top_n_subreddits_subset_diag,
            publication_style=cfg.publication_style,
            skip_subreddit_subset_diagnostics=cfg.skip_subreddit_subset_diagnostics,
        )
        run_meta = write_run_metadata(cfg.out_root, cfg, duck, script_name=os.path.basename(__file__))
        finished_json = os.path.join(cfg.out_root, "audit", f"descriptive_plots_finished_{cfg.run_id}.json")
        safe_json_dump(
            {
                "run_id": cfg.run_id,
                "timestamp_utc": utc_now_iso(),
                "mode": "replot_only",
                "skip_subreddit_subset_diagnostics": cfg.skip_subreddit_subset_diagnostics,
                "months": months,
                "author_col": author_col,
                "tables_loaded": sorted(tables.keys()),
                "figures_written": planned_figures,
                "config_json": config_json,
                "run_metadata_json": run_meta,
            },
            finished_json,
        )
        logger.info("Rebuilt %d figures from saved CSV tables.", len(planned_figures))
        logger.info("Done.")
        return 0

    if not cfg.in_root:
        raise RuntimeError("--in-root is required unless --replot-only is enabled.")

    check_output_collisions(
        cfg.out_root,
        cfg.overwrite,
        cfg.write_large_intermediates,
        skip_subreddit_subset_diagnostics=cfg.skip_subreddit_subset_diagnostics,
    )

    inv = discover_core_files(cfg.in_root, cfg.year)
    inv = filter_month_inventory(inv, cfg.only_months, cfg.limit_months)
    if not inv:
        raise RuntimeError("No monthly core files found after applying filters.")
    require_complete_inventory(inv)

    months = list(sorted(inv.keys()))
    logger.info("Selected months (%d): %s", len(months), ", ".join(months))
    logger.info("Smoke mode: %s | smoke_max_rows=%s", cfg.smoke, cfg.smoke_max_rows)
    logger.info("Write large intermediates: %s", cfg.write_large_intermediates)

    con = duckdb.connect()
    configure_duckdb(con, duck)

    try:
        build_base_views(con, inv, cfg.smoke_max_rows)
        comments_cols = get_table_columns(con, "comments_core_all")
        submissions_cols = get_table_columns(con, "submissions_core_all")
        author_col = resolve_author_col(comments_cols, submissions_cols, cfg.author_col)
        logger.info("Resolved author column: %s", author_col)
        checkpoint_metadata = build_checkpoint_metadata(cfg, months, author_col)
        checkpoint_status = checkpoint_reuse_status(cfg.checkpoint_dir, checkpoint_metadata) if cfg.reuse_checkpoints else {
            "allowed": False,
            "source": "disabled",
            "completed_tables": [],
            "reason": "--reuse-checkpoints was not supplied",
        }
        checkpoint_reuse_ok = bool(cfg.reuse_checkpoints) and bool(checkpoint_status["allowed"])
        if checkpoint_reuse_ok and checkpoint_status["completed_tables"]:
            logger.info(
                "Metadata-compatible checkpoint reuse enabled from %s; completed tables: %s.",
                checkpoint_status["source"],
                ", ".join(checkpoint_status["completed_tables"]),
            )
        elif checkpoint_reuse_ok:
            logger.info(
                "Checkpoint metadata compatible from %s, but no tables are marked complete yet.",
                checkpoint_status["source"],
            )
        else:
            logger.info("Checkpoint reuse not used: %s", checkpoint_status["reason"])
        write_checkpoint_started_manifest(
            cfg.checkpoint_dir,
            checkpoint_metadata,
            completed_tables=checkpoint_status["completed_tables"] if checkpoint_reuse_ok else [],
        )

        build_user_level_views(con, author_col)
        build_derived_tables(
            con,
            inv,
            author_col,
            duck.threads,
            submissions_cols,
            cfg.smoke_max_rows,
            cfg.min_monthly_active_users_for_subset,
            cfg.nsfw_submission_share_threshold_for_subset,
            cfg.checkpoint_dir,
            checkpoint_reuse_ok,
            checkpoint_metadata=checkpoint_metadata,
            skip_subreddit_subset_diagnostics=cfg.skip_subreddit_subset_diagnostics,
        )
        build_compact_lag_and_transition_outputs(
            con,
            months,
            duck.threads,
            cfg.checkpoint_dir,
            checkpoint_reuse_ok,
        )
        tables = fetch_tables(
            con,
            write_large_intermediates=cfg.write_large_intermediates,
            skip_subreddit_subset_diagnostics=cfg.skip_subreddit_subset_diagnostics,
        )
        tables_written = write_tables(
            cfg.out_root,
            tables,
            overwrite=cfg.overwrite,
            write_large_intermediates=cfg.write_large_intermediates,
        )
        if cfg.write_large_intermediates:
            tables_written["tables_written_parquet"] = write_large_intermediate_tables_from_duckdb(
                con,
                cfg.out_root,
                overwrite=cfg.overwrite,
            )
        generate_figures(
            cfg.out_root,
            tables,
            cfg.top_n_subreddits_subset_diag,
            publication_style=cfg.publication_style,
            skip_subreddit_subset_diagnostics=cfg.skip_subreddit_subset_diagnostics,
        )

        run_meta = write_run_metadata(cfg.out_root, cfg, duck, script_name=os.path.basename(__file__))
        finished_json = os.path.join(cfg.out_root, "audit", f"descriptive_plots_finished_{cfg.run_id}.json")
        safe_json_dump(
            {
                "run_id": cfg.run_id,
                "timestamp_utc": utc_now_iso(),
                "mode": "full",
                "skip_subreddit_subset_diagnostics": cfg.skip_subreddit_subset_diagnostics,
                "months": months,
                "author_col": author_col,
                "write_large_intermediates": cfg.write_large_intermediates,
                "checkpoint_dir": cfg.checkpoint_dir,
                "reuse_checkpoints_requested": cfg.reuse_checkpoints,
                "reuse_checkpoints_used": checkpoint_reuse_ok,
                "definition_notes": {
                    "subreddit_activity.n_users": (
                        "Removed rather than approximated: exact annual distinct users per subreddit "
                        "are not available from compact monthly summaries without a memory-heavy "
                        "full-year COUNT(DISTINCT user_id). Use mean/min/max_monthly_active_users."
                    )
                },
                **tables_written,
                "figures_written": planned_figures,
                "config_json": config_json,
                "run_metadata_json": run_meta,
            },
            finished_json,
        )
        write_checkpoint_manifest(cfg.checkpoint_dir, checkpoint_metadata)
        logger.info(
            "Wrote %d CSV tables, %d gzip CSV audit tables, %d Parquet intermediates, and %d figures.",
            len(tables_written["tables_written_csv"]),
            len(tables_written["tables_written_csv_gz"]),
            len(tables_written["tables_written_parquet"]),
            len(planned_figures),
        )
        logger.info("Done.")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
