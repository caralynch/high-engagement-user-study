#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Cara Lynch
# See the LICENSE file for details.
"""Run the Study 2 regularised logistic-regression baseline.

This script evaluates a learned binary classifier for the prospective
``label_high_engagement`` target. It uses the locked 2021 expanding-window
rolling-origin design, selects hyperparameters on the validation month only,
then refits a fresh selected pipeline on training plus validation months before
evaluating it once on each test month.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

try:
    import duckdb
except Exception:  # pragma: no cover - exercised only when duckdb is absent
    duckdb = None


logger = logging.getLogger(__name__)


EXPECTED_PREDICTION_MONTHS_2021 = tuple(f"2021-{month:02d}" for month in range(2, 13))
PRIMARY_TEST_MONTHS_2021 = tuple(f"2021-{month:02d}" for month in range(7, 13))
TOP_10_FRACTION = 0.10
PR_AUC_DEFINITION = "average_precision_ranked_list"
AUROC_DEFINITION = "deterministic_tie_broken_mann_whitney_rank_sum"
MODEL_NAME = "logistic_regression_l2"
DEFAULT_C_GRID = "0.01,0.1,1.0,10.0"
DEFAULT_CLASS_WEIGHT_OPTIONS = "none,balanced"
TIE_HANDLING = (
    "Sort by predicted positive-class probability DESC, then deterministic "
    "seeded tie_break_score DESC, then user_id ASC."
)

EXPECTED_FEATURE_GROUPS = (
    "activity",
    "historical_engagement",
    "subreddit_participation",
    "calendar",
)

FEATURE_GROUP_SETS = {
    "activity_only": ("activity",),
    "historical_engagement_only": ("historical_engagement",),
    "subreddit_participation_only": ("subreddit_participation",),
    "calendar_only": ("calendar",),
    "activity_plus_historical_engagement": ("activity", "historical_engagement"),
    "combined_simple": EXPECTED_FEATURE_GROUPS,
}

HEURISTIC_BRIDGE_FEATURE_SETS = {
    "activity_total_only": {
        "feature_name": "feature_n_total_activity_prev",
        "feature_group": "activity",
        "heuristic_baseline": "activity_total",
        "heuristic_primary_score_column": "feature_n_total_activity_prev",
        "heuristic_ranking_signal": "feature_n_total_activity_prev",
    },
    "activity_frequency_only": {
        "feature_name": "feature_n_active_days_prev",
        "feature_group": "activity",
        "heuristic_baseline": "activity_frequency",
        "heuristic_primary_score_column": "feature_n_active_days_prev",
        "heuristic_ranking_signal": "feature_n_active_days_prev",
    },
    "historical_engagement_total_only": {
        "feature_name": "feature_received_engagement_total_prev",
        "feature_group": "historical_engagement",
        "heuristic_baseline": "historical_engagement_total",
        "heuristic_primary_score_column": "feature_received_engagement_total_prev",
        "heuristic_ranking_signal": "feature_received_engagement_total_prev",
    },
    "historical_engagement_rate_only": {
        "feature_name": "feature_received_engagement_per_activity_prev",
        "feature_group": "historical_engagement",
        "heuristic_baseline": "historical_engagement_rate",
        "heuristic_primary_score_column": "feature_received_engagement_per_activity_prev",
        "heuristic_ranking_signal": "feature_received_engagement_per_activity_prev",
    },
}

REDUCED_COLLINEARITY_FEATURES = (
    "feature_n_submissions_prev",
    "feature_n_total_activity_prev",
    "feature_submission_share_prev",
    "feature_n_active_days_prev",
    "feature_mean_activity_per_active_day_prev",
    "feature_received_engagement_total_prev",
    "feature_received_engagement_per_activity_prev",
    "feature_log_received_engagement_rate_prev",
    "feature_received_replies_to_comments_prev",
    "feature_overlap_submission_and_comment_reply_events_prev",
    "feature_n_active_subreddits_prev",
    "feature_subreddit_hhi_prev",
    "feature_month_number",
)

REDUCED_COLLINEARITY_DROPPED_FEATURES = (
    "feature_comment_share_prev",
    "feature_n_comments_prev",
    "feature_received_comments_on_submissions_prev",
    "feature_subreddit_entropy_prev",
    "feature_top_subreddit_activity_share_prev",
    "feature_is_single_subreddit_user_prev",
)

REDUCED_COLLINEARITY_DROP_RATIONALES = {
    "feature_comment_share_prev": (
        "Exact complement of feature_submission_share_prev: submission share plus comment share equals 1."
    ),
    "feature_n_comments_prev": (
        "Near-duplicate of feature_n_total_activity_prev in the feature-correlation audit "
        "(Pearson r = 0.9997)."
    ),
    "feature_received_comments_on_submissions_prev": (
        "Near-duplicate of feature_received_engagement_total_prev in the feature-correlation audit "
        "(Pearson r = 0.9988)."
    ),
    "feature_subreddit_entropy_prev": (
        "Overlaps with feature_subreddit_hhi_prev as a concentration/specialisation measure "
        "(Pearson r = -0.9939 in the feature-correlation audit)."
    ),
    "feature_top_subreddit_activity_share_prev": (
        "Overlaps with feature_subreddit_hhi_prev as a dominance/specialisation measure "
        "(Pearson r = 0.9845 in the feature-correlation audit)."
    ),
    "feature_is_single_subreddit_user_prev": (
        "Thresholded specialisation indicator overlapping with continuous concentration measures "
        "(Pearson r = -0.9213 with feature_subreddit_entropy_prev in the feature-correlation audit)."
    ),
}

RATE_REPRESENTATION_SENSITIVITY_RATIONALES = {
    "feature_received_engagement_per_activity_prev": (
        "Removed only for the logged-rate-only sensitivity. The validated full-panel audit reported "
        "Pearson r = 0.395575529637182 and Spearman rho = 0.9251061882397199 with "
        "feature_log_received_engagement_rate_prev; the pair was flagged because absolute Spearman "
        "correlation met the 0.90 audit threshold."
    ),
    "feature_log_received_engagement_rate_prev": (
        "Removed only for the raw-rate-only sensitivity. The validated full-panel audit reported "
        "Pearson r = 0.395575529637182 and Spearman rho = 0.9251061882397199 with "
        "feature_received_engagement_per_activity_prev; the pair was flagged because absolute "
        "Spearman correlation met the 0.90 audit threshold."
    ),
}

SUBMISSION_REPRESENTATION_SENSITIVITY_RATIONALES = {
    "feature_n_submissions_prev": (
        "Removed only for the submission-share-only sensitivity. The validated selected-feature audit "
        "reported Pearson r = 0.221629 and Spearman rho = 0.993312 with "
        "feature_submission_share_prev; the pair was flagged because absolute Spearman correlation met "
        "the 0.90 audit threshold."
    ),
    "feature_submission_share_prev": (
        "Removed only for the submission-count-only sensitivity. The validated selected-feature audit "
        "reported Pearson r = 0.221629 and Spearman rho = 0.993312 with "
        "feature_n_submissions_prev; the pair was flagged because absolute Spearman correlation met the "
        "0.90 audit threshold."
    ),
}

REDUCED_COLLINEARITY_VARIANTS = {
    "activity_only_reduced_collinearity": {
        "features": (
            "feature_n_submissions_prev",
            "feature_n_total_activity_prev",
            "feature_submission_share_prev",
            "feature_n_active_days_prev",
            "feature_mean_activity_per_active_day_prev",
        ),
        "dropped_features": (
            "feature_n_comments_prev",
            "feature_comment_share_prev",
        ),
        "feature_groups": ("activity",),
        "corresponding_original_feature_set": "activity_only",
    },
    "historical_engagement_only_reduced_collinearity": {
        "features": (
            "feature_received_engagement_total_prev",
            "feature_received_engagement_per_activity_prev",
            "feature_log_received_engagement_rate_prev",
            "feature_received_replies_to_comments_prev",
            "feature_overlap_submission_and_comment_reply_events_prev",
        ),
        "dropped_features": ("feature_received_comments_on_submissions_prev",),
        "feature_groups": ("historical_engagement",),
        "corresponding_original_feature_set": "historical_engagement_only",
    },
    "subreddit_participation_only_reduced_collinearity": {
        "features": (
            "feature_n_active_subreddits_prev",
            "feature_subreddit_hhi_prev",
        ),
        "dropped_features": (
            "feature_subreddit_entropy_prev",
            "feature_top_subreddit_activity_share_prev",
            "feature_is_single_subreddit_user_prev",
        ),
        "feature_groups": ("subreddit_participation",),
        "corresponding_original_feature_set": "subreddit_participation_only",
    },
    "activity_plus_historical_engagement_reduced_collinearity": {
        "features": (
            "feature_n_submissions_prev",
            "feature_n_total_activity_prev",
            "feature_submission_share_prev",
            "feature_n_active_days_prev",
            "feature_mean_activity_per_active_day_prev",
            "feature_received_engagement_total_prev",
            "feature_received_engagement_per_activity_prev",
            "feature_log_received_engagement_rate_prev",
            "feature_received_replies_to_comments_prev",
            "feature_overlap_submission_and_comment_reply_events_prev",
        ),
        "dropped_features": (
            "feature_n_comments_prev",
            "feature_comment_share_prev",
            "feature_received_comments_on_submissions_prev",
        ),
        "feature_groups": ("activity", "historical_engagement"),
        "corresponding_original_feature_set": "activity_plus_historical_engagement",
    },
    "activity_plus_historical_engagement_reduced_collinearity_raw_rate_only": {
        "features": (
            "feature_n_submissions_prev",
            "feature_n_total_activity_prev",
            "feature_submission_share_prev",
            "feature_n_active_days_prev",
            "feature_mean_activity_per_active_day_prev",
            "feature_received_engagement_total_prev",
            "feature_received_engagement_per_activity_prev",
            "feature_received_replies_to_comments_prev",
            "feature_overlap_submission_and_comment_reply_events_prev",
        ),
        "dropped_features": (
            "feature_n_comments_prev",
            "feature_comment_share_prev",
            "feature_received_comments_on_submissions_prev",
            "feature_log_received_engagement_rate_prev",
        ),
        "feature_groups": ("activity", "historical_engagement"),
        "corresponding_original_feature_set": "activity_plus_historical_engagement",
        "sensitivity_counterpart_feature_set": (
            "activity_plus_historical_engagement_reduced_collinearity"
        ),
        "sensitivity_axis": "prior_month_received_engagement_rate_representation",
        "retained_rate_representation": "raw",
        "removed_rate_feature": "feature_log_received_engagement_rate_prev",
    },
    "activity_plus_historical_engagement_reduced_collinearity_logged_rate_only": {
        "features": (
            "feature_n_submissions_prev",
            "feature_n_total_activity_prev",
            "feature_submission_share_prev",
            "feature_n_active_days_prev",
            "feature_mean_activity_per_active_day_prev",
            "feature_received_engagement_total_prev",
            "feature_log_received_engagement_rate_prev",
            "feature_received_replies_to_comments_prev",
            "feature_overlap_submission_and_comment_reply_events_prev",
        ),
        "dropped_features": (
            "feature_n_comments_prev",
            "feature_comment_share_prev",
            "feature_received_comments_on_submissions_prev",
            "feature_received_engagement_per_activity_prev",
        ),
        "feature_groups": ("activity", "historical_engagement"),
        "corresponding_original_feature_set": "activity_plus_historical_engagement",
        "sensitivity_counterpart_feature_set": (
            "activity_plus_historical_engagement_reduced_collinearity"
        ),
        "sensitivity_axis": "prior_month_received_engagement_rate_representation",
        "retained_rate_representation": "logged",
        "removed_rate_feature": "feature_received_engagement_per_activity_prev",
    },
    "activity_plus_historical_engagement_reduced_collinearity_logged_rate_only_submission_count_only": {
        "features": (
            "feature_n_submissions_prev",
            "feature_n_total_activity_prev",
            "feature_n_active_days_prev",
            "feature_mean_activity_per_active_day_prev",
            "feature_received_engagement_total_prev",
            "feature_log_received_engagement_rate_prev",
            "feature_received_replies_to_comments_prev",
            "feature_overlap_submission_and_comment_reply_events_prev",
        ),
        "dropped_features": (
            "feature_n_comments_prev",
            "feature_comment_share_prev",
            "feature_submission_share_prev",
            "feature_received_comments_on_submissions_prev",
            "feature_received_engagement_per_activity_prev",
        ),
        "feature_groups": ("activity", "historical_engagement"),
        "corresponding_original_feature_set": "activity_plus_historical_engagement",
        "sensitivity_counterpart_feature_set": (
            "activity_plus_historical_engagement_reduced_collinearity_logged_rate_only"
        ),
        "sensitivity_axis": "prior_month_submission_representation",
        "retained_submission_representation": "count",
        "retained_submission_feature": "feature_n_submissions_prev",
        "removed_submission_feature": "feature_submission_share_prev",
        "sensitivity_status": "planned_awaiting_full_protocol_run_and_author_review",
    },
    "activity_plus_historical_engagement_reduced_collinearity_logged_rate_only_submission_share_only": {
        "features": (
            "feature_n_total_activity_prev",
            "feature_submission_share_prev",
            "feature_n_active_days_prev",
            "feature_mean_activity_per_active_day_prev",
            "feature_received_engagement_total_prev",
            "feature_log_received_engagement_rate_prev",
            "feature_received_replies_to_comments_prev",
            "feature_overlap_submission_and_comment_reply_events_prev",
        ),
        "dropped_features": (
            "feature_n_submissions_prev",
            "feature_n_comments_prev",
            "feature_comment_share_prev",
            "feature_received_comments_on_submissions_prev",
            "feature_received_engagement_per_activity_prev",
        ),
        "feature_groups": ("activity", "historical_engagement"),
        "corresponding_original_feature_set": "activity_plus_historical_engagement",
        "sensitivity_counterpart_feature_set": (
            "activity_plus_historical_engagement_reduced_collinearity_logged_rate_only"
        ),
        "sensitivity_axis": "prior_month_submission_representation",
        "retained_submission_representation": "composition",
        "retained_submission_feature": "feature_submission_share_prev",
        "removed_submission_feature": "feature_n_submissions_prev",
        "sensitivity_status": "planned_awaiting_full_protocol_run_and_author_review",
    },
    "combined_simple_reduced_collinearity": {
        "features": REDUCED_COLLINEARITY_FEATURES,
        "dropped_features": REDUCED_COLLINEARITY_DROPPED_FEATURES,
        "feature_groups": EXPECTED_FEATURE_GROUPS,
        "corresponding_original_feature_set": "combined_simple",
        "equivalent_to_feature_set_under_current_schema": "reduced_collinearity",
        "equivalence_note": (
            "Under the current schema, combined_simple contains only activity, historical_engagement, "
            "subreddit_participation, and calendar features, so its reduced-collinearity variant is "
            "identical to the existing reduced_collinearity feature set."
        ),
    },
}

SUPPORTED_FEATURE_SETS = (
    "all",
    "reduced_collinearity",
    *FEATURE_GROUP_SETS.keys(),
    *REDUCED_COLLINEARITY_VARIANTS.keys(),
    *HEURISTIC_BRIDGE_FEATURE_SETS.keys(),
)

FEATURE_SET_DESCRIPTIONS = {
    "all": "Use every feature listed in feature_schema.csv.",
    "reduced_collinearity": (
        "Use a curated low-collinearity feature list documented in docs/feature_set_reduction_plan.md."
    ),
    "activity_only": "Use feature_schema.csv rows in the activity feature group.",
    "historical_engagement_only": "Use feature_schema.csv rows in the historical_engagement feature group.",
    "subreddit_participation_only": "Use feature_schema.csv rows in the subreddit_participation feature group.",
    "calendar_only": "Use feature_schema.csv rows in the calendar feature group.",
    "activity_plus_historical_engagement": (
        "Use feature_schema.csv rows in the activity and historical_engagement feature groups."
    ),
    "combined_simple": (
        "Use feature_schema.csv rows in the activity, historical_engagement, subreddit_participation, "
        "and calendar feature groups."
    ),
    "activity_only_reduced_collinearity": (
        "Sensitivity/interpretable activity-only variant dropping exact-complement and near-duplicate "
        "activity features documented in docs/feature_set_reduction_plan.md."
    ),
    "historical_engagement_only_reduced_collinearity": (
        "Sensitivity/interpretable historical-engagement-only variant dropping the near-duplicate "
        "submission-comments component documented in docs/feature_set_reduction_plan.md."
    ),
    "subreddit_participation_only_reduced_collinearity": (
        "Sensitivity/interpretable subreddit-participation-only variant retaining HHI as the main "
        "specialisation measure."
    ),
    "activity_plus_historical_engagement_reduced_collinearity": (
        "Sensitivity/interpretable activity plus historical-engagement variant using the reduced "
        "activity and historical-engagement feature lists."
    ),
    "activity_plus_historical_engagement_reduced_collinearity_raw_rate_only": (
        "Targeted sensitivity variant of the accepted ten-feature grouped configuration: retain the "
        "raw prior-month received-engagement rate and remove only its logged representation."
    ),
    "activity_plus_historical_engagement_reduced_collinearity_logged_rate_only": (
        "Targeted sensitivity variant of the accepted ten-feature grouped configuration: retain the "
        "logged prior-month received-engagement rate and remove only its raw representation."
    ),
    "activity_plus_historical_engagement_reduced_collinearity_logged_rate_only_submission_count_only": (
        "Targeted sensitivity of the selected nine-feature logged-rate-only configuration: retain prior-month "
        "submission count and remove only prior-month submission share."
    ),
    "activity_plus_historical_engagement_reduced_collinearity_logged_rate_only_submission_share_only": (
        "Targeted sensitivity of the selected nine-feature logged-rate-only configuration: retain prior-month "
        "submission share and remove only prior-month submission count."
    ),
    "combined_simple_reduced_collinearity": (
        "Sensitivity/interpretable combined-simple variant. Under the current schema it is identical "
        "to reduced_collinearity."
    ),
    "activity_total_only": (
        "Single-feature bridge model using the same primary score column as the activity_total heuristic."
    ),
    "activity_frequency_only": (
        "Single-feature bridge model using the same primary score column as the activity_frequency heuristic."
    ),
    "historical_engagement_total_only": (
        "Single-feature bridge model using the same primary score column as the historical_engagement_total heuristic."
    ),
    "historical_engagement_rate_only": (
        "Single-feature bridge model using the same primary score column as the historical_engagement_rate heuristic."
    ),
}

KNOWN_OUTPUTS = (
    "run_metadata.json",
    "resolved_config.json",
    "selected_features.csv",
    "rolling_origin_splits.csv",
    "logistic_hyperparameter_grid.csv",
    "logistic_validation_metrics_by_candidate.csv",
    "logistic_selected_hyperparameters_by_fold.csv",
    "logistic_metrics_by_fold.csv",
    "logistic_metrics_summary.csv",
    "logistic_coefficients_by_fold.csv",
    "logistic_ranked_predictions.parquet",
    "logistic_final_pipeline_state_by_fold.json",
    "logistic_final_pipelines.joblib",
)

SELECTED_FEATURE_FIELDS = (
    "feature_set",
    "feature_name",
    "feature_group",
    "source_window",
    "included",
    "exclusion_reason",
)

HYPERPARAMETER_GRID_FIELDS = (
    "candidate_id",
    "model_name",
    "penalty",
    "regularization",
    "l1_ratio",
    "solver",
    "C",
    "class_weight",
    "max_iter",
    "imputer_strategy",
    "scaler",
)

VALIDATION_METRIC_FIELDS = (
    "model_name",
    "feature_set",
    "fold_id",
    "candidate_id",
    "C",
    "class_weight",
    "penalty",
    "regularization",
    "l1_ratio",
    "solver",
    "prediction_month",
    "feature_month",
    "split_role",
    "metric_scope",
    "candidate_fit_start_prediction_month",
    "candidate_fit_end_prediction_month",
    "candidate_fit_includes_validation",
    "candidate_fit_n_rows",
    "candidate_fit_n_positive",
    "candidate_fit_n_negative",
    "train_start_prediction_month",
    "train_end_prediction_month",
    "validation_prediction_month",
    "test_prediction_month",
    "n_users",
    "n_positive",
    "n_negative",
    "positive_prevalence",
    "k_at_10pct",
    "top_k_positive_count",
    "tp_at_10pct",
    "fp_at_10pct",
    "fn_at_10pct",
    "tn_at_10pct",
    "precision_at_10pct",
    "recall_at_10pct",
    "mcc_at_10pct",
    "average_precision",
    "auprc_ap",
    "auroc",
    "mean_target_score_top_k",
    "mean_target_score_all",
    "selection_metric",
    "selection_rank_within_fold",
    "selected_by_validation",
    "selection_tie_break_order",
    "top_k_rule",
    "pr_auc_definition",
    "auroc_definition",
)

SELECTED_HYPERPARAMETER_FIELDS = (
    "model_name",
    "feature_set",
    "fold_id",
    "selected_candidate_id",
    "selected_C",
    "selected_class_weight",
    "penalty",
    "regularization",
    "l1_ratio",
    "solver",
    "selection_metric",
    "selected_validation_metric_value",
    "validation_average_precision",
    "validation_auroc",
    "validation_precision_at_10pct",
    "validation_recall_at_10pct",
    "validation_mcc_at_10pct",
    "validation_C_tie_break_rank",
    "validation_class_weight_tie_break_rank",
    "selection_rank_within_fold",
    "selection_tie_break_order",
    "train_start_prediction_month",
    "train_end_prediction_month",
    "validation_prediction_month",
    "test_prediction_month",
    "candidate_fit_start_prediction_month",
    "candidate_fit_end_prediction_month",
    "candidate_fit_includes_validation",
    "candidate_fit_n_rows",
    "candidate_fit_n_positive",
    "candidate_fit_n_negative",
    "final_refit_start_prediction_month",
    "final_refit_end_prediction_month",
    "final_refit_includes_validation",
    "final_refit_n_rows",
    "final_refit_n_positive",
    "final_refit_n_negative",
    "configured_max_iter",
    "actual_n_iter",
    "reached_max_iter",
)

FOLD_METRIC_FIELDS = (
    "model_name",
    "feature_set",
    "fold_id",
    "prediction_month",
    "test_prediction_month",
    "feature_month",
    "split_role",
    "metric_scope",
    "train_start_prediction_month",
    "train_end_prediction_month",
    "validation_prediction_month",
    "candidate_fit_start_prediction_month",
    "candidate_fit_end_prediction_month",
    "candidate_fit_includes_validation",
    "candidate_fit_n_rows",
    "candidate_fit_n_positive",
    "candidate_fit_n_negative",
    "final_refit_start_prediction_month",
    "final_refit_end_prediction_month",
    "final_refit_includes_validation",
    "final_refit_n_rows",
    "final_refit_n_positive",
    "final_refit_n_negative",
    "configured_max_iter",
    "actual_n_iter",
    "reached_max_iter",
    "selected_candidate_id",
    "selected_C",
    "selected_class_weight",
    "penalty",
    "regularization",
    "l1_ratio",
    "solver",
    "n_users",
    "n_positive",
    "n_negative",
    "positive_prevalence",
    "k_at_10pct",
    "top_k_positive_count",
    "tp_at_10pct",
    "fp_at_10pct",
    "fn_at_10pct",
    "tn_at_10pct",
    "precision_at_10pct",
    "recall_at_10pct",
    "mcc_at_10pct",
    "average_precision",
    "auprc_ap",
    "auroc",
    "mean_target_score_top_k",
    "mean_target_score_all",
    "top_k_rule",
    "pr_auc_definition",
    "auroc_definition",
)

SUMMARY_FIELDS = (
    "model_name",
    "feature_set",
    "metric_scope",
    "source_split_role",
    "n_folds",
    "test_prediction_month_start",
    "test_prediction_month_end",
    "n_users_pooled",
    "n_positive_pooled",
    "k_at_10pct_pooled",
    "top_k_positive_count_pooled",
    "tp_at_10pct_pooled",
    "fp_at_10pct_pooled",
    "fn_at_10pct_pooled",
    "tn_at_10pct_pooled",
    "mcc_at_10pct_pooled",
    "n_users_macro_mean",
    "n_users_macro_stddev",
    "n_positive_macro_mean",
    "n_positive_macro_stddev",
    "positive_prevalence_macro_mean",
    "positive_prevalence_macro_stddev",
    "precision_at_10pct_macro_mean",
    "precision_at_10pct_macro_stddev",
    "recall_at_10pct_macro_mean",
    "recall_at_10pct_macro_stddev",
    "mcc_at_10pct_macro_mean",
    "mcc_at_10pct_macro_stddev",
    "average_precision_macro_mean",
    "average_precision_macro_stddev",
    "auprc_ap_macro_mean",
    "auprc_ap_macro_stddev",
    "auroc_macro_mean",
    "auroc_macro_stddev",
    "mean_target_score_top_k_macro_mean",
    "mean_target_score_top_k_macro_stddev",
    "mean_target_score_all_macro_mean",
    "mean_target_score_all_macro_stddev",
    "aggregation_note",
    "pr_auc_definition",
    "auroc_definition",
)

COEFFICIENT_FIELDS = (
    "model_name",
    "feature_set",
    "fold_id",
    "feature_name",
    "coefficient",
    "odds_ratio",
    "intercept",
    "selected_candidate_id",
    "selected_C",
    "selected_class_weight",
    "penalty",
    "regularization",
    "l1_ratio",
    "solver",
    "candidate_fit_start_prediction_month",
    "candidate_fit_end_prediction_month",
    "candidate_fit_includes_validation",
    "candidate_fit_n_rows",
    "candidate_fit_n_positive",
    "candidate_fit_n_negative",
    "final_refit_start_prediction_month",
    "final_refit_end_prediction_month",
    "final_refit_includes_validation",
    "final_refit_n_rows",
    "final_refit_n_positive",
    "final_refit_n_negative",
    "configured_max_iter",
    "actual_n_iter",
    "reached_max_iter",
    "coefficient_note",
)


@dataclass(frozen=True)
class DuckDBConfig:
    """DuckDB runtime configuration."""

    memory_limit: str
    threads: int
    temp_dir: str


@dataclass(frozen=True)
class ModelConfig:
    """Resolved logistic model configuration."""

    penalty: str
    regularization: str
    l1_ratio: float
    solver: str
    max_iter: int
    imputer_strategy: str
    scaler: str


@dataclass(frozen=True)
class LogisticConfig:
    """Resolved logistic baseline run configuration."""

    panel_path: str
    feature_schema_csv: str
    out_root: str
    feature_set: str
    reference_splits_csv: Optional[str]
    c_grid: tuple[float, ...]
    class_weight_options: tuple[str, ...]
    selection_metric: str
    random_seed: int
    target_score_col: str
    overwrite: bool
    write_ranked_predictions: bool
    write_final_pipelines: bool
    top_k_fraction: float
    smoke: bool
    smoke_max_rows: int
    min_train_months: int
    validation_window_months: int
    test_window_months: int


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
    """Return a stable UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def quote_sql_string(value: str | Path) -> str:
    """Return a single-quoted SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def quote_identifier(name: str) -> str:
    """Return a double-quoted SQL identifier."""
    return '"' + str(name).replace('"', '""') + '"'


def sql_string_list(values: Sequence[str]) -> str:
    """Return a SQL tuple of string literals."""
    if not values:
        raise ValueError("Cannot build SQL string list from an empty sequence.")
    return "(" + ", ".join(quote_sql_string(value) for value in values) + ")"


def ensure_dir(path: str | Path) -> None:
    """Create a directory if needed."""
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_json_dump(obj: dict[str, Any], path: str | Path) -> None:
    """Write stable JSON."""
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def write_csv_rows(path: str | Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write dictionaries as CSV."""
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


def package_versions() -> dict[str, Optional[str]]:
    """Return package versions relevant to this script."""
    versions: dict[str, Optional[str]] = {
        "python": sys.version.replace("\n", " "),
        "duckdb": getattr(duckdb, "__version__", None),
    }
    for package in ("numpy", "pandas", "sklearn"):
        try:
            module = __import__(package)
        except Exception:
            versions[package] = None
        else:
            versions[package] = getattr(module, "__version__", "installed")
    return versions


def known_output_paths(out_root: Path) -> list[Path]:
    """Return all known output paths for collision checks."""
    return [out_root / name for name in KNOWN_OUTPUTS]


def check_output_collisions(paths: Sequence[Path], overwrite: bool) -> None:
    """Fail if outputs exist and overwrite is disabled."""
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


def configure_duckdb(con: duckdb.DuckDBPyConnection, duck: DuckDBConfig) -> None:
    """Apply DuckDB runtime settings."""
    ensure_dir(duck.temp_dir)
    con.execute(f"SET memory_limit={quote_sql_string(duck.memory_limit)}")
    con.execute(f"SET threads={int(duck.threads)}")
    con.execute(f"SET temp_directory={quote_sql_string(duck.temp_dir)}")
    con.execute("SET preserve_insertion_order=false")
    con.execute("PRAGMA disable_object_cache")


def month_add(month: str, offset: int) -> str:
    """Add calendar months to a YYYY-MM string."""
    year_s, month_s = month.split("-")
    total = int(year_s) * 12 + int(month_s) - 1 + offset
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def previous_month(month: str) -> str:
    """Return the previous calendar month for a YYYY-MM string."""
    return month_add(month, -1)


def parse_c_grid(value: str) -> tuple[float, ...]:
    """Parse a comma-separated positive C grid."""
    try:
        grid = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--c-grid must contain comma-separated numeric values.") from exc
    if not grid:
        raise argparse.ArgumentTypeError("--c-grid must contain at least one value.")
    if any((not math.isfinite(c)) or c <= 0.0 for c in grid):
        raise argparse.ArgumentTypeError("--c-grid values must be finite and positive.")
    return grid


def parse_class_weight_options(value: str) -> tuple[str, ...]:
    """Parse class-weight options, preserving order and normalising None."""
    options: list[str] = []
    for raw in value.split(","):
        option = raw.strip().lower()
        if not option:
            continue
        if option in {"none", "null"}:
            option = "none"
        if option not in {"none", "balanced"}:
            raise argparse.ArgumentTypeError("--class-weight-options supports only none and balanced.")
        if option not in options:
            options.append(option)
    if not options:
        raise argparse.ArgumentTypeError("--class-weight-options must contain at least one value.")
    return tuple(options)


def validate_cli(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate CLI arguments before work starts."""
    if not Path(args.panel_path).exists():
        parser.error(f"--panel-path not found: {args.panel_path}")
    if not Path(args.feature_schema_csv).exists():
        parser.error(f"--feature-schema-csv not found: {args.feature_schema_csv}")
    if args.reference_splits_csv and not Path(args.reference_splits_csv).exists():
        parser.error(f"--reference-splits-csv not found: {args.reference_splits_csv}")
    if not (0.0 < float(args.top_k_fraction) < 1.0):
        parser.error("--top-k-fraction must be greater than 0 and less than 1.")
    if not math.isclose(float(args.top_k_fraction), TOP_10_FRACTION, rel_tol=0.0, abs_tol=1e-12):
        parser.error(
            "--top-k-fraction is currently fixed at 0.10 because the constructed "
            "label and Precision@10%/Recall@10% convention use the monthly positive count."
        )
    if int(args.smoke_max_rows) < 0:
        parser.error("--smoke-max-rows cannot be negative.")
    if args.smoke and 0 < int(args.smoke_max_rows) < 2:
        parser.error("--smoke-max-rows must be at least 2 when row limiting is enabled.")
    if int(args.min_train_months) < 1:
        parser.error("--min-train-months must be at least 1.")
    if int(args.validation_window_months) != 1:
        parser.error("--validation-window-months currently supports only 1 for the locked 2021 protocol.")
    if int(args.test_window_months) != 1:
        parser.error("--test-window-months currently supports only 1 for the locked 2021 protocol.")
    if int(args.duckdb_threads) < 1:
        parser.error("--duckdb-threads must be at least 1.")
    if int(args.max_iter) < 1:
        parser.error("--max-iter must be positive.")


def create_panel_source_view(con: duckdb.DuckDBPyConnection, cfg: LogisticConfig) -> None:
    """Create the input panel view, optionally with a stratified smoke row cap."""
    relation = f"read_parquet({quote_sql_string(cfg.panel_path)})"
    duplicate_rows = con.execute(
        f"""
        SELECT
          CAST(user_id AS VARCHAR) AS user_id,
          CAST(prediction_month AS VARCHAR) AS prediction_month,
          COUNT(*) AS n_rows
        FROM {relation}
        GROUP BY 1, 2
        HAVING COUNT(*) > 1
        ORDER BY prediction_month, user_id
        LIMIT 5
        """
    ).fetchall()
    if duplicate_rows:
        duplicate_key_count = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM (
                  SELECT 1
                  FROM {relation}
                  GROUP BY CAST(user_id AS VARCHAR), CAST(prediction_month AS VARCHAR)
                  HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
        )
        examples = ", ".join(
            f"({row[0]!r}, {row[1]!r}; rows={int(row[2])})" for row in duplicate_rows
        )
        raise ValueError(
            "Panel violates the one-user-month contract: "
            f"duplicate (user_id, prediction_month) key count={duplicate_key_count}; examples={examples}."
        )
    if cfg.smoke and int(cfg.smoke_max_rows) > 0:
        per_label_limit = max(1, int(cfg.smoke_max_rows) // 2)
        logger.warning(
            "Smoke row limiting enabled: keeping up to %s rows per label per prediction month.",
            per_label_limit,
        )
        relation = f"""
        (
          SELECT *
          FROM (
            SELECT
              *,
              ROW_NUMBER() OVER (
                PARTITION BY CAST(prediction_month AS VARCHAR), CAST(label_high_engagement AS INTEGER)
                ORDER BY CAST(user_id AS VARCHAR) ASC
              ) AS _smoke_row_num
            FROM {relation}
          )
          WHERE _smoke_row_num <= {per_label_limit}
        )
        """
    con.execute(f"CREATE OR REPLACE TEMP VIEW panel_source AS SELECT * FROM {relation}")


def relation_columns(con: duckdb.DuckDBPyConnection, relation_name: str) -> list[str]:
    """Return relation columns."""
    rows = con.execute(f"DESCRIBE SELECT * FROM {quote_identifier(relation_name)} LIMIT 0").fetchall()
    return [str(row[0]) for row in rows]


def observed_prediction_months(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Return distinct prediction months in the panel source."""
    rows = con.execute(
        """
        SELECT DISTINCT CAST(prediction_month AS VARCHAR) AS prediction_month
        FROM panel_source
        ORDER BY 1
        """
    ).fetchall()
    return [str(row[0]) for row in rows]


def prediction_month_feature_map(con: duckdb.DuckDBPyConnection) -> dict[str, str]:
    """Return the single feature month for each prediction month."""
    rows = con.execute(
        """
        SELECT
          CAST(prediction_month AS VARCHAR) AS prediction_month,
          MIN(CAST(feature_month AS VARCHAR)) AS feature_month,
          COUNT(DISTINCT CAST(feature_month AS VARCHAR)) AS n_feature_months
        FROM panel_source
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchall()
    bad = [str(row[0]) for row in rows if int(row[2]) != 1]
    if bad:
        raise ValueError(
            "Each prediction month must map to exactly one feature month. "
            "Problem month(s): " + ", ".join(bad)
        )
    return {str(row[0]): str(row[1]) for row in rows}


def read_feature_schema(path: str | Path) -> list[dict[str, str]]:
    """Read feature schema rows."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "feature_name" not in (reader.fieldnames or []):
            raise ValueError(f"Feature schema CSV lacks feature_name column: {path}")
        rows = []
        for row in reader:
            feature_name = str(row.get("feature_name", "")).strip()
            if not feature_name:
                continue
            rows.append(
                {
                    "feature_name": feature_name,
                    "feature_group": str(row.get("feature_group", "")).strip(),
                    "source_window": str(row.get("source_window", "")).strip(),
                    "description": str(row.get("description", "")).strip(),
                }
            )
    if not rows:
        raise ValueError(f"Feature schema CSV contains no features: {path}")
    return rows


def supported_feature_sets_text() -> str:
    """Return a stable feature-set list for user-facing errors."""
    return ", ".join(SUPPORTED_FEATURE_SETS)


def feature_set_definition(feature_set: str) -> dict[str, Any]:
    """Return the pre-specified feature-set definition used for a run."""
    if feature_set == "all":
        return {
            "feature_set": feature_set,
            "description": FEATURE_SET_DESCRIPTIONS[feature_set],
            "selection_type": "all_schema_features",
            "selection_rule": "Include every feature listed in feature_schema.csv, preserving schema order.",
            "uses_feature_schema_groups": False,
            "feature_groups": ["all"],
            "curated_feature_names": [],
            "curated_excluded_feature_names": [],
        }
    if feature_set == "reduced_collinearity":
        return {
            "feature_set": feature_set,
            "description": FEATURE_SET_DESCRIPTIONS[feature_set],
            "selection_type": "curated_feature_list",
            "selection_rule": "Include the documented reduced-collinearity feature list, preserving curated order.",
            "uses_feature_schema_groups": False,
            "feature_groups": list(EXPECTED_FEATURE_GROUPS),
            "curated_feature_names": list(REDUCED_COLLINEARITY_FEATURES),
            "curated_excluded_feature_names": list(REDUCED_COLLINEARITY_DROPPED_FEATURES),
            "curated_excluded_feature_rationales": {
                feature: REDUCED_COLLINEARITY_DROP_RATIONALES[feature]
                for feature in REDUCED_COLLINEARITY_DROPPED_FEATURES
            },
            "methodological_role": (
                "Sensitivity/interpretable variant; not a replacement for the original predictive "
                "feature-group comparisons."
            ),
        }
    if feature_set in FEATURE_GROUP_SETS:
        return {
            "feature_set": feature_set,
            "description": FEATURE_SET_DESCRIPTIONS[feature_set],
            "selection_type": "feature_schema_groups",
            "selection_rule": (
                "Include feature_schema.csv rows whose feature_group is one of feature_groups, "
                "preserving schema order."
            ),
            "uses_feature_schema_groups": True,
            "feature_groups": list(FEATURE_GROUP_SETS[feature_set]),
            "curated_feature_names": [],
            "curated_excluded_feature_names": [],
        }
    if feature_set in REDUCED_COLLINEARITY_VARIANTS:
        variant = REDUCED_COLLINEARITY_VARIANTS[feature_set]
        dropped_features = tuple(str(feature) for feature in variant["dropped_features"])
        drop_rationales = {
            **REDUCED_COLLINEARITY_DROP_RATIONALES,
            **RATE_REPRESENTATION_SENSITIVITY_RATIONALES,
            **SUBMISSION_REPRESENTATION_SENSITIVITY_RATIONALES,
        }
        definition = {
            "feature_set": feature_set,
            "description": FEATURE_SET_DESCRIPTIONS[feature_set],
            "selection_type": "curated_feature_list",
            "selection_rule": (
                "Include the documented reduced-collinearity feature list for this feature group, "
                "preserving curated order."
            ),
            "uses_feature_schema_groups": False,
            "feature_groups": list(variant["feature_groups"]),
            "curated_feature_names": list(variant["features"]),
            "curated_excluded_feature_names": list(dropped_features),
            "curated_excluded_feature_rationales": {
                feature: drop_rationales[feature]
                for feature in dropped_features
            },
            "corresponding_original_feature_set": variant["corresponding_original_feature_set"],
            "methodological_role": (
                "Sensitivity/interpretable variant; not a replacement for the original predictive "
                "feature-group comparison."
            ),
        }
        if "equivalent_to_feature_set_under_current_schema" in variant:
            definition["equivalent_to_feature_set_under_current_schema"] = variant[
                "equivalent_to_feature_set_under_current_schema"
            ]
            definition["equivalence_note"] = variant["equivalence_note"]
        if "sensitivity_counterpart_feature_set" in variant:
            definition.update(
                {
                    "sensitivity_counterpart_feature_set": variant[
                        "sensitivity_counterpart_feature_set"
                    ],
                    "sensitivity_axis": variant["sensitivity_axis"],
                    "sensitivity_status": variant.get(
                        "sensitivity_status", "requires_full_protocol_run_and_author_decision"
                    ),
                    "automatically_replaces_headline_model": False,
                    "author_decision_required": True,
                }
            )
            if "retained_rate_representation" in variant:
                definition.update(
                    {
                        "retained_rate_representation": variant["retained_rate_representation"],
                        "removed_rate_feature": variant["removed_rate_feature"],
                    }
                )
            if "retained_submission_representation" in variant:
                definition.update(
                    {
                        "retained_submission_representation": variant[
                            "retained_submission_representation"
                        ],
                        "retained_submission_feature": variant["retained_submission_feature"],
                        "removed_submission_feature": variant["removed_submission_feature"],
                        "ready_for_thesis_use": False,
                    }
                )
        return definition
    if feature_set in HEURISTIC_BRIDGE_FEATURE_SETS:
        bridge = HEURISTIC_BRIDGE_FEATURE_SETS[feature_set]
        feature_name = str(bridge["feature_name"])
        return {
            "feature_set": feature_set,
            "description": FEATURE_SET_DESCRIPTIONS[feature_set],
            "selection_type": "single_feature_heuristic_bridge",
            "selection_rule": (
                "Include exactly the feature column used as the matching non-learned heuristic's "
                "primary_score_column in Modelling/1_0_run_baselines.py."
            ),
            "uses_feature_schema_groups": False,
            "feature_groups": [str(bridge["feature_group"])],
            "curated_feature_names": [feature_name],
            "curated_excluded_feature_names": [],
            "heuristic_baseline_name": str(bridge["heuristic_baseline"]),
            "heuristic_primary_score_column": str(bridge["heuristic_primary_score_column"]),
            "heuristic_ranking_signal": str(bridge["heuristic_ranking_signal"]),
            "comparison_purpose": (
                "Bridge model for direct learned-vs-unlearned comparison with the corresponding "
                "single-score heuristic baseline; not a primary feature-group model."
            ),
        }
    raise ValueError(
        f"Unsupported feature set {feature_set!r}. Supported values: {supported_feature_sets_text()}."
    )


def resolve_feature_set(feature_set: str, schema_rows: Sequence[dict[str, str]]) -> tuple[list[str], list[dict[str, Any]]]:
    """Resolve selected features and selected_features.csv rows."""
    schema_by_name = {row["feature_name"]: row for row in schema_rows}
    schema_order = [row["feature_name"] for row in schema_rows]

    if feature_set == "all":
        selected = list(schema_order)
    elif feature_set == "reduced_collinearity":
        selected = list(REDUCED_COLLINEARITY_FEATURES)
    elif feature_set in FEATURE_GROUP_SETS:
        groups = set(FEATURE_GROUP_SETS[feature_set])
        selected = [row["feature_name"] for row in schema_rows if row.get("feature_group") in groups]
    elif feature_set in REDUCED_COLLINEARITY_VARIANTS:
        selected = list(REDUCED_COLLINEARITY_VARIANTS[feature_set]["features"])
    elif feature_set in HEURISTIC_BRIDGE_FEATURE_SETS:
        selected = [str(HEURISTIC_BRIDGE_FEATURE_SETS[feature_set]["feature_name"])]
    else:
        raise ValueError(
            f"Unsupported feature set {feature_set!r}. Supported values: {supported_feature_sets_text()}."
        )

    missing_from_schema = [feature for feature in selected if feature not in schema_by_name]
    if missing_from_schema:
        raise ValueError(
            f"Feature set {feature_set!r} requests feature(s) missing from feature_schema.csv: "
            + ", ".join(missing_from_schema)
        )
    if not selected:
        raise ValueError(f"Feature set {feature_set!r} resolved to zero features.")

    selected_set = set(selected)
    selected_rows: list[dict[str, Any]] = []
    for feature_name in schema_order:
        row = schema_by_name[feature_name]
        included = feature_name in selected_set
        if included:
            reason = ""
        elif feature_set == "reduced_collinearity" and feature_name in REDUCED_COLLINEARITY_DROPPED_FEATURES:
            reason = "dropped_for_reduced_collinearity"
        elif (
            feature_set in REDUCED_COLLINEARITY_VARIANTS
            and feature_name in REDUCED_COLLINEARITY_VARIANTS[feature_set]["dropped_features"]
        ):
            variant = REDUCED_COLLINEARITY_VARIANTS[feature_set]
            if feature_name == variant.get("removed_submission_feature"):
                reason = "dropped_for_submission_representation_sensitivity"
            else:
                reason = "dropped_for_reduced_collinearity"
        else:
            reason = "not_in_selected_feature_set"
        selected_rows.append(
            {
                "feature_set": feature_set,
                "feature_name": feature_name,
                "feature_group": row.get("feature_group", ""),
                "source_window": row.get("source_window", ""),
                "included": included,
                "exclusion_reason": reason,
            }
        )
    return selected, selected_rows


def validate_panel_preconditions(
    con: duckdb.DuckDBPyConnection,
    cfg: LogisticConfig,
    columns: Sequence[str],
    selected_features: Sequence[str],
) -> None:
    """Validate panel shape and required columns before modelling."""
    column_set = set(columns)
    required = {
        "user_id",
        "feature_month",
        "prediction_month",
        "label_high_engagement",
        "target_score",
        cfg.target_score_col,
        *selected_features,
    }
    missing = sorted(column for column in required if column not in column_set)
    if missing:
        raise ValueError("Panel is missing required selected feature/base column(s): " + ", ".join(missing))

    n_rows = int(con.execute("SELECT COUNT(*) FROM panel_source").fetchone()[0])
    if n_rows < 1:
        raise ValueError("Panel has no rows.")

    observed_months = observed_prediction_months(con)
    missing_expected = [month for month in EXPECTED_PREDICTION_MONTHS_2021 if month not in observed_months]
    if missing_expected:
        prefix = "Smoke row limiting left the panel" if cfg.smoke else "Panel is"
        raise ValueError(
            f"{prefix} missing prediction month(s) required for the 2021 rolling-origin protocol: "
            + ", ".join(missing_expected)
        )

    expected_sql = sql_string_list(EXPECTED_PREDICTION_MONTHS_2021)
    invalid_label_count = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM panel_source
            WHERE CAST(prediction_month AS VARCHAR) IN {expected_sql}
              AND (
                label_high_engagement IS NULL
                OR CAST(label_high_engagement AS INTEGER) NOT IN (0, 1)
              )
            """
        ).fetchone()[0]
    )
    if invalid_label_count:
        raise ValueError(
            "label_high_engagement must be non-NULL binary 0/1. "
            f"Invalid row count: {invalid_label_count}"
        )

    month_balance = con.execute(
        f"""
        SELECT
          CAST(prediction_month AS VARCHAR) AS prediction_month,
          COUNT(*) AS n_users,
          SUM(CAST(label_high_engagement AS INTEGER)) AS n_positive
        FROM panel_source
        WHERE CAST(prediction_month AS VARCHAR) IN {expected_sql}
        GROUP BY 1
        HAVING SUM(CAST(label_high_engagement AS INTEGER)) <= 0
          OR SUM(CAST(label_high_engagement AS INTEGER)) >= COUNT(*)
        ORDER BY 1
        """
    ).fetchall()
    if month_balance:
        details = ", ".join(f"{row[0]} (n={row[1]}, positives={row[2]})" for row in month_balance)
        prefix = "Smoke row limiting produced" if cfg.smoke else "Panel contains"
        raise ValueError(prefix + " degenerate prediction month(s) with only one label class: " + details)

    feature_map = prediction_month_feature_map(con)
    wrong_feature_months = [
        f"{prediction_month}->{feature_month}"
        for prediction_month, feature_month in feature_map.items()
        if prediction_month in EXPECTED_PREDICTION_MONTHS_2021 and feature_month != previous_month(prediction_month)
    ]
    if wrong_feature_months:
        raise ValueError(
            "Expected one-month-lag feature_month values for the 2021 protocol. Problem mapping(s): "
            + ", ".join(wrong_feature_months)
        )

    logger.info("Panel preconditions passed for %s row(s).", n_rows)


def build_rolling_origin_splits(
    feature_map: dict[str, str],
    cfg: LogisticConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build fold split rows and a compact primary-test lookup."""
    months = list(EXPECTED_PREDICTION_MONTHS_2021)
    if int(cfg.min_train_months) != 4:
        logger.warning(
            "Using non-default --min-train-months=%s; the locked 2021 protocol uses 4.",
            int(cfg.min_train_months),
        )
    split_rows: list[dict[str, object]] = []
    primary_test_rows: list[dict[str, object]] = []

    first_test_index = int(cfg.min_train_months) + int(cfg.validation_window_months)
    fold_id = 1
    for test_index in range(first_test_index, len(months), int(cfg.test_window_months)):
        train_months = months[: test_index - int(cfg.validation_window_months)]
        validation_month = months[test_index - int(cfg.validation_window_months)]
        test_month = months[test_index]
        fold_common = {
            "fold_id": fold_id,
            "train_start_prediction_month": train_months[0],
            "train_end_prediction_month": train_months[-1],
            "validation_prediction_month": validation_month,
            "test_prediction_month": test_month,
        }

        for prediction_month in train_months:
            split_rows.append(
                {
                    **fold_common,
                    "prediction_month": prediction_month,
                    "feature_month": feature_map[prediction_month],
                    "split_role": "train",
                    "is_primary_test_month": False,
                }
            )
        split_rows.append(
            {
                **fold_common,
                "prediction_month": validation_month,
                "feature_month": feature_map[validation_month],
                "split_role": "validation",
                "is_primary_test_month": False,
            }
        )
        test_row = {
            **fold_common,
            "prediction_month": test_month,
            "feature_month": feature_map[test_month],
            "split_role": "test",
            "is_primary_test_month": True,
        }
        split_rows.append(test_row)
        primary_test_rows.append(test_row)
        fold_id += 1

    observed_tests = tuple(row["test_prediction_month"] for row in primary_test_rows)
    if observed_tests != PRIMARY_TEST_MONTHS_2021:
        raise ValueError(
            "Rolling-origin test months do not match the locked 2021 protocol. "
            f"Observed={observed_tests}; expected={PRIMARY_TEST_MONTHS_2021}"
        )

    for prediction_month, feature_month in sorted(feature_map.items()):
        split_rows.append(
            {
                "fold_id": "",
                "prediction_month": prediction_month,
                "feature_month": feature_month,
                "split_role": "diagnostic_only",
                "train_start_prediction_month": "",
                "train_end_prediction_month": "",
                "validation_prediction_month": "",
                "test_prediction_month": "",
                "is_primary_test_month": False,
            }
        )

    return split_rows, primary_test_rows


def split_fieldnames() -> tuple[str, ...]:
    """Return the rolling-origin split artefact columns."""
    return (
        "fold_id",
        "prediction_month",
        "feature_month",
        "split_role",
        "train_start_prediction_month",
        "train_end_prediction_month",
        "validation_prediction_month",
        "test_prediction_month",
        "is_primary_test_month",
    )


def compare_reference_splits(generated_rows: Sequence[dict[str, object]], reference_path: str | Path) -> None:
    """Fail if generated splits differ from a reference split artefact."""
    with open(reference_path, "r", encoding="utf-8", newline="") as f:
        reference_rows = list(csv.DictReader(f))
    normalised_generated = [
        {field: str(row.get(field, "")) for field in split_fieldnames()}
        for row in generated_rows
    ]
    normalised_reference = [
        {field: str(row.get(field, "")) for field in split_fieldnames()}
        for row in reference_rows
    ]
    if normalised_generated != normalised_reference:
        mismatch = "row counts differ"
        if len(normalised_generated) == len(normalised_reference):
            for index, (left, right) in enumerate(zip(normalised_generated, normalised_reference), start=1):
                if left != right:
                    mismatch = f"first mismatch at row {index}: generated={left}; reference={right}"
                    break
        raise ValueError(
            f"Generated rolling-origin splits do not match reference {reference_path}: {mismatch}"
        )


def validate_fold_label_balance(
    con: duckdb.DuckDBPyConnection,
    cfg: LogisticConfig,
    primary_test_rows: Sequence[dict[str, object]],
) -> None:
    """Ensure every fold role has both positive and negative labels."""
    failures: list[str] = []
    months = list(EXPECTED_PREDICTION_MONTHS_2021)
    for fold in primary_test_rows:
        fold_id = int(fold["fold_id"])
        test_month = str(fold["test_prediction_month"])
        validation_month = str(fold["validation_prediction_month"])
        train_end = str(fold["train_end_prediction_month"])
        train_months = months[: months.index(train_end) + 1]
        role_months = {
            "train": train_months,
            "validation": [validation_month],
            "test": [test_month],
        }
        for role, role_values in role_months.items():
            rows = con.execute(
                f"""
                SELECT
                  COUNT(*) AS n_users,
                  SUM(CAST(label_high_engagement AS INTEGER)) AS n_positive
                FROM panel_source
                WHERE CAST(prediction_month AS VARCHAR) IN {sql_string_list(role_values)}
                """
            ).fetchone()
            n_users = int(rows[0] or 0)
            n_positive = int(rows[1] or 0)
            if n_users <= 0 or n_positive <= 0 or n_positive >= n_users:
                failures.append(
                    f"fold={fold_id} role={role} months={','.join(role_values)} "
                    f"n_users={n_users} n_positive={n_positive}"
                )
    if failures:
        prefix = "Smoke row limiting produced degenerate fold partition(s): " if cfg.smoke else (
            "Degenerate fold partition(s); each train, validation, and test role needs both labels: "
        )
        raise ValueError(prefix + "; ".join(failures))


def make_hyperparameter_grid(cfg: LogisticConfig, model_cfg: ModelConfig) -> list[dict[str, Any]]:
    """Return candidate hyperparameter rows."""
    rows: list[dict[str, Any]] = []
    candidate_id = 1
    for c_value in cfg.c_grid:
        for class_weight in cfg.class_weight_options:
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "model_name": MODEL_NAME,
                    "penalty": model_cfg.penalty,
                    "regularization": model_cfg.regularization,
                    "l1_ratio": model_cfg.l1_ratio,
                    "solver": model_cfg.solver,
                    "C": c_value,
                    "class_weight": class_weight,
                    "max_iter": model_cfg.max_iter,
                    "imputer_strategy": model_cfg.imputer_strategy,
                    "scaler": model_cfg.scaler,
                }
            )
            candidate_id += 1
    return rows


def stable_hash_score(user_id: object, prediction_month: object, seed: int, scope: str) -> float:
    """Return a deterministic score in [0, 1) for tie-breaking."""
    payload = f"{user_id}|{prediction_month}|{scope}|{seed}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) / 18446744073709551616.0


def matthews_corrcoef_from_counts(tp: int, fp: int, fn: int, tn: int) -> float:
    """Return MCC for binary counts, using NaN for a zero denominator."""
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator <= 0.0:
        return float("nan")
    return ((float(tp) * float(tn)) - (float(fp) * float(fn))) / math.sqrt(denominator)


def fetch_partition_frame(
    con: duckdb.DuckDBPyConnection,
    cfg: LogisticConfig,
    selected_features: Sequence[str],
    prediction_months: Sequence[str],
):
    """Fetch one fold partition with only required columns."""
    feature_exprs = [
        f"CAST({quote_identifier(feature)} AS DOUBLE) AS {quote_identifier(feature)}"
        for feature in selected_features
    ]
    query = f"""
    SELECT
      CAST(user_id AS VARCHAR) AS user_id,
      CAST(feature_month AS VARCHAR) AS feature_month,
      CAST(prediction_month AS VARCHAR) AS prediction_month,
      CAST(label_high_engagement AS INTEGER) AS label_high_engagement,
      CAST(target_score AS DOUBLE) AS target_score,
      CAST({quote_identifier(cfg.target_score_col)} AS DOUBLE) AS target_score_for_summary,
      {", ".join(feature_exprs)}
    FROM panel_source
    WHERE CAST(prediction_month AS VARCHAR) IN {sql_string_list(prediction_months)}
    ORDER BY CAST(prediction_month AS VARCHAR), CAST(user_id AS VARCHAR)
    """
    return con.execute(query).fetchdf()


def label_count_summary(frame) -> dict[str, int]:
    """Return row and binary-label counts for a modelling partition."""
    n_rows = int(len(frame))
    n_positive = int(frame["label_high_engagement"].astype(int).sum())
    return {
        "n_rows": n_rows,
        "n_positive": n_positive,
        "n_negative": n_rows - n_positive,
    }


def fold_fit_provenance(
    fold: dict[str, object],
    train_df,
    train_plus_validation_df,
) -> dict[str, Any]:
    """Return explicit candidate-fit and final-refit provenance for one fold."""
    train_counts = label_count_summary(train_df)
    final_counts = label_count_summary(train_plus_validation_df)
    return {
        "candidate_fit_start_prediction_month": str(fold["train_start_prediction_month"]),
        "candidate_fit_end_prediction_month": str(fold["train_end_prediction_month"]),
        "candidate_fit_includes_validation": False,
        "candidate_fit_n_rows": train_counts["n_rows"],
        "candidate_fit_n_positive": train_counts["n_positive"],
        "candidate_fit_n_negative": train_counts["n_negative"],
        "final_refit_start_prediction_month": str(fold["train_start_prediction_month"]),
        "final_refit_end_prediction_month": str(fold["validation_prediction_month"]),
        "final_refit_includes_validation": True,
        "final_refit_n_rows": final_counts["n_rows"],
        "final_refit_n_positive": final_counts["n_positive"],
        "final_refit_n_negative": final_counts["n_negative"],
    }


def final_fit_iteration_provenance(pipeline, model_cfg: ModelConfig) -> dict[str, Any]:
    """Return observed iteration diagnostics from a fitted final logistic model."""
    import numpy as np

    logistic = pipeline.named_steps["logistic"]
    n_iter_values = np.asarray(getattr(logistic, "n_iter_", []), dtype=int).reshape(-1)
    if n_iter_values.size == 0:
        raise ValueError("Fitted LogisticRegression did not expose a non-empty n_iter_.")
    if np.any(n_iter_values < 1):
        raise ValueError(f"Fitted LogisticRegression exposed invalid n_iter_={n_iter_values.tolist()}.")
    configured_max_iter = int(model_cfg.max_iter)
    actual_n_iter = int(n_iter_values.max())
    if actual_n_iter > configured_max_iter:
        raise ValueError(
            "Observed LogisticRegression iterations exceed configured max_iter: "
            f"actual_n_iter={actual_n_iter}, configured_max_iter={configured_max_iter}."
        )
    return {
        "configured_max_iter": configured_max_iter,
        "actual_n_iter": actual_n_iter,
        "reached_max_iter": actual_n_iter >= configured_max_iter,
    }


def json_safe_value(value: Any) -> Any:
    """Convert fitted-estimator state to stable JSON-compatible values."""
    import numpy as np

    if isinstance(value, np.ndarray):
        return [json_safe_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def final_pipeline_state(
    pipeline,
    *,
    fold: dict[str, object],
    selected_features: Sequence[str],
    selected_candidate: dict[str, Any],
    fit_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Export a durable, non-pickle representation of one fitted final pipeline."""
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    logistic = pipeline.named_steps["logistic"]
    return {
        "fold_id": int(fold["fold_id"]),
        "test_prediction_month": str(fold["test_prediction_month"]),
        "feature_order": list(selected_features),
        "selected_candidate": json_safe_value(selected_candidate),
        "fit_provenance": json_safe_value(fit_provenance),
        "imputer": {
            "class": type(imputer).__name__,
            "parameters": json_safe_value(imputer.get_params(deep=False)),
            "statistics": json_safe_value(getattr(imputer, "statistics_", [])),
        },
        "scaler": {
            "class": type(scaler).__name__,
            "parameters": json_safe_value(scaler.get_params(deep=False)),
            "mean": json_safe_value(getattr(scaler, "mean_", [])),
            "scale": json_safe_value(getattr(scaler, "scale_", [])),
            "variance": json_safe_value(getattr(scaler, "var_", [])),
            "n_samples_seen": json_safe_value(getattr(scaler, "n_samples_seen_", None)),
        },
        "logistic": {
            "class": type(logistic).__name__,
            "parameters": json_safe_value(logistic.get_params(deep=False)),
            "classes": json_safe_value(getattr(logistic, "classes_", [])),
            "coefficients": json_safe_value(getattr(logistic, "coef_", [])),
            "intercept": json_safe_value(getattr(logistic, "intercept_", [])),
            "n_iter": json_safe_value(getattr(logistic, "n_iter_", [])),
        },
    }


def rank_and_score(
    frame,
    scores,
    cfg: LogisticConfig,
    *,
    fold: dict[str, object],
    split_role: str,
    metric_scope: str,
    candidate: Optional[dict[str, Any]],
    selected_candidate: Optional[dict[str, Any]],
    include_ranked_rows: bool,
    fit_provenance: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Rank scores within a month and compute ranking metrics."""
    rank_frame = frame[
        [
            "user_id",
            "feature_month",
            "prediction_month",
            "label_high_engagement",
            "target_score",
            "target_score_for_summary",
        ]
    ].copy()
    rank_frame["ranking_score"] = scores
    rank_frame["tie_break_score"] = [
        stable_hash_score(user_id, month, cfg.random_seed, f"{MODEL_NAME}_{split_role}")
        for user_id, month in zip(rank_frame["user_id"], rank_frame["prediction_month"])
    ]
    rank_frame = rank_frame.sort_values(
        ["ranking_score", "tie_break_score", "user_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    rank_frame["rank_within_month"] = range(1, len(rank_frame) + 1)
    rank_frame["label_high_engagement"] = rank_frame["label_high_engagement"].astype(int)
    rank_frame["cumulative_positive"] = rank_frame["label_high_engagement"].cumsum()

    n_users = int(len(rank_frame))
    n_positive = int(rank_frame["label_high_engagement"].sum())
    n_negative = n_users - n_positive
    if n_users <= 0 or n_positive <= 0 or n_negative <= 0:
        raise ValueError(
            f"Cannot score fold={fold['fold_id']} split={split_role}; "
            f"n_users={n_users}, n_positive={n_positive}, n_negative={n_negative}."
        )

    k_at_10pct = n_positive
    top_k = rank_frame.iloc[:k_at_10pct]
    top_k_positive_count = int(top_k["label_high_engagement"].sum())
    tp_at_10pct = top_k_positive_count
    fp_at_10pct = k_at_10pct - tp_at_10pct
    fn_at_10pct = n_positive - tp_at_10pct
    tn_at_10pct = n_negative - fp_at_10pct
    mcc_at_10pct = matthews_corrcoef_from_counts(tp_at_10pct, fp_at_10pct, fn_at_10pct, tn_at_10pct)
    positive_rows = rank_frame[rank_frame["label_high_engagement"] == 1]
    average_precision = float((positive_rows["cumulative_positive"] / positive_rows["rank_within_month"]).sum())
    average_precision /= float(n_positive)
    positive_rank_sum = float((n_users - positive_rows["rank_within_month"] + 1).sum())
    auroc = (
        positive_rank_sum - (float(n_positive) * (float(n_positive) + 1.0) / 2.0)
    ) / (float(n_positive) * float(n_negative))

    selected = selected_candidate or candidate or {}
    candidate_values = candidate or selected
    prediction_month = str(rank_frame["prediction_month"].iloc[0])
    feature_month = str(rank_frame["feature_month"].iloc[0])
    metric: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "feature_set": cfg.feature_set,
        "fold_id": int(fold["fold_id"]),
        "candidate_id": candidate_values.get("candidate_id", ""),
        "C": candidate_values.get("C", ""),
        "class_weight": candidate_values.get("class_weight", ""),
        "penalty": candidate_values.get("penalty", selected.get("penalty", "")),
        "regularization": candidate_values.get("regularization", selected.get("regularization", "")),
        "l1_ratio": candidate_values.get("l1_ratio", selected.get("l1_ratio", "")),
        "solver": candidate_values.get("solver", selected.get("solver", "")),
        "prediction_month": prediction_month,
        "test_prediction_month": str(fold["test_prediction_month"]),
        "feature_month": feature_month,
        "split_role": split_role,
        "metric_scope": metric_scope,
        **(fit_provenance or {}),
        "train_start_prediction_month": str(fold["train_start_prediction_month"]),
        "train_end_prediction_month": str(fold["train_end_prediction_month"]),
        "validation_prediction_month": str(fold["validation_prediction_month"]),
        "n_users": n_users,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_prevalence": float(n_positive) / float(n_users),
        "k_at_10pct": k_at_10pct,
        "top_k_positive_count": top_k_positive_count,
        "tp_at_10pct": tp_at_10pct,
        "fp_at_10pct": fp_at_10pct,
        "fn_at_10pct": fn_at_10pct,
        "tn_at_10pct": tn_at_10pct,
        "precision_at_10pct": float(top_k_positive_count) / float(k_at_10pct),
        "recall_at_10pct": float(top_k_positive_count) / float(n_positive),
        "mcc_at_10pct": mcc_at_10pct,
        "average_precision": average_precision,
        "auprc_ap": average_precision,
        "auroc": auroc,
        "mean_target_score_top_k": float(top_k["target_score_for_summary"].mean()),
        "mean_target_score_all": float(rank_frame["target_score_for_summary"].mean()),
        "top_k_rule": "k_at_10pct_equals_monthly_n_positive",
        "pr_auc_definition": PR_AUC_DEFINITION,
        "auroc_definition": AUROC_DEFINITION,
    }

    ranked_rows: list[dict[str, Any]] = []
    if include_ranked_rows:
        selected_candidate_id = selected.get("candidate_id", candidate_values.get("candidate_id", ""))
        selected_c = selected.get("C", candidate_values.get("C", ""))
        selected_class_weight = selected.get("class_weight", candidate_values.get("class_weight", ""))
        for row in rank_frame.itertuples(index=False):
            ranked_rows.append(
                {
                    "model_name": MODEL_NAME,
                    "feature_set": cfg.feature_set,
                    "fold_id": int(fold["fold_id"]),
                    "prediction_month": row.prediction_month,
                    "test_prediction_month": str(fold["test_prediction_month"]),
                    "feature_month": row.feature_month,
                    "split_role": split_role,
                    "metric_scope": metric_scope,
                    "user_id": row.user_id,
                    "selected_candidate_id": selected_candidate_id,
                    "selected_C": selected_c,
                    "selected_class_weight": selected_class_weight,
                    **(fit_provenance or {}),
                    "ranking_score": float(row.ranking_score),
                    "tie_break_score": float(row.tie_break_score),
                    "rank_within_month": int(row.rank_within_month),
                    "label_high_engagement": int(row.label_high_engagement),
                    "target_score": float(row.target_score),
                }
            )
    return metric, ranked_rows


def class_weight_for_sklearn(value: str) -> Optional[str]:
    """Map output class-weight labels to sklearn values."""
    return None if value == "none" else value


def parse_version_major_minor(version: str) -> tuple[int, int]:
    """Parse a package version into a major/minor pair."""
    def leading_int(part: str) -> Optional[int]:
        digits = []
        for char in part:
            if char.isdigit():
                digits.append(char)
            elif digits:
                break
        return int("".join(digits)) if digits else None

    parts = str(version).split(".")
    if len(parts) < 2:
        return (0, 0)
    major = leading_int(parts[0])
    minor = leading_int(parts[1])
    if major is None or minor is None:
        return (0, 0)
    return (major, minor)


def logistic_regression_kwargs(
    candidate: dict[str, Any],
    model_cfg: ModelConfig,
    *,
    sklearn_version: Optional[str] = None,
) -> dict[str, Any]:
    """Return sklearn LogisticRegression kwargs without using deprecated API on sklearn 1.8+."""
    if model_cfg.regularization != "l2_equivalent":
        raise ValueError(f"Unsupported logistic regularization: {model_cfg.regularization}")
    if float(model_cfg.l1_ratio) != 0.0:
        raise ValueError("Only l1_ratio=0.0 is supported for the L2-equivalent baseline.")

    kwargs: dict[str, Any] = {
        "solver": model_cfg.solver,
        "C": float(candidate["C"]),
        "class_weight": class_weight_for_sklearn(str(candidate["class_weight"])),
        "max_iter": int(model_cfg.max_iter),
        "random_state": 0,
    }
    if sklearn_version is None:
        import sklearn

        sklearn_version = getattr(sklearn, "__version__", "")
    if parse_version_major_minor(str(sklearn_version)) >= (1, 8):
        kwargs["l1_ratio"] = float(model_cfg.l1_ratio)
    else:
        kwargs["penalty"] = model_cfg.penalty
    return kwargs


def make_pipeline(candidate: dict[str, Any], model_cfg: ModelConfig):
    """Create a fresh sklearn Pipeline for one candidate."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=model_cfg.imputer_strategy)),
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    **logistic_regression_kwargs(candidate, model_cfg),
                ),
            ),
        ]
    )


def c_tie_break_rank(c_value: float, c_grid: Sequence[float]) -> int:
    """Return deterministic C tie-break rank, smaller C first."""
    return sorted(float(c) for c in c_grid).index(float(c_value)) + 1


def class_weight_tie_break_rank(class_weight: str) -> int:
    """Return deterministic class-weight tie-break rank."""
    return 1 if class_weight == "none" else 2


def selection_sort_key(row: dict[str, Any], cfg: LogisticConfig) -> tuple[float, float, float, int, int]:
    """Sort key for selecting candidates; lower tuple is better."""
    metric_value = float(row[cfg.selection_metric])
    auroc = float(row["auroc"])
    return (
        -metric_value,
        -auroc,
        float(row["C"]),
        class_weight_tie_break_rank(str(row["class_weight"])),
        int(row["candidate_id"]),
    )


def safe_odds_ratio(coefficient: float) -> float:
    """Exponentiate a coefficient without raising overflow."""
    if coefficient > 709:
        return float("inf")
    if coefficient < -745:
        return 0.0
    return math.exp(coefficient)


def coefficient_rows(
    pipeline,
    selected_features: Sequence[str],
    cfg: LogisticConfig,
    fold: dict[str, object],
    selected_candidate: dict[str, Any],
    fit_provenance: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return coefficient diagnostics for the selected train-plus-validation refit model."""
    logistic = pipeline.named_steps["logistic"]
    coefficients = logistic.coef_[0]
    intercept = float(logistic.intercept_[0])
    rows: list[dict[str, Any]] = []
    for feature_name, coefficient in zip(selected_features, coefficients):
        coef = float(coefficient)
        rows.append(
            {
                "model_name": MODEL_NAME,
                "feature_set": cfg.feature_set,
                "fold_id": int(fold["fold_id"]),
                "feature_name": feature_name,
                "coefficient": coef,
                "odds_ratio": safe_odds_ratio(coef),
                "intercept": intercept,
                "selected_candidate_id": selected_candidate["candidate_id"],
                "selected_C": selected_candidate["C"],
                "selected_class_weight": selected_candidate["class_weight"],
                "penalty": selected_candidate["penalty"],
                "regularization": selected_candidate["regularization"],
                "l1_ratio": selected_candidate["l1_ratio"],
                "solver": selected_candidate["solver"],
                **fit_provenance,
                "coefficient_note": (
                    "diagnostic_scaled_feature_space_from_final_train_plus_validation_refit; "
                    "interpret all-feature coefficients cautiously"
                ),
            }
        )
    return rows


def macro_mean(rows: Sequence[dict[str, Any]], column: str) -> float:
    """Return a float mean for a metric column."""
    return sum(float(row[column]) for row in rows) / float(len(rows))


def macro_stddev(rows: Sequence[dict[str, Any]], column: str) -> Optional[float]:
    """Return sample standard deviation for a metric column."""
    if len(rows) < 2:
        return None
    mean = macro_mean(rows, column)
    variance = sum((float(row[column]) - mean) ** 2 for row in rows) / float(len(rows) - 1)
    return math.sqrt(variance)


def summary_rows(fold_metrics: Sequence[dict[str, Any]], cfg: LogisticConfig) -> list[dict[str, Any]]:
    """Aggregate test metrics across folds."""
    test_rows = [row for row in fold_metrics if row["split_role"] == "test"]
    if len(test_rows) != len(PRIMARY_TEST_MONTHS_2021):
        raise ValueError(
            f"Expected {len(PRIMARY_TEST_MONTHS_2021)} test metric rows, observed {len(test_rows)}."
        )
    tp_pooled = sum(int(row["tp_at_10pct"]) for row in test_rows)
    fp_pooled = sum(int(row["fp_at_10pct"]) for row in test_rows)
    fn_pooled = sum(int(row["fn_at_10pct"]) for row in test_rows)
    tn_pooled = sum(int(row["tn_at_10pct"]) for row in test_rows)
    return [
        {
            "model_name": MODEL_NAME,
            "feature_set": cfg.feature_set,
            "metric_scope": "primary_rolling_origin_test",
            "source_split_role": "test",
            "n_folds": len(test_rows),
            "test_prediction_month_start": min(str(row["test_prediction_month"]) for row in test_rows),
            "test_prediction_month_end": max(str(row["test_prediction_month"]) for row in test_rows),
            "n_users_pooled": sum(int(row["n_users"]) for row in test_rows),
            "n_positive_pooled": sum(int(row["n_positive"]) for row in test_rows),
            "k_at_10pct_pooled": sum(int(row["k_at_10pct"]) for row in test_rows),
            "top_k_positive_count_pooled": sum(int(row["top_k_positive_count"]) for row in test_rows),
            "tp_at_10pct_pooled": tp_pooled,
            "fp_at_10pct_pooled": fp_pooled,
            "fn_at_10pct_pooled": fn_pooled,
            "tn_at_10pct_pooled": tn_pooled,
            "mcc_at_10pct_pooled": matthews_corrcoef_from_counts(tp_pooled, fp_pooled, fn_pooled, tn_pooled),
            "n_users_macro_mean": macro_mean(test_rows, "n_users"),
            "n_users_macro_stddev": macro_stddev(test_rows, "n_users"),
            "n_positive_macro_mean": macro_mean(test_rows, "n_positive"),
            "n_positive_macro_stddev": macro_stddev(test_rows, "n_positive"),
            "positive_prevalence_macro_mean": macro_mean(test_rows, "positive_prevalence"),
            "positive_prevalence_macro_stddev": macro_stddev(test_rows, "positive_prevalence"),
            "precision_at_10pct_macro_mean": macro_mean(test_rows, "precision_at_10pct"),
            "precision_at_10pct_macro_stddev": macro_stddev(test_rows, "precision_at_10pct"),
            "recall_at_10pct_macro_mean": macro_mean(test_rows, "recall_at_10pct"),
            "recall_at_10pct_macro_stddev": macro_stddev(test_rows, "recall_at_10pct"),
            "mcc_at_10pct_macro_mean": macro_mean(test_rows, "mcc_at_10pct"),
            "mcc_at_10pct_macro_stddev": macro_stddev(test_rows, "mcc_at_10pct"),
            "average_precision_macro_mean": macro_mean(test_rows, "average_precision"),
            "average_precision_macro_stddev": macro_stddev(test_rows, "average_precision"),
            "auprc_ap_macro_mean": macro_mean(test_rows, "auprc_ap"),
            "auprc_ap_macro_stddev": macro_stddev(test_rows, "auprc_ap"),
            "auroc_macro_mean": macro_mean(test_rows, "auroc"),
            "auroc_macro_stddev": macro_stddev(test_rows, "auroc"),
            "mean_target_score_top_k_macro_mean": macro_mean(test_rows, "mean_target_score_top_k"),
            "mean_target_score_top_k_macro_stddev": macro_stddev(test_rows, "mean_target_score_top_k"),
            "mean_target_score_all_macro_mean": macro_mean(test_rows, "mean_target_score_all"),
            "mean_target_score_all_macro_stddev": macro_stddev(test_rows, "mean_target_score_all"),
            "aggregation_note": "macro month-wise means over rolling-origin test folds; pooled totals are counts only",
            "pr_auc_definition": PR_AUC_DEFINITION,
            "auroc_definition": AUROC_DEFINITION,
        }
    ]


def validate_output_rows(
    validation_metrics: Sequence[dict[str, Any]],
    selected_rows: Sequence[dict[str, Any]],
    fold_metrics: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    hyperparameter_grid: Sequence[dict[str, Any]],
) -> None:
    """Validate output rows before writing files."""
    expected_validation = len(PRIMARY_TEST_MONTHS_2021) * len(hyperparameter_grid)
    if len(validation_metrics) != expected_validation:
        raise ValueError(
            f"Unexpected validation metric row count: observed={len(validation_metrics)}, "
            f"expected={expected_validation}."
        )
    if len(selected_rows) != len(PRIMARY_TEST_MONTHS_2021):
        raise ValueError(
            f"Unexpected selected hyperparameter row count: observed={len(selected_rows)}, "
            f"expected={len(PRIMARY_TEST_MONTHS_2021)}."
        )
    if len(fold_metrics) != len(PRIMARY_TEST_MONTHS_2021):
        raise ValueError(
            f"Unexpected test metric row count: observed={len(fold_metrics)}, "
            f"expected={len(PRIMARY_TEST_MONTHS_2021)}."
        )
    observed_test_months = tuple(sorted(str(row["test_prediction_month"]) for row in fold_metrics))
    if observed_test_months != PRIMARY_TEST_MONTHS_2021:
        raise ValueError(f"Unexpected test months: observed={observed_test_months}; expected={PRIMARY_TEST_MONTHS_2021}.")
    if any(row["split_role"] != "test" or row["metric_scope"] != "primary_rolling_origin_test" for row in fold_metrics):
        raise ValueError("logistic_metrics_by_fold.csv must contain only primary rolling-origin test rows.")
    for row in fold_metrics:
        if row.get("final_refit_includes_validation") is not True:
            raise ValueError("Final logistic test models must be refit with the validation month included.")
        if str(row.get("final_refit_end_prediction_month")) != str(row.get("validation_prediction_month")):
            raise ValueError("Final refit end month must equal the fold validation month.")
        if str(row.get("final_refit_end_prediction_month")) >= str(row.get("test_prediction_month")):
            raise ValueError("Final refit month range must not include the fold test month.")
        final_rows = int(row.get("final_refit_n_rows", 0))
        candidate_rows = int(row.get("candidate_fit_n_rows", 0))
        if final_rows <= candidate_rows:
            raise ValueError("Final refit row count must exceed the training-only candidate fit row count.")
        configured_max_iter = int(row.get("configured_max_iter", 0))
        actual_n_iter = int(row.get("actual_n_iter", 0))
        reached_max_iter = row.get("reached_max_iter") is True
        if configured_max_iter < 1 or actual_n_iter < 1 or actual_n_iter > configured_max_iter:
            raise ValueError(
                "Final logistic iteration provenance is invalid: "
                f"actual_n_iter={actual_n_iter}, configured_max_iter={configured_max_iter}."
            )
        if reached_max_iter != (actual_n_iter >= configured_max_iter):
            raise ValueError("Final logistic reached_max_iter is inconsistent with observed iteration counts.")
    if len(summaries) != 1 or summaries[0]["source_split_role"] != "test":
        raise ValueError("logistic_metrics_summary.csv must aggregate only test rows.")
    for collection_name, rows in {
        "validation": validation_metrics,
        "test": fold_metrics,
    }.items():
        for row in rows:
            for column in ("precision_at_10pct", "recall_at_10pct", "average_precision", "auprc_ap", "auroc"):
                value = float(row[column])
                if value < -1e-12 or value > 1.0 + 1e-12:
                    raise ValueError(f"{collection_name} metric {column} outside [0, 1]: {value}")
            mcc = float(row["mcc_at_10pct"])
            if not math.isfinite(mcc) or mcc < -1.0 - 1e-12 or mcc > 1.0 + 1e-12:
                raise ValueError(f"{collection_name} metric mcc_at_10pct outside [-1, 1] or non-finite: {mcc}")
            tp = int(row["tp_at_10pct"])
            fp = int(row["fp_at_10pct"])
            fn = int(row["fn_at_10pct"])
            tn = int(row["tn_at_10pct"])
            top_k_positive = int(row["top_k_positive_count"])
            k_at_10pct = int(row["k_at_10pct"])
            n_positive = int(row["n_positive"])
            n_negative = int(row["n_negative"])
            if tp != top_k_positive or fp != k_at_10pct - tp or fn != n_positive - tp or tn != n_negative - fp:
                raise ValueError(f"{collection_name} top-k confusion counts are inconsistent.")
            if any(value < 0 for value in (tp, fp, fn, tn)):
                raise ValueError(f"{collection_name} top-k confusion counts must be non-negative.")
            denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            if denominator <= 0:
                raise ValueError(f"{collection_name} MCC denominator is zero.")


def write_ranked_predictions(con: duckdb.DuckDBPyConnection, out_root: Path, ranked_rows: Sequence[dict[str, Any]]) -> None:
    """Write optional ranked predictions to Parquet."""
    import pandas as pd

    ranked_path = out_root / "logistic_ranked_predictions.parquet"
    ranked_df = pd.DataFrame(ranked_rows)
    con.register("logistic_ranked_predictions_output", ranked_df)
    con.execute(
        f"""
        COPY (
          SELECT *
          FROM logistic_ranked_predictions_output
          ORDER BY fold_id, prediction_month, rank_within_month
        )
        TO {quote_sql_string(ranked_path)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    con.unregister("logistic_ranked_predictions_output")


def metadata_payload(
    cfg: LogisticConfig,
    model_cfg: ModelConfig,
    duck: DuckDBConfig,
    selected_features: Sequence[str],
    hyperparameter_grid: Sequence[dict[str, Any]],
    outputs: dict[str, str],
    run_started_at: str,
    status: str,
    panel_row_count: Optional[int] = None,
) -> dict[str, Any]:
    """Build run metadata."""
    return {
        "script_name": "Modelling/1_1_run_logistic_baseline.py",
        "run_timestamp_utc": run_started_at,
        "git": git_metadata(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "duckdb": asdict(duck),
        "status": status,
        "panel_path": cfg.panel_path,
        "feature_schema_csv": cfg.feature_schema_csv,
        "output_root": cfg.out_root,
        "panel_row_count": panel_row_count,
        "truth_label_col": "label_high_engagement",
        "target_score_col_for_descriptive_summaries": cfg.target_score_col,
        "model_name": MODEL_NAME,
        "model_type": "regularised_logistic_regression",
        "model_fitting_performed": True,
        "model_config": asdict(model_cfg),
        "ranking_score": "predict_proba(X)[:, 1]",
        "feature_set": cfg.feature_set,
        "feature_set_definition": feature_set_definition(cfg.feature_set),
        "selected_feature_count": len(selected_features),
        "selected_features": list(selected_features),
        "hyperparameter_grid": list(hyperparameter_grid),
        "selection_metric": cfg.selection_metric,
        "selection_protocol": (
            "Candidates are fitted on training rows only and selected using validation-month metrics only. "
            "After selection, a fresh preprocessing-and-logistic pipeline with the selected hyperparameters is "
            "refitted on training plus validation rows and evaluated once on the test month."
        ),
        "final_refit_protocol": {
            "fresh_pipeline_per_fold": True,
            "refit_months": "training_prediction_months_plus_validation_prediction_month",
            "final_refit_includes_validation": True,
            "test_month_included_in_fit": False,
            "coefficient_source": "final_train_plus_validation_refit",
            "ranked_prediction_source": "final_train_plus_validation_refit",
        },
        "rolling_origin_protocol": {
            "prediction_months_required": list(EXPECTED_PREDICTION_MONTHS_2021),
            "minimum_training_window_months": cfg.min_train_months,
            "validation_window_months": cfg.validation_window_months,
            "test_window_months": cfg.test_window_months,
            "primary_test_months": list(PRIMARY_TEST_MONTHS_2021),
            "primary_metric_filter": 'split_role = "test"',
        },
        "metric_definitions": {
            "k_at_10pct": "monthly n_positive from label_high_engagement; equivalent to the constructed top-10% label count",
            "tp_at_10pct": "top_k_positive_count; true positives among users selected by the top-10% ranked cutoff.",
            "fp_at_10pct": "k_at_10pct - tp_at_10pct; false positives among top-10% selected users.",
            "fn_at_10pct": "n_positive - tp_at_10pct; labelled positives outside the top-10% selected users.",
            "tn_at_10pct": "n_negative - fp_at_10pct; labelled negatives outside the top-10% selected users.",
            "precision_at_10pct": "top_k_positive_count / k_at_10pct",
            "recall_at_10pct": "top_k_positive_count / n_positive",
            "mcc_at_10pct": (
                "Matthews correlation coefficient for the top-10% ranked cutoff using "
                "tp_at_10pct, fp_at_10pct, fn_at_10pct, and tn_at_10pct. "
                "If the denominator is zero, the implementation returns NaN; full Study 2 "
                "validation asserts non-zero denominators for evaluated folds."
            ),
            "average_precision": (
                "After deterministic descending ranking, mean precision at ranks where "
                "label_high_engagement == 1."
            ),
            "auprc_ap": "Same value as average_precision.",
            "pr_auc_definition": PR_AUC_DEFINITION,
            "auroc": "Mann-Whitney/rank-sum AUROC after deterministic tie-breaking; pooled AUROC is not reported.",
            "auroc_definition": AUROC_DEFINITION,
        },
        "tie_handling": TIE_HANDLING,
        "top_k_fraction": cfg.top_k_fraction,
        "top_k_fraction_note": (
            "The CLI value is fixed at 0.10 in this implementation to match the constructed label; "
            "metric k uses monthly n_positive."
        ),
        "smoke": cfg.smoke,
        "smoke_max_rows": cfg.smoke_max_rows if cfg.smoke else 0,
        "smoke_note": (
            "When --smoke --smoke-max-rows is used, the cap is applied per prediction month with label-stratified "
            "sampling to preserve fold structure where possible."
            if cfg.smoke
            else ""
        ),
        "outputs": outputs,
        "known_caveats": [
            "Coefficients are diagnostics in scaled/imputed feature space; all-feature coefficients are not independent effects.",
            "Fold summaries are descriptive because users may appear in multiple months.",
        ],
    }


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run the Study 2 L2-regularised logistic-regression baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--panel-path", required=True, help="Path to modelling_panel.parquet.")
    parser.add_argument("--feature-schema-csv", required=True, help="Path to feature_schema.csv.")
    parser.add_argument("--out-root", required=True, help="Output root for logistic baseline artefacts.")
    parser.add_argument(
        "--feature-set",
        required=True,
        choices=SUPPORTED_FEATURE_SETS,
        help="Pre-specified feature set to evaluate.",
    )
    parser.add_argument(
        "--reference-splits-csv",
        default=None,
        help="Optional rolling_origin_splits.csv to compare against before modelling.",
    )
    parser.add_argument("--c-grid", type=parse_c_grid, default=parse_c_grid(DEFAULT_C_GRID), help="Comma-separated C grid.")
    parser.add_argument(
        "--class-weight-options",
        type=parse_class_weight_options,
        default=parse_class_weight_options(DEFAULT_CLASS_WEIGHT_OPTIONS),
        help="Comma-separated class_weight options: none, balanced.",
    )
    parser.add_argument(
        "--selection-metric",
        default="average_precision",
        choices=["average_precision", "auprc_ap", "auroc", "precision_at_10pct", "recall_at_10pct"],
        help="Validation metric used for hyperparameter selection.",
    )
    parser.add_argument("--random-seed", type=int, default=2021, help="Random seed for deterministic tie-breaking.")
    parser.add_argument(
        "--target-score-col",
        default="target_score",
        help="Target-score column used only for optional mean target-score summaries.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite deterministic logistic outputs.")
    parser.add_argument(
        "--write-ranked-predictions",
        action="store_true",
        help="Write full selected test-month logistic_ranked_predictions.parquet.",
    )
    parser.add_argument(
        "--write-final-pipelines",
        action="store_true",
        help=(
            "Persist the six fitted final pipelines as a joblib bundle and a durable JSON "
            "parameter/state export. Joblib is version-dependent and must not be loaded from untrusted sources."
        ),
    )
    parser.add_argument(
        "--top-k-fraction",
        type=float,
        default=TOP_10_FRACTION,
        help="Reserved top-k fraction; currently fixed at 0.10 to match the constructed label.",
    )
    parser.add_argument("--smoke", action="store_true", help="Enable smoke-mode metadata and optional row limiting.")
    parser.add_argument(
        "--smoke-max-rows",
        type=int,
        default=0,
        help="If >0 and --smoke, keep up to this many rows per prediction month using label-stratified sampling.",
    )
    parser.add_argument("--min-train-months", type=int, default=4, help="Minimum rolling-origin training months.")
    parser.add_argument("--validation-window-months", type=int, default=1, help="Rolling-origin validation window.")
    parser.add_argument("--test-window-months", type=int, default=1, help="Rolling-origin test window.")
    parser.add_argument(
        "--penalty",
        default="l2",
        choices=["l2"],
        help="Legacy reporting label for the L2-equivalent regularisation path.",
    )
    parser.add_argument("--solver", default="lbfgs", choices=["lbfgs"], help="LogisticRegression solver.")
    parser.add_argument("--max-iter", type=int, default=5000, help="Maximum LogisticRegression iterations.")
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
        help="DuckDB temp_directory.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run logistic baseline evaluations."""
    parser = build_argparser()
    args = parser.parse_args(argv)
    validate_cli(args, parser)
    setup_logging(args.log_level)
    if duckdb is None:
        raise RuntimeError("duckdb is required to run logistic baseline metrics.")

    cfg = LogisticConfig(
        panel_path=str(args.panel_path),
        feature_schema_csv=str(args.feature_schema_csv),
        out_root=str(args.out_root),
        feature_set=str(args.feature_set),
        reference_splits_csv=str(args.reference_splits_csv) if args.reference_splits_csv else None,
        c_grid=tuple(args.c_grid),
        class_weight_options=tuple(args.class_weight_options),
        selection_metric=str(args.selection_metric),
        random_seed=int(args.random_seed),
        target_score_col=str(args.target_score_col),
        overwrite=bool(args.overwrite),
        write_ranked_predictions=bool(args.write_ranked_predictions),
        write_final_pipelines=bool(args.write_final_pipelines),
        top_k_fraction=float(args.top_k_fraction),
        smoke=bool(args.smoke),
        smoke_max_rows=int(args.smoke_max_rows),
        min_train_months=int(args.min_train_months),
        validation_window_months=int(args.validation_window_months),
        test_window_months=int(args.test_window_months),
    )
    model_cfg = ModelConfig(
        penalty=str(args.penalty),
        regularization="l2_equivalent",
        l1_ratio=0.0,
        solver=str(args.solver),
        max_iter=int(args.max_iter),
        imputer_strategy="median",
        scaler="StandardScaler",
    )
    duck = DuckDBConfig(
        memory_limit=str(args.duckdb_memory_limit),
        threads=int(args.duckdb_threads),
        temp_dir=str(args.duckdb_temp_dir),
    )
    run_started_at = utc_now_iso()

    out_root = Path(cfg.out_root)
    output_paths = known_output_paths(out_root)
    check_output_collisions(output_paths, cfg.overwrite)
    remove_outputs(output_paths, cfg.overwrite)
    ensure_dir(out_root)

    schema_rows = read_feature_schema(cfg.feature_schema_csv)
    selected_features, selected_feature_rows = resolve_feature_set(cfg.feature_set, schema_rows)
    feature_set_def = feature_set_definition(cfg.feature_set)
    hyperparameter_grid = make_hyperparameter_grid(cfg, model_cfg)

    outputs: dict[str, str] = {
        "resolved_config.json": str(out_root / "resolved_config.json"),
        "run_metadata.json": str(out_root / "run_metadata.json"),
        "selected_features.csv": str(out_root / "selected_features.csv"),
        "logistic_hyperparameter_grid.csv": str(out_root / "logistic_hyperparameter_grid.csv"),
    }
    safe_json_dump(
        {
            "config": asdict(cfg),
            "model": asdict(model_cfg),
            "duckdb": asdict(duck),
            "feature_set_definition": feature_set_def,
            "feature_selection": {
                "feature_set": cfg.feature_set,
                "selected_feature_count": len(selected_features),
                "selected_features": list(selected_features),
            },
        },
        out_root / "resolved_config.json",
    )
    write_csv_rows(out_root / "selected_features.csv", selected_feature_rows, SELECTED_FEATURE_FIELDS)
    write_csv_rows(out_root / "logistic_hyperparameter_grid.csv", hyperparameter_grid, HYPERPARAMETER_GRID_FIELDS)

    logger.info("Reading panel: %s", cfg.panel_path)
    logger.info("Feature set: %s (%s features)", cfg.feature_set, len(selected_features))
    logger.info("Output root: %s", out_root)
    if cfg.smoke:
        logger.warning(
            "Smoke mode enabled. Per-month row cap: %s",
            int(cfg.smoke_max_rows) if int(cfg.smoke_max_rows) > 0 else "none",
        )

    con = duckdb.connect()
    try:
        configure_duckdb(con, duck)
        create_panel_source_view(con, cfg)
        columns = relation_columns(con, "panel_source")
        validate_panel_preconditions(con, cfg, columns, selected_features)

        feature_map = prediction_month_feature_map(con)
        split_rows, primary_test_rows = build_rolling_origin_splits(feature_map, cfg)
        if cfg.reference_splits_csv:
            compare_reference_splits(split_rows, cfg.reference_splits_csv)
        write_csv_rows(out_root / "rolling_origin_splits.csv", split_rows, split_fieldnames())
        outputs["rolling_origin_splits.csv"] = str(out_root / "rolling_origin_splits.csv")
        validate_fold_label_balance(con, cfg, primary_test_rows)

        validation_metrics: list[dict[str, Any]] = []
        selected_hyperparameters: list[dict[str, Any]] = []
        fold_metrics: list[dict[str, Any]] = []
        coefficients: list[dict[str, Any]] = []
        ranked_predictions: list[dict[str, Any]] = []
        final_pipeline_states: list[dict[str, Any]] = []
        final_pipelines: dict[int, Any] = {}

        all_months = list(EXPECTED_PREDICTION_MONTHS_2021)
        for fold in primary_test_rows:
            fold_id = int(fold["fold_id"])
            train_end = str(fold["train_end_prediction_month"])
            train_months = all_months[: all_months.index(train_end) + 1]
            validation_month = str(fold["validation_prediction_month"])
            test_month = str(fold["test_prediction_month"])
            logger.info(
                "Fold %s: train=%s--%s validation=%s test=%s",
                fold_id,
                train_months[0],
                train_months[-1],
                validation_month,
                test_month,
            )

            train_df = fetch_partition_frame(con, cfg, selected_features, train_months)
            validation_df = fetch_partition_frame(con, cfg, selected_features, [validation_month])
            test_df = fetch_partition_frame(con, cfg, selected_features, [test_month])
            x_train = train_df[list(selected_features)]
            y_train = train_df["label_high_engagement"].astype(int)
            x_validation = validation_df[list(selected_features)]
            x_test = test_df[list(selected_features)]

            fold_validation_metrics: list[dict[str, Any]] = []
            for candidate in hyperparameter_grid:
                pipeline = make_pipeline(candidate, model_cfg)
                pipeline.fit(x_train, y_train)
                validation_scores = pipeline.predict_proba(x_validation)[:, 1]
                metric, _ = rank_and_score(
                    validation_df,
                    validation_scores,
                    cfg,
                    fold=fold,
                    split_role="validation",
                    metric_scope="validation_model_selection",
                    candidate=candidate,
                    selected_candidate=None,
                    fit_provenance={
                        "candidate_fit_start_prediction_month": str(fold["train_start_prediction_month"]),
                        "candidate_fit_end_prediction_month": str(fold["train_end_prediction_month"]),
                        "candidate_fit_includes_validation": False,
                        "candidate_fit_n_rows": len(train_df),
                        "candidate_fit_n_positive": int(y_train.sum()),
                        "candidate_fit_n_negative": int(len(y_train) - int(y_train.sum())),
                    },
                    include_ranked_rows=False,
                )
                metric["selection_metric"] = cfg.selection_metric
                metric["selection_tie_break_order"] = (
                    f"{cfg.selection_metric} desc, auroc desc, C asc, class_weight none before balanced, candidate_id asc"
                )
                fold_validation_metrics.append(metric)

            ranked_candidates = sorted(fold_validation_metrics, key=lambda row: selection_sort_key(row, cfg))
            selected_metric = ranked_candidates[0]
            for rank, metric in enumerate(ranked_candidates, start=1):
                metric["selection_rank_within_fold"] = rank
                metric["selected_by_validation"] = rank == 1
            validation_metrics.extend(sorted(ranked_candidates, key=lambda row: int(row["candidate_id"])))

            selected_candidate_id = int(selected_metric["candidate_id"])
            selected_candidate = next(
                row for row in hyperparameter_grid if int(row["candidate_id"]) == selected_candidate_id
            )
            import pandas as pd

            train_plus_validation_df = pd.concat([train_df, validation_df], ignore_index=True)
            fit_provenance = fold_fit_provenance(fold, train_df, train_plus_validation_df)
            x_train_plus_validation = train_plus_validation_df[list(selected_features)]
            y_train_plus_validation = train_plus_validation_df["label_high_engagement"].astype(int)
            selected_pipeline = make_pipeline(selected_candidate, model_cfg)
            selected_pipeline.fit(x_train_plus_validation, y_train_plus_validation)
            fit_provenance.update(final_fit_iteration_provenance(selected_pipeline, model_cfg))
            if cfg.write_final_pipelines:
                final_pipeline_states.append(
                    final_pipeline_state(
                        selected_pipeline,
                        fold=fold,
                        selected_features=selected_features,
                        selected_candidate=selected_candidate,
                        fit_provenance=fit_provenance,
                    )
                )
                final_pipelines[fold_id] = selected_pipeline
            selected_hyperparameters.append(
                {
                    "model_name": MODEL_NAME,
                    "feature_set": cfg.feature_set,
                    "fold_id": fold_id,
                    "selected_candidate_id": selected_candidate_id,
                    "selected_C": selected_candidate["C"],
                    "selected_class_weight": selected_candidate["class_weight"],
                    "penalty": selected_candidate["penalty"],
                    "regularization": selected_candidate["regularization"],
                    "l1_ratio": selected_candidate["l1_ratio"],
                    "solver": selected_candidate["solver"],
                    "selection_metric": cfg.selection_metric,
                    "selected_validation_metric_value": selected_metric[cfg.selection_metric],
                    "validation_average_precision": selected_metric["average_precision"],
                    "validation_auroc": selected_metric["auroc"],
                    "validation_precision_at_10pct": selected_metric["precision_at_10pct"],
                    "validation_recall_at_10pct": selected_metric["recall_at_10pct"],
                    "validation_mcc_at_10pct": selected_metric["mcc_at_10pct"],
                    "validation_C_tie_break_rank": c_tie_break_rank(float(selected_candidate["C"]), cfg.c_grid),
                    "validation_class_weight_tie_break_rank": class_weight_tie_break_rank(str(selected_candidate["class_weight"])),
                    "selection_rank_within_fold": 1,
                    "selection_tie_break_order": selected_metric["selection_tie_break_order"],
                    "train_start_prediction_month": str(fold["train_start_prediction_month"]),
                    "train_end_prediction_month": str(fold["train_end_prediction_month"]),
                    "validation_prediction_month": str(fold["validation_prediction_month"]),
                    "test_prediction_month": str(fold["test_prediction_month"]),
                    **fit_provenance,
                }
            )
            coefficients.extend(
                coefficient_rows(selected_pipeline, selected_features, cfg, fold, selected_candidate, fit_provenance)
            )

            test_scores = selected_pipeline.predict_proba(x_test)[:, 1]
            test_metric, test_ranked_rows = rank_and_score(
                test_df,
                test_scores,
                cfg,
                fold=fold,
                split_role="test",
                metric_scope="primary_rolling_origin_test",
                candidate=None,
                selected_candidate=selected_candidate,
                fit_provenance=fit_provenance,
                include_ranked_rows=cfg.write_ranked_predictions,
            )
            test_metric.update(
                {
                    "selected_candidate_id": selected_candidate_id,
                    "selected_C": selected_candidate["C"],
                    "selected_class_weight": selected_candidate["class_weight"],
                    "penalty": selected_candidate["penalty"],
                    "regularization": selected_candidate["regularization"],
                    "l1_ratio": selected_candidate["l1_ratio"],
                    "solver": selected_candidate["solver"],
                }
            )
            fold_metrics.append(test_metric)
            ranked_predictions.extend(test_ranked_rows)

        summaries = summary_rows(fold_metrics, cfg)
        validate_output_rows(
            validation_metrics,
            selected_hyperparameters,
            fold_metrics,
            summaries,
            hyperparameter_grid,
        )

        write_csv_rows(
            out_root / "logistic_validation_metrics_by_candidate.csv",
            validation_metrics,
            VALIDATION_METRIC_FIELDS,
        )
        outputs["logistic_validation_metrics_by_candidate.csv"] = str(out_root / "logistic_validation_metrics_by_candidate.csv")
        write_csv_rows(
            out_root / "logistic_selected_hyperparameters_by_fold.csv",
            selected_hyperparameters,
            SELECTED_HYPERPARAMETER_FIELDS,
        )
        outputs["logistic_selected_hyperparameters_by_fold.csv"] = str(out_root / "logistic_selected_hyperparameters_by_fold.csv")
        write_csv_rows(out_root / "logistic_metrics_by_fold.csv", fold_metrics, FOLD_METRIC_FIELDS)
        outputs["logistic_metrics_by_fold.csv"] = str(out_root / "logistic_metrics_by_fold.csv")
        write_csv_rows(out_root / "logistic_metrics_summary.csv", summaries, SUMMARY_FIELDS)
        outputs["logistic_metrics_summary.csv"] = str(out_root / "logistic_metrics_summary.csv")
        write_csv_rows(out_root / "logistic_coefficients_by_fold.csv", coefficients, COEFFICIENT_FIELDS)
        outputs["logistic_coefficients_by_fold.csv"] = str(out_root / "logistic_coefficients_by_fold.csv")

        if cfg.write_final_pipelines:
            import joblib

            state_path = out_root / "logistic_final_pipeline_state_by_fold.json"
            joblib_path = out_root / "logistic_final_pipelines.joblib"
            safe_json_dump(
                {
                    "schema_version": "study2_logistic_final_pipeline_state_v1",
                    "model_name": MODEL_NAME,
                    "feature_set": cfg.feature_set,
                    "package_versions": package_versions(),
                    "fold_models": final_pipeline_states,
                    "joblib_security_note": (
                        "Version-dependent convenience archive; load only when the file and its provenance are trusted."
                    ),
                },
                state_path,
            )
            joblib.dump(final_pipelines, joblib_path, compress=3)
            outputs[state_path.name] = str(state_path)
            outputs[joblib_path.name] = str(joblib_path)

        if cfg.write_ranked_predictions:
            write_ranked_predictions(con, out_root, ranked_predictions)
            outputs["logistic_ranked_predictions.parquet"] = str(out_root / "logistic_ranked_predictions.parquet")

        panel_row_count = int(con.execute("SELECT COUNT(*) FROM panel_source").fetchone()[0])
        safe_json_dump(
            metadata_payload(
                cfg=cfg,
                model_cfg=model_cfg,
                duck=duck,
                selected_features=selected_features,
                hyperparameter_grid=hyperparameter_grid,
                outputs=outputs,
                run_started_at=run_started_at,
                status="finished",
                panel_row_count=panel_row_count,
            ),
            out_root / "run_metadata.json",
        )
    finally:
        con.close()

    logger.info("Done. Wrote logistic baseline outputs under %s", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
