from __future__ import annotations

import logging
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


REQUIRED_OUTPUT_COLUMNS: List[str] = [
    "subreddit",
    "eligible_full_period",
    "n_months_observed",
    "min_monthly_active_users",
    "mean_monthly_active_users",
    "total_activity",
    "n_submissions",
    "n_comments",
    "mean_thread_size",
    "median_thread_size",
    "repeat_user_share_2plus",
    "repeat_user_share_5plus",
    "activity_cv",
    "nsfw_submission_share",
    "R_A",
    "R_E",
    "R_U",
    "R_T",
    "S_mean",
    "S_max",
    "rank_S_mean",
    "rank_S_max",
    "candidate_by_S_mean",
    "candidate_by_S_max",
    "language_scan_status",
    "n_language_texts_sampled",
    "english_share",
    "non_english_share",
    "dominant_language",
    "language_filter_passed",
    "strict_language_filter_passed",
    "include_in_main_modelling",
    "selection_basis",
    "selection_note",
]


ESSENTIAL_INPUT_COLUMNS: List[str] = [
    "subreddit",
    "n_submissions",
    "n_comments",
    "n_total_activity",
    "n_months_observed",
    "min_monthly_active_users",
    "mean_thread_size",
    "mean_pct_users_2plus_activity",
    "cv_monthly_total_activity",
]


def require_columns(source_name: str, available: Sequence[str], required: Iterable[str]) -> None:
    """Raise a helpful error if an input table is missing required columns."""
    available_set = set(available)
    missing = [col for col in required if col not in available_set]
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {', '.join(missing)}. "
            f"Available columns: {', '.join(available)}"
        )


def percentile_rank(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    Compute the project percentile rank with average ties.

    For ascending values:
        prank(X_i) = (1 + count(X_j < X_i) + 0.5 * count(X_j = X_i, j != i)) / N

    When ``higher_is_better`` is false, smaller raw values receive higher ranks.
    Missing values are excluded from N and remain missing.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    transformed = numeric if higher_is_better else -numeric
    out = pd.Series(np.nan, index=values.index, dtype=float)
    mask = transformed.notna()
    if not mask.any():
        return out
    out.loc[mask] = transformed.loc[mask].rank(method="average", pct=True, ascending=True)
    return out


def _number(df: pd.DataFrame, column: str, default: object = pd.NA) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(default, index=df.index, dtype="Float64")


def _first_present_number(df: pd.DataFrame, columns: Sequence[str], default: object = pd.NA) -> pd.Series:
    for column in columns:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(default, index=df.index, dtype="Float64")


def _true_mask(series: pd.Series) -> pd.Series:
    return series.map(lambda value: False if pd.isna(value) else bool(value)).astype(bool)


def prepare_candidate_frame(
    source: pd.DataFrame,
    min_monthly_active_users: int,
    nsfw_submission_share_threshold: float,
) -> pd.DataFrame:
    """Normalize 06_0 subset-selection columns and recompute quantitative eligibility."""
    require_columns("subreddit_subset_selection_summary.csv", list(source.columns), ESSENTIAL_INPUT_COLUMNS)

    df = source.copy()
    out = pd.DataFrame(index=df.index)
    out["subreddit"] = df["subreddit"].astype("string")
    out["n_submissions"] = _number(df, "n_submissions")
    out["n_comments"] = _number(df, "n_comments")
    out["total_activity"] = _number(df, "n_total_activity")
    out["n_months_observed"] = _number(df, "n_months_observed").astype("Int64")
    out["min_monthly_active_users"] = _number(df, "min_monthly_active_users").astype("Int64")
    out["mean_monthly_active_users"] = _first_present_number(
        df,
        ["mean_monthly_active_users_observed", "mean_monthly_active_users"],
    )
    out["mean_thread_size"] = _number(df, "mean_thread_size")
    out["median_thread_size"] = _number(df, "median_thread_size")
    if "median_thread_size" not in df.columns:
        logger.warning("median_thread_size is not available in the 06_0 input; writing NA.")
    out["repeat_user_share_2plus"] = _number(df, "mean_pct_users_2plus_activity")
    out["repeat_user_share_5plus"] = _number(df, "mean_pct_users_5plus_activity")
    if "mean_pct_users_5plus_activity" not in df.columns:
        logger.warning("mean_pct_users_5plus_activity is not available in the 06_0 input; writing NA.")
    out["activity_cv"] = _number(df, "cv_monthly_total_activity")
    out["nsfw_submission_share"] = _number(df, "nsfw_submission_share")
    if "nsfw_submission_share" not in df.columns:
        logger.warning("nsfw_submission_share is not available; NSFW filtering will be treated as unavailable.")

    if "n_months_expected" in df.columns:
        expected_months = pd.to_numeric(df["n_months_expected"], errors="coerce")
    else:
        max_months = out["n_months_observed"].max(skipna=True)
        logger.warning(
            "n_months_expected is not available in the 06_0 input; inferring expected coverage as max n_months_observed=%s.",
            max_months,
        )
        expected_months = pd.Series(max_months, index=out.index)
    out["n_months_expected"] = expected_months.astype("Int64")

    out["eligible_full_period"] = (
        out["n_months_observed"].notna()
        & out["n_months_expected"].notna()
        & (out["n_months_observed"] == out["n_months_expected"])
    )
    out["ordinary_subreddit"] = ~out["subreddit"].fillna("").str.lower().str.startswith("u_")
    out["active_user_threshold_passed"] = (
        out["min_monthly_active_users"].notna()
        & (out["min_monthly_active_users"] >= int(min_monthly_active_users))
    )
    out["nsfw_threshold_passed"] = (
        out["nsfw_submission_share"].isna()
        | (out["nsfw_submission_share"] < float(nsfw_submission_share_threshold))
    )
    out["eligible_quantitative"] = (
        out["ordinary_subreddit"]
        & out["eligible_full_period"]
        & out["active_user_threshold_passed"]
        & out["nsfw_threshold_passed"]
    )

    out["min_monthly_active_users_required"] = int(min_monthly_active_users)
    out["nsfw_submission_share_threshold"] = float(nsfw_submission_share_threshold)
    out["selection_note"] = ""
    return out


def _assign_rank(df: pd.DataFrame, score_col: str, rank_col: str) -> pd.Series:
    ranks = pd.Series(pd.NA, index=df.index, dtype="Int64")
    ranked = (
        df.loc[df[score_col].notna()]
        .sort_values([score_col, "total_activity", "subreddit"], ascending=[False, False, True])
    )
    if len(ranked):
        ranks.loc[ranked.index] = np.arange(1, len(ranked) + 1, dtype=int)
    return ranks


def score_candidates(df: pd.DataFrame, candidate_score_threshold: float) -> pd.DataFrame:
    """Compute R_A/R_E/R_U/R_T, S_mean/S_max, ranks, and candidate flags."""
    out = df.copy()
    for column in ["R_A", "R_E", "R_U", "R_T", "S_mean", "S_max"]:
        out[column] = np.nan

    eligible_mask = _true_mask(out["eligible_quantitative"])
    if eligible_mask.any():
        eligible = out.loc[eligible_mask]
        out.loc[eligible_mask, "R_A"] = percentile_rank(eligible["total_activity"], higher_is_better=True)
        out.loc[eligible_mask, "R_E"] = percentile_rank(eligible["mean_thread_size"], higher_is_better=True)
        out.loc[eligible_mask, "R_U"] = percentile_rank(eligible["repeat_user_share_2plus"], higher_is_better=True)
        out.loc[eligible_mask, "R_T"] = percentile_rank(eligible["activity_cv"], higher_is_better=False)

    components = out[["R_A", "R_E", "R_U", "R_T"]]
    complete_components = eligible_mask & components.notna().all(axis=1)
    out.loc[complete_components, "S_mean"] = components.loc[complete_components].mean(axis=1)
    out.loc[complete_components, "S_max"] = components.loc[complete_components].max(axis=1)
    out["rank_S_mean"] = _assign_rank(out, "S_mean", "rank_S_mean")
    out["rank_S_max"] = _assign_rank(out, "S_max", "rank_S_max")
    out["candidate_by_S_mean"] = out["S_mean"].ge(float(candidate_score_threshold))
    out["candidate_by_S_max"] = out["S_max"].ge(float(candidate_score_threshold))
    out["include_in_main_modelling"] = False
    out["selection_basis"] = ""
    return sort_candidates(out)


def sort_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Sort the audit table by primary balanced-score ranking."""
    return df.sort_values(
        ["S_mean", "total_activity", "subreddit"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def top_by_score(df: pd.DataFrame, score_col: str, top_n: int) -> pd.DataFrame:
    """Return deterministic top rows for a score column."""
    return (
        df.loc[df[score_col].notna()]
        .sort_values([score_col, "total_activity", "subreddit"], ascending=[False, False, True])
        .head(int(top_n))
        .copy()
    )


def language_candidate_pool(df: pd.DataFrame, pool_size: int) -> List[str]:
    """Return union of top S_mean and top S_max candidates before language filtering."""
    seen: set[str] = set()
    pool: List[str] = []
    for score_col in ("S_mean", "S_max"):
        for subreddit in top_by_score(df, score_col, int(pool_size))["subreddit"].dropna().astype(str):
            if subreddit not in seen:
                seen.add(subreddit)
                pool.append(subreddit)
    return pool


def apply_final_selection(
    df: pd.DataFrame,
    top_n: int,
    language_mode: str,
) -> pd.DataFrame:
    """Mark the final S_mean-selected rows for downstream review/editing."""
    out = df.copy()
    out["include_in_main_modelling"] = False
    out["selection_basis"] = ""

    selectable = out["S_mean"].notna()
    if language_mode == "sample-text":
        selectable = selectable & _true_mask(out["language_filter_passed"])
        basis = "topN_S_mean_anglophone"
    else:
        basis = "topN_S_mean_language_not_run"

    selected = (
        out.loc[selectable]
        .sort_values(["S_mean", "total_activity", "subreddit"], ascending=[False, False, True])
        .head(int(top_n))
    )
    out.loc[selected.index, "include_in_main_modelling"] = True
    out.loc[selected.index, "selection_basis"] = basis
    if language_mode == "none":
        out.loc[selected.index, "selection_note"] = "language scan not run"

    if language_mode == "sample-text":
        failed = out["S_mean"].notna() & out["language_filter_passed"].eq(False)
        out.loc[failed, "selection_note"] = "failed English threshold"
    return sort_candidates(out)


def ensure_required_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all required audit columns are present and ordered first."""
    out = df.copy()
    for column in REQUIRED_OUTPUT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    extras = [col for col in out.columns if col not in REQUIRED_OUTPUT_COLUMNS]
    return out[REQUIRED_OUTPUT_COLUMNS + extras]
