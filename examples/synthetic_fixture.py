"""Invented account-month fixture from the logistic regression tests."""

from pathlib import Path

import duckdb


FEATURE_ROWS = (
    ("feature_n_submissions_prev", "activity"),
    ("feature_n_comments_prev", "activity"),
    ("feature_n_total_activity_prev", "activity"),
    ("feature_submission_share_prev", "activity"),
    ("feature_comment_share_prev", "activity"),
    ("feature_n_active_days_prev", "activity"),
    ("feature_mean_activity_per_active_day_prev", "activity"),
    ("feature_received_engagement_total_prev", "historical_engagement"),
    ("feature_received_engagement_per_activity_prev", "historical_engagement"),
    ("feature_log_received_engagement_rate_prev", "historical_engagement"),
    ("feature_received_comments_on_submissions_prev", "historical_engagement"),
    ("feature_received_replies_to_comments_prev", "historical_engagement"),
    ("feature_overlap_submission_and_comment_reply_events_prev", "historical_engagement"),
    ("feature_n_active_subreddits_prev", "subreddit_participation"),
    ("feature_subreddit_hhi_prev", "subreddit_participation"),
    ("feature_subreddit_entropy_prev", "subreddit_participation"),
    ("feature_top_subreddit_activity_share_prev", "subreddit_participation"),
    ("feature_is_single_subreddit_user_prev", "subreddit_participation"),
    ("feature_month_number", "calendar"),
)


def previous_month(month: str) -> str:
    year_s, month_s = month.split("-")
    year = int(year_s)
    month_i = int(month_s)
    if month_i == 1:
        return f"{year - 1}-12"
    return f"{year}-{month_i - 1:02d}"


def synthetic_feature_values(user_index: int, month_index: int, label: int) -> dict[str, float]:
    signal = 10.0 if label else 0.0
    total_activity = float(user_index + month_index + 5)
    submissions = float((user_index % 4) + 1)
    comments = total_activity - submissions
    submission_share = submissions / total_activity
    comment_share = comments / total_activity
    received_total = float(signal + (31 - user_index) * 0.3 + month_index)
    replies = float(signal / 2.0 + (user_index % 7))
    comments_on_submissions = received_total + 1.0
    overlap = comments_on_submissions + replies - received_total
    active_subreddits = float((user_index % 5) + 1)
    hhi = 1.0 / active_subreddits
    return {
        "feature_n_submissions_prev": submissions,
        "feature_n_comments_prev": comments,
        "feature_n_total_activity_prev": total_activity,
        "feature_submission_share_prev": submission_share,
        "feature_comment_share_prev": comment_share,
        "feature_n_active_days_prev": float((user_index % 9) + 1),
        "feature_mean_activity_per_active_day_prev": total_activity / float((user_index % 9) + 1),
        "feature_received_engagement_total_prev": received_total,
        "feature_received_engagement_per_activity_prev": received_total / total_activity,
        "feature_log_received_engagement_rate_prev": received_total / (total_activity + 1.0),
        "feature_received_comments_on_submissions_prev": comments_on_submissions,
        "feature_received_replies_to_comments_prev": replies,
        "feature_overlap_submission_and_comment_reply_events_prev": overlap,
        "feature_n_active_subreddits_prev": active_subreddits,
        "feature_subreddit_hhi_prev": hhi,
        "feature_subreddit_entropy_prev": active_subreddits / 10.0,
        "feature_top_subreddit_activity_share_prev": min(1.0, hhi + 0.2),
        "feature_is_single_subreddit_user_prev": 1.0 if active_subreddits == 1 else 0.0,
        "feature_month_number": float(int(previous_month(f"2021-{month_index + 1:02d}").split("-")[1])),
    }


def create_synthetic_panel(panel_path: Path, drop_features: set[str] | None = None) -> None:
    assert duckdb is not None
    drop_features = drop_features or set()
    feature_names = [name for name, _ in FEATURE_ROWS if name not in drop_features]
    feature_columns = ",\n                      ".join(f"{name} DOUBLE" for name in feature_names)
    create_sql = f"""
    CREATE TABLE synthetic_panel (
      user_id VARCHAR,
      feature_month VARCHAR,
      prediction_month VARCHAR,
      label_high_engagement INTEGER,
      target_score DOUBLE,
      {feature_columns}
    )
    """
    insert_columns = [
        "user_id",
        "feature_month",
        "prediction_month",
        "label_high_engagement",
        "target_score",
        *feature_names,
    ]
    placeholders = ", ".join("?" for _ in insert_columns)
    con = duckdb.connect()
    try:
        con.execute(create_sql)
        rows = []
        prediction_months = [f"2021-{month:02d}" for month in range(2, 13)]
        for month_index, prediction_month in enumerate(prediction_months, start=1):
            feature_month = previous_month(prediction_month)
            for user_index in range(1, 31):
                label = 1 if user_index <= 3 else 0
                values = synthetic_feature_values(user_index, month_index, label)
                target_score = float(100 - user_index if label else user_index / 10.0)
                rows.append(
                    (
                        f"user_{user_index:03d}",
                        feature_month,
                        prediction_month,
                        label,
                        target_score,
                        *(values[name] for name in feature_names),
                    )
                )
        con.executemany(f"INSERT INTO synthetic_panel VALUES ({placeholders})", rows)
        con.execute(f"COPY synthetic_panel TO '{panel_path.as_posix()}' (FORMAT PARQUET)")
    finally:
        con.close()
