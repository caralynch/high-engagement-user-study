#!/usr/bin/env python3
"""
Convert existing large CSV outputs from 06_0_descriptive_plots.py to smaller formats.

Usage:
  python Preprocessing/06_1_convert_existing_large_tables.py --tables-dir ../descriptive_plots_2021/tables
  python Preprocessing/06_1_convert_existing_large_tables.py --tables-dir ../descriptive_plots_2021/tables --delete-originals
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LARGE_INTERMEDIATE_TABLE_NAMES = {
    "per_user_activity",
    "per_user_active_months",
    "user_subreddit_hhi",
    "lag_relationships",
    "lag_relationships_valid_pairs",
    "thread_size_distribution",
    "user_activity_lorenz_curve",
    "user_engagement_lorenz_curve",
    "active_subreddits_per_user_distribution",
}

TAIL_AUDIT_TABLE_NAMES = {
    "lag_past_activity_future_activity_decile10_tail_usermonths",
    "lag_past_activity_future_activity_decile10_tail_users_aggregated",
    "lag_past_activity_future_engagement_decile10_tail_usermonths",
    "lag_past_activity_future_engagement_decile10_tail_users_aggregated",
    "lag_past_engagement_future_engagement_decile10_tail_usermonths",
    "lag_past_engagement_future_engagement_decile10_tail_users_aggregated",
    "active_subreddits_tail_users",
}


def convert_csv_to_parquet(csv_path: Path, delete_original: bool) -> None:
    out_path = csv_path.with_suffix(".parquet")

    print(f"[parquet] {csv_path.name} -> {out_path.name}")
    df = pd.read_csv(csv_path)
    df.to_parquet(out_path, index=False, compression="zstd")

    if delete_original:
        csv_path.unlink()
        print(f"  deleted original: {csv_path.name}")


def convert_csv_to_gzip_csv(csv_path: Path, delete_original: bool) -> None:
    out_path = csv_path.with_suffix(csv_path.suffix + ".gz")

    print(f"[csv.gz]  {csv_path.name} -> {out_path.name}")
    df = pd.read_csv(csv_path)
    df.to_csv(out_path, index=False, compression="gzip")

    if delete_original:
        csv_path.unlink()
        print(f"  deleted original: {csv_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", required=True, help="Path to the tables directory.")
    parser.add_argument(
        "--delete-originals",
        action="store_true",
        help="Delete original CSV files after successful conversion.",
    )
    args = parser.parse_args()

    tables_dir = Path(args.tables_dir).expanduser().resolve()
    if not tables_dir.exists():
        raise FileNotFoundError(f"Tables directory not found: {tables_dir}")

    for name in sorted(LARGE_INTERMEDIATE_TABLE_NAMES):
        csv_path = tables_dir / f"{name}.csv"
        if csv_path.exists():
            convert_csv_to_parquet(csv_path, args.delete_originals)

    for name in sorted(TAIL_AUDIT_TABLE_NAMES):
        csv_path = tables_dir / f"{name}.csv"
        if csv_path.exists():
            convert_csv_to_gzip_csv(csv_path, args.delete_originals)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
