from __future__ import annotations

from pathlib import Path

import pandas as pd


PRIMARY_SELECTION_TABLE = "subreddit_subset_selection_summary.csv"


def load_primary_selection_table(tables_dir: Path) -> pd.DataFrame:
    """Load the primary 06_0 subreddit-selection table."""
    path = Path(tables_dir) / PRIMARY_SELECTION_TABLE
    if not path.exists():
        raise FileNotFoundError(
            f"Required 06_0 selection table not found: {path}. "
            "Run 06_0_descriptive_plots.py without --skip-subreddit-subset-diagnostics first."
        )
    return pd.read_csv(path)

