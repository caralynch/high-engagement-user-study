#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Cara Lynch
# See the LICENSE file for details.
"""
Preprocessing step 6.3: plot subreddit threshold-audit diagnostics.

This script generates publication-quality diagnostic plots from the already
written outputs of `06_2_make_subreddit_monthly_activity.py`.

Methodological notes
--------------------
- The plots are descriptive diagnostics for choosing subreddit eligibility
  thresholds.
- The script does not recompute subreddit-month counts.
- The selected-subreddit retention heatmap answers:
  "If a subreddit must satisfy the threshold in all/N months, what share of
  annual activity is retained when all activity from those selected subreddits
  is kept?"
- The qualifying-month retention heatmap answers:
  "What share of activity lies specifically in subreddit-months that
  individually satisfy the threshold?"
- These are not inferential plots and should not include confidence intervals.
- Monthly quantile plots use the positive support of the plotted quantity
  because the y-axis is log-scaled. The distribution plot separately reports
  positive n_users values.
- User-profile subreddits beginning with u_ should already have been excluded
  by 06_2 unless that script was run with --include-user-subreddits; this
  plotting script reports the metadata setting if it can read it from
  run_metadata.json, but does not silently re-filter.

Inputs
------
Expected under --audit-root:

  subreddit_monthly_activity.parquet
  threshold_retention_summary.csv
  threshold_retention_by_month.csv
  run_metadata.json

Outputs
-------
Under <out-root>/figures/:

  subreddit_monthly_active_users_distribution
  subreddit_monthly_active_users_quantiles
  subreddit_monthly_total_activity_quantiles
  threshold_retention_annual_activity_heatmap
  threshold_retention_qualifying_months_heatmap
  threshold_selected_subreddits_heatmap
  threshold_retention_by_month_lines, if compatible input columns are present

Under <out-root>/tables/:

  subreddit_monthly_active_users_quantiles.csv
  subreddit_monthly_total_activity_quantiles.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


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
        force=True,
    )


logger = logging.getLogger(__name__)


# ----------------------------- Constants ---------------------------------


REQUIRED_ACTIVITY_COLUMNS = ("month", "subreddit", "n_users")
OPTIONAL_ACTIVITY_COLUMNS = ("n_submissions", "n_comments", "n_total_activity")
REQUIRED_SUMMARY_COLUMNS = (
    "min_users_per_month",
    "min_activity_per_month",
    "pct_activity_retained_annual",
    "pct_activity_in_qualifying_subreddit_months",
)
SELECTED_SUBREDDITS_CANDIDATES = ("n_selected_subreddits", "n_subreddits_retained")
CORE_FIGURE_STEMS = (
    "subreddit_monthly_active_users_distribution",
    "subreddit_monthly_active_users_quantiles",
    "subreddit_monthly_total_activity_quantiles",
    "threshold_retention_annual_activity_heatmap",
    "threshold_retention_qualifying_months_heatmap",
    "threshold_selected_subreddits_heatmap",
)
OPTIONAL_FIGURE_STEMS = ("threshold_retention_by_month_lines",)
TABLE_STEMS = (
    "subreddit_monthly_active_users_quantiles",
    "subreddit_monthly_total_activity_quantiles",
)
COLORBLIND_SAFE_PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
)
PLOT_SIZES = {
    # Match the thesis/report-oriented defaults used by 06_0_descriptive_plots.py.
    "title": 14,
    "axis_label": 15,
    "tick": 13,
    "legend": 13,
    "annotation": 12,
    "direct_label": 13,
}

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
        "axes.prop_cycle": cycler(color=COLORBLIND_SAFE_PALETTE),
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.labelsize": PLOT_SIZES["tick"],
        "ytick.labelsize": PLOT_SIZES["tick"],
        "axes.titlesize": PLOT_SIZES["title"],
        "axes.labelsize": PLOT_SIZES["axis_label"],
        "legend.fontsize": PLOT_SIZES["legend"],
        "legend.title_fontsize": PLOT_SIZES["legend"],
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.dpi": 100,
        "grid.alpha": 0.0,
        "axes.grid.which": "major",
    }
)


# ----------------------------- Utilities ---------------------------------


def ensure_dir(path: Path) -> None:
    """Create a directory if needed."""
    path.mkdir(parents=True, exist_ok=True)


def parse_fig_formats(value: str) -> tuple[str, ...]:
    """Parse and validate comma-separated figure formats."""
    formats = tuple(dict.fromkeys(part.strip().lower() for part in value.split(",") if part.strip()))
    if not formats:
        raise argparse.ArgumentTypeError("At least one figure format is required.")
    allowed = {"png", "svg", "pdf"}
    invalid = [fmt for fmt in formats if fmt not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unsupported figure format(s): {', '.join(invalid)}. Supported: png, svg, pdf."
        )
    return formats


def quote_sql_string(value: str) -> str:
    """Safely quote a string literal for DuckDB SQL."""
    return "'" + value.replace("'", "''") + "'"


def quote_ident(value: str) -> str:
    """Safely quote a DuckDB SQL identifier."""
    return '"' + value.replace('"', '""') + '"'


def format_int(value: float | int) -> str:
    """Format integer-like values with thousands separators."""
    if pd.isna(value):
        return "NA"
    return f"{int(round(float(value))):,}"


def human_tick(value: float, _pos: int | None = None) -> str:
    """Compact tick formatter for large counts."""
    if not np.isfinite(value):
        return ""
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:g}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def maybe_percent(series: pd.Series) -> pd.Series:
    """Convert fraction-style percentages to 0-100 scale when needed."""
    numeric = pd.to_numeric(series, errors="coerce")
    max_value = numeric.max(skipna=True)
    if pd.notna(max_value) and max_value <= 1.0:
        return numeric * 100.0
    return numeric


def save_figure(
    fig: plt.Figure,
    stem: str,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    overwrite: bool,
) -> None:
    """Save a figure in all requested formats."""
    ensure_dir(out_dir)
    fig.tight_layout()
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing figure: {path}")
        save_kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(path, **save_kwargs)
        logger.info("Wrote figure: %s", path)
    plt.close(fig)


def style_standard_axes(ax: plt.Axes) -> None:
    """Apply the clean no-grid axis styling used by the main descriptive plots."""
    ax.grid(False)
    ax.grid(False, which="minor")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", labelsize=PLOT_SIZES["tick"])


def add_endpoint_label(
    ax: plt.Axes,
    x: Sequence[object],
    y: Sequence[object],
    label: str,
    color: str,
    x_offset: int = 8,
    y_offset: int = 0,
) -> None:
    """Place a direct label at the last non-missing point without changing limits."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_vals = pd.Series(x).reset_index(drop=True)
    y_vals = pd.to_numeric(pd.Series(y), errors="coerce").reset_index(drop=True)
    mask = x_vals.notna() & y_vals.notna()
    if not mask.any():
        return
    x_last = x_vals.loc[mask].iloc[-1]
    y_last = float(y_vals.loc[mask].iloc[-1])
    ax.annotate(
        label,
        xy=(x_last, y_last),
        xytext=(x_offset, y_offset),
        textcoords="offset points",
        color=color,
        va="center",
        ha="left",
        clip_on=False,
        fontsize=PLOT_SIZES["direct_label"],
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def heatmap_text_color(value: float, norm: Normalize, cmap_name: str) -> str:
    """Choose black or white annotation text based on mapped cell luminance."""
    rgba = plt.get_cmap(cmap_name)(norm(value))
    red, green, blue = rgba[:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance > 0.55 else "white"


def annotate_heatmap(
    ax: plt.Axes,
    data: pd.DataFrame,
    fmt: str = ".0f",
    cmap: str = "viridis",
    fontsize: int = 9,
) -> None:
    """Annotate heatmap cells with contrast-aware formatted values."""
    values = data.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size:
        norm = Normalize(vmin=float(np.nanmin(finite_values)), vmax=float(np.nanmax(finite_values)))
    else:
        norm = Normalize(vmin=0.0, vmax=1.0)

    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            value = values[row_idx, col_idx]
            if not np.isfinite(value):
                continue
            text_color = heatmap_text_color(float(value), norm, cmap)
            ax.text(
                col_idx,
                row_idx,
                format(value, fmt),
                ha="center",
                va="center",
                fontsize=fontsize,
                color=text_color,
            )


def expected_output_paths(out_root: Path, formats: Sequence[str], include_optional: bool = True) -> list[Path]:
    """Return deterministic figure and table paths."""
    figures_dir = out_root / "figures"
    tables_dir = out_root / "tables"
    stems = list(CORE_FIGURE_STEMS)
    if include_optional:
        stems.extend(OPTIONAL_FIGURE_STEMS)
    paths = [figures_dir / f"{stem}.{fmt}" for stem in stems for fmt in formats]
    paths.extend(tables_dir / f"{stem}.csv" for stem in TABLE_STEMS)
    return paths


def check_output_collisions(
    out_root: Path,
    formats: Sequence[str],
    overwrite: bool,
    include_optional: bool = True,
) -> None:
    """Fail before plotting if any deterministic target already exists."""
    if overwrite:
        return
    collisions = [
        path
        for path in expected_output_paths(out_root, formats, include_optional=include_optional)
        if path.exists()
    ]
    if collisions:
        raise FileExistsError(
            "Refusing to overwrite existing outputs without --overwrite. Existing files:\n  - "
            + "\n  - ".join(str(path) for path in collisions)
        )


def require_file(path: Path, description: str) -> None:
    """Raise a clear error for a missing required file."""
    if not path.exists():
        raise FileNotFoundError(f"Required {description} not found: {path}")


def read_parquet_columns(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """Read selected columns from Parquet via DuckDB."""
    select_sql = ", ".join(quote_ident(col) for col in columns)
    query = f"SELECT {select_sql} FROM read_parquet({quote_sql_string(str(path))})"
    return duckdb.connect(database=":memory:").execute(query).fetchdf()


def parquet_columns(path: Path) -> list[str]:
    """Return Parquet schema column names using DuckDB."""
    con = duckdb.connect(database=":memory:")
    try:
        return [
            row[0]
            for row in con.execute(
                f"DESCRIBE SELECT * FROM read_parquet({quote_sql_string(str(path))})"
            ).fetchall()
        ]
    finally:
        con.close()


def load_run_metadata(path: Path) -> dict:
    """Load run metadata if available."""
    if not path.exists():
        logger.warning("Optional run metadata not found: %s", path)
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except Exception as exc:
        logger.warning("Could not read run metadata %s: %s", path, exc)
        return {}

    config = metadata.get("config", metadata)
    include_user_subreddits = config.get("include_user_subreddits")
    if include_user_subreddits is not None:
        logger.info("06_2 include_user_subreddits metadata setting: %s", include_user_subreddits)
    else:
        logger.info("Run metadata read, but include_user_subreddits setting was not found.")
    required_months = config.get("required_months")
    if required_months is not None:
        logger.info("06_2 required_months metadata setting: %s", required_months)
    return metadata


# ----------------------------- Validation ---------------------------------


def validate_activity(activity: pd.DataFrame) -> None:
    """Run lightweight consistency checks on subreddit-month activity."""
    if activity.empty:
        raise ValueError("subreddit_monthly_activity.parquet contains no rows.")

    month_count = activity["month"].nunique(dropna=True)
    if month_count != 12:
        logger.warning("Expected 12 unique months for a full-year run, found %d.", month_count)

    if activity["n_users"].isna().any():
        raise ValueError("n_users contains missing values.")
    if (activity["n_users"] < 0).any():
        raise ValueError("n_users contains negative values.")

    if {"n_submissions", "n_comments", "n_total_activity"}.issubset(activity.columns):
        expected_total = activity["n_submissions"].fillna(0) + activity["n_comments"].fillna(0)
        mismatch = activity["n_total_activity"].fillna(0) != expected_total
        if mismatch.any():
            logger.warning(
                "n_total_activity differs from n_submissions + n_comments for %s rows; continuing.",
                format_int(int(mismatch.sum())),
            )

    logger.info(
        "Subreddit-month rows: %s; distinct subreddits: %s.",
        format_int(len(activity)),
        format_int(activity["subreddit"].nunique(dropna=True)),
    )


def validate_threshold_summary(summary: pd.DataFrame) -> str:
    """Validate threshold summary and return selected-subreddit count column."""
    missing = [col for col in REQUIRED_SUMMARY_COLUMNS if col not in summary.columns]
    if missing:
        raise ValueError(f"threshold_retention_summary.csv missing required columns: {', '.join(missing)}")

    selected_col = next((col for col in SELECTED_SUBREDDITS_CANDIDATES if col in summary.columns), None)
    if selected_col is None:
        raise ValueError(
            "threshold_retention_summary.csv missing selected subreddit count column. "
            "Expected one of: " + ", ".join(SELECTED_SUBREDDITS_CANDIDATES)
        )

    grid = summary[["min_users_per_month", "min_activity_per_month"]].drop_duplicates()
    if grid.empty:
        raise ValueError("Threshold grid is empty in threshold_retention_summary.csv.")
    logger.info(
        "Threshold grid: %d user thresholds x %d activity thresholds (%d rows).",
        summary["min_users_per_month"].nunique(dropna=True),
        summary["min_activity_per_month"].nunique(dropna=True),
        len(grid),
    )
    return selected_col


# ----------------------------- Tables ---------------------------------


def compute_monthly_quantiles(activity: pd.DataFrame, value_col: str, positive_only: bool) -> pd.DataFrame:
    """Compute monthly q25, median, q75, q90, and q99 for a value column."""
    df = activity[["month", value_col]].copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["month", value_col])
    if positive_only:
        df = df[df[value_col] > 0]
    if df.empty:
        raise ValueError(f"No rows available for monthly quantiles of {value_col}.")

    counts = df.groupby("month", sort=True)[value_col].size().rename("n_rows")
    quantiles = (
        df.groupby("month", sort=True)[value_col]
        .quantile([0.25, 0.50, 0.75, 0.90, 0.99])
        .unstack()
        .rename(columns={0.25: "q25", 0.50: "median", 0.75: "q75", 0.90: "q90", 0.99: "q99"})
        .reset_index()
    )
    quantiles = quantiles.merge(counts.reset_index(), on="month", how="left")
    return quantiles[["month", "n_rows", "q25", "median", "q75", "q90", "q99"]]


def write_table(df: pd.DataFrame, path: Path, overwrite: bool) -> None:
    """Write a compact CSV table with overwrite protection."""
    ensure_dir(path.parent)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing table: {path}")
    df.to_csv(path, index=False)
    logger.info("Wrote table: %s", path)


# ----------------------------- Plot helpers -----------------------------


def set_month_axis(ax: plt.Axes, months: Sequence[str]) -> None:
    """Use stable month labels on a categorical x-axis."""
    x = np.arange(len(months))
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right")


def plot_monthly_quantiles(
    quantiles: pd.DataFrame,
    ylabel: str,
    title: str,
    publication_style: bool,
) -> plt.Figure:
    """Plot monthly median, IQR ribbon, q90, and q99."""
    q = quantiles.sort_values("month").reset_index(drop=True)
    months = q["month"].astype(str).tolist()
    x = np.arange(len(q))

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    iqr_color = "#0072B2"
    ax.fill_between(
        x,
        q["q25"].to_numpy(dtype=float),
        q["q75"].to_numpy(dtype=float),
        color=iqr_color,
        alpha=0.18,
        linewidth=0,
    )
    median_line, = ax.plot(x, q["median"], marker="o", linewidth=1.8, color=iqr_color)
    q90_line, = ax.plot(x, q["q90"], linestyle="--", linewidth=1.6, color="#D55E00", alpha=0.9)
    q99_line, = ax.plot(x, q["q99"], linestyle=":", linewidth=1.8, color="#009E73", alpha=0.95)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(human_tick))
    ax.yaxis.set_minor_formatter(NullFormatter())
    set_month_axis(ax, months)
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.4, max(1.0, len(q) - 1 + 1.75))
    q25_last = pd.to_numeric(q["q25"], errors="coerce").dropna()
    q75_last = pd.to_numeric(q["q75"], errors="coerce").dropna()
    if len(q25_last) and len(q75_last):
        ax.text(
            len(q) - 1 + 0.18,
            float(np.sqrt(q25_last.iloc[-1] * q75_last.iloc[-1])),
            "IQR",
            color=iqr_color,
            va="center",
            ha="left",
            fontsize=PLOT_SIZES["direct_label"],
            clip_on=True,
        )
    add_endpoint_label(ax, x, q["median"], "Median", median_line.get_color())
    add_endpoint_label(ax, x, q["q90"], "90th percentile", q90_line.get_color())
    add_endpoint_label(ax, x, q["q99"], "99th percentile", q99_line.get_color())
    style_standard_axes(ax)
    if not publication_style:
        ax.set_title(title)
    return fig


def plot_active_users_distribution(
    activity: pd.DataFrame,
    publication_style: bool,
) -> plt.Figure:
    """Plot distribution of active users across positive subreddit-months."""
    values = pd.to_numeric(activity["n_users"], errors="coerce")
    values = values[values > 0].dropna()
    if values.empty:
        raise ValueError("No positive n_users values available for distribution plot.")

    max_value = int(values.max())
    # Log-spaced bins keep the long-tailed distribution readable across orders of magnitude.
    bins = np.unique(np.logspace(0, np.log10(max_value), 55).astype(int))
    if len(bins) < 2:
        bins = np.array([1, max_value + 1])

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.hist(values, bins=bins, color="#0072B2", edgecolor="white", linewidth=0.35, alpha=0.95)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(human_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(human_tick))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Active users per subreddit-month")
    ax.set_ylabel("Number of subreddit-months")

    stats = {
        "Subreddit-months": len(values),
        "Median": values.median(),
        "90th percentile": values.quantile(0.90),
        "99th percentile": values.quantile(0.99),
    }
    annotation = "\n".join(f"{label}: {format_int(value)}" for label, value in stats.items())
    ax.text(
        0.98,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=PLOT_SIZES["annotation"],
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95},
    )
    style_standard_axes(ax)
    if not publication_style:
        ax.set_title("Distribution of subreddit-month active users")
    return fig


def pivot_threshold_grid(summary: pd.DataFrame, value_col: str, percent: bool) -> pd.DataFrame:
    """Create a rows=user thresholds, columns=activity thresholds matrix."""
    df = summary[["min_users_per_month", "min_activity_per_month", value_col]].copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    if percent:
        df[value_col] = maybe_percent(df[value_col])

    if df.duplicated(["min_users_per_month", "min_activity_per_month"]).any():
        raise ValueError(
            "threshold_retention_summary.csv appears to contain multiple rows per threshold pair "
            "(min_users_per_month, min_activity_per_month), possibly from multiple required-month "
            "settings. Filter to one required-month setting before plotting."
        )

    grid = (
        df.pivot(
            index="min_users_per_month",
            columns="min_activity_per_month",
            values=value_col,
        )
        .sort_index()
        .sort_index(axis=1)
    )
    if grid.empty:
        raise ValueError(f"Threshold heatmap grid for {value_col} is empty.")
    return grid


def plot_heatmap(
    grid: pd.DataFrame,
    colorbar_label: str,
    xlabel: str,
    ylabel: str,
    title: str,
    annotate: bool,
    annotation_fmt: str,
    publication_style: bool,
    cmap: str = "viridis",
) -> plt.Figure:
    """Plot a threshold matrix heatmap with smallest thresholds at bottom-left."""
    width = max(7.4, 0.72 * grid.shape[1] + 3.3)
    height = max(5.2, 0.48 * grid.shape[0] + 2.6)
    fig, ax = plt.subplots(figsize=(width, height))

    masked = np.ma.masked_invalid(grid.to_numpy(dtype=float))
    image = ax.imshow(masked, aspect="auto", cmap=cmap, origin="lower")
    cbar = fig.colorbar(image, ax=ax, shrink=0.92)
    cbar.set_label(colorbar_label, fontsize=PLOT_SIZES["axis_label"])
    cbar.ax.tick_params(labelsize=PLOT_SIZES["tick"])
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(human_tick))

    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_yticks(np.arange(grid.shape[0]))
    ax.set_xticklabels([format_int(col) for col in grid.columns], rotation=45, ha="right")
    ax.set_yticklabels([format_int(idx) for idx in grid.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        annotation_fontsize = 8 if grid.size > 64 else 9
        annotate_heatmap(ax, grid, fmt=annotation_fmt, cmap=cmap, fontsize=annotation_fontsize)
    if not publication_style:
        ax.set_title(title)
    return fig


# ----------------------------- Optional plot -----------------------------


def choose_representative_thresholds(by_month: pd.DataFrame) -> list[tuple[int, int]]:
    """Choose a small set of threshold combinations for monthly line plots."""
    combos = by_month[["min_users_per_month", "min_activity_per_month"]].drop_duplicates()
    activity_values = set(pd.to_numeric(combos["min_activity_per_month"], errors="coerce").dropna().astype(int))
    # This line plot is a simple monthly stability diagnostic for representative
    # user thresholds; the full user/activity threshold trade-off is shown in the heatmaps.
    if 0 in activity_values:
        combos = combos[pd.to_numeric(combos["min_activity_per_month"], errors="coerce") == 0]

    users_present = sorted(pd.to_numeric(combos["min_users_per_month"], errors="coerce").dropna().astype(int).unique())
    chosen_users: list[int] = []
    non_zero = [value for value in users_present if value > 0]
    if non_zero:
        chosen_users.append(non_zero[0])
    for target in (100, 250, 500):
        if target in users_present and target not in chosen_users:
            chosen_users.append(target)
    if not chosen_users and users_present:
        chosen_users.append(users_present[0])

    chosen: list[tuple[int, int]] = []
    for user_threshold in chosen_users:
        possible_activity = combos.loc[
            pd.to_numeric(combos["min_users_per_month"], errors="coerce") == user_threshold,
            "min_activity_per_month",
        ]
        if possible_activity.empty:
            continue
        activity_threshold = 0 if 0 in set(possible_activity.astype(int)) else int(possible_activity.astype(int).min())
        chosen.append((int(user_threshold), activity_threshold))
    return chosen[:4]


def threshold_line_label(user_threshold: int, activity_threshold: int) -> str:
    """Return concise direct labels for representative threshold lines."""
    if activity_threshold == 0:
        return f"users >= {format_int(user_threshold)}"
    return f"users >= {format_int(user_threshold)}, activity >= {format_int(activity_threshold)}"


def monthly_retention_ylabel(value_col: str) -> str:
    """Return the y-axis label that matches the selected monthly retention column."""
    if value_col == "pct_activity_qualifying":
        return "% activity in qualifying subreddit-months"
    if value_col in {
        "pct_activity_retained_by_selected_subreddits_month",
        "pct_activity_retained_month",
        "pct_activity_retained",
    }:
        return "% monthly activity retained"
    return "% activity"


def plot_retention_by_month_lines(
    by_month: pd.DataFrame,
    publication_style: bool,
) -> plt.Figure | None:
    """Plot monthly retained activity percentages for representative thresholds."""
    required = {"month", "min_users_per_month", "min_activity_per_month"}
    value_candidates = (
        "pct_activity_qualifying",
        "pct_activity_retained_by_selected_subreddits_month",
        "pct_activity_retained_month",
        "pct_activity_retained",
    )
    value_col = next((col for col in value_candidates if col in by_month.columns), None)
    missing = sorted(required.difference(by_month.columns))
    if missing or value_col is None:
        logger.warning(
            "Skipping threshold_retention_by_month_lines; missing columns: %s%s",
            ", ".join(missing) if missing else "",
            "; no monthly retained-activity percentage column found" if value_col is None else "",
        )
        return None

    chosen = choose_representative_thresholds(by_month)
    if not chosen:
        logger.warning("Skipping threshold_retention_by_month_lines; no representative thresholds found.")
        return None

    df = by_month.copy()
    df[value_col] = maybe_percent(df[value_col])
    df["month"] = df["month"].astype(str)
    months = sorted(df["month"].dropna().unique())
    x = np.arange(len(months))

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    plotted = 0
    plotted_values: list[float] = []
    direct_labels: list[tuple[np.ndarray, pd.Series, str, str]] = []
    for user_threshold, activity_threshold in chosen:
        subset = df[
            (pd.to_numeric(df["min_users_per_month"], errors="coerce") == user_threshold)
            & (pd.to_numeric(df["min_activity_per_month"], errors="coerce") == activity_threshold)
        ].copy()
        if subset.empty:
            continue
        subset = subset.set_index("month").reindex(months)
        label = threshold_line_label(user_threshold, activity_threshold)
        line, = ax.plot(x, subset[value_col], marker="o", linewidth=1.8)
        plotted_values.extend(pd.to_numeric(subset[value_col], errors="coerce").dropna().astype(float).tolist())
        direct_labels.append((x, subset[value_col], label, line.get_color()))
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        logger.warning("Skipping threshold_retention_by_month_lines; chosen thresholds had no rows.")
        return None

    set_month_axis(ax, months)
    ax.set_xlabel("Month")
    # The by-month audit may contain either qualifying-month retention or
    # selected-subreddit monthly retention, so the axis label must follow value_col.
    ax.set_ylabel(monthly_retention_ylabel(value_col))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
    ax.set_xlim(-0.4, max(1.0, len(months) - 1 + 1.9))
    if plotted_values:
        observed_min = min(plotted_values)
        observed_max = max(plotted_values)
        pad = max(1.0, abs(observed_min) * 0.04) if observed_min == observed_max else (observed_max - observed_min) * 0.12
        lower = observed_min - pad
        upper = observed_max + pad
        if observed_min >= 0 and observed_max <= 100:
            lower = max(0.0, lower)
            upper = min(105.0, max(upper, observed_max + 0.5))
        elif observed_max <= 100:
            upper = min(105.0, upper)
        ax.set_ylim(lower, upper)
    for x_values, y_values, label, color in direct_labels:
        add_endpoint_label(ax, x_values, y_values, label, color)
    style_standard_axes(ax)
    if not publication_style:
        ax.set_title("Monthly activity retention for representative thresholds")
    return fig


# ----------------------------- Main workflow -----------------------------


def build_argparser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(
        description="Plot subreddit threshold-audit diagnostics from 06_2 outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--audit-root", required=True, type=Path, help="Directory containing threshold audit outputs.")
    parser.add_argument("--out-root", type=Path, default=None, help="Output root. Defaults to --audit-root.")
    parser.add_argument(
        "--fig-formats",
        type=parse_fig_formats,
        default=parse_fig_formats("png,svg,pdf"),
        help="Comma-separated output formats.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing outputs.")
    parser.add_argument(
        "--publication-style",
        action="store_true",
        help="Suppress or simplify standalone plot titles where appropriate.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    parser.add_argument(
        "--max-heatmap-annotations",
        type=int,
        default=100,
        help="Maximum heatmap cells to annotate with values.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for raster figure outputs.")
    parser.add_argument(
        "--top-n-label",
        type=int,
        default=0,
        help="Reserved for optional rank/diagnostic plot labels; currently no rank plot is produced.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = build_argparser().parse_args(argv)
    setup_logging(args.log_level)

    audit_root = args.audit_root.expanduser().resolve()
    out_root = (args.out_root if args.out_root is not None else args.audit_root).expanduser().resolve()
    figures_dir = out_root / "figures"
    tables_dir = out_root / "tables"
    formats = args.fig_formats

    if args.max_heatmap_annotations < 0:
        raise ValueError("--max-heatmap-annotations must be non-negative.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    if args.top_n_label:
        logger.info("--top-n-label is accepted for future diagnostics; no current plot uses labels.")

    activity_path = audit_root / "subreddit_monthly_activity.parquet"
    summary_path = audit_root / "threshold_retention_summary.csv"
    by_month_path = audit_root / "threshold_retention_by_month.csv"
    metadata_path = audit_root / "run_metadata.json"

    logger.info("Audit root: %s", audit_root)
    logger.info("Output root: %s", out_root)
    require_file(activity_path, "subreddit-month activity Parquet")
    require_file(summary_path, "threshold retention summary CSV")
    if not by_month_path.exists():
        logger.warning("Optional threshold retention by-month CSV not found: %s", by_month_path)

    include_optional_collision_check = by_month_path.exists()
    check_output_collisions(
        out_root,
        formats,
        overwrite=args.overwrite,
        include_optional=include_optional_collision_check,
    )
    ensure_dir(figures_dir)
    ensure_dir(tables_dir)

    load_run_metadata(metadata_path)

    logger.info("Inspecting subreddit monthly activity schema.")
    schema_cols = parquet_columns(activity_path)
    missing_activity = [col for col in REQUIRED_ACTIVITY_COLUMNS if col not in schema_cols]
    if missing_activity:
        raise ValueError(f"subreddit_monthly_activity.parquet missing required columns: {', '.join(missing_activity)}")

    activity_cols = list(REQUIRED_ACTIVITY_COLUMNS)
    activity_cols.extend(col for col in OPTIONAL_ACTIVITY_COLUMNS if col in schema_cols)
    if "n_total_activity" not in activity_cols:
        raise ValueError("subreddit_monthly_activity.parquet missing required column for activity plot: n_total_activity")

    logger.info("Reading selected activity columns from Parquet: %s", ", ".join(activity_cols))
    activity = read_parquet_columns(activity_path, activity_cols)
    validate_activity(activity)

    logger.info("Reading threshold summary: %s", summary_path)
    summary = pd.read_csv(summary_path)
    selected_col = validate_threshold_summary(summary)

    logger.info("Computing monthly quantile plot tables.")
    active_users_quantiles = compute_monthly_quantiles(activity, "n_users", positive_only=True)
    total_activity_quantiles = compute_monthly_quantiles(activity, "n_total_activity", positive_only=True)
    write_table(
        active_users_quantiles,
        tables_dir / "subreddit_monthly_active_users_quantiles.csv",
        overwrite=args.overwrite,
    )
    write_table(
        total_activity_quantiles,
        tables_dir / "subreddit_monthly_total_activity_quantiles.csv",
        overwrite=args.overwrite,
    )

    logger.info("Plotting active-user distribution.")
    fig = plot_active_users_distribution(activity, publication_style=args.publication_style)
    save_figure(
        fig,
        "subreddit_monthly_active_users_distribution",
        figures_dir,
        formats,
        dpi=args.dpi,
        overwrite=args.overwrite,
    )

    logger.info("Plotting active-user monthly quantiles.")
    fig = plot_monthly_quantiles(
        active_users_quantiles,
        ylabel="Active users per subreddit",
        title="Monthly active-user quantiles across subreddits",
        publication_style=args.publication_style,
    )
    save_figure(
        fig,
        "subreddit_monthly_active_users_quantiles",
        figures_dir,
        formats,
        dpi=args.dpi,
        overwrite=args.overwrite,
    )

    logger.info("Plotting total-activity monthly quantiles.")
    fig = plot_monthly_quantiles(
        total_activity_quantiles,
        ylabel="Total activity events per subreddit",
        title="Monthly total-activity quantiles across subreddits",
        publication_style=args.publication_style,
    )
    save_figure(
        fig,
        "subreddit_monthly_total_activity_quantiles",
        figures_dir,
        formats,
        dpi=args.dpi,
        overwrite=args.overwrite,
    )

    # The two retention heatmaps deliberately answer different descriptive questions:
    # annual retention keeps all activity from selected subreddits, while qualifying
    # month retention counts only subreddit-months that individually pass thresholds.
    heatmap_specs = (
        (
            "threshold_retention_annual_activity_heatmap",
            "pct_activity_retained_annual",
            "% annual activity retained",
            "Selected-subreddit annual activity retention",
            True,
            ".1f",
        ),
        (
            "threshold_retention_qualifying_months_heatmap",
            "pct_activity_in_qualifying_subreddit_months",
            "% activity in individually qualifying subreddit-months",
            "Activity in individually qualifying subreddit-months",
            True,
            ".1f",
        ),
        (
            "threshold_selected_subreddits_heatmap",
            selected_col,
            "Selected subreddits",
            "Selected subreddits by threshold",
            False,
            ".0f",
        ),
    )

    for stem, value_col, colorbar_label, title, percent, annotation_fmt in heatmap_specs:
        logger.info("Plotting heatmap: %s", stem)
        grid = pivot_threshold_grid(summary, value_col=value_col, percent=percent)
        annotate = grid.size <= args.max_heatmap_annotations
        if not annotate:
            logger.info(
                "Skipping cell annotations for %s because %d cells exceeds --max-heatmap-annotations=%d.",
                stem,
                grid.size,
                args.max_heatmap_annotations,
            )
        fig = plot_heatmap(
            grid,
            colorbar_label=colorbar_label,
            xlabel="Minimum total activity per subreddit-month",
            ylabel="Minimum active users per subreddit-month",
            title=title,
            annotate=annotate,
            annotation_fmt=annotation_fmt,
            publication_style=args.publication_style,
            cmap="viridis",
        )
        save_figure(fig, stem, figures_dir, formats, dpi=args.dpi, overwrite=args.overwrite)

    if by_month_path.exists():
        logger.info("Reading threshold by-month summary: %s", by_month_path)
        by_month = pd.read_csv(by_month_path)
        fig = plot_retention_by_month_lines(by_month, publication_style=args.publication_style)
        if fig is not None:
            save_figure(
                fig,
                "threshold_retention_by_month_lines",
                figures_dir,
                formats,
                dpi=args.dpi,
                overwrite=args.overwrite,
            )

    logger.info("Threshold audit plotting complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
