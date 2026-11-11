from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.ticker import FuncFormatter

from .config import COLORBLIND_SAFE_PALETTE, PLOT_SIZES, SUBSET_DIAG_TOP_LABEL_COUNT
from .output_io import ensure_dir


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
        "grid.alpha": 0.0,
        "axes.grid.which": "major",
    }
)


def save_figure(fig: plt.Figure, out_root: str, name: str) -> None:
    """Save PNG, SVG, and PDF versions of a figure."""
    fig_dir = os.path.join(out_root, "figures")
    ensure_dir(fig_dir)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{name}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(fig_dir, f"{name}.svg"), bbox_inches="tight")
    fig.savefig(os.path.join(fig_dir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)

def _rank_size_plot(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    publication_style: bool = False,
) -> plt.Figure:
    ranked = df.sort_values(value_col, ascending=False).reset_index(drop=True).copy()
    ranked = ranked.loc[ranked[value_col] > 0].copy()
    # Rank 1 denotes the largest subreddit by the plotted activity quantity.
    ranked["rank"] = range(1, len(ranked) + 1)

    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    ax.plot(ranked["rank"], ranked[value_col], linewidth=1.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Subreddit rank (1 = largest)")
    ax.set_ylabel(ylabel)
    _apply_title_or_panel_label(ax, title, publication_style)
    _style_axes(ax)
    return fig

def _survival_table_plot(
    df: pd.DataFrame,
    value_col: str,
    survival_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    publication_style: bool = False,
    annotation: Optional[str] = None,
) -> plt.Figure:
    if annotation is None and "zero_share" in df.columns:
        zero_share = pd.to_numeric(df["zero_share"], errors="coerce").dropna()
        if len(zero_share) and zero_share.iloc[0] > 0:
            annotation = f"Zero-value share: {zero_share.iloc[0]:.1%}"

    plot_df = df.copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df[survival_col] = pd.to_numeric(plot_df[survival_col], errors="coerce")
    plot_df = plot_df.loc[
        plot_df[value_col].notna()
        & plot_df[survival_col].notna()
        & (plot_df[value_col] > 0)
        & (plot_df[survival_col] > 0)
    ].sort_values(value_col)

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    if len(plot_df):
        ax.step(plot_df[value_col], plot_df[survival_col], where="post", linewidth=1.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if annotation and not publication_style:
        ax.text(
            0.98,
            0.98,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=PLOT_SIZES["annotation"],
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
    _apply_title_or_panel_label(ax, title, publication_style)
    _style_axes(ax)
    return fig

def _stratified_rate_survival_plot(
    df: pd.DataFrame,
    title: str,
    publication_style: bool = False,
) -> plt.Figure:
    """Plot compact survival curves for received-engagement rates by activity threshold."""
    plot_df = df.copy()
    plot_df["rate_threshold"] = pd.to_numeric(plot_df["rate_threshold"], errors="coerce")
    plot_df["survival_share"] = pd.to_numeric(plot_df["survival_share"], errors="coerce")
    plot_df["min_total_activity"] = pd.to_numeric(plot_df["min_total_activity"], errors="coerce")
    plot_df = plot_df.loc[
        plot_df["rate_threshold"].notna()
        & plot_df["survival_share"].notna()
        & (plot_df["rate_threshold"] > 0)
        & (plot_df["survival_share"] > 0)
    ].copy()

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    label_map = {
        1: "All active",
        2: ">=2 activities",
        5: ">=5 activities",
        10: ">=10 activities",
    }
    for min_activity in sorted(plot_df["min_total_activity"].dropna().unique()):
        subset = plot_df.loc[plot_df["min_total_activity"] == min_activity].sort_values("rate_threshold")
        if not len(subset):
            continue
        label = label_map.get(int(min_activity), f">={int(min_activity)} activities")
        ax.step(
            subset["rate_threshold"],
            subset["survival_share"],
            where="post",
            linewidth=1.8,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Received engagement per authored activity")
    ax.set_ylabel("Share of user-months at or above threshold")
    _apply_title_or_panel_label(ax, title, publication_style)
    if publication_style:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            x_data = handle.get_xdata()
            y_data = handle.get_ydata()
            if len(x_data) and len(y_data):
                _add_series_endpoint_label(ax, x_data, y_data, label, handle.get_color())
    else:
        ax.legend(frameon=False)
    _style_axes(ax)
    return fig

def _decile_quantile_ribbon_plot(
    summary: pd.DataFrame,
    x_col: str,
    median_col: str,
    lower_col: str,
    upper_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    publication_style: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    x = pd.to_numeric(summary[x_col], errors="coerce")
    median = pd.to_numeric(summary[median_col], errors="coerce")
    lower = pd.to_numeric(summary[lower_col], errors="coerce")
    upper = pd.to_numeric(summary[upper_col], errors="coerce")
    mask = x.notna() & median.notna() & lower.notna() & upper.notna()
    if mask.any():
        ax.fill_between(
            x[mask],
            lower[mask],
            upper[mask],
            color=COLORBLIND_SAFE_PALETTE[0],
            alpha=0.12 if publication_style else 0.18,
            linewidth=0,
        )
        ax.plot(x[mask], median[mask], marker="o", linewidth=1.8, color=COLORBLIND_SAFE_PALETTE[0])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _apply_title_or_panel_label(ax, title, publication_style)
    _style_axes(ax)
    return fig

def _monthly_platform_activity_plot(
    df: pd.DataFrame,
    title: str,
    publication_style: bool = False,
) -> plt.Figure:
    monthly = _prepare_monthly_plot_frame(df)
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(9.2, 9.2),
        sharex=True,
        gridspec_kw={"hspace": 0.18},
    )
    comments_color = COLORBLIND_SAFE_PALETTE[0]
    activity_color = COLORBLIND_SAFE_PALETTE[1]
    users_color = COLORBLIND_SAFE_PALETTE[2]
    subreddits_color = COLORBLIND_SAFE_PALETTE[3]

    line_com, = axes[0].plot(monthly["month"], monthly["n_comments"], marker="o", linewidth=1.8, color=comments_color)
    axes[0].set_ylabel("Comments")
    _apply_title_or_panel_label(axes[0], title, publication_style)
    _add_series_endpoint_label(axes[0], monthly["month"], monthly["n_comments"], "Comments", line_com.get_color())

    line_sub, = axes[1].plot(monthly["month"], monthly["n_submissions"], marker="o", linewidth=1.8, color=activity_color)
    axes[1].set_ylabel("Submissions")
    _add_series_endpoint_label(axes[1], monthly["month"], monthly["n_submissions"], "Submissions", line_sub.get_color())

    line_users, = axes[2].plot(monthly["month"], monthly["n_active_users"], marker="o", linewidth=1.8, color=users_color)
    axes[2].set_ylabel("Active users")
    _add_series_endpoint_label(axes[2], monthly["month"], monthly["n_active_users"], "Active users", line_users.get_color())

    line_subreddits, = axes[3].plot(monthly["month"], monthly["n_active_subreddits"], marker="o", linewidth=1.8, color=subreddits_color)
    axes[3].set_xlabel("Month")
    axes[3].set_ylabel("Active subreddits")
    _add_series_endpoint_label(axes[3], monthly["month"], monthly["n_active_subreddits"], "Active subreddits", line_subreddits.get_color())

    for ax in axes:
        ax.yaxis.set_major_formatter(FuncFormatter(_format_readable_number))
        _style_axes(ax)
    _format_month_axis(axes[-1])
    return fig

def _monthly_quantile_ribbon_plot(
    summary: pd.DataFrame,
    x_col: str,
    median_col: str,
    lower_col: str,
    upper_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    publication_style: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    monthly = _prepare_monthly_plot_frame(summary)
    median = pd.to_numeric(monthly[median_col], errors="coerce")
    lower = pd.to_numeric(monthly[lower_col], errors="coerce")
    upper = pd.to_numeric(monthly[upper_col], errors="coerce")
    mask = median.notna() & lower.notna() & upper.notna()
    if mask.any():
        x = monthly.loc[mask, "month"]
        lower_vals = lower[mask].to_numpy(dtype=float)
        upper_vals = upper[mask].to_numpy(dtype=float)
        ax.fill_between(
            x,
            lower_vals,
            upper_vals,
            color=COLORBLIND_SAFE_PALETTE[0],
            alpha=0.12 if publication_style else 0.18,
            linewidth=0,
        )
        ax.plot(x, median[mask], marker="o", linewidth=1.8, color=COLORBLIND_SAFE_PALETTE[0])
        finite_lower = lower_vals[np.isfinite(lower_vals)]
        finite_upper = upper_vals[np.isfinite(upper_vals)]
        if len(finite_lower) and len(finite_upper):
            y_min = float(finite_lower.min())
            y_max = float(finite_upper.max())
            if y_min <= 0:
                y_min = 0.0
            y_range = y_max - y_min
            if y_range > 0:
                pad = 0.09 * y_range
            else:
                pad = max(0.25, 0.09 * max(abs(y_max), 1.0))
            ax.set_ylim(y_min - pad if y_min > 0 else 0.0, y_max + pad)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _format_month_axis(ax)
    _apply_title_or_panel_label(ax, title, publication_style)
    _style_axes(ax)
    return fig

def _decile_mean_median_plot(
    summary: pd.DataFrame,
    x_col: str,
    mean_col: str,
    median_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    publication_style: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    mean_line, = ax.plot(
        summary[x_col],
        summary[mean_col],
        marker="o",
        linewidth=1.8,
        linestyle="--",
        alpha=0.8,
        label="Mean at t+1",
    )
    median_line, = ax.plot(
        summary[x_col],
        summary[median_col],
        marker="o",
        linewidth=1.8,
        linestyle="--",
        alpha=0.8,
        label="Median at t+1",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _apply_title_or_panel_label(ax, title, publication_style)
    if publication_style:
        _add_series_endpoint_label(ax, summary[x_col], summary[mean_col], "Mean", mean_line.get_color())
        _add_series_endpoint_label(ax, summary[x_col], summary[median_col], "Median", median_line.get_color())
    else:
        ax.legend(frameon=False)
    _style_axes(ax)
    return fig

def _scatter_log_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    publication_style: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = x.notna() & y.notna() & (x > 0) & (y >= 0)
    ax.scatter(x[mask], y[mask], s=12, alpha=0.6)
    ax.set_xscale("log")
    positive_y = (y[mask] > 0).any()
    if positive_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _apply_title_or_panel_label(ax, title, publication_style)
    _style_axes(ax)
    return fig

def _hexbin_log_with_median_trend_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    publication_style: bool = False,
    gridsize: int = 36,
    n_trend_bins: int = 14,
) -> plt.Figure:
    """Plot a dense subreddit-level relationship with a robust median trend."""
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = x.notna() & y.notna() & (x > 0) & (y > 0)
    x_plot = x[mask].to_numpy(dtype=float)
    y_plot = y[mask].to_numpy(dtype=float)

    if len(x_plot):
        hb = ax.hexbin(
            x_plot,
            y_plot,
            gridsize=gridsize,
            xscale="log",
            yscale="log",
            mincnt=1,
            cmap="cividis",
            linewidths=0.0,
        )
        cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Number of subreddits", fontsize=PLOT_SIZES["axis_label"])
        cbar.ax.tick_params(labelsize=PLOT_SIZES["tick"])

        if x_plot.min() < x_plot.max():
            x_edges = np.logspace(np.log10(x_plot.min()), np.log10(x_plot.max()), n_trend_bins + 1)
            x_centers: List[float] = []
            y_medians: List[float] = []
            for i in range(len(x_edges) - 1):
                left = x_edges[i]
                right = x_edges[i + 1]
                if i == len(x_edges) - 2:
                    in_bin = (x_plot >= left) & (x_plot <= right)
                else:
                    in_bin = (x_plot >= left) & (x_plot < right)
                if not np.any(in_bin):
                    continue
                y_bin = y_plot[in_bin]
                positive_y_bin = y_bin[y_bin > 0]
                if not len(positive_y_bin):
                    continue
                x_centers.append(float(np.exp(np.mean(np.log(x_plot[in_bin])))))
                y_medians.append(float(np.median(positive_y_bin)))
            if x_centers and y_medians:
                ax.plot(
                    x_centers,
                    y_medians,
                    color="white",
                    linewidth=1.6,
                    marker="o",
                    markersize=3.0,
                    markerfacecolor="white",
                    markeredgecolor="#333333",
                    zorder=3,
                )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _apply_title_or_panel_label(ax, title, publication_style)
    _style_axes(ax)
    return fig

def _lorenz_curve_plot(
    df: pd.DataFrame,
    title: str,
    value_share_col: str,
    ylabel: str,
    summary: Optional[pd.DataFrame] = None,
    publication_style: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.plot([0, 1], [0, 1], color="#999999", linewidth=1.2, linestyle="--")
    if not publication_style:
        ax.text(
            0.60,
            0.62,
            "Perfect equality",
            color="#777777",
            fontsize=PLOT_SIZES["annotation"],
            rotation=45,
            rotation_mode="anchor",
        )
    if len(df):
        x = np.concatenate(([0.0], df["cum_user_share"].to_numpy(dtype=float)))
        y = np.concatenate(([0.0], df[value_share_col].to_numpy(dtype=float)))
        ax.plot(x, y, linewidth=2.0, color=COLORBLIND_SAFE_PALETTE[0])
    if summary is not None and len(summary) and not publication_style:
        row = summary.iloc[0]
        gini = pd.to_numeric(pd.Series([row.get("gini")]), errors="coerce").iloc[0]
        top_10 = pd.to_numeric(pd.Series([row.get("top_10pct_share")]), errors="coerce").iloc[0]
        parts: List[str] = []
        if pd.notna(gini):
            parts.append(f"Gini: {gini:.3f}")
        if pd.notna(top_10):
            parts.append(f"Top 10% share: {top_10:.1%}")
        if parts:
            ax.text(
                0.98,
                0.15,
                "\n".join(parts),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=PLOT_SIZES["annotation"],
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Cumulative share of users")
    ax.set_ylabel(ylabel)
    _apply_title_or_panel_label(ax, title, publication_style)
    _style_axes(ax)
    return fig

def _transition_heatmap_plot(
    df: pd.DataFrame,
    title: str,
    y_col: str,
    x_col: str,
    y_label: str,
    x_label: str,
    publication_style: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    col_order = ["Inactive"] + [str(i) for i in range(1, 11)]
    row_order = list(range(1, 11))
    plot_df = df.copy()
    plot_df[x_col] = plot_df[x_col].astype(str)
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    heatmap = (
        plot_df.pivot(index=y_col, columns=x_col, values="row_share")
        .reindex(index=row_order, columns=col_order)
        .fillna(0.0)
    )
    im = ax.imshow(heatmap.to_numpy(), cmap="cividis", vmin=0.0, vmax=max(heatmap.to_numpy().max(), 1e-12), aspect="auto", origin="lower")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order)
    ax.set_yticks(range(10))
    ax.set_yticklabels(row_order)
    _apply_title_or_panel_label(ax, title, publication_style)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalised share", fontsize=PLOT_SIZES["axis_label"])
    cbar.ax.tick_params(labelsize=PLOT_SIZES["tick"])
    _style_axes(ax)
    # Heatmaps read more cleanly with a full but still unobtrusive rectangular frame.
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_linewidth(0.8)
    return fig

def _active_subreddits_distribution_plot(
    df: pd.DataFrame,
    title: str,
    publication_style: bool = False,
) -> plt.Figure:
    summary = df.copy()
    summary["n_active_subreddits"] = pd.to_numeric(summary["n_active_subreddits"], errors="coerce")
    summary["survival_user_share"] = pd.to_numeric(summary["survival_user_share"], errors="coerce")
    summary = summary.loc[
        summary["n_active_subreddits"].notna()
        & summary["survival_user_share"].notna()
        & (summary["n_active_subreddits"] > 0)
        & (summary["survival_user_share"] > 0)
    ].sort_values("n_active_subreddits")

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    if len(summary):
        ax.plot(
            summary["n_active_subreddits"],
            summary["survival_user_share"],
            linewidth=1.8,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of distinct subreddits a user was active in")
    ax.set_ylabel("Share of users active in at least this many subreddits")
    _apply_title_or_panel_label(ax, title, publication_style)
    _style_axes(ax)
    return fig

def _user_subreddit_hhi_two_panel_plot(
    hhi_plot_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    title: str,
    publication_style: bool = False,
) -> plt.Figure:
    """Plot the HHI point mass at 1 separately from the continuous remainder."""
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(8.0, 6.8),
        gridspec_kw={"height_ratios": [1.0, 2.1], "hspace": 0.28},
    )

    summary_row = summary_df.iloc[0] if len(summary_df) else {}
    share_eq_1 = pd.to_numeric(pd.Series([summary_row.get("share_hhi_eq_1")]), errors="coerce").iloc[0] if len(summary_df) else np.nan
    share_lt_1 = pd.to_numeric(pd.Series([summary_row.get("share_hhi_lt_1")]), errors="coerce").iloc[0] if len(summary_df) else np.nan

    categories = ["HHI < 1", "HHI = 1"]
    shares = [
        0.0 if pd.isna(share_lt_1) else float(share_lt_1),
        0.0 if pd.isna(share_eq_1) else float(share_eq_1),
    ]
    hhi_lt_1_color = COLORBLIND_SAFE_PALETTE[2]
    ax_top.bar(categories, shares, color=[hhi_lt_1_color, COLORBLIND_SAFE_PALETTE[0]], width=0.6)
    ax_top.set_ylabel("Share of users")
    ax_top.set_ylim(0, max(max(shares) * 1.15, 0.05))
    _apply_title_or_panel_label(ax_top, title, publication_style)
    for i, share in enumerate(shares):
        ax_top.text(
            i,
            share + max(shares) * 0.035,
            f"{share:.1%}",
            ha="center",
            va="bottom",
            fontsize=PLOT_SIZES["annotation"],
        )
    _style_axes(ax_top)

    hhi_bins = hhi_plot_df.copy()
    for col in ("bin_left", "bin_right", "user_share_lt_1"):
        hhi_bins[col] = pd.to_numeric(hhi_bins[col], errors="coerce")
    hhi_bins = hhi_bins.dropna(subset=["bin_left", "bin_right", "user_share_lt_1"]).sort_values("bin_left")
    if len(hhi_bins):
        widths = hhi_bins["bin_right"] - hhi_bins["bin_left"]
        ax_bottom.bar(
            hhi_bins["bin_left"],
            hhi_bins["user_share_lt_1"],
            width=widths,
            align="edge",
            color=hhi_lt_1_color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
        )
    else:
        ax_bottom.text(
            0.5,
            0.5,
            "No users with HHI < 1",
            transform=ax_bottom.transAxes,
            ha="center",
            va="center",
        )
    ax_bottom.set_xlabel("User subreddit concentration (HHI)")
    ax_bottom.set_ylabel("Share of multi-subreddit users")
    _style_axes(ax_bottom)
    return fig

def _monthly_engagement_bin_prevalence_plot(df: pd.DataFrame, title: str, publication_style: bool = False) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    monthly = _prepare_monthly_plot_frame(df)
    for bin_name in ["0", "1", "2-5", "6-20", "21+"]:
        subset = monthly.loc[monthly["engagement_bin"] == bin_name].copy()
        if len(subset):
            line, = ax.plot(subset["month"], subset["user_share"], marker="o", linewidth=1.8, label=bin_name)
            if publication_style:
                _add_series_endpoint_label(ax, subset["month"], subset["user_share"], bin_name, line.get_color())
    ax.set_xlabel("Month")
    ax.set_ylabel("Share of active users")
    _apply_title_or_panel_label(ax, title, publication_style)
    _format_month_axis(ax)
    if not publication_style:
        ax.legend(
            title="Received engagement",
            frameon=False,
            bbox_to_anchor=(1.02, 0.5),
            loc="center left",
            borderaxespad=0.0,
        )
    _style_axes(ax)
    return fig

def _add_series_endpoint_label(
    ax: plt.Axes,
    x: pd.Series,
    y: pd.Series,
    label: str,
    color: str,
) -> None:
    """Place a direct label near the last visible point of a line series."""
    x_vals = pd.Series(x)
    y_vals = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = x_vals.notna() & y_vals.notna()
    if not mask.any():
        return
    x_last = x_vals.loc[mask].iloc[-1]
    y_last = float(y_vals.loc[mask].iloc[-1])
    ax.annotate(
        label,
        xy=(x_last, y_last),
        xytext=(8, 0),
        textcoords="offset points",
        color=color,
        va="center",
        ha="left",
        clip_on=False,
        fontsize=PLOT_SIZES["direct_label"],
    )

def _subreddit_subset_selection_ranked_plot(
    df: pd.DataFrame,
    title: str,
    top_n: int,
    publication_style: bool = False,
) -> plt.Figure:
    ranked = df.loc[df["is_eligible_for_subset_selection"].fillna(False).astype(bool)].copy()
    ranked = ranked.loc[ranked["candidate_score"].notna()].copy()

    if not len(ranked):
        fig, ax = plt.subplots(figsize=(8.6, 5.0))
        ax.text(
            0.5,
            0.5,
            "No eligible subreddits available\nfor subset selection",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        _apply_title_or_panel_label(ax, title, publication_style)
        return fig

    ranked = ranked.sort_values(["candidate_score", "n_total_activity", "subreddit"], ascending=[False, False, True]).head(top_n).copy()
    ranked = ranked.sort_values("candidate_score", ascending=True).reset_index(drop=True)
    fig_height = min(13.0, max(6.5, 0.42 * len(ranked) + 2.0))
    fig, ax = plt.subplots(figsize=(8.6, fig_height))
    colors = [
        COLORBLIND_SAFE_PALETTE[0] if bool(flag) else "#B3B3B3"
        for flag in ranked["candidate_flag"].tolist()
    ]

    ax.barh(ranked["subreddit"], ranked["candidate_score"], color=colors)
    if len(ranked):
        ax.set_xlim(0, min(max(1.0, float(ranked["candidate_score"].max()) * 1.05), 1.15))
    ax.set_xlabel("Subset-selection candidate score")
    ax.set_ylabel("Subreddit")
    _apply_title_or_panel_label(ax, title, publication_style)
    _style_axes(ax)
    return fig

def _subset_selection_diagnostic_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    publication_style: bool = False,
    x_log: bool = True,
    y_log: bool = False,
    top_n_labels: int = SUBSET_DIAG_TOP_LABEL_COUNT,
    legend_loc: str = "best",
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Plot one eligible subreddit per point with candidate highlights and a descriptive trend."""
    eligible = df.loc[df["is_eligible_for_subset_selection"].fillna(False).astype(bool)].copy()
    fig, ax = plt.subplots(figsize=(7.8, 5.8))

    if not len(eligible):
        ax.text(
            0.5,
            0.5,
            "No eligible subreddits available\nfor subset diagnostics",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        _apply_title_or_panel_label(ax, title, publication_style)
        return fig

    eligible[x_col] = pd.to_numeric(eligible[x_col], errors="coerce")
    eligible[y_col] = pd.to_numeric(eligible[y_col], errors="coerce")
    mask = eligible[x_col].notna() & eligible[y_col].notna()
    if x_log:
        mask &= eligible[x_col] > 0
    if y_log:
        mask &= eligible[y_col] > 0
    eligible = eligible.loc[mask].copy()

    if not len(eligible):
        ax.text(
            0.5,
            0.5,
            "No eligible subreddits with\nplottable values",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        _apply_title_or_panel_label(ax, title, publication_style)
        return fig

    subset_point_size = 10
    base_color = "#B8BDC7"
    highlight_color = "#E69F00"
    top_highlight_color = "#A50F15"
    trend_color = "#6E7781"

    base = eligible.loc[~eligible["candidate_flag"].fillna(False).astype(bool)].copy()
    shortlisted = eligible.loc[eligible["candidate_flag"].fillna(False).astype(bool)].copy()
    if len(shortlisted):
        shortlisted["candidate_rank"] = pd.to_numeric(shortlisted["candidate_rank"], errors="coerce")
        top_shortlisted_mask = shortlisted["candidate_rank"].le(top_n_labels)
        top_shortlisted = shortlisted.loc[top_shortlisted_mask].copy()
        other_shortlisted = shortlisted.loc[~top_shortlisted_mask].copy()
    else:
        top_shortlisted = shortlisted
        other_shortlisted = shortlisted

    if len(base):
        ax.scatter(
            base[x_col],
            base[y_col],
            s=subset_point_size,
            alpha=0.28,
            color=base_color,
            edgecolors="none",
            label="Eligible subreddits",
            zorder=1,
        )

    if len(other_shortlisted):
        ax.scatter(
            other_shortlisted[x_col],
            other_shortlisted[y_col],
            s=subset_point_size,
            alpha=0.9,
            color=highlight_color,
            edgecolors="none",
            label="Shortlisted candidates",
            zorder=3,
        )

    if len(top_shortlisted):
        ax.scatter(
            top_shortlisted[x_col],
            top_shortlisted[y_col],
            s=subset_point_size,
            alpha=0.95,
            color=top_highlight_color,
            edgecolors="none",
            label=f"Top {top_n_labels} shortlisted candidates",
            zorder=4,
        )

    x_vals = eligible[x_col].to_numpy(dtype=float)
    y_vals = eligible[y_col].to_numpy(dtype=float)
    if len(eligible) >= 5 and x_vals.min() < x_vals.max():
        n_bins = min(14, max(6, int(np.sqrt(len(eligible)))))
        if x_log:
            x_edges = np.logspace(np.log10(x_vals.min()), np.log10(x_vals.max()), n_bins + 1)
        else:
            x_edges = np.linspace(x_vals.min(), x_vals.max(), n_bins + 1)
        x_centers: List[float] = []
        y_medians: List[float] = []
        for i in range(len(x_edges) - 1):
            left = x_edges[i]
            right = x_edges[i + 1]
            if i == len(x_edges) - 2:
                in_bin = (x_vals >= left) & (x_vals <= right)
            else:
                in_bin = (x_vals >= left) & (x_vals < right)
            if not np.any(in_bin):
                continue
            y_bin = y_vals[in_bin]
            if y_log:
                y_bin = y_bin[y_bin > 0]
            if not len(y_bin):
                continue
            if x_log:
                x_centers.append(float(np.exp(np.mean(np.log(x_vals[in_bin])))))
            else:
                x_centers.append(float(np.mean(x_vals[in_bin])))
            y_medians.append(float(np.median(y_bin)))
        if x_centers and y_medians:
            ax.plot(
                x_centers,
                y_medians,
                color=trend_color,
                linewidth=1.1,
                alpha=0.9,
                zorder=2,
            )

    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    elif "repeat" in y_col:
        ax.set_ylim(bottom=0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _apply_title_or_panel_label(ax, title, publication_style)
    _style_axes(ax)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend_kwargs = {
            "frameon": False,
            "loc": legend_loc,
        }
        if legend_bbox_to_anchor is not None:
            legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
        ax.legend(handles, labels, **legend_kwargs)
    return fig

def _format_readable_number(value: float, _pos: int) -> str:
    """Format large counts without scientific notation."""
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:g}"

def _format_month_axis(ax: plt.Axes) -> None:
    """Keep month labels readable after figures are scaled in LaTeX."""
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

def _style_axes(ax: plt.Axes) -> None:
    """Apply consistent publication-style axis formatting."""
    ax.grid(False)
    ax.grid(False, which="minor")
    ax.minorticks_off()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", labelsize=PLOT_SIZES["tick"])

def _apply_title_or_panel_label(
    ax: plt.Axes,
    text: str,
    publication_style: bool,
    panel_label: Optional[str] = None,
) -> None:
    """Handle title text under standard versus publication-style output."""
    if publication_style:
        ax.set_title("")
        if panel_label:
            ax.set_title(panel_label, loc="left")
    else:
        ax.set_title(text)

def _prepare_monthly_plot_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the month column to datetime and sort chronologically for plotting."""
    out = df.copy()
    out["month"] = pd.to_datetime(out["month"], format="%Y-%m", errors="raise")
    return out.sort_values("month").reset_index(drop=True)

def generate_figures(
    out_root: str,
    tables: Dict[str, pd.DataFrame],
    top_n_subset_diag: int,
    publication_style: bool = False,
    skip_subreddit_subset_diagnostics: bool = False,
) -> None:
    """Generate and save all figures."""
    # 1. monthly platform activity
    save_figure(
        _monthly_platform_activity_plot(
            tables["monthly_activity"].copy(),
            "Monthly platform activity",
            publication_style=publication_style,
        ),
        out_root,
        "monthly_platform_activity",
    )

    sr = tables["subreddit_activity"].copy()
    save_figure(
        _rank_size_plot(
            sr,
            "n_submissions",
            "Subreddit rank-size distribution by submissions",
            "Submissions",
            publication_style=publication_style,
        ),
        out_root,
        "subreddit_rank_size_submissions",
    )
    save_figure(
        _rank_size_plot(
            sr,
            "n_comments",
            "Subreddit rank-size distribution by comments",
            "Comments",
            publication_style=publication_style,
        ),
        out_root,
        "subreddit_rank_size_comments",
    )

    save_figure(
        _survival_table_plot(
            tables["user_activity_distribution_total_plot"].copy(),
            "activity_value",
            "survival_share",
            "Per-user total-activity survival distribution",
            "Total activity per user",
            "Share of users with at least this much total activity",
            publication_style=publication_style,
        ),
        out_root,
        "user_activity_distribution_total",
    )
    save_figure(
        _survival_table_plot(
            tables["user_activity_distribution_submissions_plot"].copy(),
            "submission_count",
            "survival_share",
            "Per-user submission-count survival distribution",
            "Submissions per user",
            "Share of users with at least this many submissions",
            publication_style=publication_style,
        ),
        out_root,
        "user_activity_distribution_submissions",
    )
    save_figure(
        _survival_table_plot(
            tables["user_activity_distribution_comments_plot"].copy(),
            "comment_count",
            "survival_share",
            "Per-user comment-count survival distribution",
            "Comments per user",
            "Share of users with at least this many comments",
            publication_style=publication_style,
        ),
        out_root,
        "user_activity_distribution_comments",
    )
    save_figure(
        _lorenz_curve_plot(
            tables["user_activity_lorenz_curve_plot"].copy(),
            "Lorenz curve of user activity",
            "cum_activity_share",
            "Cumulative share of activity",
            summary=tables["user_activity_lorenz_summary"].copy(),
            publication_style=publication_style,
        ),
        out_root,
        "user_activity_lorenz_curve",
    )

    save_figure(
        _survival_table_plot(
            tables["user_engagement_received_distribution_plot"].copy(),
            "engagement_value",
            "survival_share",
            "Per-user received-engagement survival distribution",
            "Received engagement per user",
            "Share of users with at least this much received engagement",
            publication_style=publication_style,
        ),
        out_root,
        "user_engagement_received_distribution",
    )
    save_figure(
        _stratified_rate_survival_plot(
            tables["user_engagement_rate_distribution_plot"].copy(),
            "Received engagement per activity distribution",
            publication_style=publication_style,
        ),
        out_root,
        "received_engagement_per_activity_distribution",
    )
    monthly_rate = tables["monthly_engagement_rate_summary"].copy()
    monthly_rate = monthly_rate.loc[
        pd.to_numeric(monthly_rate["min_total_activity"], errors="coerce") == 1
    ].copy()
    save_figure(
        _monthly_quantile_ribbon_plot(
            monthly_rate,
            "month",
            "median_received_engagement_per_activity",
            "p25_received_engagement_per_activity",
            "p75_received_engagement_per_activity",
            "Monthly received engagement per activity",
            "Month",
            "Received engagement per authored activity",
            publication_style=publication_style,
        ),
        out_root,
        "monthly_received_engagement_per_activity_iqr",
    )
    save_figure(
        _lorenz_curve_plot(
            tables["user_engagement_lorenz_curve_plot"].copy(),
            "Lorenz curve of received engagement",
            "cum_engagement_share",
            "Cumulative share of received engagement",
            summary=tables["user_engagement_lorenz_summary"].copy(),
            publication_style=publication_style,
        ),
        out_root,
        "user_engagement_lorenz_curve",
    )

    thread_size_plot = tables["thread_size_distribution_plot"].copy()
    thread_size_plot["thread_size"] = pd.to_numeric(thread_size_plot["thread_size"], errors="coerce")
    thread_size_plot["n_submissions"] = pd.to_numeric(thread_size_plot["n_submissions"], errors="coerce")
    total_thread_submissions = thread_size_plot["n_submissions"].sum()
    zero_comment_share = (
        float(thread_size_plot.loc[thread_size_plot["thread_size"] == 0, "n_submissions"].sum() / total_thread_submissions)
        if total_thread_submissions > 0
        else float("nan")
    )
    thread_size_annotation = (
        f"Zero-comment share: {zero_comment_share:.1%}"
        if np.isfinite(zero_comment_share)
        else None
    )
    save_figure(
        _survival_table_plot(
            thread_size_plot,
            "thread_size",
            "survival_share",
            "Thread-size survival distribution",
            "Thread size (comments per submission)",
            "Share of submissions with at least this many comments",
            publication_style=publication_style,
            annotation=thread_size_annotation,
        ),
        out_root,
        "thread_size_distribution",
    )

    mt = _prepare_monthly_plot_frame(tables["monthly_thread_size"])
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    mean_line, = ax.plot(mt["month"], mt["mean_thread_size"], marker="o", linewidth=1.8)
    median_line, = ax.plot(mt["month"], mt["median_thread_size"], marker="o", linewidth=1.8)
    ax.set_xlabel("Month")
    thread_values = pd.to_numeric(pd.concat([mt["mean_thread_size"], mt["median_thread_size"]]), errors="coerce")
    if (thread_values.dropna() > 0).all():
        ax.set_yscale("log")
        positive_max = float(thread_values.dropna().max())
        candidate_ticks = [1, 2, 5, 10, 20, 50, 100]
        y_ticks = [tick for tick in candidate_ticks if tick <= positive_max * 1.25]
        if 1 not in y_ticks:
            y_ticks.insert(0, 1)
        upper_tick = next((tick for tick in candidate_ticks if tick > positive_max), None)
        if upper_tick is not None and upper_tick not in y_ticks:
            y_ticks.append(upper_tick)
        if not y_ticks:
            y_ticks = [1]
        ax.set_yticks(sorted(set(y_ticks)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax.set_ylabel("Comments per submission (log scale)")
    else:
        ax.set_ylabel("Comments per submission")
    _apply_title_or_panel_label(ax, "Monthly thread size summary", publication_style)
    _format_month_axis(ax)
    _style_axes(ax)
    _add_series_endpoint_label(ax, mt["month"], mt["mean_thread_size"], "Mean", mean_line.get_color())
    _add_series_endpoint_label(ax, mt["month"], mt["median_thread_size"], "Median", median_line.get_color())
    save_figure(fig, out_root, "monthly_thread_size")
    save_figure(
        _monthly_quantile_ribbon_plot(
            tables["monthly_thread_size_quantile_ribbon_summary"].copy(),
            "month",
            "median_thread_size",
            "p25_thread_size",
            "p75_thread_size",
            "Monthly thread size distribution summary",
            "Month",
            "Comments per submission",
            publication_style=publication_style,
        ),
        out_root,
        "monthly_thread_size_quantile_ribbon",
    )

    am = tables["active_months_per_user_summary"].copy()
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    ax.bar(am["n_active_months"].astype(str), am["user_share"])
    ax.set_xlabel("Number of active months in selected period")
    ax.set_ylabel("Share of users")
    ax.set_ylim(bottom=0)
    _apply_title_or_panel_label(ax, "Distribution of active months per user", publication_style)
    _style_axes(ax)
    save_figure(fig, out_root, "active_months_per_user_distribution")
    save_figure(
        _active_subreddits_distribution_plot(
            tables["active_subreddits_per_user_summary"].copy(),
            "Distribution of active subreddits per user",
            publication_style=publication_style,
        ),
        out_root,
        "active_subreddits_per_user_distribution",
    )

    save_figure(
        _user_subreddit_hhi_two_panel_plot(
            tables["user_subreddit_hhi_plot"].copy(),
            tables["user_subreddit_hhi_summary"].copy(),
            "Distribution of user subreddit concentration",
            publication_style=publication_style,
        ),
        out_root,
        "user_subreddit_hhi_distribution",
    )

    repeat = tables["subreddit_repeat_user_share"].copy()
    save_figure(
        _hexbin_log_with_median_trend_plot(
            repeat,
            "mean_monthly_active_users",
            "mean_pct_users_2plus_activity",
            "Subreddit repeat-user share versus size",
            "Mean monthly active users per subreddit",
            "Mean monthly share of active users with >=2 activities",
            publication_style=publication_style,
        ),
        out_root,
        "repeat_user_share_by_subreddit",
    )

    lag_summary_1 = tables["lag_past_activity_future_activity_summary"].copy()
    save_figure(
        _decile_mean_median_plot(
            lag_summary_1,
            "activity_t_decile",
            "mean_activity_t1",
            "median_activity_t1",
            "Past activity versus future activity",
            "Activity decile at t",
            "Activity at t+1",
            publication_style=publication_style,
        ),
        out_root,
        "lag_past_activity_future_activity",
    )
    save_figure(
        _decile_quantile_ribbon_plot(
            tables["lag_past_activity_future_activity_quantile_ribbon_summary"].copy(),
            "activity_t_decile",
            "median_activity_t1",
            "p25_activity_t1",
            "p75_activity_t1",
            "Past activity versus future activity",
            "Activity decile at t",
            "Activity at t+1",
            publication_style=publication_style,
        ),
        out_root,
        "lag_past_activity_future_activity_quantile_ribbon",
    )

    lag_summary_2 = tables["lag_past_engagement_future_engagement_summary"].copy()
    save_figure(
        _decile_mean_median_plot(
            lag_summary_2,
            "engagement_t_decile",
            "mean_engagement_t1",
            "median_engagement_t1",
            "Past engagement versus future engagement",
            "Engagement decile at t",
            "Engagement at t+1",
            publication_style=publication_style,
        ),
        out_root,
        "lag_past_engagement_future_engagement",
    )
    save_figure(
        _decile_quantile_ribbon_plot(
            tables["lag_past_engagement_future_engagement_quantile_ribbon_summary"].copy(),
            "engagement_t_decile",
            "median_engagement_t1",
            "p25_engagement_t1",
            "p75_engagement_t1",
            "Past engagement versus future engagement",
            "Engagement decile at t",
            "Engagement at t+1",
            publication_style=publication_style,
        ),
        out_root,
        "lag_past_engagement_future_engagement_quantile_ribbon",
    )

    lag_summary_3 = tables["lag_past_activity_future_engagement_summary"].copy()
    save_figure(
        _decile_mean_median_plot(
            lag_summary_3,
            "activity_t_decile",
            "mean_engagement_t1",
            "median_engagement_t1",
            "Past activity versus future engagement",
            "Activity decile at t",
            "Engagement at t+1",
            publication_style=publication_style,
        ),
        out_root,
        "lag_past_activity_future_engagement",
    )
    save_figure(
        _decile_quantile_ribbon_plot(
            tables["lag_past_activity_future_engagement_quantile_ribbon_summary"].copy(),
            "activity_t_decile",
            "median_engagement_t1",
            "p25_engagement_t1",
            "p75_engagement_t1",
            "Past activity versus future engagement",
            "Activity decile at t",
            "Engagement at t+1",
            publication_style=publication_style,
        ),
        out_root,
        "lag_past_activity_future_engagement_quantile_ribbon",
    )
    save_figure(
        _transition_heatmap_plot(
            tables["activity_transition_matrix"].copy(),
            "Activity transition heatmap from t to t+1",
            "activity_t_decile",
            "activity_t1_state",
            "Activity decile at t",
            "Activity state at t+1",
            publication_style=publication_style,
        ),
        out_root,
        "activity_transition_heatmap_t_to_t1",
    )
    save_figure(
        _transition_heatmap_plot(
            tables["engagement_transition_matrix"].copy(),
            "Engagement transition heatmap from t to t+1",
            "engagement_t_decile",
            "engagement_t1_state",
            "Engagement decile at t",
            "Engagement state at t+1",
            publication_style=publication_style,
        ),
        out_root,
        "engagement_transition_heatmap_t_to_t1",
    )
    save_figure(
        _monthly_engagement_bin_prevalence_plot(
            tables["monthly_engagement_bin_prevalence"].copy(),
            "Monthly engagement-bin prevalence among active users",
            publication_style=publication_style,
        ),
        out_root,
        "monthly_engagement_bin_prevalence",
    )
    if skip_subreddit_subset_diagnostics:
        return

    save_figure(
        _subreddit_subset_selection_ranked_plot(
            tables["subreddit_subset_selection_summary"].copy(),
            "Ranked subreddit subset-selection candidates",
            top_n=top_n_subset_diag,
            publication_style=publication_style,
        ),
        out_root,
        "subreddit_subset_selection_ranked",
    )

    subset = tables["subreddit_subset_selection_summary"].copy()

    save_figure(
        _subset_selection_diagnostic_plot(
            subset,
            "n_total_activity",
            "mean_thread_size",
            "Subset-selection diagnostic: subreddit size versus engagement",
            "Total subreddit activity (selected period)",
            "Mean thread size",
            publication_style=publication_style,
            x_log=True,
            y_log=True,
            top_n_labels=SUBSET_DIAG_TOP_LABEL_COUNT,
            legend_loc="upper left",
        ),
        out_root,
        "subset_diag_size_vs_engagement",
    )
    save_figure(
        _subset_selection_diagnostic_plot(
            subset,
            "n_total_activity",
            "mean_pct_users_2plus_activity",
            "Subset-selection diagnostic: subreddit size versus repeat-user share",
            "Total subreddit activity (selected period)",
            "Mean monthly repeat-user share",
            publication_style=publication_style,
            x_log=True,
            y_log=False,
            top_n_labels=SUBSET_DIAG_TOP_LABEL_COUNT,
            legend_loc="lower right",
        ),
        out_root,
        "subset_diag_size_vs_repeat_share",
    )
    save_figure(
        _subset_selection_diagnostic_plot(
            subset,
            "mean_thread_size",
            "mean_pct_users_2plus_activity",
            "Subset-selection diagnostic: engagement versus repeat-user share",
            "Mean thread size",
            "Mean monthly repeat-user share",
            publication_style=publication_style,
            x_log=True,
            y_log=False,
            top_n_labels=SUBSET_DIAG_TOP_LABEL_COUNT,
            legend_loc="lower right",
        ),
        out_root,
        "subset_diag_engagement_vs_repeat_share",
    )
