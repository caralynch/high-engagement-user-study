from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import pandas as pd

from .scoring import ensure_required_output_columns, top_by_score


OUTPUT_FILES: List[str] = [
    "tables/subreddit_candidates_all.csv",
    "tables/subreddit_candidates_anglophone.csv",
    "tables/selected_subreddits_final.csv",
    "tables/subreddit_candidates_smax_top.csv",
    "tables/subreddit_selection_summary.csv",
    "run_config.json",
    "selection_notes.md",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def expected_output_paths(out_root: Path) -> List[Path]:
    return [Path(out_root) / rel for rel in OUTPUT_FILES]


def check_output_collisions(out_root: Path, overwrite: bool) -> None:
    if overwrite:
        return
    collisions = [path for path in expected_output_paths(out_root) if path.exists()]
    if collisions:
        raise FileExistsError(
            "Refusing to overwrite existing outputs without --overwrite. Existing files:\n  - "
            + "\n  - ".join(str(path) for path in collisions)
        )


def true_mask(series: pd.Series) -> pd.Series:
    """Return a bool mask where only explicit truthy non-missing values are true."""
    return series.map(lambda value: False if pd.isna(value) else bool(value)).astype(bool)


def true_count(series: pd.Series) -> int:
    """Count explicit truthy values without pandas fillna downcasting warnings."""
    return int(true_mask(series).sum())


def _git_value(args: List[str]) -> Optional[str]:
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
    return result.stdout.strip()


def git_metadata() -> Dict[str, object]:
    status = _git_value(["status", "--short"])
    return {
        "commit": _git_value(["rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_short": status,
    }


def package_versions(language_detector: Optional[str]) -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {"python": sys.version.replace("\n", " "), "language_detector": language_detector}
    for package in ("pandas", "numpy", "duckdb", "langid", "lingua", "langdetect"):
        try:
            module = __import__(package)
        except Exception:
            versions[package] = None
        else:
            versions[package] = getattr(module, "__version__", "installed")
    return versions


def build_summary(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    top_n: int,
    language_mode: str,
) -> pd.DataFrame:
    eligible = candidates.loc[candidates["S_mean"].notna()].copy()
    selected_subreddits = set(selected["subreddit"].astype(str))
    top_mean = set(top_by_score(candidates, "S_mean", int(top_n))["subreddit"].astype(str))
    top_max = set(top_by_score(candidates, "S_max", int(top_n))["subreddit"].astype(str))

    denominator_activity = pd.to_numeric(eligible["total_activity"], errors="coerce").sum()
    denominator_submissions = pd.to_numeric(eligible["n_submissions"], errors="coerce").sum()
    denominator_comments = pd.to_numeric(eligible["n_comments"], errors="coerce").sum()
    retained = eligible.loc[eligible["subreddit"].astype(str).isin(selected_subreddits)]

    def share(numer: float, denom: float) -> object:
        return float(numer / denom) if denom else pd.NA

    payload = {
        "n_eligible_before_language_filtering": int(len(eligible)),
        "n_scanned_for_language": int(candidates["language_scan_status"].isin(["scanned", "insufficient_text", "not_available", "error"]).sum()),
        "n_passing_min_english_share": true_count(candidates["language_filter_passed"]),
        "n_passing_strict_min_english_share": true_count(candidates["strict_language_filter_passed"]),
        "top_n_selected_under_S_mean": int(len(selected)),
        "overlap_top_n_S_mean_S_max": int(len(top_mean & top_max)),
        "retained_total_activity_share": share(
            pd.to_numeric(retained["total_activity"], errors="coerce").sum(),
            denominator_activity,
        ),
        "retained_submissions_share": share(
            pd.to_numeric(retained["n_submissions"], errors="coerce").sum(),
            denominator_submissions,
        ),
        "retained_comments_share": share(
            pd.to_numeric(retained["n_comments"], errors="coerce").sum(),
            denominator_comments,
        ),
        "language_mode": language_mode,
        "anglophone_filter_applied": bool(language_mode == "sample-text"),
        "subreddit_candidates_anglophone_note": (
            "Anglophone filter not applied; file is intentionally empty because language_mode=none."
            if language_mode == "none"
            else "Rows passing language_filter_passed."
        ),
        "thread_size_quality_note": "; ".join(thread_size_quality_notes(candidates)),
    }
    return pd.DataFrame([payload])


def thread_size_quality_notes(candidates: pd.DataFrame) -> List[str]:
    """Return non-fatal notes about missing or zero thread-size metrics."""
    notes: List[str] = []
    groups = [
        ("scored rows", candidates.loc[candidates["S_mean"].notna()]),
        ("final selected rows", candidates.loc[true_mask(candidates["include_in_main_modelling"])]),
    ]
    for label, frame in groups:
        if frame.empty:
            continue
        for column in ("mean_thread_size", "median_thread_size"):
            values = pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else pd.Series(dtype=float)
            nonmissing = values.dropna()
            if nonmissing.empty:
                notes.append(f"{column} is entirely missing among {label}.")
            elif nonmissing.eq(0).all():
                notes.append(f"{column} is entirely zero among {label}.")
    return notes


def write_run_config(
    out_root: Path,
    args_mapping: Mapping[str, object],
    resolved_paths: Mapping[str, str],
    language_detector: Optional[str],
) -> None:
    payload = {
        "timestamp_utc": utc_now_iso(),
        "script": "Preprocessing/07_0_select_candidate_subreddits.py",
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "args": dict(args_mapping),
        "resolved_paths": dict(resolved_paths),
        "git": git_metadata(),
        "package_versions": package_versions(language_detector),
    }
    path = Path(out_root) / "run_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def write_selection_notes(
    out_root: Path,
    language_mode: str,
    language_detector: Optional[str],
    min_monthly_active_users: int,
    nsfw_submission_share_threshold: float,
    min_english_share: float,
    strict_min_english_share: float,
    thread_size_notes: Optional[List[str]] = None,
) -> None:
    detector_text = language_detector or ("not run" if language_mode == "none" else "not available")
    if language_mode == "none":
        language_detail = (
            "- Anglophone filtering was not applied. "
            "`tables/subreddit_candidates_anglophone.csv` is intentionally empty in this mode."
        )
    else:
        language_detail = (
            "- `tables/subreddit_candidates_anglophone.csv` contains rows where `language_filter_passed` is true."
        )
    thread_detail = "\n".join(f"- {note}" for note in (thread_size_notes or [])) or "- No thread-size quality warnings were generated."
    text = f"""# Subreddit Selection Notes

This directory was produced by `Preprocessing/07_0_select_candidate_subreddits.py`.

Eligibility criteria:
- ordinary subreddit names only; user-profile-style names beginning `u_` are excluded;
- observed in every selected month;
- at least {int(min_monthly_active_users)} active users in every selected month;
- NSFW exclusion where metadata are available: `nsfw_submission_share < {float(nsfw_submission_share_threshold):.3f}`.

Scores:
- `R_A`: percentile rank of total activity, higher is better;
- `R_E`: percentile rank of mean thread size, higher is better;
- `R_U`: percentile rank of repeat-user share, higher is better;
- `R_T`: percentile rank of activity CV, lower is better;
- `S_mean = mean(R_A, R_E, R_U, R_T)`;
- `S_max = max(R_A, R_E, R_U, R_T)` for comparison/sensitivity ranking.

Selection flags:
- `candidate_by_S_mean` and `candidate_by_S_max` are threshold flags based on the configured candidate-score threshold;
- `include_in_main_modelling` marks only the final top-N rows selected by `S_mean` for downstream review/filtering.

Language status:
- language mode: `{language_mode}`;
- detector: `{detector_text}`;
- main English threshold: `{float(min_english_share):.3f}`;
- strict reporting threshold: `{float(strict_min_english_share):.3f}`.
{language_detail}

Thread-size notes:
{thread_detail}

Manual review:
Review `tables/selected_subreddits_final.csv` before downstream filtering. It includes
`include_in_main_modelling`, `selection_basis`, `selection_note`, score/rank columns,
and language audit columns so the selected list can be inspected or manually edited.
"""
    path = Path(out_root) / "selection_notes.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_selection_outputs(
    out_root: Path,
    candidates: pd.DataFrame,
    top_n: int,
    language_mode: str,
    overwrite: bool,
) -> Dict[str, Path]:
    """Write all deterministic selection outputs."""
    check_output_collisions(out_root, overwrite)
    tables_dir = Path(out_root) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    candidates_all = ensure_required_output_columns(candidates)
    selected = candidates_all.loc[true_mask(candidates_all["include_in_main_modelling"])].copy()
    smax_top = top_by_score(candidates_all, "S_max", int(top_n))
    if language_mode == "sample-text":
        anglophone = candidates_all.loc[true_mask(candidates_all["language_filter_passed"])].copy()
    else:
        anglophone = candidates_all.iloc[0:0].copy()
        anglophone["selection_note"] = "language scan not run"
    summary = build_summary(candidates_all, selected, top_n=top_n, language_mode=language_mode)

    outputs = {
        "subreddit_candidates_all": tables_dir / "subreddit_candidates_all.csv",
        "subreddit_candidates_anglophone": tables_dir / "subreddit_candidates_anglophone.csv",
        "selected_subreddits_final": tables_dir / "selected_subreddits_final.csv",
        "subreddit_candidates_smax_top": tables_dir / "subreddit_candidates_smax_top.csv",
        "subreddit_selection_summary": tables_dir / "subreddit_selection_summary.csv",
    }
    candidates_all.to_csv(outputs["subreddit_candidates_all"], index=False)
    anglophone.to_csv(outputs["subreddit_candidates_anglophone"], index=False)
    selected.to_csv(outputs["selected_subreddits_final"], index=False)
    smax_top.to_csv(outputs["subreddit_candidates_smax_top"], index=False)
    summary.to_csv(outputs["subreddit_selection_summary"], index=False)
    return outputs
