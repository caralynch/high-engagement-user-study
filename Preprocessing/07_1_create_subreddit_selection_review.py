#!/usr/bin/env python3
"""Create a manual subreddit-selection review workbook.

This helper consumes the CSV outputs from
`Preprocessing/07_0_select_candidate_subreddits.py` and creates a reviewable
Excel workbook for manual Anglophone/language-script decisions. It does not run
language detection and does not modify the candidate scoring logic.

The intended use case is a manual, auditable final-selection step:

1. Start from quantitatively eligible candidate subreddits produced by 07_0.
2. Review candidates in ascending `rank_S_mean` order.
3. Exclude candidates that fall outside the predefined Anglophone/Latin-script
   modelling scope, recording the reason.
4. Continue down the `S_mean` ranking until 20 included communities are obtained.
5. Export the narrow final manifest as `selected_subreddits_final.csv` for the
   downstream selected-subreddit extraction step.

The `S_max` ranking is included for comparison/sensitivity only. It is not used
to fill gaps in the main `S_mean` selection unless the study design is explicitly
changed.

Required input, relative to `--in-dir`:
- `subreddit_candidates_all.csv`
    Wide 07_0 audit file containing all candidate rows, eligibility fields,
    quantitative metrics, scores, ranks, language-status columns, and selection
    metadata.

Optional inputs, relative to `--in-dir`:
- `subreddit_candidates_smax_top.csv`
    Top candidates under the `S_max` comparison ranking. If absent, the script
    derives the `S_max` review sheet from `subreddit_candidates_all.csv`.
- `selected_subreddits_final.csv`
    Existing 07_0 selected-subreddit output. If present, it is used only for
    comparison/prefill context; the manual review workbook remains the source
    for the reviewed final manifest.

Outputs, relative to `--out-dir`:
- `subreddit_selection_review.xlsx`
    Excel workbook containing:
    - `S_mean_review`: top `--top-k` quantitatively eligible candidates sorted
      by `rank_S_mean`, with compact quantitative columns and manual review
      fields.
    - `S_max_review`: top `--top-k` quantitatively eligible candidates sorted
      by `rank_S_max`, for sensitivity/comparison review.
    - `final_selected_manifest`: narrow scaffold intended to contain only the
      final included subreddits and the minimal metadata needed downstream.
    - `README`: embedded description of the selection rule, manual fields, and
      intended workflow.
- `selected_subreddits_final_manual_template.csv`
    CSV scaffold matching the narrow final manifest structure. This is a
    starting template only; after manual review, export the completed final
    manifest as `selected_subreddits_final.csv`.

Main CLI arguments:
- `--in-dir PATH`
    Directory containing the 07_0 CSV outputs.
- `--out-dir PATH`
    Directory where the review workbook and manifest template should be written.
- `--top-k INT`
    Number of candidates to include in each review sheet. Default should usually
    be at least 50 so that replacements are available after exclusions.
- `--overwrite`
    Allow existing output files to be replaced.

Example on Linux:

    python Preprocessing/07_1_create_subreddit_selection_review.py \\
      --in-dir runs/subreddit_selection/tables \\
      --out-dir runs/subreddit_selection_review \\
      --top-k 50 \\
      --overwrite

Example, Windows PowerShell:

    python .\\Preprocessing\\07_1_create_subreddit_selection_review.py `
      --in-dir "..\\data\\outputs\\subreddit_selection_07_0\\tables" `
      --out-dir "..\\data\\outputs\\subreddit_selection_review" `
      --top-k 50 `
      --overwrite

Example, if running the helper from inside the 07_0 output directory:

    python 07_1_create_subreddit_selection_review.py \\
      --in-dir tables \\
      --out-dir manual_review \\
      --top-k 50 \\
      --overwrite

Manual review fields added to the review sheets:
- `manual_review_status`
- `manual_language_passed`
- `manual_language_label`
- `manual_script_label`
- `include_in_main_modelling`
- `selection_rank`
- `selection_reason`
- `exclusion_reason`
- `manual_review_note`

Recommended controlled values:
- `manual_review_status`: `not_reviewed`, `reviewed`, `uncertain`
- `manual_language_passed`: `TRUE`, `FALSE`, `UNCERTAIN`
- `manual_language_label`: `English`, `Mostly English`, `Mixed`,
  `Non-English`, `Unclear`
- `manual_script_label`: `Latin`, `Cyrillic`, `Mixed`, `Other`, `Unclear`

The full wide candidate CSV should be kept unchanged as the audit trail. The
review workbook is the manual decision layer, and the final exported manifest
should be narrow, stable, and suitable for downstream pipeline use.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script needs openpyxl. Install with: pip install openpyxl"
    ) from exc


REVIEW_COLUMNS = [
    "subreddit",
    "rank_S_mean",
    "S_mean",
    "rank_S_max",
    "S_max",
    "total_activity",
    "mean_monthly_active_users",
    "min_monthly_active_users",
    "mean_thread_size",
    "median_thread_size",
    "repeat_user_share_2plus",
    "activity_cv",
    "nsfw_submission_share",
    "eligible_full_period",
    "ordinary_subreddit",
    "active_user_threshold_passed",
    "nsfw_threshold_passed",
    "eligible_quantitative",
    "candidate_by_S_mean",
    "candidate_by_S_max",
    # Manual review fields.
    "manual_review_status",
    "manual_language_passed",
    "manual_language_label",
    "manual_script_label",
    "include_in_main_modelling",
    "selection_rank",
    "selection_reason",
    "exclusion_reason",
    "manual_review_note",
]

FINAL_COLUMNS = [
    "selection_rank",
    "subreddit",
    "rank_S_mean",
    "S_mean",
    "rank_S_max",
    "S_max",
    "total_activity",
    "mean_monthly_active_users",
    "min_monthly_active_users",
    "mean_thread_size",
    "median_thread_size",
    "repeat_user_share_2plus",
    "activity_cv",
    "manual_language_passed",
    "manual_language_label",
    "manual_script_label",
    "selection_reason",
    "manual_review_note",
]

README_ROWS = [
    ["Purpose", "Manual review workbook for final subreddit subset selection."],
    ["Main rule", "Select the first 20 quantitatively eligible ordinary subreddits by S_mean after manual Anglophone/script review."],
    ["Replacement rule", "If a candidate is excluded, replacement proceeds strictly down the S_mean ranking."],
    ["S_max role", "S_max is a sensitivity/comparison ranking only; it is not used to choose replacements for the main subset unless the study design is explicitly changed."],
    ["Language review", "No automated language scanning is assumed. Fill manual language/script fields manually."],
    ["Final export", "After review, export final_selected_manifest as selected_subreddits_final.csv for downstream scripts."],
]

LIST_VALUES = {
    "manual_review_status": ["not_reviewed", "reviewed", "uncertain"],
    "manual_language_passed": ["TRUE", "FALSE", "UNCERTAIN"],
    "manual_language_label": ["English", "Mostly English", "Mixed", "Non-English", "Unclear"],
    "manual_script_label": ["Latin", "Cyrillic", "Mixed", "Other", "Unclear"],
    "include_in_main_modelling": ["TRUE", "FALSE", "UNCERTAIN"],
    "exclusion_reason": [
        "",
        "outside_anglophone_scope",
        "non_latin_script",
        "mixed_language_uncertain",
        "outside_top20_after_replacement",
        "other_manual_exclusion",
    ],
}

DEFAULT_SELECTION_REASON = (
    "Top-20 S_mean candidate after manual Anglophone/script screening; retained in rank order."
)


def _read_csv(path: Path, required: bool = True) -> pd.DataFrame | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required input: {path}")
        return None
    return pd.read_csv(path)


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out.loc[:, list(columns)]


def _prepare_review_frame(df: pd.DataFrame, rank_col: str, top_k: int) -> pd.DataFrame:
    if "eligible_quantitative" in df.columns:
        df = df.loc[df["eligible_quantitative"].fillna(False).astype(bool)].copy()
    else:
        df = df.copy()

    df[rank_col] = pd.to_numeric(df[rank_col], errors="coerce")
    df = df.sort_values(rank_col, na_position="last").head(top_k).copy()

    # Initialise manual fields. Keep strings so Excel validation behaves cleanly.
    df["manual_review_status"] = "not_reviewed"
    df["manual_language_passed"] = "UNCERTAIN"
    df["manual_language_label"] = "Unclear"
    df["manual_script_label"] = "Unclear"
    df["include_in_main_modelling"] = "UNCERTAIN"
    df["selection_rank"] = pd.NA
    df["selection_reason"] = ""
    df["exclusion_reason"] = ""
    df["manual_review_note"] = ""

    return _ensure_columns(df, REVIEW_COLUMNS)


def _prepare_final_template(s_mean_review: pd.DataFrame, selected_existing: pd.DataFrame | None) -> pd.DataFrame:
    """Create an initial final manifest template.

    If selected_subreddits_final.csv exists, use it as the prefilled top-20 list,
    but mark language fields as requiring manual review. Otherwise, create an empty
    20-row template.
    """
    if selected_existing is not None and len(selected_existing):
        src = selected_existing.copy().head(20)
        src["selection_rank"] = range(1, len(src) + 1)
        src["manual_language_passed"] = "UNCERTAIN"
        src["manual_language_label"] = "Unclear"
        src["manual_script_label"] = "Unclear"
        src["selection_reason"] = DEFAULT_SELECTION_REASON
        src["manual_review_note"] = "Prefilled from current selected_subreddits_final.csv; manual language/script review still required."
        return _ensure_columns(src, FINAL_COLUMNS)

    empty = pd.DataFrame({col: [pd.NA] * 20 for col in FINAL_COLUMNS})
    empty["selection_rank"] = list(range(1, 21))
    return empty


def _write_workbook(
    out_xlsx: Path,
    s_mean_review: pd.DataFrame,
    s_max_review: pd.DataFrame,
    final_template: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        s_mean_review.to_excel(writer, sheet_name="S_mean_review", index=False)
        s_max_review.to_excel(writer, sheet_name="S_max_review", index=False)
        final_template.to_excel(writer, sheet_name="final_selected_manifest", index=False)
        pd.DataFrame(README_ROWS, columns=["Field", "Description"]).to_excel(writer, sheet_name="README", index=False)

    wb = load_workbook(out_xlsx)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    manual_fill = PatternFill("solid", fgColor="FFF2CC")
    header_font = Font(color="FFFFFF", bold=True)
    manual_cols = {
        "manual_review_status",
        "manual_language_passed",
        "manual_language_label",
        "manual_script_label",
        "include_in_main_modelling",
        "selection_rank",
        "selection_reason",
        "exclusion_reason",
        "manual_review_note",
    }

    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.row_dimensions[1].height = 30

        headers = [cell.value for cell in ws[1]]
        for idx, header in enumerate(headers, start=1):
            letter = get_column_letter(idx)
            if header in manual_cols:
                for row in range(2, ws.max_row + 1):
                    ws[f"{letter}{row}"].fill = manual_fill
                    ws[f"{letter}{row}"].alignment = Alignment(wrap_text=True, vertical="top")
            width = 14
            if header in {"subreddit", "selection_reason", "exclusion_reason", "manual_review_note", "Description"}:
                width = 36 if header != "manual_review_note" else 52
            elif header in {"total_activity", "mean_monthly_active_users", "min_monthly_active_users"}:
                width = 18
            ws.column_dimensions[letter].width = width

        # Data validation for editable categorical fields.
        if ws.title in {"S_mean_review", "S_max_review"}:
            for col_name, allowed in LIST_VALUES.items():
                if col_name not in headers:
                    continue
                col_idx = headers.index(col_name) + 1
                letter = get_column_letter(col_idx)
                formula = '"' + ",".join(allowed) + '"'
                dv = DataValidation(type="list", formula1=formula, allow_blank=True)
                ws.add_data_validation(dv)
                dv.add(f"{letter}2:{letter}{ws.max_row}")

    wb.save(out_xlsx)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=Path("."), help="Directory containing 07_0 CSV outputs.")
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="Directory for review workbook outputs.")
    parser.add_argument("--top-k", type=int, default=50, help="Number of rows to include in each ranking review sheet.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_xlsx = args.out_dir / "subreddit_selection_review.xlsx"
    out_csv = args.out_dir / "selected_subreddits_final_manual_template.csv"
    if not args.overwrite and (out_xlsx.exists() or out_csv.exists()):
        raise FileExistsError("Output exists. Re-run with --overwrite if you want to replace it.")

    candidates = _read_csv(args.in_dir / "subreddit_candidates_all.csv", required=True)
    smax = _read_csv(args.in_dir / "subreddit_candidates_smax_top.csv", required=False)
    selected = _read_csv(args.in_dir / "selected_subreddits_final.csv", required=False)

    s_mean_review = _prepare_review_frame(candidates, "rank_S_mean", args.top_k)
    s_max_source = smax if smax is not None and len(smax) else candidates
    s_max_review = _prepare_review_frame(s_max_source, "rank_S_max", args.top_k)

    # Mark overlap with the prefilled S_mean top-k review for convenience.
    s_mean_subreddits = set(s_mean_review["subreddit"].dropna().astype(str))
    s_max_review.insert(
        s_max_review.columns.get_loc("manual_review_status"),
        "already_in_S_mean_review_top_k",
        s_max_review["subreddit"].astype(str).isin(s_mean_subreddits),
    )

    final_template = _prepare_final_template(s_mean_review, selected)

    _write_workbook(out_xlsx, s_mean_review, s_max_review, final_template)
    final_template.to_csv(out_csv, index=False)

    print(f"Wrote {out_xlsx}")
    print(f"Wrote {out_csv}")
    print("Next: manually review S_mean_review, then export final_selected_manifest as selected_subreddits_final.csv.")


if __name__ == "__main__":
    main()
