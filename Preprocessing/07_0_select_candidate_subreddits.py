#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Cara Lynch
# See the LICENSE file for details.
"""
Preprocessing step 7.0: select and rank candidate subreddits for modelling.

Purpose
-------
This standalone step moves final subreddit candidate selection out of the
descriptive plotting pipeline. It reads existing CSV tables produced by
`06_0_descriptive_plots.py`, recomputes explicit candidate score components,
optionally adds local language-identification audit columns, and writes
reviewable selection artefacts for downstream modelling and selected-subreddit
filtering steps.

Inputs
------
--tables-dir
    Directory containing 06_0 table outputs. The primary required input is
    `subreddit_subset_selection_summary.csv`.
--in-root
    Optional cleaned/pseudonymised Parquet root. Required only with
    `--language-mode sample-text`.

Outputs
-------
Under --out-root:
    tables/subreddit_candidates_all.csv
    tables/subreddit_candidates_anglophone.csv
    tables/selected_subreddits_final.csv
    tables/subreddit_candidates_smax_top.csv
    tables/subreddit_selection_summary.csv
    run_config.json
    selection_notes.md

Methodological assumptions
--------------------------
Eligibility is recomputed from 06_0 audit columns: ordinary subreddits only,
full selected-month coverage, a minimum monthly active-user threshold, and an
NSFW submission-share threshold where metadata are available. The score
components are percentile ranks among quantitatively eligible subreddits:
R_A for total activity, R_E for mean thread size, R_U for repeat-user share, and
R_T for activity stability, where lower activity CV is better. The balanced
score is `S_mean = mean(R_A, R_E, R_U, R_T)`. The comparison score is
`S_max = max(R_A, R_E, R_U, R_T)`.

Smoke examples
--------------
python Preprocessing/07_0_select_candidate_subreddits.py \\
  --tables-dir /path/to/06_0_outputs/tables \\
  --out-root /path/to/07_0_subreddit_selection \\
  --language-mode none \\
  --top-n 20 \\
  --overwrite

python Preprocessing/07_0_select_candidate_subreddits.py \\
  --tables-dir /path/to/06_0_outputs/tables \\
  --out-root /path/to/07_0_subreddit_selection_lang \\
  --language-mode sample-text \\
  --in-root /path/to/parquet_pseudo \\
  --top-n 20 \\
  --overwrite
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Preprocessing.subreddit_selection.io import load_primary_selection_table
from Preprocessing.subreddit_selection.language import (
    apply_language_summary,
    find_language_detector,
    initialize_language_columns,
    mark_language_not_available,
    sample_language_texts,
    summarise_detected_languages,
    validate_text_sidecar_conventions,
)
from Preprocessing.subreddit_selection.outputs import (
    check_output_collisions,
    thread_size_quality_notes,
    write_run_config,
    write_selection_notes,
    write_selection_outputs,
)
from Preprocessing.subreddit_selection.scoring import (
    apply_final_selection,
    language_candidate_pool,
    prepare_candidate_frame,
    score_candidates,
)


logger = logging.getLogger(__name__)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and rank candidate subreddits from existing 06_0 output tables."
    )
    parser.add_argument("--tables-dir", required=True, help="Directory containing 06_0 CSV table outputs.")
    parser.add_argument("--out-root", required=True, help="Output root for 07_0 selection artefacts.")
    parser.add_argument("--year", type=int, default=2021, help="Year used for optional text sidecar discovery.")
    parser.add_argument("--min-monthly-active-users", type=int, default=100)
    parser.add_argument("--nsfw-submission-share-threshold", type=float, default=0.30)
    parser.add_argument("--candidate-score-threshold", type=float, default=0.75)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--language-mode", choices=("none", "sample-text"), default="none")
    parser.add_argument("--in-root", default=None, help="Cleaned/pseudonymised Parquet root; required for sample-text.")
    parser.add_argument("--text-sample-size-per-subreddit", type=int, default=200)
    parser.add_argument("--language-candidate-pool-size", type=int, default=250)
    parser.add_argument("--min-english-share", type=float, default=0.80)
    parser.add_argument("--strict-min-english-share", type=float, default=0.90)
    parser.add_argument("--random-seed", type=int, default=2021)
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting deterministic outputs.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _resolve_path(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def _apply_language_mode(candidates, args: argparse.Namespace):
    language_detector_name: Optional[str] = None
    if args.language_mode == "none":
        logger.info("Language scanning disabled; writing not_run language audit columns.")
        return initialize_language_columns(candidates, status="not_run"), language_detector_name

    if not args.in_root:
        raise ValueError("--in-root is required when --language-mode sample-text is used.")

    in_root = _resolve_path(args.in_root)
    assert in_root is not None
    logger.info("Validating text sidecar conventions under %s", in_root)
    validate_text_sidecar_conventions(in_root, args.year)
    pool = language_candidate_pool(candidates, args.language_candidate_pool_size)
    logger.info("Language candidate pool contains %s subreddits.", len(pool))

    detector = find_language_detector()
    if detector is None:
        logger.warning(
            "No supported local language detector is installed. Install langid, lingua, or langdetect "
            "to enable automated language scanning."
        )
        return mark_language_not_available(candidates, pool), language_detector_name

    language_detector_name = detector.name
    logger.info("Sampling text for language detection with detector=%s", detector.name)
    samples = sample_language_texts(
        in_root=in_root,
        year=args.year,
        pool=pool,
        sample_size_per_subreddit=args.text_sample_size_per_subreddit,
        random_seed=args.random_seed,
    )
    logger.info("Sampled %s text rows for language detection.", len(samples))
    summary = summarise_detected_languages(
        samples,
        pool=pool,
        detector=detector,
        min_english_share=args.min_english_share,
        strict_min_english_share=args.strict_min_english_share,
    )
    return apply_language_summary(candidates, summary), language_detector_name


def run(args: argparse.Namespace) -> int:
    tables_dir = _resolve_path(args.tables_dir)
    out_root = _resolve_path(args.out_root)
    assert tables_dir is not None
    assert out_root is not None

    check_output_collisions(out_root, overwrite=bool(args.overwrite))
    logger.info("Loading 06_0 selection table from %s", tables_dir)
    source = load_primary_selection_table(tables_dir)
    logger.info("Loaded %s subreddit rows.", len(source))

    candidates = prepare_candidate_frame(
        source,
        min_monthly_active_users=args.min_monthly_active_users,
        nsfw_submission_share_threshold=args.nsfw_submission_share_threshold,
    )
    candidates = score_candidates(
        candidates,
        candidate_score_threshold=args.candidate_score_threshold,
    )
    logger.info("Quantitatively eligible scored rows: %s", int(candidates["S_mean"].notna().sum()))

    candidates, language_detector_name = _apply_language_mode(candidates, args)
    candidates = apply_final_selection(
        candidates,
        top_n=args.top_n,
        language_mode=args.language_mode,
    )
    thread_notes = thread_size_quality_notes(candidates)
    for note in thread_notes:
        logger.warning(note)
    outputs = write_selection_outputs(
        out_root=out_root,
        candidates=candidates,
        top_n=args.top_n,
        language_mode=args.language_mode,
        overwrite=bool(args.overwrite),
    )
    write_run_config(
        out_root=out_root,
        args_mapping=vars(args),
        resolved_paths={
            "tables_dir": str(tables_dir),
            "out_root": str(out_root),
            "in_root": str(_resolve_path(args.in_root)) if args.in_root else "",
        },
        language_detector=language_detector_name,
    )
    write_selection_notes(
        out_root=out_root,
        language_mode=args.language_mode,
        language_detector=language_detector_name,
        min_monthly_active_users=args.min_monthly_active_users,
        nsfw_submission_share_threshold=args.nsfw_submission_share_threshold,
        min_english_share=args.min_english_share,
        strict_min_english_share=args.strict_min_english_share,
        thread_size_notes=thread_notes,
    )
    for name, path in outputs.items():
        logger.info("Wrote %s: %s", name, path)
    logger.info("Selection complete: %s final rows selected.", int(candidates["include_in_main_modelling"].sum()))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    try:
        return run(args)
    except Exception as exc:
        logger.error("%s", exc)
        logger.debug("Full traceback", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
