#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Cara Lynch
# See the LICENSE file for details.
"""
Preprocessing step 7.2: filter monthly Parquets to selected subreddits.

This derived-data step reads a reviewed selected-subreddit list from the 07_0 /
07_1 selection workflow and writes monthly comments/submissions Parquet files
containing only rows from those subreddits. It also keeps backwards-compatible
`candidate_flag` and `top_n` modes for older 06_0 subset-summary outputs.

Expected input layout
---------------------
<in-root>/comments/comments_core_YYYY-MM.parquet
<in-root>/submissions/submissions_core_YYYY-MM.parquet

Subset source
-------------
For the current reviewed workflow, use:

--selection-mode explicit_csv
--explicit-subreddit-csv <path-to-reviewed-selected_subreddits_final.csv>

The required --subset-summary-csv argument is retained for provenance and for
legacy candidate_flag/top_n modes. In explicit_csv mode it can point at the same
reviewed selected-subreddit manifest.

Output layout
-------------
<out-root>/comments/comments_core_YYYY-MM.parquet
<out-root>/submissions/submissions_core_YYYY-MM.parquet
<out-root>/audit/selected_subreddits.csv
<out-root>/audit/selected_subreddit_filter_manifest.json

Required CLI arguments
----------------------
--in-root
    Root containing cleaned monthly comments/submissions Parquet files.
--subset-summary-csv
    Path to 06_0's tables/subreddit_subset_selection_summary.csv. This is
    still required in explicit_csv mode for provenance in the manifest.
--out-root
    Root where filtered monthly Parquet files and audit files will be written.

Important optional CLI arguments
--------------------------------
--year
    Year to discover and filter. Defaults to 2021.
--selection-mode {candidate_flag,top_n,explicit_csv}
    Selection rule. Defaults to candidate_flag.
--top-n N
    Required only with --selection-mode top_n.
--explicit-subreddit-csv PATH
    Required only with --selection-mode explicit_csv. The CSV must contain a
    subreddit column.
--subreddit-col
    Subreddit column in the cleaned monthly Parquet files. Defaults to
    subreddit.
--overwrite
    Required to replace existing outputs.
--smoke --smoke-max-rows N
    Limit rows read per file for plumbing checks only.
--duckdb-memory-limit, --duckdb-threads, --duckdb-temp-dir
    DuckDB runtime settings; each defaults from the matching DUCKDB_* env var
    where available.
--log-level
    Logging verbosity. Defaults to INFO.

Selection modes
---------------
candidate_flag
    Select all rows in the subset summary where candidate_flag is true.
top_n
    Select the top N rows where is_eligible_for_subset_selection is true and
    candidate_score is non-null, ordered by candidate_score descending,
    n_total_activity descending when available, then subreddit ascending.
explicit_csv
    Select the distinct subreddit names listed in --explicit-subreddit-csv.

Examples
--------
Reviewed final manifest from the 07_0 / 07_1 workflow.

python Preprocessing/07_2_make_selected_subreddit_parquets.py \
  --in-root /path/to/parquet_pseudo \
  --subset-summary-csv /path/to/selected_subreddits_final.csv \
  --out-root /path/to/selected_subreddit_parquets \
  --year 2021 \
  --selection-mode explicit_csv \
  --explicit-subreddit-csv /path/to/selected_subreddits_final.csv

Legacy 06_0 candidate_flag selection.

python Preprocessing/07_2_make_selected_subreddit_parquets.py \
  --in-root /path/to/parquet_clean \
  --subset-summary-csv /path/to/descriptive_plots/tables/subreddit_subset_selection_summary.csv \
  --out-root /path/to/selected_subreddit_parquets \
  --year 2021

Top 20 eligible candidates by candidate_score.

python Preprocessing/07_2_make_selected_subreddit_parquets.py \
  --in-root /path/to/parquet_clean \
  --subset-summary-csv /path/to/descriptive_plots/tables/subreddit_subset_selection_summary.csv \
  --out-root /path/to/selected_subreddit_parquets_top20 \
  --year 2021 \
  --selection-mode top_n \
  --top-n 20

Explicit hand-picked subreddit list.

python Preprocessing/07_2_make_selected_subreddit_parquets.py \
  --in-root /path/to/parquet_clean \
  --subset-summary-csv /path/to/selected_subreddits.csv \
  --out-root /path/to/selected_subreddit_parquets_explicit \
  --year 2021 \
  --selection-mode explicit_csv \
  --explicit-subreddit-csv /path/to/selected_subreddits.csv

Notes
-----
- Monthly Parquet files are filtered and written with DuckDB COPY using ZSTD
  compression. Full Parquet files are not loaded into pandas.
- Smoke mode limits rows read per input file and is for plumbing checks only;
  smoke outputs and row counts are approximate and not suitable for reporting.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import duckdb

from Preprocessing.descriptive_plots.config import COMMENTS_KIND, SUBMISSIONS_KIND, DuckDBConfig, Inventory
from Preprocessing.descriptive_plots.derived_tables import build_single_file_relation, discover_core_files, require_complete_inventory
from Preprocessing.descriptive_plots.duckdb_helpers import configure_duckdb
from Preprocessing.descriptive_plots.output_io import ensure_dir, quote_sql_string, safe_json_dump, utc_now_iso


logger = logging.getLogger(__name__)


KIND_TO_DIR = {
    COMMENTS_KIND: "comments",
    SUBMISSIONS_KIND: "submissions",
}


@dataclass(frozen=True)
class FilterConfig:
    """Resolved runtime configuration."""

    in_root: str
    subset_summary_csv: str
    out_root: str
    year: str
    selection_mode: str
    top_n: Optional[int]
    explicit_subreddit_csv: Optional[str]
    subreddit_col: str
    overwrite: bool
    smoke: bool
    smoke_max_rows: int


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


def quote_identifier(name: str) -> str:
    """Return a double-quoted SQL identifier."""
    return '"' + name.replace('"', '""') + '"'


def read_csv_expr(path: str | Path) -> str:
    """Return a DuckDB CSV scan expression using all-varchar inference."""
    return f"read_csv_auto({quote_sql_string(str(path))}, all_varchar=true)"


def detect_csv_columns(con: duckdb.DuckDBPyConnection, path: str | Path) -> list[str]:
    """Return column names from a CSV file via DuckDB."""
    try:
        rows = con.execute(f"DESCRIBE SELECT * FROM {read_csv_expr(path)} LIMIT 0").fetchall()
    except duckdb.Error as exc:
        raise RuntimeError(f"Could not read CSV schema for {path}: {exc}") from exc
    return [str(row[0]) for row in rows]


def detect_parquet_columns(con: duckdb.DuckDBPyConnection, path: str | Path) -> list[str]:
    """Return column names from one Parquet file via DuckDB."""
    try:
        rows = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({quote_sql_string(str(path))}) LIMIT 0"
        ).fetchall()
    except duckdb.Error as exc:
        raise RuntimeError(f"Could not read Parquet schema for {path}: {exc}") from exc
    return [str(row[0]) for row in rows]


def require_columns(source_label: str, columns: Sequence[str], required: Sequence[str]) -> None:
    """Raise a clear error if required columns are absent."""
    missing = [col for col in required if col not in set(columns)]
    if missing:
        raise ValueError(
            f"{source_label} is missing required column(s): {', '.join(missing)}. "
            f"Available columns: {', '.join(columns)}"
        )


def optional_number_expr(columns: Sequence[str], column: str) -> str:
    """Return a robust numeric expression for an optional all-varchar CSV column."""
    if column not in set(columns):
        return "CAST(NULL AS DOUBLE)"
    qcol = quote_identifier(column)
    return f"TRY_CAST(NULLIF(trim(CAST({qcol} AS VARCHAR)), '') AS DOUBLE)"


def bool_expr(column: str) -> str:
    """Return a robust boolean expression for an all-varchar CSV column."""
    qcol = quote_identifier(column)
    return f"COALESCE(TRY_CAST({qcol} AS BOOLEAN), FALSE)"


def subreddit_expr(column: str = "subreddit") -> str:
    """Return a normalised subreddit-name expression."""
    qcol = quote_identifier(column)
    return f"NULLIF(trim(CAST({qcol} AS VARCHAR)), '')"


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    p = argparse.ArgumentParser(
        description="Filter monthly cleaned Parquet files to a selected subreddit subset using DuckDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in-root", required=True, help="Input root containing cleaned monthly Parquet outputs.")
    p.add_argument(
        "--subset-summary-csv",
        required=True,
        help="Path to 06_0 tables/subreddit_subset_selection_summary.csv.",
    )
    p.add_argument("--out-root", required=True, help="Output root for filtered monthly Parquet dataset.")
    p.add_argument("--year", default="2021", help="Year to process (YYYY).")
    p.add_argument(
        "--selection-mode",
        default="candidate_flag",
        choices=["candidate_flag", "top_n", "explicit_csv"],
        help="How to choose selected subreddits.",
    )
    p.add_argument("--top-n", type=int, default=None, help="Number of eligible subreddits to select in top_n mode.")
    p.add_argument(
        "--explicit-subreddit-csv",
        default=None,
        help="CSV with a subreddit column to use in explicit_csv mode.",
    )
    p.add_argument("--subreddit-col", default="subreddit", help="Subreddit column in monthly cleaned Parquet files.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing derived outputs.")
    p.add_argument("--smoke", action="store_true", help="Enable smoke mode for quick plumbing checks.")
    p.add_argument(
        "--smoke-max-rows",
        type=int,
        default=10000,
        help="Maximum rows read per input file in smoke mode.",
    )
    p.add_argument(
        "--duckdb-memory-limit",
        default=os.environ.get("DUCKDB_MEMORY_LIMIT", "16GB"),
        help="DuckDB memory limit.",
    )
    p.add_argument(
        "--duckdb-threads",
        type=int,
        default=int(os.environ.get("DUCKDB_THREADS", "4")),
        help="DuckDB threads.",
    )
    p.add_argument(
        "--duckdb-temp-dir",
        default=os.environ.get("DUCKDB_TEMP_DIR", "duckdb_tmp"),
        help="DuckDB temp_directory. Prefer node-local scratch on HPC.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return p


def validate_cli(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate cross-argument requirements."""
    if args.selection_mode == "top_n":
        if args.top_n is None:
            parser.error("--top-n is required when --selection-mode=top_n.")
        if int(args.top_n) <= 0:
            parser.error("--top-n must be a positive integer.")
    if args.selection_mode == "explicit_csv" and not args.explicit_subreddit_csv:
        parser.error("--explicit-subreddit-csv is required when --selection-mode=explicit_csv.")
    if args.smoke and int(args.smoke_max_rows) <= 0:
        parser.error("--smoke-max-rows must be positive when --smoke is set.")


def expected_year_months(year: str) -> list[str]:
    """Return YYYY-MM month labels for a complete calendar year."""
    return [f"{year}-{month:02d}" for month in range(1, 13)]


def log_inventory(inv: Inventory, year: str) -> None:
    """Log discovered and skipped monthly inputs."""
    discovered = sorted(inv.keys())
    logger.info("Discovered months with at least one core file: %s", ", ".join(discovered) or "(none)")
    for month in expected_year_months(year):
        if month not in inv:
            logger.warning("No input files discovered for %s; skipping this month.", month)
    for month, files in sorted(inv.items()):
        if files.comments_core:
            logger.info("Input %s comments: %s", month, files.comments_core)
        else:
            logger.error("Input %s comments: missing", month)
        if files.submissions_core:
            logger.info("Input %s submissions: %s", month, files.submissions_core)
        else:
            logger.error("Input %s submissions: missing", month)


def output_path_for(out_root: Path, kind: str, input_path: str) -> Path:
    """Return the output path preserving subdirectory and original filename."""
    return out_root / KIND_TO_DIR[kind] / Path(input_path).name


def iter_inventory_files(inv: Inventory, out_root: Path):
    """Yield month/kind input/output file descriptors."""
    for month, files in sorted(inv.items()):
        if files.comments_core:
            yield {
                "month": month,
                "kind": "comments",
                "core_kind": COMMENTS_KIND,
                "input_path": files.comments_core,
                "output_path": output_path_for(out_root, COMMENTS_KIND, files.comments_core),
            }
        if files.submissions_core:
            yield {
                "month": month,
                "kind": "submissions",
                "core_kind": SUBMISSIONS_KIND,
                "input_path": files.submissions_core,
                "output_path": output_path_for(out_root, SUBMISSIONS_KIND, files.submissions_core),
            }


def year_output_globs(out_root: Path, year: str) -> list[Path]:
    """Return existing year outputs that would make a rerun ambiguous."""
    paths: list[Path] = []
    for dirname, prefix in (("comments", "comments_core"), ("submissions", "submissions_core")):
        base = out_root / dirname
        if base.exists():
            paths.extend(sorted(base.glob(f"{prefix}_{year}-*.parquet")))
    return paths


def check_output_collisions(paths: Sequence[Path], overwrite: bool) -> None:
    """Fail fast if deterministic outputs already exist and overwrite is disabled."""
    existing = sorted({path for path in paths if path.exists()})
    if existing and not overwrite:
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


def create_selected_subreddits_table(con: duckdb.DuckDBPyConnection, cfg: FilterConfig) -> list[str]:
    """Create temp selected_subreddits table and return selected names in order."""
    if cfg.selection_mode in {"candidate_flag", "top_n"}:
        source = Path(cfg.subset_summary_csv)
        if not source.exists():
            raise FileNotFoundError(f"Subset summary CSV not found: {source}")
        columns = detect_csv_columns(con, source)
        logger.info("Subset summary columns: %s", ", ".join(columns))
        require_columns("subset summary CSV", columns, ["subreddit"])

        score_expr = optional_number_expr(columns, "candidate_score")
        activity_expr = optional_number_expr(columns, "n_total_activity")
        rank_expr = optional_number_expr(columns, "candidate_rank")

        if cfg.selection_mode == "candidate_flag":
            require_columns("subset summary CSV", columns, ["candidate_flag"])
            con.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE selected_subreddits AS
                WITH source AS (
                    SELECT
                        {subreddit_expr("subreddit")} AS subreddit,
                        {bool_expr("candidate_flag")} AS candidate_flag,
                        {rank_expr} AS candidate_rank,
                        {score_expr} AS candidate_score,
                        {activity_expr} AS n_total_activity
                    FROM {read_csv_expr(source)}
                ),
                dedup AS (
                    SELECT
                        subreddit,
                        MIN(candidate_rank) AS candidate_rank,
                        MAX(candidate_score) AS candidate_score,
                        MAX(n_total_activity) AS n_total_activity
                    FROM source
                    WHERE subreddit IS NOT NULL
                      AND candidate_flag
                    GROUP BY subreddit
                ),
                ranked AS (
                    SELECT
                        ROW_NUMBER() OVER (
                            ORDER BY
                                candidate_rank ASC NULLS LAST,
                                candidate_score DESC NULLS LAST,
                                n_total_activity DESC NULLS LAST,
                                subreddit ASC
                        ) AS selection_order,
                        subreddit
                    FROM dedup
                )
                SELECT selection_order, subreddit
                FROM ranked
                ORDER BY selection_order
                """
            )
        else:
            require_columns(
                "subset summary CSV",
                columns,
                ["is_eligible_for_subset_selection", "candidate_score"],
            )
            top_n = int(cfg.top_n or 0)
            con.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE selected_subreddits AS
                WITH source AS (
                    SELECT
                        {subreddit_expr("subreddit")} AS subreddit,
                        {bool_expr("is_eligible_for_subset_selection")} AS is_eligible,
                        {score_expr} AS candidate_score,
                        {activity_expr} AS n_total_activity
                    FROM {read_csv_expr(source)}
                ),
                dedup AS (
                    SELECT
                        subreddit,
                        MAX(candidate_score) AS candidate_score,
                        MAX(n_total_activity) AS n_total_activity
                    FROM source
                    WHERE subreddit IS NOT NULL
                      AND is_eligible
                      AND candidate_score IS NOT NULL
                    GROUP BY subreddit
                ),
                ranked AS (
                    SELECT
                        ROW_NUMBER() OVER (
                            ORDER BY
                                candidate_score DESC NULLS LAST,
                                n_total_activity DESC NULLS LAST,
                                subreddit ASC
                        ) AS selection_order,
                        subreddit
                    FROM dedup
                )
                SELECT selection_order, subreddit
                FROM ranked
                WHERE selection_order <= {top_n}
                ORDER BY selection_order
                """
            )
    else:
        source = Path(str(cfg.explicit_subreddit_csv))
        if not source.exists():
            raise FileNotFoundError(f"Explicit subreddit CSV not found: {source}")
        columns = detect_csv_columns(con, source)
        logger.info("Explicit subreddit CSV columns: %s", ", ".join(columns))
        require_columns("explicit subreddit CSV", columns, ["subreddit"])
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE selected_subreddits AS
            WITH source AS (
                SELECT {subreddit_expr("subreddit")} AS subreddit
                FROM {read_csv_expr(source)}
            ),
            dedup AS (
                SELECT DISTINCT subreddit
                FROM source
                WHERE subreddit IS NOT NULL
            )
            SELECT
                ROW_NUMBER() OVER (ORDER BY subreddit ASC) AS selection_order,
                subreddit
            FROM dedup
            ORDER BY selection_order
            """
        )

    selected = [
        str(row[0])
        for row in con.execute(
            "SELECT subreddit FROM selected_subreddits ORDER BY selection_order"
        ).fetchall()
    ]
    if not selected:
        raise RuntimeError(
            "Selected subreddit list is empty. Check selection mode and source CSV filters."
        )
    logger.info("Selected subreddit count: %s", f"{len(selected):,}")
    return selected


def write_selected_subreddits(con: duckdb.DuckDBPyConnection, path: Path) -> None:
    """Write selected subreddit audit CSV."""
    ensure_dir(str(path.parent))
    con.execute(
        f"""
        COPY (
            SELECT selection_order, subreddit
            FROM selected_subreddits
            ORDER BY selection_order
        )
        TO {quote_sql_string(str(path))}
        (FORMAT CSV, HEADER, DELIMITER ',')
        """
    )
    logger.info("Wrote selected subreddit list: %s", path)


def count_parquet_rows(con: duckdb.DuckDBPyConnection, path: Path) -> int:
    """Count rows in a Parquet file."""
    return int(
        con.execute(
            f"SELECT COUNT(*) FROM read_parquet({quote_sql_string(str(path))})"
        ).fetchone()[0]
    )


def validate_output_rows_selected(
    con: duckdb.DuckDBPyConnection,
    output_path: Path,
    subreddit_col: str,
) -> None:
    """Ensure every written row belongs to the selected subreddit list."""
    qcol = quote_identifier(subreddit_col)
    invalid_rows = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet({quote_sql_string(str(output_path))}) AS out_rows
            WHERE NOT EXISTS (
                SELECT 1
                FROM selected_subreddits selected
                WHERE selected.subreddit = CAST(out_rows.{qcol} AS VARCHAR)
            )
            """
        ).fetchone()[0]
    )
    if invalid_rows:
        raise RuntimeError(
            f"Validation failed for {output_path}: {invalid_rows:,} output row(s) "
            "have a subreddit outside the selected list."
        )


def filter_one_file(
    con: duckdb.DuckDBPyConnection,
    cfg: FilterConfig,
    item: dict,
) -> dict:
    """Filter one monthly input file and return row-count metadata."""
    input_path = Path(str(item["input_path"]))
    output_path = Path(str(item["output_path"]))
    month = str(item["month"])
    kind = str(item["kind"])

    columns = detect_parquet_columns(con, input_path)
    if cfg.subreddit_col not in columns:
        raise ValueError(
            f"Input file {input_path} is missing requested --subreddit-col "
            f"{cfg.subreddit_col!r}. Available columns: {', '.join(columns)}"
        )

    ensure_dir(str(output_path.parent))
    source_relation = build_single_file_relation(
        str(input_path),
        cfg.smoke_max_rows if cfg.smoke else 0,
    )
    qcol = quote_identifier(cfg.subreddit_col)

    logger.info("Filtering %s %s input: %s", month, kind, input_path)
    input_rows = int(
        con.execute(f"SELECT COUNT(*) FROM {source_relation}").fetchone()[0]
    )
    con.execute(
        f"""
        COPY (
            SELECT src.*
            FROM {source_relation} AS src
            WHERE EXISTS (
                SELECT 1
                FROM selected_subreddits selected
                WHERE selected.subreddit = CAST(src.{qcol} AS VARCHAR)
            )
        )
        TO {quote_sql_string(str(output_path))}
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    output_rows = count_parquet_rows(con, output_path)
    validate_output_rows_selected(con, output_path, cfg.subreddit_col)
    if output_rows > input_rows:
        raise RuntimeError(
            f"Validation failed for {output_path}: output rows ({output_rows:,}) "
            f"exceed input rows ({input_rows:,})."
        )

    logger.info(
        "Wrote %s %s output: %s rows (%s input rows) -> %s",
        month,
        kind,
        f"{output_rows:,}",
        f"{input_rows:,}",
        output_path,
    )
    return {
        "month": month,
        "kind": kind,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_rows": input_rows,
        "output_rows": output_rows,
        "smoke_limited": bool(cfg.smoke),
    }


def nested_row_counts(row_counts: Sequence[dict]) -> dict:
    """Return row-count records keyed by month and kind."""
    out: dict[str, dict[str, dict]] = {}
    for record in row_counts:
        month = str(record["month"])
        kind = str(record["kind"])
        out.setdefault(month, {})[kind] = {
            "input_path": record["input_path"],
            "output_path": record["output_path"],
            "input_rows": int(record["input_rows"]),
            "output_rows": int(record["output_rows"]),
            "smoke_limited": bool(record["smoke_limited"]),
        }
    return out


def validate_manifest_counts(
    con: duckdb.DuckDBPyConnection,
    manifest_path: Path,
) -> None:
    """Ensure manifest output row counts match the files currently on disk."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    mismatches: list[str] = []
    for month, kinds in manifest["row_counts_by_month_kind"].items():
        for kind, info in kinds.items():
            output_path = Path(info["output_path"])
            manifest_rows = int(info["output_rows"])
            actual_rows = count_parquet_rows(con, output_path)
            if actual_rows != manifest_rows:
                mismatches.append(
                    f"{month} {kind}: manifest={manifest_rows:,}, actual={actual_rows:,}"
                )
    if mismatches:
        raise RuntimeError(
            "Manifest row-count validation failed:\n  - " + "\n  - ".join(mismatches)
        )
    logger.info("Manifest row counts match written Parquet files.")


def manifest_payload(
    cfg: FilterConfig,
    duck: DuckDBConfig,
    selected_subreddits: Sequence[str],
    row_counts: Sequence[dict],
) -> dict:
    """Build reproducibility manifest."""
    total_input_rows = sum(int(record["input_rows"]) for record in row_counts)
    total_output_rows = sum(int(record["output_rows"]) for record in row_counts)
    return {
        "timestamp_utc": utc_now_iso(),
        "script_name": Path(__file__).name,
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "duckdb": {"version": duckdb.__version__, **asdict(duck)},
        "input_roots": {
            "in_root": cfg.in_root,
            "out_root": cfg.out_root,
        },
        "subset_source": {
            "subset_summary_csv": cfg.subset_summary_csv,
            "explicit_subreddit_csv": cfg.explicit_subreddit_csv,
        },
        "selection_mode": cfg.selection_mode,
        "selection_parameters": {
            "top_n": cfg.top_n,
            "subreddit_col": cfg.subreddit_col,
        },
        "year": cfg.year,
        "smoke": cfg.smoke,
        "smoke_max_rows": cfg.smoke_max_rows if cfg.smoke else 0,
        "smoke_note": (
            "Smoke outputs are approximate plumbing checks and are not suitable for reporting."
            if cfg.smoke
            else ""
        ),
        "selected_subreddits_count": len(selected_subreddits),
        "selected_subreddit_names": list(selected_subreddits),
        "row_counts_by_month_kind": nested_row_counts(row_counts),
        "row_count_totals": {
            "input_rows": total_input_rows,
            "output_rows": total_output_rows,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    validate_cli(args, parser)
    setup_logging(args.log_level)

    cfg = FilterConfig(
        in_root=str(args.in_root),
        subset_summary_csv=str(args.subset_summary_csv),
        out_root=str(args.out_root),
        year=str(args.year),
        selection_mode=str(args.selection_mode),
        top_n=int(args.top_n) if args.top_n is not None else None,
        explicit_subreddit_csv=(
            str(args.explicit_subreddit_csv) if args.explicit_subreddit_csv else None
        ),
        subreddit_col=str(args.subreddit_col),
        overwrite=bool(args.overwrite),
        smoke=bool(args.smoke),
        smoke_max_rows=int(args.smoke_max_rows),
    )
    duck = DuckDBConfig(
        memory_limit=str(args.duckdb_memory_limit),
        threads=int(args.duckdb_threads),
        temp_dir=str(args.duckdb_temp_dir),
    )

    if cfg.smoke:
        logger.warning(
            "Smoke mode enabled: limiting each input file to %s rows. "
            "Outputs are approximate and not for reporting.",
            f"{cfg.smoke_max_rows:,}",
        )

    out_root = Path(cfg.out_root)
    audit_dir = out_root / "audit"
    selected_csv = audit_dir / "selected_subreddits.csv"
    manifest_json = audit_dir / "selected_subreddit_filter_manifest.json"

    inv = discover_core_files(cfg.in_root, cfg.year)
    if not inv:
        raise FileNotFoundError(
            "No cleaned monthly Parquet files found under "
            f"{cfg.in_root!r} for year {cfg.year}."
        )
    log_inventory(inv, cfg.year)
    require_complete_inventory(inv)

    inventory_items = list(iter_inventory_files(inv, out_root))
    desired_outputs = [Path(str(item["output_path"])) for item in inventory_items]
    desired_outputs.extend([selected_csv, manifest_json])
    collision_paths = desired_outputs + year_output_globs(out_root, cfg.year)
    check_output_collisions(collision_paths, cfg.overwrite)
    remove_outputs(collision_paths, cfg.overwrite)

    ensure_dir(str(out_root))
    ensure_dir(str(out_root / "comments"))
    ensure_dir(str(out_root / "submissions"))
    ensure_dir(str(audit_dir))

    logger.info("Input root: %s", cfg.in_root)
    logger.info("Output root: %s", cfg.out_root)
    logger.info("Subset summary CSV: %s", cfg.subset_summary_csv)
    logger.info("Selection mode: %s", cfg.selection_mode)
    logger.info("DuckDB memory limit: %s", duck.memory_limit)
    logger.info("DuckDB threads: %s", duck.threads)
    logger.info("DuckDB temp dir: %s", duck.temp_dir)

    con = duckdb.connect()
    try:
        configure_duckdb(con, duck)
        selected_subreddits = create_selected_subreddits_table(con, cfg)
        write_selected_subreddits(con, selected_csv)

        row_counts: list[dict] = []
        for item in inventory_items:
            row_counts.append(filter_one_file(con, cfg, item))

        payload = manifest_payload(cfg, duck, selected_subreddits, row_counts)
        safe_json_dump(payload, str(manifest_json))
        validate_manifest_counts(con, manifest_json)
    finally:
        con.close()

    logger.info("Wrote manifest: %s", manifest_json)
    logger.info("Done. Total output rows: %s", f"{payload['row_count_totals']['output_rows']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
