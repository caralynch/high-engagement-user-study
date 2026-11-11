#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Cara Lynch
# See the LICENSE file for details.
"""
Preprocessing step 3.1: validate cleaned monthly Parquet outputs.

This script validates Parquet integrity for monthly cleaned datasets produced by
`03_0_cleaning.py`, and can optionally compare cleaned outputs against the
corresponding raw (pre-clean) Parquet outputs.

Fast integrity checks (default)
-------------------------------
- Parquet magic bytes: header/footer signature is "PAR1" (detects truncation/corruption).
- PyArrow can open the file (footer/metadata readable).
- DuckDB "touch" (LIMIT 0) to catch subtle read issues.

Optional checks (tunable cost)
------------------------------
- Row counts (DuckDB COUNT(*)).
- Prefix checks on core IDs (comments_core):
  - id starts with t1_
  - link_id starts with t3_
  - parent_id starts with t1_ or t3_
- Flag checks:
  - is_deleted_author exists in *_core files
  - is_deleted_author does NOT exist in sidecar text files
- Subset checks comparing cleaned vs raw for the same month and kind:
  - Sampled subset: a sample of cleaned IDs exist in raw (cheap).
  - Full subset: verify that every cleaned ID occurs in the raw IDs (heavy).

Command-line interface
----------------------
--root                    Cleaned parquet root (e.g., parquet_clean).
--kinds                   One or more dataset kinds to validate (default: all).
--year                    Optional year filter (YYYY). If provided, only months from that year are used.
--only-months             Optional: comma-separated list of YYYY-MM months to keep (e.g. "2021-06,2021-07").
--limit-months            Optional: validate only first N discovered months.
--smoke                   Quick-run mode: limits months and disables heavy checks unless explicitly requested.

Checks
------
--counts                  Compute row counts (DuckDB COUNT(*); IO heavy).
--prefix-checks           Run ID prefix sanity checks on comments_core.
--flag-checks             Verify is_deleted_author presence/absence (core vs sidecars).

Raw comparisons
---------------
--raw-root                Raw parquet root for comparisons (e.g., parquet).
--sample                  Sample size for subset check (0 = skip).
--full-subset             Check every cleaned ID against the raw IDs (heavy).

DuckDB tuning (optional)
------------------------
--duckdb-memory-limit     DuckDB memory limit (e.g. "4GB").
--duckdb-threads          DuckDB threads (int).
--duckdb-temp-dir         DuckDB temp_directory (recommended on HPC scratch).

Logging
-------
--log-level               DEBUG/INFO/WARNING/ERROR/CRITICAL.

Environment variables (optional)
--------------------------------
DUCKDB_MEMORY_LIMIT="4GB"     Alternative default for --duckdb-memory-limit.
DUCKDB_THREADS=1              Alternative default for --duckdb-threads.
DUCKDB_TEMP_DIR="/tmp"        Alternative default for --duckdb-temp-dir.

Inputs
------
Cleaned outputs (expected layout):
  <root>/comments/comments_core_YYYY-MM.parquet
  <root>/comments/comments_body_YYYY-MM.parquet
  <root>/submissions/submissions_core_YYYY-MM.parquet
  <root>/submissions/submissions_selftext_YYYY-MM.parquet

Raw outputs for comparison (optional):
  <raw-root>/comments/comments_core_YYYY-MM.parquet
  ...

Outputs
-------
This script does not modify data. It writes validation results to stdout/stderr
(via logging). Use shell redirection if desired:
  python 03_1_check_cleaned_files.py ... > validate.log 2>&1

Examples
--------
Integrity-only over cleaned (all kinds):
  python 03_1_check_cleaned_files.py --root parquet_clean --log-level INFO

Integrity + counts for comments_core:
  python 03_1_check_cleaned_files.py --root parquet_clean --kinds comments_core --counts

Integrate the useful manual checks from the DuckDB snippet (prefix + flag checks):
  python 03_1_check_cleaned_files.py --root parquet_clean --kinds comments_core submissions_core --prefix-checks --flag-checks

Compare cleaned vs raw with a sample subset check:
  python 03_1_check_cleaned_files.py --root parquet_clean --raw-root parquet --kinds comments_core --sample 200000

Full subset check (heavy!):
  python 03_1_check_cleaned_files.py --root parquet_clean --raw-root parquet --kinds comments_core --full-subset

Smoke test (fast):
  python 03_1_check_cleaned_files.py --root parquet_clean --smoke
"""

from __future__ import annotations

import argparse
import glob
import logging
import mmap
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pyarrow.parquet as pq

try:
    import duckdb  # optional but strongly recommended
except Exception:
    duckdb = None


# ----------------------------- Constants -------------------------------------

KINDS: Tuple[str, ...] = (
    "comments_core",
    "comments_body",
    "submissions_core",
    "submissions_selftext",
)

MONTH_RE = re.compile(r"(?<!\d)(20\d{2})[-_](0[1-9]|1[0-2])(?!\d)")

# Prefix patterns
RE_T1 = r"^t1_"
RE_T3 = r"^t3_"
RE_T1_OR_T3 = r"^(t1_|t3_)"

# Flag column
DELETED_AUTHOR_COL = "is_deleted_author"


# ----------------------------- Data structures --------------------------------


@dataclass(frozen=True)
class MonthFile:
    """One discovered parquet file for a given (kind, month)."""

    kind: str
    month: str  # YYYY-MM
    path: str


# ----------------------------- Utility helpers --------------------------------


def month_of(path: str) -> str:
    """Extract YYYY-MM token from a file path, tolerant of '-' or '_' separators."""
    m = MONTH_RE.search(path)
    return f"{m.group(1)}-{m.group(2)}" if m else "UNKNOWN"


def parse_only_months(s: str) -> Optional[set]:
    """Parse comma-separated YYYY-MM months; returns None if empty."""
    if not s:
        return None
    keep = {m.strip() for m in s.split(",") if m.strip()}
    return keep or None


def ensure_duckdb_available() -> None:
    """Exit with a clear error if DuckDB is required but not installed."""
    if duckdb is None:
        raise RuntimeError(
            "duckdb Python package is not installed. "
            "Install it (e.g. pip install duckdb) or disable DuckDB-dependent checks."
        )


def configure_duckdb(
    con, memory_limit: str, threads: int, temp_dir: Optional[str]
) -> None:
    """Apply per-connection DuckDB settings (safe to call per file)."""
    if memory_limit:
        con.execute(f"SET memory_limit='{memory_limit}'")
    if threads and threads > 0:
        con.execute(f"SET threads={threads}")
    if temp_dir:
        con.execute(f"SET temp_directory='{temp_dir}'")
    # Ensure we actually touch files (avoid caching masking corruption)
    con.execute("PRAGMA disable_object_cache")


# ----------------------------- Discovery --------------------------------------


def discover_files(root: str, kinds: Sequence[str]) -> List[MonthFile]:
    """
    Discover parquet files under <root>/comments and <root>/submissions.

    Returns a list of MonthFile entries (kind, month, path).
    """
    patterns = {
        "comments_core": f"{root}/comments/comments_core_*.parquet",
        "comments_body": f"{root}/comments/comments_body_*.parquet",
        "submissions_core": f"{root}/submissions/submissions_core_*.parquet",
        "submissions_selftext": f"{root}/submissions/submissions_selftext_*.parquet",
    }

    out: List[MonthFile] = []
    for kind in kinds:
        for p in sorted(glob.glob(patterns[kind])):
            m = month_of(p)
            if m == "UNKNOWN":
                continue
            out.append(MonthFile(kind=kind, month=m, path=p))
    return out


def group_by_kind_month(files: Iterable[MonthFile]) -> Dict[str, Dict[str, str]]:
    """
    Convert list of MonthFile -> mapping: kind -> month -> path.
    """
    out: Dict[str, Dict[str, str]] = {}
    for f in files:
        out.setdefault(f.kind, {})[f.month] = f.path
    return out


# ----------------------------- Parquet checks ---------------------------------


def parquet_has_par1(path: str) -> Tuple[bool, bool, int]:
    """
    Return (has_header, has_footer, meta_len) for a Parquet file.

    - header/footer are checked for b"PAR1"
    - meta_len is parsed from the 4 bytes preceding the footer (little-endian int)
      and is best-effort (may be -1 on failure).
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size < 8:
        return (False, False, -1)

    with p.open("rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        header = mm[:4]
        footer = mm[-4:]
        meta_len = -1
        if mm.size() >= 8:
            try:
                meta_len = struct.unpack("<i", mm[-8:-4])[0]
            except Exception:
                meta_len = -1
        mm.close()

    return (header == b"PAR1", footer == b"PAR1", meta_len)


def pyarrow_open_ok(path: str) -> Tuple[bool, str]:
    """
    Check that PyArrow can open a Parquet file.

    Returns (ok, message).
    """
    try:
        pq.ParquetFile(path)
        return True, "ok"
    except Exception as e:
        return False, str(e)


def duckdb_touch_ok(path: str, con) -> Tuple[bool, str]:
    """
    DuckDB schema touch: `SELECT * FROM read_parquet(...) LIMIT 0`.

    Returns (ok, message).
    """
    try:
        con.execute(f"SELECT * FROM read_parquet('{path}') LIMIT 0;").fetchall()
        return True, "ok"
    except Exception as e:
        return False, str(e)


def duckdb_count(path: str, con) -> int:
    """DuckDB COUNT(*) for a parquet file (IO heavy)."""
    return con.execute(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]


# ----------------------------- Cleaning-specific checks -----------------------


def duckdb_prefix_checks_comments_core(path: str, con) -> Tuple[bool, str]:
    """
    Prefix sanity checks for comments_core:
      - id starts with t1_
      - link_id starts with t3_
      - parent_id starts with t1_ or t3_

    Returns (ok, message).
    """
    try:
        q = f"""
        SELECT
          count_if(NOT regexp_matches(id, '{RE_T1}')) AS bad_comment_ids,
          count_if(NOT regexp_matches(link_id, '{RE_T3}')) AS bad_link_ids,
          count_if(NOT regexp_matches(parent_id, '{RE_T1_OR_T3}')) AS bad_parent_ids
        FROM read_parquet('{path}');
        """
        bad_id, bad_link, bad_parent = con.execute(q).fetchone()
        ok = bad_id == 0 and bad_link == 0 and bad_parent == 0
        msg = f"bad_comment_ids={bad_id:,} bad_link_ids={bad_link:,} bad_parent_ids={bad_parent:,}"
        return ok, msg
    except Exception as e:
        return False, f"prefix check failed: {e}"


def duckdb_has_column(path: str, col: str, con) -> Tuple[bool, str]:
    """
    Check whether a parquet file contains a given column using DuckDB DESCRIBE.

    Returns (has_col, message).
    """
    try:
        # DESCRIBE SELECT is stable for read_parquet; column_name is the first column in output
        rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}');").fetchall()
        cols = {r[0] for r in rows}  # column_name
        has = col in cols
        return has, "present" if has else "absent"
    except Exception as e:
        return False, f"describe failed: {e}"


# ----------------------------- Subset checks (cleaned vs raw) -----------------


def sampled_subset_check(
    cleaned_path: str, raw_path: str, sample_n: int, con
) -> Tuple[bool, str]:
    """
    Check that a sample of cleaned IDs exist in raw for the same kind/month.

    Implementation notes:
    - Uses ORDER BY random() LIMIT N for portability across DuckDB versions.
    - Checks that all sampled IDs are found in raw via SEMI JOIN.

    Returns (ok, message).
    """
    if sample_n <= 0:
        return True, "skipped"

    try:
        con.execute("DROP TABLE IF EXISTS cleaned_sample;")
        con.execute(
            f"""
            CREATE TEMP TABLE cleaned_sample AS
            SELECT id
            FROM read_parquet('{cleaned_path}')
            ORDER BY random()
            LIMIT {int(sample_n)};
        """
        )

        got = con.execute("SELECT COUNT(*) FROM cleaned_sample;").fetchone()[0]
        if got == 0:
            return True, "cleaned sample had 0 rows (treated as OK)"

        matched = con.execute(
            f"""
            SELECT COUNT(*)
            FROM cleaned_sample c
            SEMI JOIN read_parquet('{raw_path}') r
            ON c.id = r.id;
        """
        ).fetchone()[0]

        ok = matched == got
        return ok, f"matched {matched}/{got} sampled IDs"
    except Exception as e:
        return False, f"sampled subset check failed: {e}"


def full_subset_check(cleaned_path: str, raw_path: str, con) -> Tuple[bool, str]:
    """
    Full check: verify that every cleaned ID occurs in the raw IDs (can be expensive).

    Returns (ok, message).
    """
    try:
        missing = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{cleaned_path}') c
            ANTI JOIN read_parquet('{raw_path}') r
            ON c.id = r.id;
        """
        ).fetchone()[0]
        ok = missing == 0
        return ok, f"cleaned IDs missing from raw: {missing}"
    except Exception as e:
        return False, f"full subset check failed: {e}"


# ----------------------------- CLI --------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI for the validator."""
    p = argparse.ArgumentParser(
        description="Validate cleaned monthly Parquet integrity and optionally compare cleaned vs raw.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--root", required=True, help="Cleaned parquet root (e.g., parquet_clean)."
    )
    p.add_argument(
        "--kinds",
        nargs="+",
        default=list(KINDS),
        choices=list(KINDS),
        help="One or more kinds to validate.",
    )
    p.add_argument("--year", default="", help="Optional year filter (YYYY).")
    p.add_argument(
        "--only-months", default="", help="Comma-separated subset like 2021-01,2021-06"
    )
    p.add_argument(
        "--limit-months",
        type=int,
        default=None,
        help="Validate only first N months per kind.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Quick-run mode (limits months, avoids heavy checks).",
    )

    p.add_argument(
        "--counts", action="store_true", help="Compute row counts (IO heavy)."
    )
    p.add_argument(
        "--prefix-checks",
        action="store_true",
        help="Run comments_core ID prefix checks.",
    )
    p.add_argument(
        "--flag-checks",
        action="store_true",
        help="Verify is_deleted_author is present in core and absent in text sidecars.",
    )

    p.add_argument(
        "--raw-root",
        default=None,
        help="Raw parquet root for comparisons (e.g., parquet).",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Sample size for checking cleaned IDs against raw IDs (0=skip).",
    )
    p.add_argument(
        "--full-subset", action="store_true", help="Check every cleaned ID against the raw IDs (heavy)."
    )

    p.add_argument(
        "--duckdb-memory-limit",
        default=os.environ.get("DUCKDB_MEMORY_LIMIT", "4GB"),
        help="DuckDB memory limit per run (e.g. '4GB').",
    )
    p.add_argument(
        "--duckdb-threads",
        type=int,
        default=int(os.environ.get("DUCKDB_THREADS", "1")),
        help="DuckDB threads setting.",
    )
    p.add_argument(
        "--duckdb-temp-dir",
        default=os.environ.get("DUCKDB_TEMP_DIR", ""),
        help="DuckDB temp_directory (blank = do not set).",
    )

    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return p


def apply_smoke_preset(args: argparse.Namespace) -> None:
    """
    Apply a conservative preset for smoke tests.

    - Limits months validated.
    - Disables full-subset unless explicitly requested.
    - Reduces sample size if enabled.
    """
    args.limit_months = args.limit_months or 1
    if args.sample and args.sample > 50_000:
        args.sample = 50_000
    # counts can stay if user asked; but smoke defaults to integrity-only
    # (we don't forcibly disable user-specified flags)


# ----------------------------- Main -------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint. Returns a process exit code."""
    args = build_argparser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    log = logging.getLogger("validate_cleaned")

    if args.smoke:
        apply_smoke_preset(args)

    keep_months = parse_only_months(args.only_months)
    year_filter = args.year.strip()

    # Discover cleaned files
    cleaned_files = discover_files(args.root, args.kinds)
    if not cleaned_files:
        log.error("No files found under %s for kinds=%s", args.root, args.kinds)
        return 2

    # Group cleaned by kind/month, and apply month filters
    cleaned_map = group_by_kind_month(cleaned_files)

    def month_allowed(m: str) -> bool:
        if year_filter and not m.startswith(year_filter + "-"):
            return False
        if keep_months is not None and m not in keep_months:
            return False
        return True

    # If doing comparisons, discover raw map once
    raw_map: Optional[Dict[str, Dict[str, str]]] = None
    if args.raw_root:
        raw_files = discover_files(args.raw_root, args.kinds)
        raw_map = group_by_kind_month(raw_files)

    # DuckDB connection (shared) if any DuckDB-dependent checks are requested
    needs_duckdb = any(
        [
            args.counts,
            args.prefix_checks,
            args.flag_checks,
            args.sample > 0,
            args.full_subset,
            True,  # we always do duckdb touch by default, mirroring your original script :contentReference[oaicite:2]{index=2}
        ]
    )

    con = None
    if needs_duckdb:
        ensure_duckdb_available()
        con = duckdb.connect()
        configure_duckdb(
            con,
            memory_limit=args.duckdb_memory_limit,
            threads=args.duckdb_threads,
            temp_dir=(args.duckdb_temp_dir or None),
        )

    failures = 0

    try:
        for kind in args.kinds:
            months = sorted(
                m for m in cleaned_map.get(kind, {}).keys() if month_allowed(m)
            )
            if args.limit_months is not None:
                months = months[: args.limit_months]

            if not months:
                log.warning("No months to validate for kind=%s after filtering.", kind)
                continue

            log.info("=== KIND: %s | months=%d ===", kind, len(months))

            for m in months:
                p = cleaned_map[kind][m]
                log.info("--- %s %s ---", kind, m)

                # 1) PAR1 signature
                h_ok, f_ok, meta_len = parquet_has_par1(p)
                if not (h_ok and f_ok):
                    log.error(
                        "PAR1 signature missing (header=%s footer=%s meta_len=%s) file=%s",
                        h_ok,
                        f_ok,
                        meta_len,
                        p,
                    )
                    failures += 1
                    continue
                log.debug("PAR1 ok meta_len=%s", meta_len)

                # 2) PyArrow open
                ok, msg = pyarrow_open_ok(p)
                if not ok:
                    log.error("pyarrow open failed: %s | file=%s", msg, p)
                    failures += 1
                    continue
                log.debug("pyarrow open ok")

                # 3) DuckDB touch
                assert con is not None
                ok, msg = duckdb_touch_ok(p, con)
                if not ok:
                    log.error("duckdb touch failed: %s | file=%s", msg, p)
                    failures += 1
                    continue
                log.debug("duckdb touch ok")

                # 4) Optional counts
                if args.counts:
                    try:
                        n = duckdb_count(p, con)
                        log.info("row_count: %s", f"{n:,}")
                    except Exception as e:
                        log.error("count failed: %s | file=%s", e, p)
                        failures += 1

                # 5) Optional prefix checks
                if args.prefix_checks and kind == "comments_core":
                    ok, msg = duckdb_prefix_checks_comments_core(p, con)
                    if ok:
                        log.info("prefix_checks: OK (%s)", msg)
                    else:
                        log.error("prefix_checks: FAIL (%s)", msg)
                        failures += 1

                # 6) Optional flag checks (core has flag; sidecars do not)
                if args.flag_checks:
                    if kind in ("comments_core", "submissions_core"):
                        has, msg = duckdb_has_column(p, DELETED_AUTHOR_COL, con)
                        if has:
                            log.info(
                                "flag_check(core): OK (%s present)", DELETED_AUTHOR_COL
                            )
                        else:
                            log.error(
                                "flag_check(core): FAIL (%s absent) | %s",
                                DELETED_AUTHOR_COL,
                                msg,
                            )
                            failures += 1
                    if kind in ("comments_body", "submissions_selftext"):
                        has, msg = duckdb_has_column(p, DELETED_AUTHOR_COL, con)
                        if has:
                            log.error(
                                "flag_check(text): FAIL (%s unexpectedly present)",
                                DELETED_AUTHOR_COL,
                            )
                            failures += 1
                        else:
                            log.info(
                                "flag_check(text): OK (%s absent)", DELETED_AUTHOR_COL
                            )

                # 7) Optional raw comparisons (sample/full subset)
                if args.raw_root and raw_map is not None:
                    raw_p = raw_map.get(kind, {}).get(m)
                    if not raw_p:
                        log.warning(
                            "no raw file for %s %s under %s", kind, m, args.raw_root
                        )
                    else:
                        if args.sample > 0:
                            ok, msg = sampled_subset_check(p, raw_p, args.sample, con)
                            if ok:
                                log.info("subset(sample=%d): OK (%s)", args.sample, msg)
                            else:
                                log.error(
                                    "subset(sample=%d): FAIL (%s)", args.sample, msg
                                )
                                failures += 1
                        if args.full_subset:
                            ok, msg = full_subset_check(p, raw_p, con)
                            if ok:
                                log.info("subset(full): OK (%s)", msg)
                            else:
                                log.error("subset(full): FAIL (%s)", msg)
                                failures += 1

        if failures:
            log.error("SUMMARY: completed with %d failures.", failures)
            return 1
        log.info("SUMMARY: all checks passed.")
        return 0

    finally:
        if con is not None:
            con.close()


if __name__ == "__main__":
    raise SystemExit(main())
