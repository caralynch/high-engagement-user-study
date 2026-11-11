#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Cara Lynch
# See the LICENSE file for details.
"""
Preprocessing step 3: stream-safe monthly cleaning for Pushshift-to-Parquet outputs (DuckDB-only).

This script cleans monthly Parquet outputs produced by `01_convert_to_parquet.py`:

- Cleans core files (comments_core, submissions_core):
  - Drops unusable rows (missing essential IDs).
  - Repairs missing fullname prefixes (t1_/t3_) where needed.
  - Adds `is_deleted_author` flag in core outputs.
- Filters text sidecars (comments_body, submissions_selftext) by cleaned core via SEMI JOIN
  (or INNER JOIN when `--move-title` requires core columns).
- Adds placeholder flags for text fields:
  - comments: is_removed_body, is_deleted_body
  - submissions: is_removed_selftext, is_deleted_selftext
- By default, moves `title` from submissions_core into submissions_selftext (so all post text
  lives in the text sidecar). Use `--no-move-title` to keep `title` in submissions_core.
  When title-moving is enabled, the script writes a temporary cleaned submissions_core
  *with* `title` for joining, while writing the final submissions_core *without* `title`
  directly (avoids rewriting submissions_core).
- Optional: emits dropped rows as Parquet under <out-root>/dropped/{comments,submissions}/
  (core unusable rows, plus text rows not present in cleaned core).
- Supports multiprocessing per month via --jobs N (each month processed independently).
- Writes cleaned Parquet to an output root mirroring the input layout, plus audit
  and provenance metadata.

This version avoids loading full files into pandas and is designed to scale to
multi-GB monthly files.

Post-conditions (guarantees):
- All core IDs are non-null and have canonical fullnames:
  - comments.id      starts with "t1_"
  - submissions.id   starts with "t3_"
- comments.link_id always refers to a submissions.id (t3_)
- comments.parent_id always refers to either a comments.id (t1_) or submissions.id (t3_)
- submissions_core does not contain a title column by default (use --no-move-title to keep it)
- text sidecars contain only rows whose IDs exist in cleaned core

Schema changes introduced by this step:
- Adds is_deleted_author to core tables
- Adds is_removed_* and is_deleted_* flags to text tables
- Relocates submissions.title from core to the text sidecar by default (use --no-move-title to disable)



Command-line interface
----------------------
--parquet-root            Input root containing Parquet outputs (comments/ and submissions/).
--out-root                Output root for cleaned outputs (creates comments/, submissions/, audit/, dropped/).
--year                    Year to process (YYYY).
--only-months             Optional: comma-separated list of YYYY-MM months to keep
                          (e.g. "2021-06,2021-07").
--limit-months            Optional: process only the first N discovered months
                          (useful for smoke tests).
--jobs                    Number of worker processes (months processed independently).
--move-title / --no-move-title
                          Move title into submissions_selftext and omit it from submissions_core
                          (default: move-title). Use --no-move-title to keep title in core.
--emit-dropped            Write dropped rows as separate Parquet files under out-root/dropped/.
--overwrite               Overwrite existing outputs if present.
--dry-run                 Print planned work and exit (no writes).
--smoke                   Quick-run mode with conservative defaults.
--smoke-max-rows-core     Optional: when --smoke, limit rows written for core tables.
--smoke-max-rows-text     Optional: when --smoke, limit rows written for text tables.
--smoke-skip-audit        When --smoke, skip expensive audit COUNT scans.
--log-level               DEBUG / INFO / WARNING / ERROR / CRITICAL.

Failure behavior:
------------------
- Raises an error if --move-title is enabled and the temporary core is missing
- Skips months only when all expected outputs already exist (unless --overwrite)


Note on smoke mode
------------------
When --smoke is enabled, the script applies conservative defaults for speed and
low I/O:
- single worker
- low DuckDB thread count
- uncompressed Parquet output
- default row limits for core and text outputs (unless overridden)
- audit COUNT scans are skipped unless explicitly enabled

- When --smoke-skip-audit is used, the script may take a no-join fast path for text sidecars;
  in this case, sidecar outputs may not be strictly filtered to cleaned core membership.

WARNING: --smoke mode is not intended for analysis or publication outputs.
It may truncate outputs and skip audit validation.


DuckDB tuning (optional)
------------------------
--duckdb-memory-limit     DuckDB memory limit (e.g. "4GB"). Applied per worker.
--duckdb-threads          DuckDB threads setting (integer). Applied per worker.
--duckdb-temp-dir         DuckDB temp_directory (recommended for node-local scratch on HPC).
--compression             Parquet compression codec (default: ZSTD).

Environment variables (optional)
--------------------------------
PARQUET_ROOT="parquet"    Alternative default for --parquet-root.
OUT_ROOT="parquet_clean"  Alternative default for --out-root.
JOBS=4                    Alternative default for --jobs.

Inputs
------
Parquet outputs from `01_convert_to_parquet.py` (either layout supported):
  <parquet-root>/comments/comments_core_YYYY-MM.parquet
  <parquet-root>/comments/comments_body_YYYY-MM.parquet
  <parquet-root>/submissions/submissions_core_YYYY-MM.parquet
  <parquet-root>/submissions/submissions_selftext_YYYY-MM.parquet

Also supports:
  <parquet-root>/parquet/comments/*.parquet
  <parquet-root>/parquet/submissions/*.parquet

Assumptions:
- Input Parquet files follow the schema produced by 01_convert_to_parquet.py
- IDs may be raw Pushshift IDs or fullnames
- Text sidecars may contain rows not present in core

Outputs
-------
Cleaned Parquet outputs:
  <out-root>/comments/comments_core_YYYY-MM.parquet
  <out-root>/comments/comments_body_YYYY-MM.parquet
  <out-root>/submissions/submissions_core_YYYY-MM.parquet
  <out-root>/submissions/submissions_selftext_YYYY-MM.parquet

Optional dropped rows:
  <out-root>/dropped/comments/comments_core_dropped_YYYY-MM.parquet
  <out-root>/dropped/comments/comments_body_dropped_YYYY-MM.parquet
  <out-root>/dropped/submissions/submissions_core_dropped_YYYY-MM.parquet
  <out-root>/dropped/submissions/submissions_selftext_dropped_YYYY-MM.parquet

Audit / provenance:
  <out-root>/audit/clean_audit_<year>.csv
  <out-root>/audit/clean_config_<run_id>.json
  <out-root>/audit/clean_run_<run_id>.jsonl
  <out-root>/audit/clean_finished_<run_id>.json

Examples
--------
Run on 4 workers and move post titles to selftext output:
  python 03_0_cleaning.py \
    --parquet-root parquet \
    --out-root parquet_clean \
    --year 2021 \
    --emit-dropped \
    --jobs 4 \
    --duckdb-temp-dir "$TMPDIR" \
    --duckdb-memory-limit "4GB" \
    --log-level INFO

Dry run a single month:
  python 03_0_cleaning.py \
    --parquet-root parquet \
    --out-root parquet_clean \
    --year 2021 \
    --only-months "2021-01" \
    --dry-run \
    --log-level INFO

Smoke test:
  python 03_0_cleaning.py \
    --parquet-root parquet \
    --out-root parquet_clean \
    --year 2021 \
    --smoke \
    --smoke-max-rows-core 100000 \
    --smoke-max-rows-text 100000
"""


from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import duckdb

import json
import socket
import platform
import getpass
import hashlib
import subprocess


# ----------------------------- Regex & constants -----------------------------

MONTH_RE = re.compile(r"(?<!\d)(20\d{2})[-_](0[1-9]|1[0-2])(?!\d)")
# avoid regex for ID checks: Pushshift fullnames are usually "t1_<id>" (comments)
# or "t3_<id>" (submissions). "Repair" means adding the expected prefix if missing.
T1_PREFIX = "t1_"
T3_PREFIX = "t3_"


AUDIT_FIELDS = [
    "month",
    # comments core
    "comments_original_len",
    "comments_dropped_unusable",
    "comments_id_repaired",
    "comments_link_id_repaired",
    "comments_parent_id_repaired",
    "comments_deleted_authors",
    # comments body
    "comments_text_original_len",
    "comments_text_clean_len",
    "comments_text_dropped_not_in_core",
    "comments_text_ids_repaired",
    # submissions core
    "submissions_original_len",
    "submissions_dropped_unusable",
    "submissions_id_repaired",
    "submissions_deleted_authors",
    # submissions selftext
    "submissions_text_original_len",
    "submissions_text_clean_len",
    "submissions_text_dropped_not_in_core",
    "submissions_text_ids_repaired",
    # present only when --move-title
    "submissions_title_moved_to_text",
    # timing
    "started_at",
    "finished_at",
    "elapsed_seconds",
]


# ----------------------------- Data structures ------------------------------


@dataclass(frozen=True)
class MonthFiles:
    """Core + text parquet paths for a given month."""

    core: Optional[str] = None
    text: Optional[str] = None


Inventory = Dict[str, MonthFiles]  # month -> MonthFiles


# ----------------------------- Audit utilities ------------------------------


def now_iso() -> str:
    """UTC timestamp in stable ISO format for audit logs."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dir(path: str) -> None:
    """Create directory (and parents) if missing."""
    os.makedirs(path, exist_ok=True)


def ensure_audit_csv(audit_csv: str) -> None:
    """Create the audit CSV with header if missing."""
    ensure_dir(os.path.dirname(audit_csv))
    if not os.path.exists(audit_csv):
        with open(audit_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
            writer.writeheader()


def append_audit_row(audit_csv: str, row: dict) -> None:
    """
    Append one row to the audit CSV.

    Intended to be called by the parent process only (safe with multiprocessing).
    """
    out = {k: row.get(k, 0) for k in AUDIT_FIELDS}
    with open(audit_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        writer.writerow(out)


def ensure_output_dirs(out_root: str) -> None:
    """Create the directory tree used by this pipeline step."""
    ensure_dir(os.path.join(out_root, "comments"))
    ensure_dir(os.path.join(out_root, "submissions"))
    ensure_dir(os.path.join(out_root, "audit"))
    ensure_dir(os.path.join(out_root, "dropped", "comments"))
    ensure_dir(os.path.join(out_root, "dropped", "submissions"))


def safe_json_dump(obj: dict, path: str) -> None:
    """Write JSON with stable formatting."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def utc_run_id(prefix: str, year: str) -> str:
    """Stable-ish run identifier string (UTC timestamp)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{year}_{ts}"


def get_git_commit() -> Optional[str]:
    """Best-effort git commit hash for provenance (returns None if unavailable)."""
    try:
        out = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        return out if out else None
    except Exception:
        return None


def file_sha256(path: str, max_bytes: int = 2_000_000) -> Optional[str]:
    """
    Best-effort content hash for a file (reads up to max_bytes to avoid heavy IO).
    Useful to confirm script version when not using git.
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            chunk = f.read(max_bytes)
            h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def append_jsonl(path: str, record: dict) -> None:
    """
    Append one JSON record as a single line (JSONL).
    Intended to be called by the parent process only.
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True))
        f.write("\n")


def build_run_metadata(
    args: argparse.Namespace, months: List[str], script_path: str
) -> dict:
    """Build run-level metadata dict for reproducibility."""
    return {
        "script": os.path.basename(script_path),
        "script_path": os.path.abspath(script_path),
        "script_sha256_head2mb": file_sha256(script_path),
        "git_commit": get_git_commit(),
        "started_at": now_iso(),
        "user": getpass.getuser(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "pid": os.getpid(),
        "args": vars(args),
        "months_planned": months,
        "paths": {
            "parquet_root": os.path.abspath(args.parquet_root),
            "out_root": os.path.abspath(args.out_root),
        },
        "duckdb": {
            "memory_limit": args.duckdb_memory_limit,
            "threads": args.duckdb_threads,
            "temp_dir": args.duckdb_temp_dir,
        },
    }


# ----------------------------- Discovery helpers ----------------------------


def month_of(path: str) -> str:
    """Extract YYYY-MM token from a file path; returns 'UNKNOWN' if not found."""
    m = MONTH_RE.search(path)
    return f"{m.group(1)}-{m.group(2)}" if m else "UNKNOWN"


def discover_month_files(
    parquet_root: str, year: str, log: logging.Logger
) -> Tuple[Inventory, Inventory]:
    """
    Discover monthly parquet files for comments and submissions.

    Supports either layout:
      - <parquet_root>/comments/*.parquet
      - <parquet_root>/parquet/comments/*.parquet
    and same for submissions.

    Returns:
      (comments_inventory, submissions_inventory)
    """

    def _roots(kind: str) -> List[str]:
        return [
            os.path.join(parquet_root, kind),
            os.path.join(parquet_root, "parquet", kind),
        ]

    def _build(kind: str) -> Inventory:
        files: List[str] = []
        for r in _roots(kind):
            files.extend(glob.glob(os.path.join(r, f"*_{year}-*.parquet")))
            files.extend(glob.glob(os.path.join(r, f"*_{year}_*.parquet")))

        inv: Dict[str, Dict[str, Optional[str]]] = defaultdict(lambda: {"core": None, "text": None})  # type: ignore
        for p in sorted(files):
            mm = month_of(p)
            if mm == "UNKNOWN":
                continue
            b = os.path.basename(p)
            if kind == "comments":
                if "comments_core" in b:
                    inv[mm]["core"] = p
                elif "comments_body" in b:
                    inv[mm]["text"] = p
            else:
                if "submissions_core" in b:
                    inv[mm]["core"] = p
                elif "submissions_selftext" in b:
                    inv[mm]["text"] = p

        out: Inventory = {
            m: MonthFiles(core=d["core"], text=d["text"])
            for m, d in inv.items()
            if d["core"] or d["text"]
        }
        for m in sorted(out)[:12]:
            mf = out[m]
            if mf.core:
                log.info("%s %s: %s (core)", kind, m, mf.core)
            if mf.text:
                log.info("%s %s: %s (text)", kind, m, mf.text)
        if len(out) > 12:
            log.info("%s inventory: ... (%d more months)", kind, len(out) - 12)
        return out

    from collections import defaultdict  # local to keep top tidy

    return _build("comments"), _build("submissions")


# ----------------------------- SQL builders ---------------------------------


def sql_clean_comments_core(src: str) -> str:
    """DuckDB SQL: clean comments core (drop unusable, repair IDs, add is_deleted_author)."""
    return f"""
    WITH orig AS (
      SELECT * FROM read_parquet('{src}')
    ),
    base AS (
      SELECT * FROM orig
      WHERE id IS NOT NULL AND link_id IS NOT NULL AND parent_id IS NOT NULL
    ),
    fixed AS (
      SELECT
        -- comments ids should be t1_*
        CASE WHEN starts_with(id, '{T1_PREFIX}') THEN id ELSE '{T1_PREFIX}' || id END AS id,

        -- link_id should be t3_*
        CASE WHEN starts_with(link_id, '{T3_PREFIX}') THEN link_id ELSE '{T3_PREFIX}' || link_id END AS link_id,

        -- parent_id is either a comment (t1_) or a submission (t3_). Infer using relationship to link_id.
        CASE
          -- if parent refers to the submission (either exact link_id or raw id part), treat as t3_
          WHEN parent_id = link_id
            OR parent_id = substr(link_id, 4)  -- raw id part of t3_<raw>
          THEN CASE WHEN starts_with(parent_id, '{T3_PREFIX}') THEN parent_id ELSE '{T3_PREFIX}' || parent_id END

          -- otherwise it’s a comment parent -> t1_
          ELSE CASE WHEN starts_with(parent_id, '{T1_PREFIX}') THEN parent_id ELSE '{T1_PREFIX}' || parent_id END
        END AS parent_id,

        lower(trim(coalesce(author,''))) = '[deleted]' AS is_deleted_author,
        * EXCLUDE(id, link_id, parent_id)
      FROM base
    )
    SELECT * FROM fixed
    """


def sql_comments_core_dropped(src: str) -> str:
    """DuckDB SQL: unusable comments core rows."""
    return f"SELECT * FROM read_parquet('{src}') WHERE id IS NULL OR link_id IS NULL OR parent_id IS NULL"


def sql_comments_core_audit(src: str) -> str:
    """Single-pass audit query for comments core (prefix-based, no regex)."""
    return f"""
    SELECT
      COUNT(*)::BIGINT AS comments_original_len,
      SUM(id IS NULL OR link_id IS NULL OR parent_id IS NULL)::BIGINT AS comments_dropped_unusable,

      -- "repaired" means it didn't already have the expected prefix
      SUM(id IS NOT NULL AND NOT starts_with(id, '{T1_PREFIX}'))::BIGINT AS comments_id_repaired,
      SUM(link_id IS NOT NULL AND NOT starts_with(link_id, '{T3_PREFIX}'))::BIGINT AS comments_link_id_repaired,

      -- parent can be t1_ or t3_; we count as "repaired" if it lacks either prefix (roughly matches old behavior)
      SUM(
        parent_id IS NOT NULL
        AND NOT starts_with(parent_id, '{T1_PREFIX}')
        AND NOT starts_with(parent_id, '{T3_PREFIX}')
      )::BIGINT AS comments_parent_id_repaired,

      -- deleted authors among usable rows (matches cleaned output definition)
      SUM(
        (id IS NOT NULL AND link_id IS NOT NULL AND parent_id IS NOT NULL)
        AND (lower(trim(coalesce(author,''))) = '[deleted]')
      )::BIGINT AS comments_deleted_authors
    FROM read_parquet('{src}');
    """


def sql_clean_submissions_core(src: str) -> str:
    """DuckDB SQL: clean submissions core (drop unusable, repair IDs, add is_deleted_author)."""
    return f"""
    WITH orig AS (SELECT * FROM read_parquet('{src}')),
    base AS (SELECT * FROM orig WHERE id IS NOT NULL),
    fixed AS (
      SELECT
        CASE WHEN starts_with(id, '{T3_PREFIX}') THEN id ELSE '{T3_PREFIX}' || id END AS id,
        lower(trim(coalesce(author,''))) = '[deleted]' AS is_deleted_author,
        * EXCLUDE(id)
      FROM base
    )
    SELECT * FROM fixed
    """


def sql_submissions_core_dropped(src: str) -> str:
    """DuckDB SQL: unusable submissions core rows."""
    return f"SELECT * FROM read_parquet('{src}') WHERE id IS NULL"


def sql_submissions_core_audit(src: str) -> str:
    """Single-pass audit query for submissions core."""
    return f"""
    SELECT
      COUNT(*)::BIGINT AS submissions_original_len,
      SUM(id IS NULL)::BIGINT AS submissions_dropped_unusable,
      SUM(id IS NOT NULL AND NOT starts_with(id, '{T3_PREFIX}'))::BIGINT AS submissions_id_repaired,
      -- deleted authors among usable rows (matches cleaned output definition)
      SUM(
        (id IS NOT NULL)
        AND (lower(trim(coalesce(author,''))) = '[deleted]')
      )::BIGINT AS submissions_deleted_authors
    FROM read_parquet('{src}');
    """


def sql_comments_body_filtered(src_text: str, cleaned_core: str) -> str:
    return f"""
    SELECT
      b.*,
      lower(trim(b.body)) = '[removed]'  AS is_removed_body,
      lower(trim(b.body)) = '[deleted]'  AS is_deleted_body
    FROM read_parquet('{src_text}') b
    SEMI JOIN read_parquet('{cleaned_core}') c
    ON (
      (CASE WHEN starts_with(b.id, '{T1_PREFIX}') THEN b.id ELSE '{T1_PREFIX}' || b.id END) = c.id
    )
    """

def sql_comments_body_nojoin(src_text: str) -> str:
    """
    DuckDB SQL: rewrite comments_body without joining to core.
    Safe ONLY when comments_core cleaning did not change membership (no drops/repairs).
    """
    return f"""
    SELECT
      b.*,
      lower(trim(b.body)) = '[removed]'  AS is_removed_body,
      lower(trim(b.body)) = '[deleted]'  AS is_deleted_body
    FROM read_parquet('{src_text}') b
    """


def sql_submissions_selftext_nojoin(src_text: str, move_title: bool) -> str:
    """
    DuckDB SQL: rewrite submissions_selftext without joining to core.

    IMPORTANT:
    - Only valid when submissions_core cleaning did not change membership (no drops/repairs),
      AND move_title is False (because move_title needs title from core).
    """
    if move_title:
        raise ValueError("sql_submissions_selftext_nojoin cannot be used when move_title=True")

    return f"""
    SELECT
      sst.*,
      lower(trim(sst.selftext)) = '[removed]'  AS is_removed_selftext,
      lower(trim(sst.selftext)) = '[deleted]'  AS is_deleted_selftext
    FROM read_parquet('{src_text}') sst
    """


def sql_comments_body_dropped(src_text: str, cleaned_core: str) -> str:
    """DuckDB SQL: body rows not present in cleaned comments core."""
    return f"""
    SELECT b.*
    FROM read_parquet('{src_text}') b
    ANTI JOIN read_parquet('{cleaned_core}') c
    ON (
      (CASE WHEN starts_with(b.id, '{T1_PREFIX}') THEN b.id ELSE '{T1_PREFIX}' || b.id END) = c.id
    )
    """


def sql_comments_body_audit(src_text: str) -> str:
    """Single-pass audit query for comments body input."""
    return f"""
    SELECT
      COUNT(*)::BIGINT AS comments_text_original_len,
      SUM(id IS NOT NULL AND NOT starts_with(id, '{T1_PREFIX}'))::BIGINT AS comments_text_ids_repaired
    FROM read_parquet('{src_text}');
    """


def sql_submissions_selftext_filtered(
    src_text: str, cleaned_core: str, move_title: bool
) -> str:
    """
    DuckDB SQL: keep only selftext rows present in cleaned submissions core; add placeholder flags.

    When move_title=True, uses INNER JOIN so title can be selected from core.
    Otherwise uses SEMI JOIN for existence-only filtering.
    """
    if move_title:
        return f"""
        SELECT
          sst.*,
          sc.title,
          lower(trim(sst.selftext)) = '[removed]'  AS is_removed_selftext,
          lower(trim(sst.selftext)) = '[deleted]'  AS is_deleted_selftext
        FROM read_parquet('{src_text}') sst
        INNER JOIN read_parquet('{cleaned_core}') sc
          ON ( (CASE WHEN starts_with(sst.id, '{T3_PREFIX}') THEN sst.id ELSE '{T3_PREFIX}' || sst.id END)= sc.id )
        """
    return f"""
    SELECT
      sst.*,
      lower(trim(sst.selftext)) = '[removed]'  AS is_removed_selftext,
      lower(trim(sst.selftext)) = '[deleted]'  AS is_deleted_selftext
    FROM read_parquet('{src_text}') sst
    SEMI JOIN read_parquet('{cleaned_core}') sc
      ON ( (CASE WHEN starts_with(sst.id, '{T3_PREFIX}') THEN sst.id ELSE '{T3_PREFIX}' || sst.id END) = sc.id )
    """


def sql_submissions_selftext_dropped(src_text: str, cleaned_core: str) -> str:
    """DuckDB SQL: selftext rows not present in cleaned submissions core."""
    return f"""
    SELECT sst.*
    FROM read_parquet('{src_text}') sst
    ANTI JOIN read_parquet('{cleaned_core}') sc
    ON (
      (CASE WHEN starts_with(sst.id, '{T3_PREFIX}') THEN sst.id ELSE '{T3_PREFIX}' || sst.id END) = sc.id
    )
    """


def sql_submissions_selftext_audit(src_text: str) -> str:
    """Single-pass audit query for submissions selftext input."""
    return f"""
    SELECT
      COUNT(*)::BIGINT AS submissions_text_original_len,
      SUM(id IS NOT NULL AND NOT starts_with(id, '{T3_PREFIX}'))::BIGINT AS submissions_text_ids_repaired
    FROM read_parquet('{src_text}');
    """


# ----------------------------- DuckDB helpers --------------------------------


def configure_duckdb(
    con: duckdb.DuckDBPyConnection,
    memory_limit: str,
    threads: int,
    temp_dir: Optional[str],
) -> None:
    """Apply per-connection DuckDB settings (safe to call in each worker)."""
    # Helps reduce memory overhead for large scans/joins.
    con.execute("SET preserve_insertion_order=false")

    if memory_limit:
        con.execute(f"SET memory_limit='{memory_limit}'")
    if threads and threads > 0:
        con.execute(f"SET threads={threads}")
    if temp_dir:
        con.execute(f"SET temp_directory='{temp_dir}'")



def wrap_with_limit(query: str, limit: Optional[int]) -> str:
    """
    Wrap a query with a LIMIT clause (used to speed up smoke tests).

    Important: We wrap the *whole* query so it applies regardless of query shape.
    """
    if limit is None or limit <= 0:
        return query
    return f"SELECT * FROM ( {query} ) LIMIT {int(limit)}"


def fetch_single_row_dict(con: duckdb.DuckDBPyConnection, query: str) -> Dict[str, int]:
    """
    Execute a query expected to return exactly one row and return it as a dict
    keyed by column alias. Values are returned as Python ints where possible.
    """
    cur = con.execute(query)
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    if row is None:
        return {c: 0 for c in cols}
    out: Dict[str, int] = {}
    for c, v in zip(cols, row):
        # duckdb may return numpy ints; int(...) normalizes
        out[c] = int(v) if v is not None else 0
    return out


def copy_parquet(
    con: duckdb.DuckDBPyConnection, query: str, out_path: str, compression: str
) -> None:
    """Write a query result to Parquet using DuckDB COPY."""
    if compression == "UNCOMPRESSED":
        con.execute(f"COPY ( {query} ) TO '{out_path}' (FORMAT PARQUET);")
    else:
        con.execute(
            f"COPY ( {query} ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION {compression});"
        )


# ----------------------------- Per-month processing --------------------------


def process_month(
    con: duckdb.DuckDBPyConnection,
    month: str,
    paths: Dict[str, Optional[str]],
    out_root: str,
    move_title: bool,
    emit_dropped: bool,
    overwrite: bool,
    compression: str,
    log: logging.Logger,
    smoke_skip_audit: bool = False,
    smoke_max_rows_core: Optional[int] = None,
    smoke_max_rows_text: Optional[int] = None,
) -> Dict[str, object]:
    """
    Clean a single month (YYYY-MM).

    Returns an audit dict (counts + timing). This function performs all writes
    for a month. It is safe to run in parallel across months (independent outputs).
    """
    audit: Dict[str, object] = {}
    started_at = now_iso()
    t0 = time.time()

    # Whether we can safely skip filtering text sidecars by core membership.
    # Defaults to False until proven otherwise.
    comments_core_unchanged = False
    submissions_core_unchanged = False


    log.info(
        "[%s] START move_title=%s emit_dropped=%s smoke_skip_audit=%s core_limit=%s text_limit=%s",
        month,
        move_title,
        emit_dropped,
        smoke_skip_audit,
        smoke_max_rows_core,
        smoke_max_rows_text,
    )

    p_cc = paths.get("comments_core")
    p_cb = paths.get("comments_body")
    p_sc = paths.get("submissions_core")
    p_st = paths.get("submissions_selftext")

    out_cc = os.path.join(out_root, "comments", f"comments_core_{month}.parquet")
    out_cb = os.path.join(out_root, "comments", f"comments_body_{month}.parquet")
    out_sc = os.path.join(out_root, "submissions", f"submissions_core_{month}.parquet")
    out_st = os.path.join(
        out_root, "submissions", f"submissions_selftext_{month}.parquet"
    )

    drop_dir = os.path.join(out_root, "dropped")
    out_cc_drop = os.path.join(
        drop_dir, "comments", f"comments_core_dropped_{month}.parquet"
    )
    out_cb_drop = os.path.join(
        drop_dir, "comments", f"comments_body_dropped_{month}.parquet"
    )
    out_sc_drop = os.path.join(
        drop_dir, "submissions", f"submissions_core_dropped_{month}.parquet"
    )
    out_st_drop = os.path.join(
        drop_dir, "submissions", f"submissions_selftext_dropped_{month}.parquet"
    )

    # Skip only if ALL expected outputs exist and not overwriting.
    # "Expected" outputs depend on which inputs exist for this month:
    # - core outputs are expected only if the corresponding core input exists
    # - text outputs are expected only if the corresponding text input exists
    expected_outs: List[str] = []

    # core outputs
    if p_cc and os.path.exists(p_cc):
        expected_outs.append(out_cc)
    if p_sc and os.path.exists(p_sc):
        expected_outs.append(out_sc)

    # text outputs (only expected if the text input exists)
    if p_cb and os.path.exists(p_cb) and (p_cc and os.path.exists(p_cc)):
        expected_outs.append(out_cb)
    if p_st and os.path.exists(p_st) and (p_sc and os.path.exists(p_sc)):
        expected_outs.append(out_st)

    if (
        not overwrite
        and expected_outs
        and all(os.path.exists(p) for p in expected_outs)
    ):
        log.warning(
            "[%s] All expected outputs already exist and --overwrite not set; skipping month.",
            month,
        )
        audit["started_at"] = started_at
        audit["finished_at"] = now_iso()
        audit["elapsed_seconds"] = 0
        return audit

    # ---- comments_core ----
    if p_cc and os.path.exists(p_cc):
        if not smoke_skip_audit:
            audit.update(fetch_single_row_dict(con, sql_comments_core_audit(p_cc)))

        # NOTE (smoke mode beware!): If smoke_skip_audit=True, don't know whether core cleaning
        # dropped rows or repaired IDs. Treating core as "unchanged" enables no-join
        # fast path for sidecars, which may produce sidecars that are not strictly filtered to
        # cleaned core membership. Fine for quick smoke tests, not for real outputs.
        comments_core_unchanged = (
            smoke_skip_audit
            or (
                int(audit.get("comments_dropped_unusable", 0)) == 0
                and int(audit.get("comments_id_repaired", 0)) == 0
                and int(audit.get("comments_link_id_repaired", 0)) == 0
                and int(audit.get("comments_parent_id_repaired", 0)) == 0
            )
        )

        clean_q = sql_clean_comments_core(p_cc)
        clean_q = wrap_with_limit(clean_q, smoke_max_rows_core)
        copy_parquet(con, clean_q, out_cc, compression)

        if emit_dropped:
            dropped_q = sql_comments_core_dropped(p_cc)
            dropped_q = wrap_with_limit(dropped_q, smoke_max_rows_core)
            copy_parquet(con, dropped_q, out_cc_drop, compression)


        log.info(
            "[%s] comments_core: orig=%s dropped=%s id_fix=%s link_fix=%s parent_fix=%s",
            month,
            f"{audit.get('comments_original_len', 0):,}",
            f"{audit.get('comments_dropped_unusable', 0):,}",
            f"{audit.get('comments_id_repaired', 0):,}",
            f"{audit.get('comments_link_id_repaired', 0):,}",
            f"{audit.get('comments_parent_id_repaired', 0):,}",
        )

    else:
        log.warning("[%s] Missing comments_core input; skipping.", month)

    # ---- submissions_core ----
    if p_sc and os.path.exists(p_sc):
        if not smoke_skip_audit:
            audit.update(fetch_single_row_dict(con, sql_submissions_core_audit(p_sc)))

        # NOTE (smoke mode beware!): If smoke_skip_audit=True, don't know whether core cleaning
        # dropped rows or repaired IDs. Treating core as "unchanged" enables no-join
        # fast path for sidecars, which may produce sidecars that are not strictly filtered to
        # cleaned core membership. Fine for quick smoke tests, not for real outputs.
        submissions_core_unchanged = (
            smoke_skip_audit
            or (
                int(audit.get("submissions_dropped_unusable", 0)) == 0
                and int(audit.get("submissions_id_repaired", 0)) == 0
            )
        )

        if move_title:
            # If we need title for --move-title, write a temporary core WITH title for joining,
            # then write the final core WITHOUT title (so no rewrite is needed later).
            tmp_sc_with_title = out_sc + ".with_title.tmp"

            # temp core WITH title (for the selftext join)
            clean_q_tmp = sql_clean_submissions_core(
                p_sc
            )  # retains original columns including title
            clean_q_tmp = wrap_with_limit(clean_q_tmp, smoke_max_rows_core)
            copy_parquet(con, clean_q_tmp, tmp_sc_with_title, compression)

            # final core WITHOUT title (direct write; avoids rewrite pass)
            clean_q_final = f"""
            WITH cleaned AS (
            {sql_clean_submissions_core(p_sc)}
            )
            SELECT * EXCLUDE(title) FROM cleaned
            """

            clean_q_final = wrap_with_limit(clean_q_final, smoke_max_rows_core)
            copy_parquet(con, clean_q_final, out_sc, compression)
        else:
            # normal: just write core as-is
            clean_q = sql_clean_submissions_core(p_sc)
            clean_q = wrap_with_limit(clean_q, smoke_max_rows_core)
            copy_parquet(con, clean_q, out_sc, compression)

        if emit_dropped:
            dropped_q = sql_submissions_core_dropped(p_sc)
            dropped_q = wrap_with_limit(dropped_q, smoke_max_rows_core)
            copy_parquet(con, dropped_q, out_sc_drop, compression)

        log.info(
            "[%s] submissions_core: orig=%s dropped=%s id_fix=%s",
            month,
            f"{audit.get('submissions_original_len', 0):,}",
            f"{audit.get('submissions_dropped_unusable', 0):,}",
            f"{audit.get('submissions_id_repaired', 0):,}",
        )

    else:
        log.warning("[%s] Missing submissions_core input; skipping.", month)

    # ---- comments_body ----
    if p_cb and os.path.exists(p_cb) and os.path.exists(out_cc):
        # Fast path when core membership didn't change: avoid SEMI JOIN (major memory win).
        if comments_core_unchanged:
            body_q = sql_comments_body_nojoin(p_cb)
        else:
            body_q = sql_comments_body_filtered(p_cb, out_cc)

        body_q = wrap_with_limit(body_q, smoke_max_rows_text)
        copy_parquet(con, body_q, out_cb, compression)


        if not smoke_skip_audit:
            audit.update(fetch_single_row_dict(con, sql_comments_body_audit(p_cb)))

            # Count from the *written output* (avoids re-running the join against raw input)
            clean_len = fetch_single_row_dict(
                con, f"SELECT COUNT(*)::BIGINT AS n FROM read_parquet('{out_cb}')",
            )["n"]

            audit["comments_text_clean_len"] = int(clean_len)
            audit["comments_text_dropped_not_in_core"] = int(
                audit["comments_text_original_len"]
            ) - int(clean_len)

        else:
            audit["comments_text_clean_len"] = 0
            audit["comments_text_dropped_not_in_core"] = 0

        if emit_dropped:
            if comments_core_unchanged:
                dropped_q = f"SELECT * FROM read_parquet('{p_cb}') WHERE 1=0"
            else:
                dropped_q = sql_comments_body_dropped(p_cb, out_cc)

            dropped_q = wrap_with_limit(dropped_q, smoke_max_rows_text)
            copy_parquet(con, dropped_q, out_cb_drop, compression)

        log.info(
            "[%s] comments_body: orig=%s dropped=%s",
            month,
            f"{audit.get('comments_text_original_len', 0):,}",
            f"{audit.get('comments_text_dropped_not_in_core', 0):,}",
        )
    else:
        log.info("[%s] Skipping comments_body (missing input or cleaned core).", month)

    # ---- submissions_selftext ----
    if p_st and os.path.exists(p_st) and os.path.exists(out_sc):
        core_for_text = out_sc
        if move_title:
            tmp_sc_with_title = out_sc + ".with_title.tmp"
            if not os.path.exists(tmp_sc_with_title):
                raise RuntimeError(
                    f"[{month}] move_title=True but temp core missing: {tmp_sc_with_title}"
                )
            core_for_text = tmp_sc_with_title

        # Fast path when core membership didn't change and we don't need title from core.
        if submissions_core_unchanged and not move_title:
            st_q = sql_submissions_selftext_nojoin(p_st, move_title=False)
        else:
            st_q = sql_submissions_selftext_filtered(p_st, core_for_text, move_title)

        st_q = wrap_with_limit(st_q, smoke_max_rows_text)
        copy_parquet(con, st_q, out_st, compression)


        if not smoke_skip_audit:
            audit.update(
                fetch_single_row_dict(con, sql_submissions_selftext_audit(p_st))
            )

            # Count from the *written output* (avoids re-running the join against raw input)
            clean_len = fetch_single_row_dict(
                con, f"SELECT COUNT(*)::BIGINT AS n FROM read_parquet('{out_st}')",
            )["n"]

            audit["submissions_text_clean_len"] = int(clean_len)
            audit["submissions_text_dropped_not_in_core"] = int(
                audit["submissions_text_original_len"]
            ) - int(clean_len)

            if move_title:
                # Count moved titles from the *written output* (only exists when move_title=True)
                moved = fetch_single_row_dict(
                    con,
                    f"SELECT COUNT(*)::BIGINT AS n FROM read_parquet('{out_st}') WHERE title IS NOT NULL",
                )["n"]
                audit["submissions_title_moved_to_text"] = int(moved)

        else:
            audit["submissions_text_clean_len"] = 0
            audit["submissions_text_dropped_not_in_core"] = 0
            if move_title:
                audit["submissions_title_moved_to_text"] = 0

        if emit_dropped:
            if submissions_core_unchanged:
                dropped_q = f"SELECT * FROM read_parquet('{p_st}') WHERE 1=0"
            else:
                dropped_q = sql_submissions_selftext_dropped(p_st, core_for_text)

            dropped_q = wrap_with_limit(dropped_q, smoke_max_rows_text)
            copy_parquet(con, dropped_q, out_st_drop, compression)

        if move_title:
            tmp_sc_with_title = out_sc + ".with_title.tmp"
            try:
                if os.path.exists(tmp_sc_with_title):
                    os.remove(tmp_sc_with_title)
            except OSError:
                log.warning(
                    "[%s] Could not delete temp file: %s", month, tmp_sc_with_title
                )

        log.info(
            "[%s] submissions_selftext: orig=%s dropped=%s",
            month,
            f"{audit.get('submissions_text_original_len', 0):,}",
            f"{audit.get('submissions_text_dropped_not_in_core', 0):,}",
        )
    else:
        log.info(
            "[%s] Skipping submissions_selftext (missing input or cleaned core).", month
        )

    elapsed = time.time() - t0
    audit["started_at"] = started_at
    audit["finished_at"] = now_iso()
    audit["elapsed_seconds"] = int(elapsed)
    log.info("[%s] DONE in %.1f min", month, elapsed / 60.0)
    return audit


def process_month_task(task: tuple) -> dict:
    """Worker wrapper for multiprocessing Pool."""
    (
        month,
        paths,
        out_root,
        move_title,
        emit_dropped,
        overwrite,
        compression,
        memory_limit,
        threads,
        temp_dir,
        log_level,
        smoke_skip_audit,
        smoke_max_rows_core,
        smoke_max_rows_text,
    ) = task

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    log = logging.getLogger("cleaning_worker")

    con = duckdb.connect()
    try:
        configure_duckdb(
            con, memory_limit=memory_limit, threads=threads, temp_dir=temp_dir
        )
        audit = process_month(
            con,
            month,
            paths,
            out_root,
            move_title,
            emit_dropped,
            overwrite,
            compression,
            log,
            smoke_skip_audit=smoke_skip_audit,
            smoke_max_rows_core=smoke_max_rows_core,
            smoke_max_rows_text=smoke_max_rows_text,
        )

        return {"month": month, **audit}
    finally:
        con.close()


# ----------------------------- CLI / Main -----------------------------------


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI parser for this script."""
    p = argparse.ArgumentParser(
        description="Clean monthly Reddit Parquet files using DuckDB (stream-safe).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--parquet-root",
        default=os.environ.get("PARQUET_ROOT", "parquet"),
        help="Input root with comments/ and submissions/ (converter outputs).",
    )
    p.add_argument(
        "--out-root",
        default=os.environ.get("OUT_ROOT", "parquet_clean"),
        help="Output root for cleaned files.",
    )
    p.add_argument("--year", default="2021", help="Year to process, e.g., 2021")
    p.add_argument(
        "--only-months", default="", help="Comma-separated subset like 2021-01,2021-06"
    )
    p.add_argument(
        "--limit-months",
        type=int,
        default=None,
        help="Process only first N discovered months.",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run id used in audit filenames (default: generated).",
    )

    p.add_argument(
        "--jobs",
        type=int,
        default=int(os.environ.get("JOBS", "1")),
        help="Process months in parallel with this many workers.",
    )
    p.add_argument(
        "--move-title",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Move submissions.title into submissions_selftext and omit it from submissions_core. "
            "Use --no-move-title to keep title in submissions_core."
        ),
    )

    p.add_argument(
        "--emit-dropped",
        action="store_true",
        help="Write dropped rows under out-root/dropped/.",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing outputs."
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned work and exit without writing.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Quick-run mode: few months, conservative settings.",
    )
    p.add_argument(
        "--smoke-max-rows-core",
        type=int,
        default=None,
        help="When --smoke, limit rows written for core outputs (e.g., 100000).",
    )
    p.add_argument(
        "--smoke-max-rows-text",
        type=int,
        default=None,
        help="When --smoke, limit rows written for text outputs (e.g., 100000).",
    )
    p.add_argument(
        "--smoke-skip-audit",
        action="store_true",
        help="When --smoke, skip expensive audit COUNT scans for speed.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )

    p.add_argument(
        "--compression",
        default="ZSTD",
        choices=["ZSTD", "SNAPPY", "GZIP", "LZ4", "UNCOMPRESSED"],
        help="Parquet compression codec used by DuckDB COPY.",
    )
    p.add_argument(
        "--duckdb-memory-limit",
        default="4GB",
        help="DuckDB memory limit applied per worker (e.g. '4GB').",
    )
    p.add_argument(
        "--duckdb-threads",
        type=int,
        default=1,
        help="DuckDB threads setting per worker.",
    )
    p.add_argument(
        "--duckdb-temp-dir",
        default=None,
        help="DuckDB temp_directory (recommended to point at node-local scratch on HPC).",
    )

    return p


def apply_smoke_preset(args: argparse.Namespace) -> None:
    """Apply a conservative preset for smoke tests (fast, low IO)."""
    args.limit_months = args.limit_months or 1
    args.jobs = min(args.jobs, 1)
    args.duckdb_threads = 1
    args.duckdb_memory_limit = args.duckdb_memory_limit or "2GB"

    # Make writes cheap in smoke mode
    if not getattr(args, "compression", None) or args.compression == "ZSTD":
        args.compression = "UNCOMPRESSED"

    # Default row limits if not explicitly set
    if getattr(args, "smoke_max_rows_core", None) is None:
        args.smoke_max_rows_core = 100_000
    if getattr(args, "smoke_max_rows_text", None) is None:
        args.smoke_max_rows_text = 100_000

    # Default to skipping expensive COUNT audits in smoke mode
    if not getattr(args, "smoke_skip_audit", False):
        args.smoke_skip_audit = True


def main() -> int:
    args = build_argparser().parse_args()

    if args.smoke:
        apply_smoke_preset(args)
    smoke_skip = bool(args.smoke and args.smoke_skip_audit)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    log = logging.getLogger("cleaning")

    log.info("Launched at %s", now_iso())
    log.info("Args: %s", args)

    ensure_output_dirs(args.out_root)

    comments_inv, subs_inv = discover_month_files(args.parquet_root, args.year, log)
    months = sorted(set(comments_inv.keys()) | set(subs_inv.keys()))

    if args.only_months:
        keep = {m.strip() for m in args.only_months.split(",") if m.strip()}
        months = [m for m in months if m in keep]

    if args.limit_months is not None:
        months = months[: args.limit_months]

    if not months:
        log.warning("No matching months discovered. Check --parquet-root / --year.")
        return 0

    log.info("Months to clean: %s", months)
    # Run id used for config + JSONL output naming
    script_path = os.path.abspath(__file__)
    run_id = args.run_id or utc_run_id("clean", args.year)

    # Write resolved config / provenance once per run
    run_meta = build_run_metadata(args, months, script_path)
    config_json = os.path.join(args.out_root, "audit", f"clean_config_{run_id}.json")
    safe_json_dump(run_meta, config_json)

    # Per-month append-only JSONL (parent process)
    run_jsonl = os.path.join(args.out_root, "audit", f"clean_run_{run_id}.jsonl")
    log.info("Run config JSON -> %s", config_json)
    log.info("Run JSONL log    -> %s", run_jsonl)

    # Build per-month tasks
    tasks = []
    for m in months:
        paths = {
            "comments_core": comments_inv.get(m, MonthFiles()).core,
            "comments_body": comments_inv.get(m, MonthFiles()).text,
            "submissions_core": subs_inv.get(m, MonthFiles()).core,
            "submissions_selftext": subs_inv.get(m, MonthFiles()).text,
        }
        tasks.append(
            (
                m,
                paths,
                args.out_root,
                args.move_title,
                args.emit_dropped,
                args.overwrite,
                args.compression,
                args.duckdb_memory_limit,
                args.duckdb_threads,
                args.duckdb_temp_dir,
                args.log_level,
                smoke_skip,
                args.smoke_max_rows_core if args.smoke else None,
                args.smoke_max_rows_text if args.smoke else None,
            )
        )

    audit_csv = os.path.join(args.out_root, "audit", f"clean_audit_{args.year}.csv")
    ensure_audit_csv(audit_csv)

    if args.dry_run:
        log.info("Dry-run enabled; planned tasks:")
        for t in tasks:
            log.info("  %s", t[0])
        log.info("Audit CSV would be written to: %s", audit_csv)
        return 0

    results: List[dict] = []

    if args.jobs > 1:
        log.info("Multiprocessing enabled: %d jobs.", args.jobs)
        with Pool(processes=args.jobs, maxtasksperchild=1) as pool:
            for r in pool.imap_unordered(process_month_task, tasks):
                append_audit_row(audit_csv, r)
                append_jsonl(
                    run_jsonl,
                    {
                        "run_id": run_id,
                        "month": r["month"],
                        "finished_at": now_iso(),
                        "result": r,
                    },
                )

                log.info("[DONE] %s", r["month"])
                results.append(r)
    else:
        log.info("Single-process mode.")
        con = duckdb.connect()
        try:
            configure_duckdb(
                con,
                memory_limit=args.duckdb_memory_limit,
                threads=args.duckdb_threads,
                temp_dir=args.duckdb_temp_dir,
            )
            for t in tasks:
                m, paths = t[0], t[1]
                r = {
                    "month": m,
                    **process_month(
                        con,
                        m,
                        paths,
                        args.out_root,
                        args.move_title,
                        args.emit_dropped,
                        args.overwrite,
                        args.compression,
                        log,
                        smoke_skip_audit=smoke_skip,
                        smoke_max_rows_core=args.smoke_max_rows_core
                        if args.smoke
                        else None,
                        smoke_max_rows_text=args.smoke_max_rows_text
                        if args.smoke
                        else None,
                    ),
                }
                append_audit_row(audit_csv, r)
                append_jsonl(
                    run_jsonl,
                    {
                        "run_id": run_id,
                        "month": r["month"],
                        "finished_at": now_iso(),
                        "result": r,
                    },
                )

                log.info("[DONE] %s", m)
                results.append(r)
        finally:
            con.close()

    log.info("Audit CSV updated at -> %s", audit_csv)
    append_jsonl(
        run_jsonl,
        {
            "run_id": run_id,
            "event": "run_finished",
            "finished_at": now_iso(),
            "audit_csv": os.path.abspath(audit_csv),
            "config_json": os.path.abspath(config_json),
        },
    )
    final_json = os.path.join(args.out_root, "audit", f"clean_finished_{run_id}.json")
    safe_json_dump({"run_id": run_id, "finished_at": now_iso()}, final_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
