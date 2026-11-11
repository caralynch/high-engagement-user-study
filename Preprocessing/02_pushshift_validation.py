# MIT License
# Copyright (c) 2026 Cara Lynch
# See the LICENSE file for details.
"""
Preprocessing step 2: validate Pushshift Reddit dumps and Parquet conversion outputs.

This script performs integrity and sanity checks for monthly Reddit comment and
submission datasets converted from Zstandard-compressed NDJSON (.zst) to Parquet
by `01_convert_to_parquet.py`.

Checks include:
- Inventory: discover Parquet outputs and pair core/text sidecars by month.
- Row counts: compare Parquet counts (core vs text; optionally vs .zst newline counts).
- Schema validation: required columns, expected Arrow types, and timestamp unit/tz.
- Corruption sniff: scan text fields for embedded null bytes ("\\x00").
- Value sanity spot checks: sample rows (DuckDB) for ID prefixes and timestamp-month coherence.
- JSON-string probe: sample-parse JSON-serialized fields (e.g., `content_categories`, `media`).

Command-line interface
----------------------
--root                    Root directory containing 'comments/' and 'submissions/' .zst dumps.
--parquet-root            Root directory containing Parquet outputs.
--year                    Year to validate (matches RC_<year>-*.zst and RS_<year>-*.zst).

--checks                  Which checks to run (space-separated):
                          inventory rowcounts schema sniff spot json
--no-zst                  Skip comparisons against .zst newline counts.
--zst-chunk-bytes         Chunk size for streaming .zst newline counting.

Sampling & limits
-----------------
--seed                    RNG seed for deterministic sampling.
--months-limit            Limit the number of months processed (useful for quick tests).
--sample-per-month        Rows sampled per month for spot checks (DuckDB).
--sniff-files             Number of text Parquet files to scan for embedded null bytes.
--sniff-max-rows          Maximum number of rows scanned per sniffed file (None for all).
--json-max-records        Max rows read per month for JSON probe (None for all).
--json-sample-values      Values per JSON column to attempt to parse per month.

Quick-run / smoke test
----------------------
--smoke                   Reduce IO for fast validation (fewer months, fewer samples).

--log-level               DEBUG/INFO/WARNING/ERROR/CRITICAL.

Environment variables (optional)
--------------------------------
PARQUET_ROOT="parquet"    Alternative default for --parquet-root (resolved to absolute path).

Inputs
------
Zstandard-compressed NDJSON (Pushshift-style monthly dumps):
  <root>/comments/RC_<year>-MM.zst
  <root>/submissions/RS_<year>-MM.zst

Parquet outputs from `01_convert_to_parquet.py` (either layout supported):
  <parquet-root>/comments/*.parquet
  <parquet-root>/submissions/*.parquet
  <parquet-root>/parquet/comments/*.parquet
  <parquet-root>/parquet/submissions/*.parquet

Outputs
-------
This script does not modify data. It writes validation results to stdout/stderr
(via logging). Optionally redirect logs to a file, e.g.:
  python 02_pushshift_validation.py ... > validation.log 2>&1

Examples
--------
Run the default validation suite (inventory + schema + corruption sniff + spot checks + JSON probe):
  python 02_pushshift_validation.py \
    --root reddit_2020_2022/reddit \
    --parquet-root parquet \
    --year 2021 \
    --checks inventory schema sniff spot json \
    --log-level INFO

Include row count comparisons against the original .zst files (slower):
  python 02_pushshift_validation.py \
    --root reddit_2020_2022/reddit \
    --parquet-root parquet \
    --year 2021 \
    --checks rowcounts \
    --log-level INFO

Smoke test (fast CI-style run):
  python 02_pushshift_validation.py \
    --root reddit_2020_2022/reddit \
    --parquet-root parquet \
    --year 2021 \
    --smoke \
    --log-level INFO
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import zstandard as zstd


# ----------------------------- Constants & Regex -----------------------------

MONTH_RE = re.compile(r"(?<!\d)(20\d{2})[-_](0[1-9]|1[0-2])(?!\d)")


# ----------------------------- Data Structures ------------------------------


@dataclass(frozen=True)
class MonthFiles:
    """Paths to core/text parquet files for a given month."""

    core: Optional[str] = None
    text: Optional[str] = None


Inventory = Dict[str, MonthFiles]  # month "YYYY-MM" -> MonthFiles


# ----------------------------- Utility Functions ----------------------------


def month_of(path: str) -> str:
    """
    Extract 'YYYY-MM' month token from a path using a tolerant regex that accepts
    either '-' or '_' separators anywhere in the filename.

    Returns 'UNKNOWN' if no token is found.
    """
    m = MONTH_RE.search(path)
    return f"{m.group(1)}-{m.group(2)}" if m else "UNKNOWN"


def parquet_rowcount(path: str) -> int:
    """Return Parquet file row count from metadata (fast; does not load data)."""
    return pq.ParquetFile(path).metadata.num_rows


def read_schema(path: str) -> pa.Schema:
    """Read Arrow schema from a Parquet file."""
    return pq.ParquetFile(path).schema_arrow


def assert_arrow_ts(
    field: pa.Field, must_have_tz: str = "UTC", unit: str = "us"
) -> Tuple[bool, str]:
    """
    Validate that an Arrow field is a timestamp with expected unit and timezone.

    Returns (ok, message).
    """
    t = field.type
    if not pa.types.is_timestamp(t):
        return False, f"{field.name} not timestamp (got {t})"
    if t.tz != must_have_tz or t.unit != unit:
        return (
            False,
            f"{field.name} timestamp unit/tz mismatch (got unit={t.unit}, tz={t.tz})",
        )
    return True, ""


# ----------------------------- Zstandard / ZST ------------------------------


def zst_files(root: str, kind: str, year: str) -> List[str]:
    """
    List expected .zst files for a given kind/year.

    Expected layouts supported:
      - {root}/comments/RC_YYYY-MM.zst
      - {root}/submissions/RS_YYYY-MM.zst
      - {root}/RC_YYYY-MM.zst  (flat fallback)
      - {root}/RS_YYYY-MM.zst  (flat fallback)
    """
    prefix = "RC" if kind == "comments" else "RS"
    paths = sorted(glob.glob(os.path.join(root, kind, f"{prefix}_{year}-*.zst")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(root, f"{prefix}_{year}-*.zst")))
    return paths


def count_zst_lines(path: str, chunk_bytes: int = 1 << 20) -> int:
    """
    Count newline characters in a .zst NDJSON file by streaming decompression.

    Note: this is an approximation to number of records and assumes one JSON per line.
    """
    n = 0
    dctx = zstd.ZstdDecompressor(max_window_size=2 ** 31)
    with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
        while True:
            chunk = reader.read(chunk_bytes)
            if not chunk:
                break
            n += chunk.count(b"\n")
    return n


# ----------------------------- Parquet Discovery ----------------------------


def find_month_files(
    parquet_root: str, kind: str, year: str, log: logging.Logger
) -> Inventory:
    """
    Discover core/text parquet files by month for a given kind.

    Supports both:
      - {parquet_root}/parquet/{kind}/*_{YYYY-MM}.parquet
      - {parquet_root}/{kind}/*_{YYYY-MM}.parquet

    Accepts either '-' or '_' separators in the month token.
    """
    roots = [
        os.path.join(parquet_root, "parquet", kind),
        os.path.join(parquet_root, kind),
    ]

    patterns = []
    for r in roots:
        patterns.append(os.path.join(r, f"*_{year}-*.parquet"))
        patterns.append(os.path.join(r, f"*_{year}_*.parquet"))

    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))

    # Pair by detected month
    bym: Dict[str, Dict[str, Optional[str]]] = defaultdict(
        lambda: {"core": None, "text": None}
    )
    for p in files:
        mm = month_of(p)
        if mm == "UNKNOWN":
            continue
        base = os.path.basename(p)

        if kind == "comments":
            if "comments_core" in base:
                bym[mm]["core"] = p
            elif "comments_body" in base:
                bym[mm]["text"] = p
        else:
            if "submissions_core" in base:
                bym[mm]["core"] = p
            elif "submissions_selftext" in base:
                bym[mm]["text"] = p

    inv: Inventory = {
        m: MonthFiles(core=d["core"], text=d["text"])
        for m, d in bym.items()
        if d["core"] or d["text"]
    }

    # Helpful preview
    preview_lines = []
    for m in sorted(inv):
        mf = inv[m]
        if mf.core:
            preview_lines.append(f"{kind} {m}: {mf.core} (core)")
        if mf.text:
            preview_lines.append(f"{kind} {m}: {mf.text} (text)")
    for line in preview_lines[:24]:
        log.info(line)
    if len(preview_lines) > 24:
        log.info("... (%d more parquet paths)", len(preview_lines) - 24)

    return inv


# ----------------------------- Schema Checks --------------------------------


def check_core_comments_schema(s: pa.Schema) -> Tuple[List[str], List[str]]:
    """
    Validate expected schema for comments core files.

    Ensures required fields exist with expected types and that 'body' is NOT present.
    """
    required = {
        "author": pa.string(),
        "author_flair_text": pa.string(),
        "author_created_utc": pa.timestamp("us", tz="UTC"),
        "author_premium": pa.bool_(),
        "can_mod_post": pa.bool_(),
        "controversiality": pa.int64(),
        "created_utc": pa.timestamp("us", tz="UTC"),
        "distinguished": pa.string(),
        "id": pa.string(),
        "is_submitter": pa.bool_(),
        "link_id": pa.string(),
        "parent_id": pa.string(),
        "removal_reason": pa.string(),
        "score": pa.int64(),
        "stickied": pa.bool_(),
        "subreddit": pa.string(),
        "subreddit_id": pa.string(),
        "subreddit_type": pa.string(),
    }

    miss, wrong = [], []
    for k, t in required.items():
        if k not in s.names:
            miss.append(k)
            continue
        if k in ("created_utc", "author_created_utc"):
            ok, msg = assert_arrow_ts(s.field(k))
            if not ok:
                wrong.append(msg)
        else:
            if s.field(k).type != t:
                wrong.append(f"{k} -> {s.field(k).type} (expected {t})")

    if "body" in s.names:
        wrong.append("core file unexpectedly contains 'body'")

    return miss, wrong


def check_text_comments_schema(s: pa.Schema) -> Tuple[List[str], List[str]]:
    """Validate expected schema for comments text files: (id, body)."""
    need = {"id": pa.string(), "body": pa.string()}
    miss, wrong = [], []
    for k, t in need.items():
        if k not in s.names:
            miss.append(k)
        elif s.field(k).type != t:
            wrong.append(f"{k} -> {s.field(k).type} (expected {t})")
    return miss, wrong


def check_core_subm_schema(s: pa.Schema) -> Tuple[List[str], List[str]]:
    """
    Validate expected schema for submissions core files.

    Ensures required fields exist with expected types and that 'selftext' is NOT present.
    """
    required = {
        "author": pa.string(),
        "author_created_utc": pa.timestamp("us", tz="UTC"),
        "author_flair_text": pa.string(),
        "author_premium": pa.bool_(),
        "content_categories": pa.string(),  # JSON-serialized
        "contest_mode": pa.bool_(),
        "created_utc": pa.timestamp("us", tz="UTC"),
        "distinguished": pa.string(),
        "domain": pa.string(),
        "id": pa.string(),
        "is_crosspostable": pa.bool_(),
        "is_original_content": pa.bool_(),
        "is_reddit_media_domain": pa.bool_(),
        "is_robot_indexable": pa.bool_(),
        "is_self": pa.bool_(),
        "is_video": pa.bool_(),
        "link_flair_text": pa.string(),
        "media": pa.string(),  # JSON-serialized
        "name": pa.string(),
        "num_comments": pa.int64(),
        "num_crossposts": pa.int64(),
        "over_18": pa.bool_(),
        "removed_by_category": pa.string(),
        "score": pa.int64(),
        "spoiler": pa.bool_(),
        "subreddit": pa.string(),
        "subreddit_id": pa.string(),
        "subreddit_subscribers": pa.int64(),
        "subreddit_type": pa.string(),
        "suggested_sort": pa.string(),
        "title": pa.string(),
        "total_awards_received": pa.int64(),
        "upvote_ratio": pa.float64(),
        "url": pa.string(),
        "crosspost_parent": pa.string(),
    }

    miss, wrong = [], []
    for k, t in required.items():
        if k not in s.names:
            miss.append(k)
            continue
        if k in ("created_utc", "author_created_utc"):
            ok, msg = assert_arrow_ts(s.field(k))
            if not ok:
                wrong.append(msg)
        else:
            if s.field(k).type != t:
                wrong.append(f"{k} -> {s.field(k).type} (expected {t})")

    if "selftext" in s.names:
        wrong.append("core file unexpectedly contains 'selftext'")

    return miss, wrong


def check_text_subm_schema(s: pa.Schema) -> Tuple[List[str], List[str]]:
    """Validate expected schema for submissions text files: (id, selftext)."""
    need = {"id": pa.string(), "selftext": pa.string()}
    miss, wrong = [], []
    for k, t in need.items():
        if k not in s.names:
            miss.append(k)
        elif s.field(k).type != t:
            wrong.append(f"{k} -> {s.field(k).type} (expected {t})")
    return miss, wrong


# ----------------------------- Content Checks -------------------------------


def scan_embedded_nulls(
    path: str, columns: Sequence[str], limit_rows: Optional[int], sample: int,
) -> Dict[str, object]:
    """
    Scan selected string columns for embedded null bytes ("\\x00") which can indicate
    decode/serialization issues.

    Reads with PyArrow, optionally slices to limit_rows, then samples up to `sample`
    values per column.

    Returns a dict with keys: checked, bad, examples.
    """
    out: Dict[str, object] = {"checked": 0, "bad": 0, "examples": []}
    try:
        tbl = pq.read_table(path, columns=list(columns))
        if limit_rows is not None and tbl.num_rows > limit_rows:
            tbl = tbl.slice(0, limit_rows)
    except Exception as e:
        out["examples"] = [{"error": f"read_table failed: {e}"}]
        return out

    out["checked"] = tbl.num_rows
    examples: List[Dict[str, str]] = []
    bad = 0

    for c in columns:
        if c not in tbl.column_names:
            continue
        col = tbl[c]
        n = min(sample, len(col))
        for i in range(n):
            v = col[i].as_py()
            if isinstance(v, str) and ("\x00" in v):
                bad += 1
                if len(examples) < 5:
                    examples.append({"col": c, "val": v[:120]})

    out["bad"] = bad
    out["examples"] = examples
    return out


def scan_embedded_nulls_streaming(
    path: str,
    columns: Sequence[str],
    limit_rows: Optional[int],
    batch_size: int = 8192,
    max_examples: int = 5,
) -> Dict[str, object]:
    """
    Stream-scan selected string columns for embedded null bytes ("\\x00") without
    loading the entire Parquet file into memory.

    Reads row batches via ParquetFile.iter_batches(columns=...), stopping after
    `limit_rows` rows if provided.

    Returns:
      {
        "checked": int,
        "bad": int,
        "examples": [{"col": str, "val": str}, ...] or [{"error": "..."}]
      }
    """
    out: Dict[str, object] = {"checked": 0, "bad": 0, "examples": []}
    examples: List[Dict[str, str]] = []
    bad_total = 0
    checked_total = 0

    try:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=list(columns), batch_size=batch_size):
            if limit_rows is not None and checked_total >= limit_rows:
                break

            # If we would exceed the limit, slice this batch down
            if limit_rows is not None:
                remaining = limit_rows - checked_total
                if batch.num_rows > remaining:
                    batch = batch.slice(0, remaining)

            checked_total += batch.num_rows

            # Scan each requested column in this batch
            for c in columns:
                if c not in batch.schema.names:
                    continue

                arr = batch.column(batch.schema.get_field_index(c))

                # Only string-like columns are relevant
                if not (
                    pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type)
                ):
                    continue

                # Efficient boolean mask where substring "\x00" occurs
                mask = pc.match_substring(arr, pattern="\x00")
                n_bad = int(pc.sum(mask).as_py())  # sum(True) counts matches
                if n_bad == 0:
                    continue

                bad_total += n_bad

                # Collect a few examples (avoid full conversion)
                if len(examples) < max_examples:
                    idx = pc.indices_nonzero(mask)
                    # idx is an Int32/Int64 array of positions
                    for j in range(min(len(idx), max_examples - len(examples))):
                        pos = int(idx[j].as_py())
                        v = arr[pos].as_py()
                        if isinstance(v, str):
                            examples.append({"col": c, "val": v[:120]})

        out["checked"] = checked_total
        out["bad"] = bad_total
        out["examples"] = examples
        return out

    except Exception as e:
        out["examples"] = [{"error": f"stream scan failed: {e}"}]
        return out


def duckdb_sample_rows(paths: Sequence[str], n_rows: int, memory_limit: str = "4GB"):
    """
    Sample `n_rows` from one or more parquet files using DuckDB.

    Uses DuckDB `USING SAMPLE {n_rows} ROWS` (reservoir-like sampling).
    Returns a pandas DataFrame.
    """
    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit='{memory_limit}'")
    plist = "[" + ", ".join(repr(p) for p in paths) + "]"
    q = f"SELECT * FROM read_parquet({plist}, union_by_name=True) USING SAMPLE {n_rows} ROWS;"
    return con.execute(q).df()


def json_string_probe(
    core_path: str,
    columns: Sequence[str],
    max_records: Optional[int],
    sample_values: int,
) -> Dict[str, object]:
    """
    Validate that selected columns contain JSON-parseable strings (when non-null).

    Returns dict keyed by column with counts and examples.
    """
    out: Dict[str, object] = {}
    tbl = pq.read_table(core_path, columns=list(columns))
    if max_records is not None and tbl.num_rows > max_records:
        tbl = tbl.slice(0, max_records)

    for col in columns:
        if col not in tbl.column_names:
            out[col] = {"present": False}
            continue

        arr = tbl[col]
        bad = 0
        examples: List[str] = []
        for i in range(min(sample_values, len(arr))):
            v = arr[i].as_py()
            if v is None:
                continue
            try:
                json.loads(v)
            except Exception:
                bad += 1
                if len(examples) < 3:
                    examples.append(str(v)[:120])

        out[col] = {"present": True, "bad": bad, "examples": examples}

    return out


# ----------------------------- High-level Checks ----------------------------


def check_inventory(
    inv_comments: Inventory, inv_subm: Inventory, log: logging.Logger
) -> None:
    """Log a high-level inventory summary."""
    log.info("=== INVENTORY ===")
    log.info("comments months: %d", len(inv_comments))
    log.info("submissions months: %d", len(inv_subm))


def check_rowcounts(
    inv: Inventory,
    kind: str,
    root: str,
    year: str,
    zst_chunk: int,
    log: logging.Logger,
    do_zst: bool,
) -> None:
    """
    Print per-month row count report for core/text parquet, and optionally compare to .zst newline counts.
    """
    log.info("=== ROW COUNTS: %s ===", kind)

    pq_month_core: Dict[str, int] = {}
    pq_month_text: Dict[str, int] = {}

    for m, d in inv.items():
        if d.core:
            pq_month_core[m] = parquet_rowcount(d.core)
        if d.text:
            pq_month_text[m] = parquet_rowcount(d.text)

    zsts = {month_of(p): p for p in zst_files(root, kind, year)} if do_zst else {}

    header = (
        f"{'Month':<8} {'Core':>12} {'Text':>12} {'Δ(core-text)':>14} "
        f"{'ZST lines':>12} {'Δ(core-zst)':>14} {'Ratio':>7}"
    )
    log.info(header)
    log.info("-" * len(header))

    for m in sorted(inv):
        core = pq_month_core.get(m, 0)
        text = pq_month_text.get(m, 0)

        dct = core - text if (core and text) else None

        zc = None
        dz = None
        ratio = None
        if do_zst and m in zsts and os.path.exists(zsts[m]):
            zc = count_zst_lines(zsts[m], chunk_bytes=zst_chunk)
            dz = core - zc
            ratio = (core / zc) if zc else None

        log.info(
            "%-8s %12s %12s %14s %12s %14s %7s",
            m,
            f"{core:,}" if core else "0",
            f"{text:,}" if text else "0",
            "" if dct is None else f"{dct:+,}",
            "" if zc is None else f"{zc:,}",
            "" if dz is None else f"{dz:+,}",
            "" if ratio is None else f"{ratio:.3f}",
        )


def check_schema(inv: Inventory, kind: str, log: logging.Logger) -> None:
    """
    Validate schemas for all available core/text parquet files in inventory.
    Logs missing columns and type mismatches.
    """
    log.info("=== SCHEMA VALIDATION: %s ===", kind)
    miss_cols: Counter = Counter()
    wrong_types: Counter = Counter()

    for m, d in inv.items():
        if d.core and os.path.exists(d.core):
            s = read_schema(d.core)
            if kind == "comments":
                miss, wrong = check_core_comments_schema(s)
            else:
                miss, wrong = check_core_subm_schema(s)
            for k in miss:
                miss_cols[(kind, "core", k)] += 1
            for w in wrong:
                wrong_types[(kind, "core", w)] += 1

        if d.text and os.path.exists(d.text):
            s = read_schema(d.text)
            if kind == "comments":
                miss, wrong = check_text_comments_schema(s)
            else:
                miss, wrong = check_text_subm_schema(s)
            for k in miss:
                miss_cols[(kind, "text", k)] += 1
            for w in wrong:
                wrong_types[(kind, "text", w)] += 1

    if miss_cols:
        log.warning("Missing columns:")
        for (k, part, col), cnt in sorted(miss_cols.items()):
            log.warning("  %s/%s: %s -> %d files", k, part, col, cnt)
    else:
        log.info("No missing required columns.")

    if wrong_types:
        log.warning("Type mismatches or unexpected cols:")
        for (k, part, msg), cnt in sorted(wrong_types.items()):
            log.warning("  %s/%s: %s -> %d files", k, part, msg, cnt)
    else:
        log.info("No type/timestamp mismatches.")


def check_sniff_nulls(
    inv: Inventory,
    kind: str,
    text_column: str,
    sample_files: int,
    limit_rows: Optional[int],
    rng: random.Random,
    log: logging.Logger,
) -> None:

    """
    Run embedded-null sniff on a random subset of text parquet files.
    """
    log.info("=== CORRUPTION SNIFF (embedded nulls): %s ===", kind)

    text_paths = [d.text for d in inv.values() if d.text and os.path.exists(d.text)]
    if not text_paths:
        log.warning("No text files found for %s; skipping sniff.", kind)
        return

    k = min(sample_files, len(text_paths))
    picks = rng.sample(text_paths, k=k)

    for p in picks:
        res = scan_embedded_nulls_streaming(
            p,
            columns=[text_column],
            limit_rows=limit_rows,
            batch_size=8192,
            max_examples=5,
        )

        log.info(
            "%s_text %s: checked=%s bad=%s examples=%s",
            kind,
            os.path.basename(p),
            res["checked"],
            res["bad"],
            res["examples"],
        )


def check_spot(
    inv: Inventory,
    kind: str,
    sample_per_month: int,
    months_limit: Optional[int],
    rng: random.Random,
    log: logging.Logger,
) -> None:
    """
    Spot-check sampled rows for basic sanity conditions using DuckDB.

    Checks include:
    - created_utc month token vs file month (logged as note)
    - ID prefix conventions (t1_/t3_) and parent/link id prefixes for comments
    - non-empty title for submissions
    """
    import pandas as pd  # local import to keep dependency optional unless this check runs

    log.info("=== VALUE SANITY SPOT CHECKS: %s ===", kind)

    months = sorted(inv.keys())
    if months_limit is not None:
        months = months[:months_limit]

    for m in months:
        mf = inv[m]
        path = mf.core or mf.text
        if not path:
            continue

        try:
            df = duckdb_sample_rows([path], sample_per_month)
        except Exception as e:
            log.error("%s %s: DuckDB read ERROR: %s", kind, m, e)
            continue

        bad = 0
        for _, r in df.iterrows():
            notes = []

            if "created_utc" in df.columns and not pd.isna(r.get("created_utc", None)):
                dt = r["created_utc"]
                mm = f"{dt.year:04d}-{dt.month:02d}"
                if mm != m:
                    notes.append(f"created_utc month {mm} != file {m}")

            if kind == "comments":
                if isinstance(r.get("id"), str) and not r["id"].startswith("t1_"):
                    notes.append("comments.id not t1_")
                if isinstance(r.get("link_id"), str) and not r["link_id"].startswith(
                    "t3_"
                ):
                    notes.append("link_id not t3_")
                if isinstance(r.get("parent_id"), str) and not (
                    r["parent_id"].startswith("t1_") or r["parent_id"].startswith("t3_")
                ):
                    notes.append("parent_id not t1_/t3_")
            else:
                if isinstance(r.get("id"), str) and not r["id"].startswith("t3_"):
                    notes.append("submissions.id not t3_")
                if isinstance(r.get("title"), str) and len(r["title"]) == 0:
                    notes.append("empty title")

            if notes:
                bad += 1
                log.warning(
                    "%s %s: %s :: id=%s subreddit=%s score=%s",
                    kind,
                    m,
                    ", ".join(notes),
                    r.get("id"),
                    r.get("subreddit"),
                    r.get("score"),
                )

        if bad == 0:
            log.info("%s %s: OK in sampled rows", kind, m)


def check_json_probe(
    inv_subm: Inventory,
    max_records: Optional[int],
    sample_values: int,
    months_limit: Optional[int],
    log: logging.Logger,
) -> None:
    """
    Probe JSON-string fields in submissions core files: `content_categories`, `media`.
    """
    log.info("=== JSON STRING PROBE: submissions ===")
    months = sorted(inv_subm.keys())
    if months_limit is not None:
        months = months[:months_limit]

    for m in months:
        mf = inv_subm[m]
        if not mf.core:
            continue
        try:
            res = json_string_probe(
                mf.core,
                columns=["content_categories", "media"],
                max_records=max_records,
                sample_values=sample_values,
            )
        except Exception as e:
            log.error("%s: JSON probe failed to read/scan core parquet: %s", m, e)
            continue

        for col, info in res.items():
            if not info.get("present", False):
                log.warning("%s %s: column not present", m, col)
                continue
            bad = info["bad"]
            if bad:
                log.warning(
                    "%s %s: %d malformed JSON strings in sample; examples=%s",
                    m,
                    col,
                    bad,
                    info["examples"],
                )
            else:
                log.info("%s %s: OK in sample", m, col)


# ----------------------------- CLI / Main -----------------------------------


def build_argparser() -> argparse.ArgumentParser:
    """Construct CLI parser."""
    p = argparse.ArgumentParser(
        description="Validate Pushshift .zst -> Parquet conversion outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--root",
        default="reddit_2020_2022/reddit",
        help="Root containing comments/ and submissions/ .zst files.",
    )
    p.add_argument(
        "--parquet-root",
        default=os.path.abspath(os.environ.get("PARQUET_ROOT", "parquet")),
        help="Root containing parquet outputs (supports {parquet-root}/parquet/{kind}/...).",
    )
    p.add_argument("--year", default="2021", help="Year to validate (YYYY).")

    p.add_argument("--seed", type=int, default=123, help="RNG seed for sampling.")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    # Rowcount vs ZST
    p.add_argument(
        "--zst-chunk-bytes",
        type=int,
        default=1 << 20,
        help="Chunk size for .zst line counting.",
    )
    p.add_argument(
        "--no-zst",
        action="store_true",
        help="Skip rowcount comparisons against .zst newline counts.",
    )

    # Sampling & limits
    p.add_argument(
        "--sample-per-month",
        type=int,
        default=6,
        help="Rows to sample per month for spot-check.",
    )
    p.add_argument(
        "--sniff-files",
        type=int,
        default=6,
        help="Number of text Parquet files to scan for embedded null bytes.",
    )
    p.add_argument(
        "--sniff-max-rows",
        type=int,
        default=200_000,
        help="Maximum number of rows scanned per sniffed file (None for all).",
    )

    p.add_argument(
        "--json-max-records",
        type=int,
        default=50_000,
        help="Max rows to load per month for JSON probe (None for all).",
    )
    p.add_argument(
        "--json-sample-values",
        type=int,
        default=1000,
        help="Values per JSON column to attempt to parse.",
    )

    p.add_argument(
        "--months-limit",
        type=int,
        default=None,
        help="Limit number of months processed (useful for quick tests).",
    )

    # Check selection
    p.add_argument(
        "--checks",
        nargs="+",
        default=["inventory", "schema", "sniff", "spot", "json"],
        choices=["inventory", "rowcounts", "schema", "sniff", "spot", "json"],
        help="Which checks to run.",
    )

    # Smoke test preset
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Quick-run mode: fewer months, fewer samples, lighter IO.",
    )
    return p


def apply_smoke_preset(args: argparse.Namespace) -> None:
    """
    Apply a conservative preset for fast CI / smoke testing.
    Keeps semantics the same but reduces IO.
    """
    args.months_limit = args.months_limit or 2
    args.sample_per_month = min(args.sample_per_month, 3)
    args.sniff_files = min(args.sniff_files, 2)
    args.sniff_max_rows = (
        min(args.sniff_max_rows, 50_000) if args.sniff_max_rows is not None else 50_000
    )

    args.json_max_records = (
        min(args.json_max_records, 10_000)
        if args.json_max_records is not None
        else 10_000
    )
    args.json_sample_values = min(args.json_sample_values, 300)


def main() -> int:
    args = build_argparser().parse_args()
    if args.smoke:
        apply_smoke_preset(args)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    log = logging.getLogger("pushshift_validation")

    rng = random.Random(args.seed)

    # 1) Inventory
    inv_comments = find_month_files(args.parquet_root, "comments", args.year, log)
    inv_subm = find_month_files(args.parquet_root, "submissions", args.year, log)

    if "inventory" in args.checks:
        check_inventory(inv_comments, inv_subm, log)

    # 2) Rowcounts (optional)
    if "rowcounts" in args.checks:
        check_rowcounts(
            inv_comments,
            "comments",
            args.root,
            args.year,
            args.zst_chunk_bytes,
            log,
            do_zst=not args.no_zst,
        )
        check_rowcounts(
            inv_subm,
            "submissions",
            args.root,
            args.year,
            args.zst_chunk_bytes,
            log,
            do_zst=not args.no_zst,
        )

    # 3) Schema
    if "schema" in args.checks:
        check_schema(inv_comments, "comments", log)
        check_schema(inv_subm, "submissions", log)

    # 4) Corruption sniff
    if "sniff" in args.checks:
        check_sniff_nulls(
            inv_comments,
            "comments",
            "body",
            sample_files=args.sniff_files,
            limit_rows=args.sniff_max_rows,
            rng=rng,
            log=log,
        )

        check_sniff_nulls(
            inv_subm,
            "submissions",
            "selftext",
            sample_files=args.sniff_files,
            limit_rows=args.sniff_max_rows,
            rng=rng,
            log=log,
        )

    # 5) Spot checks
    if "spot" in args.checks:
        check_spot(
            inv_comments, "comments", args.sample_per_month, args.months_limit, rng, log
        )
        check_spot(
            inv_subm, "submissions", args.sample_per_month, args.months_limit, rng, log
        )

    # 6) JSON probe
    if "json" in args.checks:
        check_json_probe(
            inv_subm,
            args.json_max_records,
            args.json_sample_values,
            args.months_limit,
            log,
        )

    log.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
