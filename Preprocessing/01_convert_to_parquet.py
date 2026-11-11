# MIT License
# Copyright (c) 2026 Cara Lynch
# See the LICENSE file for details.
"""
Preprocessing step 1: convert Reddit .zst (NDJSON) dumps to Parquet.

This script ingests monthly Reddit comment/submission dumps stored as Zstandard-
compressed NDJSON (.zst), performs light normalization, and writes Parquet
outputs (core + text sidecars) per month.

Command-line interface
----------------------
--input-root              Root directory containing 'comments/' and 'submissions/'.
--out-dir                 Output directory (creates 'comments/' and 'submissions/' within it).
--year                    Year to process (matches RC_<year>-*.zst and RS_<year>-*.zst).
--only-months             Optional: comma-separated list of YYYY-MM months to keep (e.g. "2021-06,2021-07").
--limit-files             Optional: limit number of monthly files per type (smoke tests).
--batch-rows              Rows to buffer before writing to Parquet.
--row-group-rows          Parquet row group size.
--compression-level-core  Zstd compression level for core outputs (default: 12).
--compression-level-text  Zstd compression level for text sidecar outputs (default: 15).
--max-records             Optional: stop after reading N records per input file (0 = no limit).
--max-batches             Optional: stop after writing N batches per input file (0 = no limit).
--jobs                    Number of worker processes.
--comments-only           Process comments only (skip submissions).
--submissions-only        Process submissions only (skip comments).
--log-level               DEBUG/INFO/WARNING/ERROR/CRITICAL.

Environment variables (optional)
--------------------------------
ONLY_MONTHS="2021-06,2021-07"   Alternative default for --only-months.
LIMIT_FILES=3                  Alternative default for --limit-files.
JOBS=8                         Alternative default for --jobs.

Inputs
------
<input-root>/comments/RC_<year>-MM.zst
<input-root>/submissions/RS_<year>-MM.zst

Outputs
-------
Comments (per month):
  <out-dir>/comments/comments_core_YYYY-MM.parquet
  <out-dir>/comments/comments_body_YYYY-MM.parquet

Submissions (per month):
  <out-dir>/submissions/submissions_core_YYYY-MM.parquet
  <out-dir>/submissions/submissions_selftext_YYYY-MM.parquet

Examples
--------
Submissions-only smoke test (first 200k records):
  python 01_convert_to_parquet.py \
    --input-root "inputs/reddit/2021" \
    --out-dir "$TMPDIR/reddit_parquet_test" \
    --year 2021 \
    --submissions-only \
    --limit-files 1 \
    --max-records 200000 \
    --compression-level-core 1 \
    --compression-level-text 1 \
    --batch-rows 5000 \
    --row-group-rows 5000 \
    --jobs 1 \
    --log-level INFO

Process specific months:
  python 01_convert_to_parquet.py \
    --input-root reddit_2020_2022/reddit \
    --out-dir parquet \
    --year 2021 \
    --only-months "2021-06,2021-07" \
    --jobs 8
"""


import os
import re
import glob
import json
import argparse
import logging
import multiprocessing
from typing import Dict, List, Optional, Tuple, Any, Iterator

import orjson
import zstandard as zstd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from concurrent.futures import ProcessPoolExecutor, as_completed

LOG = logging.getLogger(__name__)

# ---------- DEFAULTS (overridable via CLI) ----------
DEFAULT_INPUT_ROOT = "reddit_2020_2022/reddit"
DEFAULT_OUT_DIR = "parquet"  # will create {out_dir}/comments and {out_dir}/submissions

DEFAULT_BATCH_ROWS = 50_000
DEFAULT_ROW_GROUP_ROWS = 50_000

# plausible Unix seconds bounds (inclusive)
EPOCH_MIN_S = 1104537600  # 2005-01-01 00:00:00 UTC
EPOCH_MAX_S = 1757793065  # 2025-09-13 19:51:05 UTC


# **NOTE**: permalink & author_fullname removed from both keep lists
KEEP_COMMENTS = [
    "author",
    "author_flair_text",
    "author_created_utc",
    "author_premium",
    "body",
    "can_mod_post",
    "controversiality",
    "created_utc",
    "distinguished",
    "id",
    "is_submitter",
    "link_id",
    "parent_id",
    "removal_reason",
    "score",
    "stickied",
    "subreddit",
    "subreddit_id",
    "subreddit_type",  # 'permalink' dropped
]

KEEP_SUBMISSIONS = [
    "author",
    "author_created_utc",
    "author_flair_text",  # 'author_fullname' dropped
    "author_premium",
    "content_categories",
    "contest_mode",
    "created_utc",
    "distinguished",
    "domain",
    "id",
    "is_crosspostable",
    "is_original_content",
    "is_reddit_media_domain",
    "is_robot_indexable",
    "is_self",
    "is_video",
    "link_flair_text",
    "media",
    "name",
    "num_comments",
    "num_crossposts",
    "over_18",  # 'permalink' dropped
    "removed_by_category",
    "score",
    "selftext",
    "spoiler",
    "subreddit",
    "subreddit_id",
    "subreddit_subscribers",
    "subreddit_type",
    "suggested_sort",
    "title",
    "total_awards_received",
    "upvote_ratio",
    "url",
    "crosspost_parent",
]

# ---------- UTILS ----------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Parameters
    ----------
    argv:
        Optional list of arguments (for testing). If None, uses sys.argv.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    p = argparse.ArgumentParser(
        description="Convert Reddit monthly .zst NDJSON dumps into Parquet (core + text sidecars)."
    )
    p.add_argument(
        "--input-root",
        default=DEFAULT_INPUT_ROOT,
        help="Root directory containing 'comments/' and 'submissions/' subfolders.",
    )
    p.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Output root directory. Will create 'comments/' and 'submissions/' inside.",
    )
    p.add_argument(
        "--year",
        type=int,
        default=2021,
        help="Year to process (used for glob patterns RC_<year>-*.zst / RS_<year>-*.zst).",
    )

    p.add_argument(
        "--only-months",
        default=os.environ.get("ONLY_MONTHS", ""),
        help='Comma-separated months to keep, e.g. "2021-06,2021-07". Overrides env ONLY_MONTHS.',
    )
    p.add_argument(
        "--limit-files",
        type=int,
        default=int(os.environ.get("LIMIT_FILES", "0")),
        help="Limit number of monthly files per type (0 = no limit). Overrides env LIMIT_FILES.",
    )

    p.add_argument(
        "--batch-rows",
        type=int,
        default=DEFAULT_BATCH_ROWS,
        help="Rows per in-memory batch before writing to Parquet.",
    )
    p.add_argument(
        "--row-group-rows",
        type=int,
        default=DEFAULT_ROW_GROUP_ROWS,
        help="Row group size for Parquet writing (tradeoff: metadata vs scan efficiency).",
    )

    p.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Stop after reading this many records from EACH input file (0 = no limit). "
        "Useful for smoke tests.",
    )

    p.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Stop after writing this many batches from EACH input file (0 = no limit). "
        "Useful for smoke tests.",
    )

    p.add_argument(
        "--compression-level-core",
        type=int,
        default=12,
        help="Zstd compression level for core Parquet files (metadata-focused tables).",
    )
    p.add_argument(
        "--compression-level-text",
        type=int,
        default=15,
        help="Zstd compression level for text sidecar Parquet files (body/selftext).",
    )

    default_jobs = max(1, multiprocessing.cpu_count() // 2)
    p.add_argument(
        "--jobs",
        type=int,
        default=int(os.environ.get("JOBS", default_jobs)),
        help="Number of worker processes (default: half your CPU cores).",
    )

    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--comments-only",
        action="store_true",
        help="Process comments only (skip submissions).",
    )
    group.add_argument(
        "--submissions-only",
        action="store_true",
        help="Process submissions only (skip comments).",
    )

    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return p.parse_args(argv)


def _log_config(
    args: argparse.Namespace, out_dir_comments: str, out_dir_submissions: str
) -> None:
    LOG.info(
        "Resolved config: input_root=%s out_dir=%s year=%s jobs=%s "
        "batch_rows=%s row_group_rows=%s max_records=%s max_batches=%s "
        "compression_core=%s compression_text=%s only_months=%s limit_files=%s comments_only=%s submissions_only=%s "
        "out_comments=%s out_submissions=%s",
        args.input_root,
        args.out_dir,
        args.year,
        args.jobs,
        args.batch_rows,
        args.row_group_rows,
        args.max_records,
        args.max_batches,
        args.compression_level_core,
        args.compression_level_text,
        args.only_months,
        args.limit_files,
        args.comments_only,
        args.submissions_only,
        out_dir_comments,
        out_dir_submissions,
    )


def _safe_json_dump(x: Any) -> Any:
    """Serialize lists/dicts as compact JSON; pass through scalars unchanged."""
    if isinstance(x, (list, dict)):
        return json.dumps(x, separators=(",", ":"), ensure_ascii=False)
    return x


def _prefixed_id(raw_id: Optional[str], prefix: str) -> Optional[str]:
    """
    Ensure a Reddit thing ID has the expected prefix (e.g., 't1_' or 't3_').

    Accepts already-prefixed IDs and returns None for None input.
    """
    if raw_id is None:
        return None
    s = str(raw_id)
    if s.startswith(prefix):
        return s
    if re.match(r"^t\d_[0-9a-zA-Z]+$", s):
        return s
    return f"{prefix}{s}"


def _strip_prefix(s: Optional[str]) -> Optional[str]:
    """Remove a leading 't{digit}_' prefix from a Reddit thing ID."""
    if s is None:
        return None
    return re.sub(r"^t\d_", "", str(s))


def _scale_to_seconds_strict(s: pd.Series) -> pd.Series:
    """
    Coerce a numeric-like series to epoch seconds, downscaling ms/us/ns if detected.

    Notes
    -----
    The heuristic divides values > 1e12 by 1e3 up to 3 times.
    """
    s = s.copy()
    s = pd.to_numeric(s, errors="coerce").astype("float64")
    s[~np.isfinite(s)] = np.nan
    for _ in range(3):
        too_big = s > 1.0e12
        if not np.any(too_big):
            break
        s[too_big] = s[too_big] / 1_000.0
    return s


def _to_utc_timestamp(series: pd.Series) -> pd.Series:
    """
    Convert a series of epoch timestamps into UTC pandas datetimes.

    - Coerces to float
    - Downscales ms/us/ns
    - Filters values outside [EPOCH_MIN_S, EPOCH_MAX_S]
    - Uses pd.to_datetime(..., utc=True) with overflow protection
    """

    s = _scale_to_seconds_strict(series)
    ok = (s >= EPOCH_MIN_S) & (s <= EPOCH_MAX_S)
    s_ok = s.where(ok, np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        try:
            out = pd.to_datetime(s_ok, unit="s", utc=True, errors="coerce")
        except FloatingPointError:
            out = pd.Series(pd.NaT, index=s_ok.index, dtype="datetime64[ns, UTC]")
            mask = np.isfinite(s_ok.values)
            out.loc[mask] = pd.to_datetime(
                s_ok.loc[mask].astype("float64"), unit="s", utc=True, errors="coerce"
            )
    return out


def _ensure_columns(record: Dict[str, Any], keep: List[str]) -> Dict[str, Any]:
    """Return a dict containing exactly the keys in `keep`, filled with None if missing."""
    return {k: record.get(k, None) for k in keep}


def _iter_ndjson_from_zst(
    path: str, chunk_size: int = 1 << 24
) -> Iterator[Dict[str, Any]]:
    """
    Stream NDJSON objects from a Zstandard-compressed .zst file.

    Yields
    ------
    dict
        One decoded JSON object per line.
    """
    with open(path, "rb") as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=2 ** 31)
        with dctx.stream_reader(fh) as reader:
            buf = bytearray()
            while True:
                chunk = reader.read(chunk_size)
                if not chunk:
                    break
                buf.extend(chunk)
                *lines, buf = buf.split(b"\n")
                for line in lines:
                    if line:
                        yield orjson.loads(line)
            if buf.strip():
                yield orjson.loads(bytes(buf))


# ---------- SCHEMAS ----------
arrow_ts = pa.timestamp("us", tz="UTC")

# Core (no big text), plus *_len
COMMENTS_CORE_SCHEMA = pa.schema(
    [
        ("author", pa.string()),
        ("author_flair_text", pa.string()),
        ("author_created_utc", arrow_ts),
        ("author_premium", pa.bool_()),
        ("can_mod_post", pa.bool_()),
        ("controversiality", pa.int64()),
        ("created_utc", arrow_ts),
        ("distinguished", pa.string()),
        ("id", pa.string()),  # t1_
        ("is_submitter", pa.bool_()),
        ("link_id", pa.string()),  # t3_...
        ("parent_id", pa.string()),  # t1_/t3_
        ("removal_reason", pa.string()),
        ("score", pa.int64()),
        ("stickied", pa.bool_()),
        ("subreddit", pa.string()),
        ("subreddit_id", pa.string()),
        ("subreddit_type", pa.string()),
    ]
)

COMMENTS_BODY_SCHEMA = pa.schema([("id", pa.string()), ("body", pa.string()),])

SUBMISSIONS_CORE_SCHEMA = pa.schema(
    [
        ("author", pa.string()),
        ("author_created_utc", arrow_ts),
        ("author_flair_text", pa.string()),
        ("author_premium", pa.bool_()),
        ("content_categories", pa.string()),  # JSON-string
        ("contest_mode", pa.bool_()),
        ("created_utc", arrow_ts),
        ("distinguished", pa.string()),
        ("domain", pa.string()),
        ("id", pa.string()),  # t3_
        ("is_crosspostable", pa.bool_()),
        ("is_original_content", pa.bool_()),
        ("is_reddit_media_domain", pa.bool_()),
        ("is_robot_indexable", pa.bool_()),
        ("is_self", pa.bool_()),
        ("is_video", pa.bool_()),
        ("link_flair_text", pa.string()),
        ("media", pa.string()),  # JSON-string
        ("name", pa.string()),
        ("num_comments", pa.int64()),
        ("num_crossposts", pa.int64()),
        ("over_18", pa.bool_()),
        ("removed_by_category", pa.string()),
        ("score", pa.int64()),
        ("spoiler", pa.bool_()),
        ("subreddit", pa.string()),
        ("subreddit_id", pa.string()),
        ("subreddit_subscribers", pa.int64()),
        ("subreddit_type", pa.string()),
        ("suggested_sort", pa.string()),
        ("title", pa.string()),
        ("total_awards_received", pa.int64()),
        ("upvote_ratio", pa.float64()),
        ("url", pa.string()),
        ("crosspost_parent", pa.string()),
    ]
)

SUBMISSIONS_SELFTEXT_SCHEMA = pa.schema(
    [("id", pa.string()), ("title", pa.string()), ("selftext", pa.string()),]
)

# Big text columns to exclude from dict & stats
BIG_TEXT = {"body", "selftext", "title", "media"}


def _writer_for(
    out_path: str, schema: pa.schema, compression_level: int
) -> pq.ParquetWriter:

    """
    Construct a ParquetWriter with sensible defaults for Reddit data.

    Uses dictionary encoding for non-big-text columns and disables statistics
    for big-text columns to reduce metadata bloat.
    """
    use_dict = [n for n in schema.names if n not in BIG_TEXT]
    write_stats = {n: (n not in BIG_TEXT) for n in schema.names}
    kwargs = dict(
        schema=schema,
        compression="zstd",
        compression_level=compression_level,
        use_dictionary=use_dict,
        version="2.6",
        data_page_size=1 << 22,
    )
    try:
        return pq.ParquetWriter(out_path, write_statistics=write_stats, **kwargs)
    except TypeError:
        # fallback for older pyarrow that doesn't accept dict
        return pq.ParquetWriter(out_path, write_statistics=True, **kwargs)


# ---------- TRANSFORMS (return CORE_DF, TEXT_DF) ----------
def _prefix_series(s: pd.Series, prefix: str) -> pd.Series:
    """Prefix values lacking a 't{digit}_' prefix with `prefix` (string dtype)."""

    s = s.astype("string")
    already = s.str.match(r"^t\d_")
    s.loc[~already] = prefix + s.loc[~already]
    return s


def transform_comments_batch(
    rows: List[Dict[str, Any]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transform a batch of comment records into (core_df, body_df).

    Returns
    -------
    core_df:
        Comment metadata without body text.
    body_df:
        Sidecar table containing (id, body).
    """

    rows = [_ensure_columns(r, KEEP_COMMENTS) for r in rows]
    df = pd.DataFrame(rows)

    # ID normalization
    df["id"] = _prefix_series(df["id"], "t1_")

    # Timestamps
    df["created_utc"] = _to_utc_timestamp(df["created_utc"])
    if "author_created_utc" in df.columns:
        df["author_created_utc"] = _to_utc_timestamp(df["author_created_utc"])

    # Numerics
    for col in ("controversiality", "score"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Split: core (no body) + sidecar (id, body)
    core = df.drop(columns=["body"])
    body = df[["id", "body"]]
    return core, body


def transform_submissions_batch(
    rows: List[Dict[str, Any]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transform a batch of submission records into (core_df, text_df).

    Returns
    -------
    core_df:
        Submission metadata without selftext.
    text_df:
        Sidecar table containing (id, title, selftext).
    """

    # ensure keys + normalize nested
    for r in rows:
        for k in KEEP_SUBMISSIONS:
            r.setdefault(k, None)
        r["content_categories"] = _safe_json_dump(r.get("content_categories"))
        r["media"] = _safe_json_dump(r.get("media"))
    df = pd.DataFrame([{k: r.get(k, None) for k in KEEP_SUBMISSIONS} for r in rows])

    # ID normalization
    df["id"] = _prefix_series(df["id"], "t3_")

    # Timestamps
    df["created_utc"] = _to_utc_timestamp(df["created_utc"])
    if "author_created_utc" in df.columns:
        df["author_created_utc"] = _to_utc_timestamp(df["author_created_utc"])

    # Numerics
    for col in (
        "num_comments",
        "num_crossposts",
        "score",
        "subreddit_subscribers",
        "total_awards_received",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    if "upvote_ratio" in df.columns:
        df["upvote_ratio"] = pd.to_numeric(df["upvote_ratio"], errors="coerce")

    # Split: core (no selftext) + sidecar (id, selftext)
    core = df.drop(columns=["selftext"])
    text = df[["id", "title", "selftext"]]
    return core, text


# ---------- DRIVER ----------
def process_comments_month(
    zst_path: str,
    month: str,
    out_dir_comments: str,
    batch_rows: int,
    row_group_rows: int,
    compression_level_core: int,
    compression_level_text: int,
    max_records: int = 0,
    max_batches: int = 0,
) -> None:

    """Process one monthly comments .zst file into two Parquet sidecars (core + body)."""

    out_core = os.path.join(out_dir_comments, f"comments_core_{month}.parquet")
    out_body = os.path.join(out_dir_comments, f"comments_body_{month}.parquet")

    os.makedirs(os.path.dirname(out_core), exist_ok=True)

    LOG.info("Processing comments: %s -> %s + %s", zst_path, out_core, out_body)

    batch: List[Dict[str, Any]] = []
    n_in = n_core = n_body = batches = 0

    writer_core = None
    writer_body = None
    try:
        for rec in _iter_ndjson_from_zst(zst_path):
            n_in += 1
            batch.append(rec)

            if max_records and n_in >= max_records:
                LOG.info(
                    "Reached max_records=%d for %s; stopping early after reading %d records.",
                    max_records,
                    zst_path,
                    n_in,
                )
                break

            if len(batch) >= batch_rows:
                core_df, body_df = transform_comments_batch(batch)
                tbl_core = pa.Table.from_pandas(
                    core_df,
                    preserve_index=False,
                    schema=COMMENTS_CORE_SCHEMA,
                    safe=False,
                )
                tbl_body = pa.Table.from_pandas(
                    body_df,
                    preserve_index=False,
                    schema=COMMENTS_BODY_SCHEMA,
                    safe=False,
                )
                if writer_core is None:
                    writer_core = _writer_for(
                        out_core, COMMENTS_CORE_SCHEMA, compression_level_core
                    )
                    writer_body = pq.ParquetWriter(
                        out_body,
                        schema=COMMENTS_BODY_SCHEMA,
                        compression="zstd",
                        compression_level=compression_level_text,
                        use_dictionary=False,
                        write_statistics=False,
                        version="2.6",
                        data_page_size=1 << 22,
                    )

                writer_core.write_table(tbl_core, row_group_size=row_group_rows)
                writer_body.write_table(tbl_body, row_group_size=row_group_rows)
                n_core += len(core_df)
                n_body += len(body_df)
                batches += 1
                if max_batches and batches >= max_batches:
                    LOG.info(
                        "Reached max_batches=%d for %s; stopping early.",
                        max_batches,
                        zst_path,
                    )
                    batch.clear()
                    break

                batch.clear()
                if batches % 10 == 0:
                    LOG.info(
                        "  ... %s read / %s core / %s body",
                        f"{n_in:,}",
                        f"{n_core:,}",
                        f"{n_body:,}",
                    )

        if batch:
            core_df, body_df = transform_comments_batch(batch)
            tbl_core = pa.Table.from_pandas(
                core_df,
                preserve_index=False,
                schema=COMMENTS_CORE_SCHEMA,
                safe=False,
            )
            tbl_body = pa.Table.from_pandas(
                body_df,
                preserve_index=False,
                schema=COMMENTS_BODY_SCHEMA,
                safe=False,
            )

            if writer_core is None:
                writer_core = _writer_for(
                    out_core, COMMENTS_CORE_SCHEMA, compression_level_core
                )
                writer_body = pq.ParquetWriter(
                    out_body,
                    schema=COMMENTS_BODY_SCHEMA,
                    compression="zstd",
                    compression_level=compression_level_text,
                    use_dictionary=False,
                    write_statistics=False,
                    version="2.6",
                    data_page_size=1 << 22,
                )

            writer_core.write_table(tbl_core, row_group_size=row_group_rows)
            writer_body.write_table(tbl_body, row_group_size=row_group_rows)
            n_core += len(core_df)
            n_body += len(body_df)
            batches += 1

    finally:
        if writer_core is not None:
            writer_core.close()
        if writer_body is not None:
            writer_body.close()

    LOG.info(
        f"Done comments {zst_path} -> {out_core} + {out_body}: {n_in:,} read → {n_core:,} core / {n_body:,} body across {batches} batches"
    )


def process_submissions_month(
    zst_path: str,
    month: str,
    out_dir_submissions: str,
    batch_rows: int,
    row_group_rows: int,
    compression_level_core: int,
    compression_level_text: int,
    max_records: int = 0,
    max_batches: int = 0,
) -> None:

    """Process one monthly submissions .zst file into two Parquet sidecars (core + selftext)."""

    out_core = os.path.join(out_dir_submissions, f"submissions_core_{month}.parquet")
    out_text = os.path.join(
        out_dir_submissions, f"submissions_selftext_{month}.parquet"
    )

    os.makedirs(os.path.dirname(out_core), exist_ok=True)

    LOG.info("Processing submissions: %s -> %s + %s", zst_path, out_core, out_text)

    batch: List[Dict[str, Any]] = []
    n_in = n_core = n_text = batches = 0

    writer_core = None
    writer_text = None
    try:
        for rec in _iter_ndjson_from_zst(zst_path):
            n_in += 1
            batch.append(rec)

            if max_records and n_in >= max_records:
                LOG.info(
                    "Reached max_records=%d for %s; stopping early after reading %d records.",
                    max_records,
                    zst_path,
                    n_in,
                )
                break

            if len(batch) >= batch_rows:
                core_df, text_df = transform_submissions_batch(batch)
                tbl_core = pa.Table.from_pandas(
                    core_df,
                    preserve_index=False,
                    schema=SUBMISSIONS_CORE_SCHEMA,
                    safe=False,
                )
                tbl_text = pa.Table.from_pandas(
                    text_df,
                    preserve_index=False,
                    schema=SUBMISSIONS_SELFTEXT_SCHEMA,
                    safe=False,
                )
                if writer_core is None:
                    writer_core = _writer_for(
                        out_core, SUBMISSIONS_CORE_SCHEMA, compression_level_core
                    )
                    writer_text = pq.ParquetWriter(
                        out_text,
                        schema=SUBMISSIONS_SELFTEXT_SCHEMA,
                        compression="zstd",
                        compression_level=compression_level_text,
                        use_dictionary=False,
                        write_statistics=False,
                        version="2.6",
                        data_page_size=1 << 22,
                    )

                writer_core.write_table(tbl_core, row_group_size=row_group_rows)
                writer_text.write_table(tbl_text, row_group_size=row_group_rows)
                n_core += len(core_df)
                n_text += len(text_df)
                batches += 1
                if max_batches and batches >= max_batches:
                    LOG.info(
                        "Reached max_batches=%d for %s; stopping early.",
                        max_batches,
                        zst_path,
                    )
                    batch.clear()
                    break

                batch.clear()
                if batches % 10 == 0:

                    LOG.info(
                        "  ... %s read / %s core / %s selftext",
                        f"{n_in:,}",
                        f"{n_core:,}",
                        f"{n_text:,}",
                    )

        if batch:
            core_df, text_df = transform_submissions_batch(batch)
            tbl_core = pa.Table.from_pandas(
                core_df,
                preserve_index=False,
                schema=SUBMISSIONS_CORE_SCHEMA,
                safe=False,
            )
            tbl_text = pa.Table.from_pandas(
                text_df,
                preserve_index=False,
                schema=SUBMISSIONS_SELFTEXT_SCHEMA,
                safe=False,
            )
            if writer_core is None:
                writer_core = _writer_for(
                    out_core, SUBMISSIONS_CORE_SCHEMA, compression_level_core
                )
                writer_text = pq.ParquetWriter(
                    out_text,
                    schema=SUBMISSIONS_SELFTEXT_SCHEMA,
                    compression="zstd",
                    compression_level=15,
                    use_dictionary=False,
                    write_statistics=False,
                    version="2.6",
                    data_page_size=1 << 22,
                )
            writer_core.write_table(tbl_core, row_group_size=row_group_rows)
            writer_text.write_table(tbl_text, row_group_size=row_group_rows)
            n_core += len(core_df)
            n_text += len(text_df)
            batches += 1
    finally:
        if writer_core is not None:
            writer_core.close()
        if writer_text is not None:
            writer_text.close()

    LOG.info(
        f"Done submissions {zst_path} -> {out_core} + {out_text}: {n_in:,} read → {n_core:,} core / {n_text:,} selftext across {batches} batches"
    )


def discover_and_process(args: argparse.Namespace) -> None:
    """
    Discover monthly RC_/RS_ .zst files and process them in parallel.

    Parameters
    ----------
    args:
        Parsed command-line arguments.
    """
    input_root = args.input_root
    out_dir_comments = os.path.join(args.out_dir, "comments")
    out_dir_submissions = os.path.join(args.out_dir, "submissions")
    _log_config(args, out_dir_comments, out_dir_submissions)

    rc_paths = []
    rs_paths = []

    if not args.submissions_only:
        rc_paths = sorted(
            glob.glob(os.path.join(input_root, "comments", f"RC_{args.year}-*.zst"))
        )
    if not args.comments_only:
        rs_paths = sorted(
            glob.glob(os.path.join(input_root, "submissions", f"RS_{args.year}-*.zst"))
        )

    LOG.info("INPUT_ROOT=%s", input_root)
    LOG.info(
        "Found %d RC and %d RS files for year=%d",
        len(rc_paths),
        len(rs_paths),
        args.year,
    )

    # --- filter by months and/or limit number of files ---
    only_months = (args.only_months or "").strip()
    if only_months:
        keep = {m.strip() for m in only_months.split(",") if m.strip()}
        rc_paths = [
            p
            for p in rc_paths
            if re.search(r"(\d{4}-\d{2})", os.path.basename(p)).group(1) in keep
        ]
        rs_paths = [
            p
            for p in rs_paths
            if re.search(r"(\d{4}-\d{2})", os.path.basename(p)).group(1) in keep
        ]
        LOG.info(
            "After only-months filter: %d RC and %d RS files",
            len(rc_paths),
            len(rs_paths),
        )

    if args.limit_files and args.limit_files > 0:
        rc_paths = rc_paths[: args.limit_files]
        rs_paths = rs_paths[: args.limit_files]
        LOG.info(
            "After limit-files=%d: %d RC and %d RS files",
            args.limit_files,
            len(rc_paths),
            len(rs_paths),
        )

    if not rc_paths and not rs_paths:
        LOG.warning("No input files matched. Check --input-root/--year/--only-months.")
        return

    os.makedirs(out_dir_comments, exist_ok=True)
    os.makedirs(out_dir_submissions, exist_ok=True)

    tasks = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for p in rc_paths:
            month = re.search(r"(\d{4}-\d{2})", os.path.basename(p)).group(1)
            tasks.append(
                ex.submit(
                    process_comments_month,
                    p,
                    month,
                    out_dir_comments,
                    args.batch_rows,
                    args.row_group_rows,
                    args.compression_level_core,
                    args.compression_level_text,
                    args.max_records,
                    args.max_batches,
                )
            )
        for p in rs_paths:
            month = re.search(r"(\d{4}-\d{2})", os.path.basename(p)).group(1)
            tasks.append(
                ex.submit(
                    process_submissions_month,
                    p,
                    month,
                    out_dir_submissions,
                    args.batch_rows,
                    args.row_group_rows,
                    args.compression_level_core,
                    args.compression_level_text,
                    args.max_records,
                    args.max_batches,
                )
            )

        for fut in as_completed(tasks):
            # propagate any worker exceptions immediately
            fut.result()


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
    )
    discover_and_process(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
