#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Cara Lynch
# See the LICENSE file for details.
"""
Preprocessing step 4.1: scrub user mentions in Reddit text Parquet files.

This script replaces user-mention tokens in text fields with the *same pseudonymous
identifier* (`author_pid`) produced by step 04_0. Pseudonyms are deterministic given a
secret salt file, and use the same namespace as step 04_0:

  author_pid = "u_" + base32( HMAC_SHA256(secret, "author:" + NFKC(casefold(username))) )[0:k]

This ensures that:
- `author_pid` generated for an author in step 04_0 matches the pseudonym used when that
  author is mentioned in text.

It is designed to run after:
- 03_0_cleaning.py (produces cleaned monthly Parquet sidecars / cores with text fields)
- 04_0_pseudonymise_authors.py (defines the author_pid scheme + salt)

What it processes
-----------------
By default, it processes these monthly Parquet tables under <in-root>:

- comments/comments_body_YYYY-MM[<in-suffix>].parquet
    - scrubs column: `body`
- submissions/submissions_selftext_YYYY-MM[<in-suffix>].parquet
    - scrubs columns: `selftext` and (if present) `title`
- submissions/submissions_core_YYYY-MM[<in-suffix>].parquet
    - scrubs column: `title` (if present)
    - only needed for legacy layouts where `title` lives in submissions_core

This supports both layouts:
- title in submissions_core (default Pushshift layout)
- title moved to submissions_selftext (if your cleaning step moved it)

If `title` has already been moved to submissions_selftext, pass
--skip-submissions-core. In that mode the script will not discover, process, or
write submissions_core files. Raw author-bearing submissions_core tables must not
be passed through into parquet_pseudo: when submissions_core is processed, the
input must be a pseudonymized core table with `author_pid` and without `author`.

Mention matching rules
----------------------
Replaces occurrences of usernames in the following forms:

- /u/<name>
- u/<name>
- u\\<name>
- /user/<name>
- @<name>

Where <name> is 3–20 chars from [A-Za-z0-9_]. Matching is case-insensitive; normalization
uses Unicode NFKC + casefold + strip.

URLs
----
Usernames inside URLs are also scrubbed. For example, `https://reddit.com/u/SomeUser`
will become `https://reddit.com/u/u_<TOKEN>`. This may change link destinations and is
intentional for privacy.

Markdown code exclusion
-----------------------
If --skip-code is enabled, replacements are skipped inside:
- fenced code blocks delimited by ``` or ~~~
- inline code spans delimited by backticks (heuristic)

Smoke mode
----------
If --smoke is enabled, the script limits rows scanned/written per file via
--smoke-max-rows-text. Smoke mode is for plumbing checks, not stable sampling.

Outputs
-------
Writes scrubbed Parquets under <out-root> preserving relative paths/filenames, plus:
- audit/scrub_user_mentions_run_<run-id>.json
- audit/scrub_user_mentions_summary_<year>.csv

Command-line interface
----------------------
--in-root               Input root containing monthly Parquets
--out-root              Output root for scrubbed Parquets
--year                  Year to process (YYYY)
--only-months           Optional comma-separated month list (YYYY-MM,...)
--salt-file             Secret salt bytes file (same as step 04_0)
--in-suffix             Optional filename suffix (e.g. "_botflag"; default: "")
--id-bytes              Number of HMAC bytes kept before base32 encoding (default: 10)
--deleted-token         Token used for deleted-style mentions (default: u_deleted)
--skip-submissions-core Do not discover/process/write submissions_core files
--skip-code             Skip replacements inside markdown code regions
--skip-existing         Skip outputs that already exist
--smoke                 Enable smoke mode
--smoke-max-rows-text   Max rows per file in smoke mode
--batch-size            Arrow record batch size
--compression           Parquet compression (default: ZSTD)
--log-level             Logging level

Examples
--------
Scrub mentions in cleaned sidecars with title already moved into submissions_selftext:
  python 04_1_scrub_user_mentions.py \
    --in-root parquet_clean \
    --out-root parquet_pseudo_text \
    --year 2021 \
    --salt-file secret_salt.bin \
    --skip-submissions-core \
    --skip-code \
    --compression ZSTD \
    --log-level INFO

Scrub mentions in bot-flagged sidecars (if your sidecars are suffixed):
  python 04_1_scrub_user_mentions.py \
    --in-root parquet_clean_botflag \
    --out-root parquet_pseudo_text \
    --year 2021 \
    --salt-file secret_salt.bin \
    --in-suffix _botflag \
    --skip-code
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import glob
import hashlib
import hmac
import json
import logging
import os
import platform
import re
import sys
import unicodedata as ud
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


# ----------------------------- Logging ---------------------------------


def setup_logging(level: str) -> None:
    """Configure basic console logging."""
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(
        level=numeric, format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


logger = logging.getLogger(__name__)


# ----------------------------- Config ----------------------------------


MONTH_RE = re.compile(r"(20\d{2}-[01]\d)")  # YYYY-MM

# Mention / profile patterns we want to scrub.
#
# Captures the *exact* prefix so we can re-emit the same mention style.
# Note: this will also scrub usernames inside profile URLs (e.g., reddit.com/u/<name>),
# which may change link destinations. This is intentional for privacy.
#
# Supported:
#   - u/username
#   - /u/username
#   - u\username
#   - /user/username
#   - @username
#
# Username token is 3–20 chars of [A-Za-z0-9_], consistent with your earlier approach.
#
# Note: '@' mentions can collide with emails; we mitigate by requiring a non-word char
# before '@' (start-of-string or whitespace/punct) and a valid username token after.
MENTION_RE = re.compile(
    r"(?<!\w)"  # not preceded by a word char
    r"(?P<prefix>@|/user/|/u/|u/|u\\)"  # exact mention prefix
    r"(?P<name>[A-Za-z0-9_]{3,20})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ScrubConfig:
    """Non-performance configuration for mention scrubbing."""

    in_root: str
    out_root: str
    year: str
    only_months: Tuple[str, ...]
    salt_file: str
    in_suffix: str
    id_bytes: int
    deleted_token: str
    skip_submissions_core: bool
    skip_code: bool
    skip_existing: bool
    compression: str
    run_id: str

    smoke: bool
    smoke_max_rows_text: int

    batch_size: int


# ----------------------------- Utilities --------------------------------


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def utc_now_compact() -> str:
    """UTC timestamp string suitable for filenames."""
    return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def parse_only_months(s: str) -> Tuple[str, ...]:
    """Parse comma-separated YYYY-MM list."""
    if not s.strip():
        return tuple()
    return tuple(m.strip() for m in s.split(",") if m.strip())


def read_secret(path: str) -> bytes:
    """
    Read secret salt bytes from a file.

    The file should contain raw bytes; leading/trailing whitespace is stripped.
    """
    with open(path, "rb") as fh:
        secret = fh.read().strip()
    if len(secret) < 16:
        raise ValueError("Salt must be at least 16 bytes.")
    return secret


def norm_author(s: str) -> str:
    """
    Normalize a username for stable HMAC.

    Uses Unicode NFKC + casefold + strip.
    """
    return ud.normalize("NFKC", str(s)).casefold().strip()


def hmac_pid(secret: bytes, msg: str, id_bytes: int) -> str:
    """Compute a stable pseudonymous ID token from a secret and message."""
    digest = hmac.new(secret, msg.encode("utf-8"), hashlib.sha256).digest()
    token = base64.b32encode(digest[:id_bytes]).decode("ascii").rstrip("=")
    return f"u_{token}"


def author_pid_for_username(secret: bytes, username: str, id_bytes: int, deleted_token: str) -> str:
    """
    Compute the *author_pid-compatible* pseudonym for a mentioned username.

    This intentionally uses the same namespace as step 5.0: "author:".

    Special-cases deleted-style tokens (e.g., "deleted", "[deleted]") to `deleted_token`.
    """
    n = norm_author(username)
    if n in {"deleted", "[deleted]"}:
        return deleted_token
    return hmac_pid(secret, f"author:{n}", id_bytes=id_bytes)


def list_text_files(
    in_root: str,
    year: str,
    in_suffix: str,
    skip_submissions_core: bool = False,
) -> Dict[str, Dict[str, str]]:
    """
    Discover monthly Parquet inputs under an input root.

    Returns mapping:
      month -> { kind -> path }

    Kinds include:
      - comments_body
      - submissions_selftext
      - submissions_core (for title, if present)

    If skip_submissions_core is True, submissions_core is not discovered.
    """
    inv: Dict[str, Dict[str, str]] = {}

    patterns = [
        ("comments_body", os.path.join(in_root, "comments", f"comments_body_{year}-*{in_suffix}.parquet")),
        ("submissions_selftext", os.path.join(in_root, "submissions", f"submissions_selftext_{year}-*{in_suffix}.parquet")),
    ]
    if not skip_submissions_core:
        patterns.append(
            (
                "submissions_core",
                os.path.join(in_root, "submissions", f"submissions_core_{year}-*{in_suffix}.parquet"),
            )
        )

    for kind, pat in patterns:
        for path in glob.glob(pat):
            m = MONTH_RE.search(os.path.basename(path))
            if not m:
                continue
            month = m.group(1)
            inv.setdefault(month, {})
            inv[month][kind] = path

    return dict(sorted(inv.items()))


def filter_months(months: Sequence[str], only_months: Tuple[str, ...]) -> List[str]:
    """Apply an optional month whitelist."""
    if not only_months:
        return list(months)
    keep = set(only_months)
    return [m for m in months if m in keep]


def read_parquet_sql_fragment(path: str, limit: Optional[int]) -> str:
    """
    Helper for logging: describe how we will read in smoke vs non-smoke.

    (Arrow reads directly; this is just for clarity in logs.)
    """
    if limit is None:
        return f"{path} (full)"
    return f"{path} (LIMIT {limit} rows)"


def validate_submissions_core_schema(in_path: str) -> None:
    """
    Refuse to pass raw author-bearing submissions_core tables through this step.

    Step 04_1 only scrubs mention text. It must not copy a raw cleaned
    submissions_core table into parquet_pseudo because that would reintroduce
    direct author identifiers alongside the scrubbed text sidecars.
    """
    cols = set(pq.ParquetFile(in_path).schema_arrow.names)
    if "author" in cols:
        raise RuntimeError(
            "Refusing to process submissions_core with raw author identifiers: "
            f"{in_path}. The table contains 'author'. Processing it would pass "
            "raw author values into parquet_pseudo. Run 04_0_pseudonymise_authors.py "
            "first, or use --skip-submissions-core when title has been moved to "
            "submissions_selftext."
        )
    if "author_pid" not in cols:
        raise RuntimeError(
            "Refusing to process submissions_core without 'author_pid': "
            f"{in_path}. submissions_core processing is only safe for pseudonymized "
            "core tables. Use --skip-submissions-core when title has been moved to "
            "submissions_selftext."
        )


# ---------------------- Markdown code span masking ----------------------


def _mask_code_regions(text: str) -> List[Tuple[int, int]]:
    """
    Return (start,end) spans representing markdown code regions to skip replacements.

    Heuristic:
      - fenced code blocks delimited by ``` or ~~~ (first-next occurrence)
      - inline code delimited by backticks `...` (pairwise)

    This is intentionally simple and fast; it will not perfectly parse all markdown.
    """
    spans: List[Tuple[int, int]] = []

    # Fenced blocks
    for fence in ("```", "~~~"):
        start = 0
        while True:
            i = text.find(fence, start)
            if i == -1:
                break
            j = text.find(fence, i + len(fence))
            if j == -1:
                break
            spans.append((i, j + len(fence)))
            start = j + len(fence)

    # Inline code (simple)
    start = 0
    while True:
        i = text.find("`", start)
        if i == -1:
            break
        j = text.find("`", i + 1)
        if j == -1:
            break
        spans.append((i, j + 1))
        start = j + 1

    spans.sort()
    return spans


def _in_spans(idx: int, spans: List[Tuple[int, int]]) -> bool:
    """Return True if idx lies inside any (start,end) span; spans must be sorted."""
    for a, b in spans:
        if a <= idx < b:
            return True
        if idx < a:
            return False
    return False


# ----------------------------- Core logic --------------------------------


def replace_mentions_in_text(
    text: Optional[str],
    secret: bytes,
    id_bytes: int,
    deleted_token: str,
    skip_code: bool,
) -> Tuple[Optional[str], int, bool]:

    """
    Replace user mentions in a text field.

    Parameters
    ----------
    text:
        Input text (may be None).
    secret:
        Secret salt bytes.
    id_bytes:
        Number of HMAC bytes kept before base32 encoding.
    skip_code:
        If True, skip replacements inside markdown code spans.

    Returns
    -------
    (new_text, n_replaced, changed)
    """
    if text is None or text == "":
        return text, 0, False

    spans: List[Tuple[int, int]] = _mask_code_regions(text) if skip_code else []

    out: List[str] = []
    i = 0
    n_rep = 0

    for m in MENTION_RE.finditer(text):
        s, e = m.span()
        out.append(text[i:s])

        if skip_code and _in_spans(s, spans):
            out.append(text[s:e])
        else:
            uname = m.group("name")
            prefix = m.group("prefix")
            pid = author_pid_for_username(secret, uname, id_bytes=id_bytes, deleted_token=deleted_token)
            out.append(prefix + pid)
            n_rep += 1


        i = e

    out.append(text[i:])
    new_text = "".join(out)
    return new_text, n_rep, (new_text != text)


def scrub_table_columns(
    tbl: pa.Table,
    text_cols: Sequence[str],
    secret: bytes,
    id_bytes: int,
    deleted_token: str,
    skip_code: bool,
) -> Tuple[pa.Table, int, int]:
    """
    Scrub mentions in specified text columns of a PyArrow table.

    Returns
    -------
    (new_table, mentions_replaced, cells_changed)
    """
    cols = tbl.column_names
    replacements_total = 0
    changed_cells = 0

    arrays: List[pa.Array] = []
    for name in cols:
        col = tbl[name]
        if name not in text_cols:
            arrays.append(col)
            continue

        # Apply per-row transform for this column.
        # Use to_pylist() to avoid pandas overhead and preserve None values cleanly.
        py_vals = col.to_pylist()
        new_vals: List[Optional[str]] = []

        for v in py_vals:
            new_v, n_rep, changed = replace_mentions_in_text(
                v,
                secret=secret,
                id_bytes=id_bytes,
                deleted_token=deleted_token,
                skip_code=skip_code,
            )

            new_vals.append(new_v)
            replacements_total += n_rep
            if changed:
                changed_cells += 1

        arrays.append(pa.array(new_vals, type=col.type))

    return pa.Table.from_arrays(arrays, names=cols), replacements_total, changed_cells


def process_parquet_file(
    in_path: str,
    out_path: str,
    text_cols: Sequence[str],
    secret: bytes,
    id_bytes: int,
    deleted_token: str,
    skip_code: bool,
    compression: str,
    batch_size: int,
    smoke: bool,
    smoke_max_rows_text: int,
) -> Dict[str, int]:

    """
    Process one Parquet file in streaming batches and write scrubbed output.

    Returns dict of diagnostics:
      - rows_written
      - mentions_replaced
      - text_cells_changed
      - batches_written
    """
    ensure_dir(os.path.dirname(out_path))

    dataset = ds.dataset(in_path, format="parquet")
    scanner = dataset.scanner(batch_size=batch_size)

    writer: Optional[pq.ParquetWriter] = None

    rows_written = 0
    mentions_replaced = 0
    text_cells_changed = 0
    batches_written = 0

    for record_batch in scanner.to_batches():
        if smoke and rows_written >= smoke_max_rows_text:
            break

        # Convert to table
        tbl = pa.Table.from_batches([record_batch])

        # Apply smoke truncation at batch level if needed
        if smoke:
            remaining = smoke_max_rows_text - rows_written
            if remaining <= 0:
                break
            if tbl.num_rows > remaining:
                tbl = tbl.slice(0, remaining)

        # Determine which requested columns actually exist
        present_text_cols = [c for c in text_cols if c in tbl.column_names]
        if not present_text_cols:
            # No columns to scrub; pass through
            new_tbl = tbl
            rep = 0
            changed = 0
        else:
            new_tbl, rep, changed = scrub_table_columns(
                tbl,
                text_cols=present_text_cols,
                secret=secret,
                id_bytes=id_bytes,
                deleted_token=deleted_token,
                skip_code=skip_code,
            )


        if writer is None:
            writer = pq.ParquetWriter(out_path, new_tbl.schema, compression=compression.lower())

        writer.write_table(new_tbl)
        batches_written += 1

        rows_written += new_tbl.num_rows
        mentions_replaced += rep
        text_cells_changed += changed

    if writer is not None:
        writer.close()

    return {
        "rows_written": int(rows_written),
        "mentions_replaced": int(mentions_replaced),
        "text_cells_changed": int(text_cells_changed),
        "batches_written": int(batches_written),
    }


def write_run_metadata(out_root: str, cfg: ScrubConfig) -> str:
    """Write a JSON run metadata file and return its path."""
    ensure_dir(os.path.join(out_root, "audit"))
    out_path = os.path.join(out_root, "audit", f"scrub_user_mentions_run_{cfg.run_id}.json")
    payload = {
        "run_id": cfg.run_id,
        "timestamp_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "script": os.path.basename(__file__),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "pyarrow": {"version": pa.__version__},
        "config": asdict(cfg),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return out_path


# ----------------------------- CLI / Main --------------------------------


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI parser for this script."""
    p = argparse.ArgumentParser(
        description="Scrub /u/ and u/ user mentions in Reddit text Parquets (PyArrow streaming).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--in-root", required=True, help="Input root containing monthly Parquets.")
    p.add_argument("--out-root", required=True, help="Output root for scrubbed Parquets.")
    p.add_argument("--year", default="2021", help="Year to process (YYYY).")
    p.add_argument("--only-months", default="", help="Comma-separated subset like 2021-01,2021-06.")
    p.add_argument("--salt-file", required=True, help="Path to secret salt bytes (same as step 04_0).")

    p.add_argument(
        "--in-suffix",
        default="",
        help='Optional input filename suffix (e.g. "_botflag"); default "".',
    )
    p.add_argument(
        "--id-bytes",
        type=int,
        default=10,
        help="Number of HMAC digest bytes kept before base32 encoding (10 bytes ~= 80 bits).",
    )

    p.add_argument(
        "--deleted-token",
        default="u_deleted",
        help='Token used for deleted-style mentions (e.g., "u/deleted", "u/[deleted]").',
    )

    p.add_argument(
        "--skip-submissions-core",
        action="store_true",
        help=(
            "Do not discover, process, or write submissions_core files. Use when "
            "submission titles have already been moved into submissions_selftext."
        ),
    )

    p.add_argument(
        "--skip-code",
        action="store_true",
        help="Skip replacements inside markdown code blocks/spans.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip writing outputs that already exist.",
    )

    # Smoke mode
    p.add_argument("--smoke", action="store_true", help="Enable smoke mode (limits rows written per file).")
    p.add_argument(
        "--smoke-max-rows-text",
        type=int,
        default=200_000,
        help="Max rows per file in --smoke mode.",
    )

    # Performance / IO knobs
    p.add_argument("--batch-size", type=int, default=50_000, help="Arrow record batch size.")
    p.add_argument(
        "--compression",
        default="ZSTD",
        help="Parquet compression codec (e.g., ZSTD, SNAPPY, UNCOMPRESSED).",
    )
    p.add_argument("--log-level", default="INFO", help="Logging level.")

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    args = build_argparser().parse_args(argv)
    setup_logging(args.log_level)

    run_id = utc_now_compact()

    cfg = ScrubConfig(
        in_root=args.in_root,
        out_root=args.out_root,
        year=str(args.year),
        only_months=parse_only_months(args.only_months),
        salt_file=args.salt_file,
        in_suffix=str(args.in_suffix),
        id_bytes=int(args.id_bytes),
        deleted_token=str(args.deleted_token),
        skip_submissions_core=bool(args.skip_submissions_core),
        skip_code=bool(args.skip_code),
        skip_existing=bool(args.skip_existing),
        compression=str(args.compression).upper(),
        run_id=run_id,
        smoke=bool(args.smoke),
        smoke_max_rows_text=int(args.smoke_max_rows_text),
        batch_size=int(args.batch_size),
    )

    ensure_dir(cfg.out_root)
    ensure_dir(os.path.join(cfg.out_root, "comments"))
    ensure_dir(os.path.join(cfg.out_root, "submissions"))
    ensure_dir(os.path.join(cfg.out_root, "audit"))

    try:
        secret = read_secret(cfg.salt_file)
    except Exception as e:
        logger.error("Failed to read salt file: %s", e)
        return 2

    inv = list_text_files(
        cfg.in_root,
        cfg.year,
        cfg.in_suffix,
        skip_submissions_core=cfg.skip_submissions_core,
    )
    months = filter_months(list(inv.keys()), cfg.only_months)

    if not months:
        logger.warning(
            "No months found to process under in-root=%s for year=%s (suffix=%r).",
            cfg.in_root,
            cfg.year,
            cfg.in_suffix,
        )
        return 0

    logger.info("Discovered %d month(s): %s", len(months), ", ".join(months))
    logger.info("Writing outputs to: %s", cfg.out_root)
    if cfg.smoke:
        logger.warning("SMOKE MODE enabled: limiting to %d rows per file.", cfg.smoke_max_rows_text)
    if cfg.skip_code:
        logger.info("Markdown code skipping enabled.")
    if cfg.skip_submissions_core:
        logger.info("Skipping submissions_core discovery/processing/writes.")

    run_meta_path = write_run_metadata(cfg.out_root, cfg)
    logger.info("Wrote run metadata: %s", run_meta_path)

    summary_rows: List[dict] = []

    # File → columns to scrub
    # - comments_body: body
    # - submissions_selftext: selftext (+ title if present)
    # - submissions_core: title (if present)
    kind_to_columns: Dict[str, List[str]] = {
        "comments_body": ["body"],
        "submissions_selftext": ["selftext", "title"],
        "submissions_core": ["title"],
    }

    for month in months:
        month_paths = inv.get(month, {})
        if not month_paths:
            logger.info("Month %s: no files; skipping.", month)
            continue

        logger.info("=== %s ===", month)

        for kind, in_path in month_paths.items():
            base = os.path.basename(in_path)
            subdir = "comments" if kind.startswith("comments_") else "submissions"
            out_path = os.path.join(cfg.out_root, subdir, base)

            if cfg.skip_existing and os.path.exists(out_path):
                logger.info("Skip existing: %s", out_path)
                continue

            if kind == "submissions_core":
                validate_submissions_core_schema(in_path)

            text_cols = kind_to_columns.get(kind, [])
            if not text_cols:
                logger.info("Kind %s: no configured text columns; skipping.", kind)
                continue

            logger.info("Processing %s: %s", kind, read_parquet_sql_fragment(in_path, cfg.smoke_max_rows_text if cfg.smoke else None))

            try:
                diag = process_parquet_file(
                    in_path=in_path,
                    out_path=out_path,
                    text_cols=text_cols,
                    secret=secret,
                    id_bytes=cfg.id_bytes,
                    deleted_token=cfg.deleted_token,
                    skip_code=cfg.skip_code,
                    compression=cfg.compression,
                    batch_size=cfg.batch_size,
                    smoke=cfg.smoke,
                    smoke_max_rows_text=cfg.smoke_max_rows_text,
                )

            except Exception as e:
                logger.exception("Failed processing %s (%s): %s", kind, in_path, e)
                return 3

            logger.info(
                "Wrote %s -> %s (rows=%d, mentions_replaced=%d, text_cells_changed=%d, batches=%d)",
                kind,
                out_path,
                diag["rows_written"],
                diag["mentions_replaced"],
                diag["text_cells_changed"],
                diag["batches_written"],
            )

            summary_rows.append(
                {
                    "month": month,
                    "kind": kind,
                    "in_path": in_path,
                    "out_path": out_path,
                    "rows_written": diag["rows_written"],
                    "mentions_replaced": diag["mentions_replaced"],
                    "text_cells_changed": diag["text_cells_changed"],
                    "batches_written": diag["batches_written"],
                    "skip_code": int(cfg.skip_code),
                    "smoke": int(cfg.smoke),
                }
            )

    if summary_rows:
        summary_df = pa.Table.from_pylist(summary_rows)
        summary_path = os.path.join(cfg.out_root, "audit", f"scrub_user_mentions_summary_{cfg.year}.csv")
        # Use Arrow CSV writer via pandas for simplicity and compatibility
        import pandas as pd  # local import to keep top-level deps minimal

        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        logger.info("Wrote summary CSV: %s", summary_path)

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
