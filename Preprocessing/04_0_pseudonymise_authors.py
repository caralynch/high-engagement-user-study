#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Cara Lynch
# See the LICENSE file for details.
"""
Preprocessing step 4.0: pseudonymise Reddit authors in cleaned Parquet files.

This script replaces the identifying `author` field in monthly *core* Parquet tables
(comments_core, submissions_core) with a stable pseudonymous identifier `author_pid`,
computed using HMAC-SHA256 with a secret salt.

It is designed to run after:
- 03_0_cleaning.py (produces cleaned monthly Parquet; adds `is_deleted_author` flag)

If the input root contains suffixed core files such as `*_botflag.parquet`, pass
that suffix with `--in-suffix`. For unsuffixed cleaned files, pass
`--in-suffix ""`.

Design goals
------------
- Scalable: uses DuckDB SQL over Parquet and avoids loading full monthly tables into pandas.
- Reproducible: deterministic pseudonyms from a secret salt + normalized author string.
- Safe defaults: drops `author` by default; drops author_flair_text by default; can keep with --keep-flair.
- Pipeline-consistent: CLI + logging + DuckDB tuning similar to steps 03/04.
- QA-friendly: computes true mapping-miss diagnostics *before* dropping `author`.
- Smoke-testable: optional row limits and author-map limits for quick end-to-end validation.

Pseudonym design
----------------
For a non-deleted author name `a`, we compute:

  author_pid = "u_" + base32( HMAC_SHA256(secret, "author:" + NFKC(lower(a))) )[0:k]

Where k depends on --id-bytes (default: 10 bytes = 80 bits).

Deleted / missing authors
-------------------------
- Rows where `author` is NULL -> `author_pid` is NULL.
- Rows where `lower(author) == "[deleted]"` -> `author_pid` is set to --deleted-token
  (default: "u_deleted"), which is non-identifying.

Inputs (expected layout)
------------------------
<in-root>/
  comments/comments_core_YYYY-MM[_botflag].parquet
  submissions/submissions_core_YYYY-MM[_botflag].parquet

This script *only* rewrites core files (where `author` lives). Sidecars (body/selftext)
are not modified by this step.

Outputs
-------
Writes pseudonymised core Parquets under:
<out-root>/
  comments/<same-filename-as-input>.parquet
  submissions/<same-filename-as-input>.parquet
  audit/pseudonymise_authors_run_<run-id>.json
  audit/pseudonymise_authors_summary_<year>.csv

Optional (QA only; do not distribute):
  audit/author_pid_map_<year>.csv          (if --emit-map)

Smoke mode
----------
If --smoke is enabled, this script limits:
- rows scanned/written per core file (via --smoke-max-rows-core)
- distinct authors included in the month mapping (via --smoke-max-authors)

Note: in smoke mode, the author subset is not guaranteed to be stable across runs
because LIMIT is applied without ORDER BY. Smoke mode is intended for plumbing/CI checks,
not stable sampling.

Command-line interface
----------------------
--in-root               Input root directory (cleaned or bot-flagged root).
--out-root              Output root directory (pseudonymised).
--year                  Year to process (YYYY).
--only-months           Optional: comma-separated list of YYYY-MM months to keep.
--in-suffix             Input core filename suffix (default: "_botflag"; set "" for unsuffixed).
--id-bytes              Number of HMAC bytes to keep before base32 (default: 10).
--deleted-token         Token to use for "[deleted]" authors (default: "u_deleted").
--keep-flair            Keep author_flair_text column if present.
--emit-map              Write a private mapping CSV (QA only; do not distribute).
--skip-existing         Skip outputs that already exist.
--compression           Output Parquet compression (default: ZSTD).
--smoke                 Enable smoke mode (limits rows and author-map size).
--smoke-max-rows-core   Max rows per core file scanned/written in smoke mode.
--smoke-max-authors     Max distinct authors included in the author map in smoke mode.

DuckDB tuning
-------------
--threads               DuckDB threads (default: 4).
--memory-gb             DuckDB memory limit in GB (default: 16).
--temp-dir              DuckDB temp directory (default: duckdb_tmp).

Logging
-------
--log-level             DEBUG/INFO/WARNING/ERROR/CRITICAL (default: INFO).

Examples
--------
Pseudonymise suffixed core files for 2021:
  python Preprocessing/04_0_pseudonymise_authors.py \
    --in-root parquet_clean_botflag \
    --out-root parquet_pseudo \
    --year 2021 \
    --salt-file secret_salt.bin \
    --in-suffix _botflag \
    --threads 8 --memory-gb 32 --temp-dir /tmp \
    --keep-flair \
    --log-level INFO

Pseudonymise cleaned (non-botflag) cores:
  python Preprocessing/04_0_pseudonymise_authors.py \
    --in-root parquet_clean \
    --out-root parquet_pseudo \
    --year 2021 \
    --salt-file secret_salt.bin \
    --in-suffix ""
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

import duckdb
import pandas as pd


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


@dataclass(frozen=True)
class DuckDBConfig:
    """DuckDB runtime configuration."""

    threads: int = 4
    memory_gb: int = 16
    temp_dir: str = "duckdb_tmp"


@dataclass(frozen=True)
class PseudonymConfig:
    """Pseudonymisation configuration (non-DuckDB knobs)."""

    in_root: str
    out_root: str
    year: str
    only_months: Tuple[str, ...]
    salt_file: str
    in_suffix: str
    id_bytes: int
    deleted_token: str
    keep_flair: bool
    emit_map: bool
    skip_existing: bool
    compression: str
    run_id: str

    # Smoke mode
    smoke: bool
    smoke_max_rows_core: int
    smoke_max_authors: int


# ----------------------------- Utilities --------------------------------


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def utc_now_compact() -> str:
    """UTC timestamp string suitable for filenames."""
    return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def norm_author(s: str) -> str:
    """
    Normalize an author string for stable hashing.

    Uses Unicode NFKC normalization + casefold + strip.
    """
    return ud.normalize("NFKC", str(s)).casefold().strip()


def hmac_pid(secret: bytes, msg: str, id_bytes: int) -> str:
    """
    Compute a stable pseudonymous ID token from a secret and message.

    Parameters
    ----------
    secret:
        Secret salt bytes. Keep private.
    msg:
        Message to authenticate (already normalized / namespaced).
    id_bytes:
        Number of bytes to keep from the HMAC digest before base32 encoding.

    Returns
    -------
    str
        A token like "u_<BASE32...>".
    """
    digest = hmac.new(secret, msg.encode("utf-8"), hashlib.sha256).digest()
    token = base64.b32encode(digest[:id_bytes]).decode("ascii").rstrip("=")
    return f"u_{token}"


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


def configure_duckdb(con: duckdb.DuckDBPyConnection, cfg: DuckDBConfig) -> None:
    """Apply DuckDB configuration settings."""
    con.execute(f"SET threads={int(cfg.threads)}")
    con.execute(f"SET memory_limit='{int(cfg.memory_gb)}GB'")
    ensure_dir(cfg.temp_dir)
    con.execute(f"SET temp_directory='{cfg.temp_dir}'")


def parse_only_months(s: str) -> Tuple[str, ...]:
    """Parse comma-separated YYYY-MM list."""
    if not s.strip():
        return tuple()
    return tuple(m.strip() for m in s.split(",") if m.strip())


def list_core_files(
    in_root: str, year: str, in_suffix: str
) -> Dict[str, Dict[str, str]]:
    """
    Discover monthly core Parquet inputs under an input root.

    Returns a mapping:
      month -> {"comments_core": path, "submissions_core": path}
    """
    inv: Dict[str, Dict[str, str]] = {}

    patterns = [
        (
            "comments_core",
            os.path.join(
                in_root, "comments", f"comments_core_{year}-*{in_suffix}.parquet"
            ),
        ),
        (
            "submissions_core",
            os.path.join(
                in_root, "submissions", f"submissions_core_{year}-*{in_suffix}.parquet"
            ),
        ),
    ]

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


def duckdb_columns(con: duckdb.DuckDBPyConnection, parquet_path: str) -> List[str]:
    """Get column names from a Parquet file via DuckDB without scanning full data."""
    df = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet_path}')").df()
    return df["column_name"].tolist()


def build_select_list(alias: str, cols: Sequence[str], drop: Sequence[str]) -> str:
    """Build a SQL select list 'alias.col1, alias.col2, ...' excluding drop columns."""
    drop_set = set(drop)
    kept = [c for c in cols if c not in drop_set]
    return ", ".join(f"{alias}.{c}" for c in kept)


def read_parquet_sql(parquet_path: str, limit: Optional[int] = None) -> str:
    """
    Build a DuckDB SQL fragment for reading a Parquet file, optionally limiting rows.

    Returns
    -------
    str
        Either "read_parquet('<path>')" or "(SELECT * FROM read_parquet('<path>') LIMIT <n>)".
    """
    if limit is None:
        return f"read_parquet('{parquet_path}')"
    if limit <= 0:
        raise ValueError("limit must be positive")
    return f"(SELECT * FROM read_parquet('{parquet_path}') LIMIT {int(limit)})"


# ----------------------------- Core processing ---------------------------


def build_author_map(
    con: duckdb.DuckDBPyConnection,
    month_paths: Dict[str, str],
    secret: bytes,
    id_bytes: int,
    smoke: bool,
    smoke_max_rows_core: int,
    smoke_max_authors: int,
) -> pd.DataFrame:
    """
    Build a month-level author mapping table for joining back to cores.

    Reads distinct authors from available core files (comments/submissions),
    excluding NULL and "[deleted]". Join key is `author_key = lower(author)`.

    Returns a DataFrame with columns: author_key, author_pid.
    """
    unions: List[str] = []

    if "comments_core" in month_paths:
        src = read_parquet_sql(
            month_paths["comments_core"],
            limit=(smoke_max_rows_core if smoke else None),
        )
        unions.append(
            f"""
            SELECT DISTINCT lower(author) AS author_key
            FROM {src}
            WHERE author IS NOT NULL AND lower(author) <> '[deleted]'
            """
        )

    if "submissions_core" in month_paths:
        src = read_parquet_sql(
            month_paths["submissions_core"],
            limit=(smoke_max_rows_core if smoke else None),
        )
        unions.append(
            f"""
            SELECT DISTINCT lower(author) AS author_key
            FROM {src}
            WHERE author IS NOT NULL AND lower(author) <> '[deleted]'
            """
        )

    if not unions:
        return pd.DataFrame(
            {
                "author_key": pd.Series(dtype="string"),
                "author_pid": pd.Series(dtype="string"),
            }
        )

    sql = " UNION ALL ".join(unions)
    limit_clause = f" LIMIT {int(smoke_max_authors)}" if smoke else ""
    authors = con.execute(f"SELECT DISTINCT author_key FROM ({sql}){limit_clause}").df()

    if authors.empty:
        return pd.DataFrame(
            {
                "author_key": pd.Series(dtype="string"),
                "author_pid": pd.Series(dtype="string"),
            }
        )

    authors["author_norm"] = authors["author_key"].map(norm_author)
    authors["author_pid"] = authors["author_norm"].map(
        lambda s: hmac_pid(secret, f"author:{s}", id_bytes)
    )
    return authors[["author_key", "author_pid"]]


def write_pseudonymised_core(
    con: duckdb.DuckDBPyConnection,
    in_path: str,
    out_path: str,
    author_map: pd.DataFrame,
    deleted_token: str,
    keep_flair: bool,
    compression: str,
    smoke: bool,
    smoke_max_rows_core: int,
) -> Dict[str, int]:
    """
    Write a pseudonymised core Parquet file, with pre-drop QA diagnostics.

    Diagnostics are computed on the input *before* dropping `author`.

    Returns
    -------
    Dict[str, int]
        Keys:
        - rows_total
        - rows_author_null
        - rows_author_deleted
        - rows_should_map
        - rows_map_miss
    """
    ensure_dir(os.path.dirname(out_path))

    cols = duckdb_columns(con, in_path)

    if "author" not in cols:
        raise ValueError(f"Input core Parquet has no 'author' column: {in_path}")

    drop_cols: List[str] = ["author"]
    if (not keep_flair) and ("author_flair_text" in cols):
        drop_cols.append("author_flair_text")


    select_rest = build_select_list(alias="t", cols=cols, drop=drop_cols)

    # Register author_map even if empty (prevents "table not found" issues).
    con.register("author_map", author_map)
    src = read_parquet_sql(in_path, limit=(smoke_max_rows_core if smoke else None))

    diag = (
        con.execute(
            f"""
        WITH t AS (
          SELECT author
          FROM {src}
        )
        SELECT
          COUNT(*) AS rows_total,
          COALESCE(SUM(CASE WHEN author IS NULL THEN 1 ELSE 0 END), 0) AS rows_author_null,
          COALESCE(SUM(CASE WHEN author IS NOT NULL AND lower(author) = '[deleted]' THEN 1 ELSE 0 END), 0) AS rows_author_deleted,
          COALESCE(SUM(CASE WHEN author IS NOT NULL AND lower(author) <> '[deleted]' THEN 1 ELSE 0 END), 0) AS rows_should_map,
          COALESCE(SUM(
            CASE
              WHEN author IS NOT NULL AND lower(author) <> '[deleted]' AND am.author_pid IS NULL THEN 1
              ELSE 0
            END
          ), 0) AS rows_map_miss
        FROM t
        LEFT JOIN author_map AS am
          ON lower(t.author) = am.author_key
        """
        )
        .df()
        .iloc[0]
        .to_dict()
    )

    diag = {k: int(v) for k, v in diag.items()}

    con.execute(
        f"""
        COPY (
          SELECT
            CASE
              WHEN t.author IS NULL THEN NULL
              WHEN lower(t.author) = '[deleted]' THEN '{deleted_token}'
              ELSE am.author_pid
            END AS author_pid,
            {select_rest}
          FROM {src} AS t
          LEFT JOIN author_map AS am
            ON lower(t.author) = am.author_key
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION {compression})
        """
    )

    return diag


def write_run_metadata(out_root: str, cfg: PseudonymConfig, duck: DuckDBConfig) -> str:
    """Write a JSON run metadata file and return its path."""
    ensure_dir(os.path.join(out_root, "audit"))
    out_path = os.path.join(
        out_root, "audit", f"pseudonymise_authors_run_{cfg.run_id}.json"
    )
    payload = {
        "run_id": cfg.run_id,
        "timestamp_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "script": os.path.basename(__file__),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "duckdb": {"version": duckdb.__version__, **asdict(duck)},
        "config": asdict(cfg),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return out_path


# ----------------------------- CLI / Main --------------------------------


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI parser for this script."""
    p = argparse.ArgumentParser(
        description="Pseudonymise Reddit authors in monthly core Parquet files (DuckDB).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--in-root",
        required=True,
        help="Input root (cleaned or bot-flagged parquet root).",
    )
    p.add_argument(
        "--out-root", required=True, help="Output root for pseudonymised core files."
    )
    p.add_argument("--year", default="2021", help="Year to process (YYYY).")
    p.add_argument(
        "--only-months", default="", help="Comma-separated subset like 2021-01,2021-06."
    )

    p.add_argument(
        "--salt-file",
        required=True,
        help="Path to secret salt bytes (keep private; do not commit).",
    )

    p.add_argument(
        "--in-suffix",
        default="_botflag",
        help='Input core filename suffix (e.g. "_botflag" or "" for none).',
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
        help='Token used for rows where lower(author) == "[deleted]".',
    )

    p.add_argument(
        "--keep-flair",
        action="store_true",
        help="Keep author_flair_text if present (NOT recommended for publication).",
    )

    p.add_argument(
        "--emit-map",
        action="store_true",
        help="Write a private author->author_pid map CSV under audit/ (QA only; do not distribute).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip writing outputs that already exist.",
    )

    # Smoke test knobs
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run a fast smoke test: limit rows scanned/written.",
    )
    p.add_argument(
        "--smoke-max-rows-core",
        type=int,
        default=200_000,
        help="Max rows per core file to scan/write in --smoke mode.",
    )
    p.add_argument(
        "--smoke-max-authors",
        type=int,
        default=200_000,
        help="Max distinct authors to include in the month mapping in --smoke mode.",
    )

    p.add_argument(
        "--compression",
        default="ZSTD",
        help="Parquet compression codec (e.g., ZSTD, SNAPPY, UNCOMPRESSED).",
    )

    # DuckDB knobs
    p.add_argument("--threads", type=int, default=4, help="DuckDB threads.")
    p.add_argument(
        "--memory-gb", type=int, default=16, help="DuckDB memory limit (GB)."
    )
    p.add_argument("--temp-dir", default="duckdb_tmp", help="DuckDB temp directory.")
    p.add_argument("--log-level", default="INFO", help="Logging level.")

    return p


def main() -> int:
    """Entry point."""
    args = build_argparser().parse_args()
    setup_logging(args.log_level)

    run_id = utc_now_compact()

    cfg = PseudonymConfig(
        in_root=args.in_root,
        out_root=args.out_root,
        year=str(args.year),
        only_months=parse_only_months(args.only_months),
        salt_file=args.salt_file,
        in_suffix=args.in_suffix,
        id_bytes=int(args.id_bytes),
        deleted_token=str(args.deleted_token),
        keep_flair=bool(args.keep_flair),
        emit_map=bool(args.emit_map),
        skip_existing=bool(args.skip_existing),
        compression=str(args.compression).upper(),
        run_id=run_id,
        smoke=bool(args.smoke),
        smoke_max_rows_core=int(args.smoke_max_rows_core),
        smoke_max_authors=int(args.smoke_max_authors),
    )

    duck_cfg = DuckDBConfig(
        threads=int(args.threads),
        memory_gb=int(args.memory_gb),
        temp_dir=str(args.temp_dir),
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

    inv = list_core_files(cfg.in_root, cfg.year, cfg.in_suffix)
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
        logger.warning(
            "SMOKE MODE enabled: limiting to %d rows per core file and %d distinct authors.",
            cfg.smoke_max_rows_core,
            cfg.smoke_max_authors,
        )

    con = duckdb.connect()
    try:
        configure_duckdb(con, duck_cfg)

        run_meta_path = write_run_metadata(cfg.out_root, cfg, duck_cfg)
        logger.info("Wrote run metadata: %s", run_meta_path)

        summary_rows: List[dict] = []
        map_rows: List[pd.DataFrame] = []

        for month in months:
            month_paths = inv.get(month, {})
            if not month_paths:
                logger.info("Month %s: no core files; skipping.", month)
                continue

            logger.info("=== %s ===", month)

            author_map = build_author_map(
                con,
                month_paths,
                secret=secret,
                id_bytes=cfg.id_bytes,
                smoke=cfg.smoke,
                smoke_max_rows_core=cfg.smoke_max_rows_core,
                smoke_max_authors=cfg.smoke_max_authors,
            )

            if cfg.emit_map and not author_map.empty:
                tmp = author_map.copy()
                tmp["month"] = month
                map_rows.append(tmp)

            logger.info(
                "Month %s: distinct non-deleted authors = %d", month, len(author_map)
            )

            for kind in ("comments_core", "submissions_core"):
                if kind not in month_paths:
                    continue

                in_path = month_paths[kind]
                base = os.path.basename(in_path)
                out_dir = "comments" if kind == "comments_core" else "submissions"
                out_path = os.path.join(cfg.out_root, out_dir, base)

                if cfg.skip_existing and os.path.exists(out_path):
                    logger.info("Skip existing: %s", out_path)
                    continue

                diag = write_pseudonymised_core(
                    con=con,
                    in_path=in_path,
                    out_path=out_path,
                    author_map=author_map,
                    deleted_token=cfg.deleted_token,
                    keep_flair=cfg.keep_flair,
                    compression=cfg.compression,
                    smoke=cfg.smoke,
                    smoke_max_rows_core=cfg.smoke_max_rows_core,
                )


                logger.info(
                    "Wrote %s -> %s (rows=%d, should_map=%d, map_miss=%d)",
                    kind,
                    out_path,
                    diag["rows_total"],
                    diag["rows_should_map"],
                    diag["rows_map_miss"],
                )

                if (not cfg.smoke) and diag["rows_map_miss"] > 0:
                    logger.warning(
                        "Non-smoke run: %s has %d mapping misses out of %d rows that should map. "
                        "This suggests unexpected author normalization/join-key mismatch.",
                        in_path,
                        diag["rows_map_miss"],
                        diag["rows_should_map"],
                    )

                summary_rows.append(
                    {
                        "month": month,
                        "kind": kind,
                        "in_path": in_path,
                        "out_path": out_path,
                        "rows_total": diag["rows_total"],
                        "rows_author_null": diag["rows_author_null"],
                        "rows_author_deleted": diag["rows_author_deleted"],
                        "rows_should_map": diag["rows_should_map"],
                        "rows_map_miss": diag["rows_map_miss"],
                        "distinct_authors_in_map": int(len(author_map)),
                    }
                )

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = os.path.join(
                cfg.out_root, "audit", f"pseudonymise_authors_summary_{cfg.year}.csv"
            )
            summary_df.to_csv(summary_path, index=False)
            logger.info("Wrote summary CSV: %s", summary_path)

        if cfg.emit_map and map_rows:
            mp = pd.concat(map_rows, ignore_index=True).drop_duplicates(
                ["author_key", "author_pid"]
            )
            out_map = os.path.join(
                cfg.out_root, "audit", f"author_pid_map_{cfg.year}.csv"
            )
            mp.to_csv(out_map, index=False)
            logger.warning("Wrote PRIVATE author map (do not distribute): %s", out_map)

    finally:
        con.close()

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
