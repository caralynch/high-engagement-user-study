from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import pandas as pd


logger = logging.getLogger(__name__)


LANGUAGE_COLUMNS: List[str] = [
    "language_scan_status",
    "n_language_texts_sampled",
    "english_share",
    "non_english_share",
    "dominant_language",
    "language_filter_passed",
    "strict_language_filter_passed",
]


@dataclass(frozen=True)
class LanguageDetector:
    name: str
    detect: Callable[[str], Optional[str]]


@dataclass(frozen=True)
class SidecarInventory:
    comments_core: List[Path]
    comments_body: List[Path]
    submissions_core: List[Path]
    submissions_selftext: List[Path]


def initialize_language_columns(df: pd.DataFrame, status: str = "not_run") -> pd.DataFrame:
    """Add language audit columns with a common default status."""
    out = df.copy()
    out["language_scan_status"] = status
    out["n_language_texts_sampled"] = pd.NA
    out["english_share"] = pd.NA
    out["non_english_share"] = pd.NA
    out["dominant_language"] = pd.NA
    out["language_filter_passed"] = pd.NA
    out["strict_language_filter_passed"] = pd.NA
    return out


def mark_language_not_available(df: pd.DataFrame, pool: Sequence[str]) -> pd.DataFrame:
    """Mark the requested language pool as unscanned because no local detector exists."""
    out = initialize_language_columns(df, status="not_run")
    mask = out["subreddit"].astype(str).isin(set(pool))
    out.loc[mask, "language_scan_status"] = "not_available"
    out.loc[mask, "n_language_texts_sampled"] = 0
    return out


def _month_token(path: Path) -> str:
    stem = path.stem
    return stem.rsplit("_", 1)[-1]


def _discover_sidecars(in_root: Path, year: int) -> SidecarInventory:
    comments_dir = Path(in_root) / "comments"
    submissions_dir = Path(in_root) / "submissions"
    return SidecarInventory(
        comments_core=sorted(comments_dir.glob(f"comments_core_{year}-*.parquet")),
        comments_body=sorted(comments_dir.glob(f"comments_body_{year}-*.parquet")),
        submissions_core=sorted(submissions_dir.glob(f"submissions_core_{year}-*.parquet")),
        submissions_selftext=sorted(submissions_dir.glob(f"submissions_selftext_{year}-*.parquet")),
    )


def validate_text_sidecar_conventions(in_root: Path, year: int) -> SidecarInventory:
    """
    Validate the exact text sidecar conventions needed for sample-text mode.

    This function intentionally fails rather than guessing when files or schemas
    do not match the expected project layout.
    """
    inv = _discover_sidecars(Path(in_root), int(year))
    groups = {
        "comments_core": inv.comments_core,
        "comments_body": inv.comments_body,
        "submissions_core": inv.submissions_core,
        "submissions_selftext": inv.submissions_selftext,
    }
    missing_groups = [name for name, paths in groups.items() if not paths]
    if missing_groups:
        raise RuntimeError(
            "Cannot run --language-mode sample-text because expected text sidecar files are missing: "
            + ", ".join(missing_groups)
            + ". Expected comments/comments_core_YYYY-MM.parquet, comments/comments_body_YYYY-MM.parquet, "
            "submissions/submissions_core_YYYY-MM.parquet, and submissions/submissions_selftext_YYYY-MM.parquet."
        )

    month_sets = {name: {_month_token(path) for path in paths} for name, paths in groups.items()}
    first_name = next(iter(month_sets))
    first_months = month_sets[first_name]
    mismatched = [name for name, months in month_sets.items() if months != first_months]
    if mismatched:
        details = "; ".join(f"{name}={sorted(months)}" for name, months in month_sets.items())
        raise RuntimeError(
            "Cannot run --language-mode sample-text because text sidecar month coverage does not match: "
            + details
        )

    try:
        import duckdb
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("DuckDB is required for --language-mode sample-text sidecar validation.") from exc

    con = duckdb.connect()
    try:
        required = {
            "comments_core": ("id", "subreddit"),
            "comments_body": ("id", "body"),
            "submissions_core": ("id", "subreddit"),
            "submissions_selftext": ("id", "title", "selftext"),
        }
        sample_paths = {
            "comments_core": inv.comments_core[0],
            "comments_body": inv.comments_body[0],
            "submissions_core": inv.submissions_core[0],
            "submissions_selftext": inv.submissions_selftext[0],
        }
        for name, cols in required.items():
            available = list(
                con.execute(
                    f"SELECT * FROM read_parquet({_quote_sql_string(str(sample_paths[name]))}) LIMIT 0"
                ).df().columns
            )
            missing = [col for col in cols if col not in available]
            if missing:
                raise RuntimeError(
                    f"Cannot run --language-mode sample-text because {sample_paths[name]} is missing "
                    f"required column(s) {missing}. Available columns: {available}"
                )
    finally:
        con.close()
    return inv


def _quote_sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _read_parquet_list(paths: Sequence[Path]) -> str:
    quoted = ", ".join(_quote_sql_string(str(path)) for path in paths)
    return f"read_parquet([{quoted}])"


def _pool_values(pool: Sequence[str]) -> str:
    if not pool:
        return "SELECT CAST(NULL AS VARCHAR) AS subreddit WHERE FALSE"
    values = ", ".join(f"({_quote_sql_string(item)})" for item in pool)
    return f"SELECT * FROM (VALUES {values}) AS v(subreddit)"


def sample_language_texts(
    in_root: Path,
    year: int,
    pool: Sequence[str],
    sample_size_per_subreddit: int,
    random_seed: int,
) -> pd.DataFrame:
    """Sample candidate subreddit text deterministically from validated sidecars."""
    inv = validate_text_sidecar_conventions(in_root, year)
    if not pool:
        return pd.DataFrame(columns=["subreddit", "text"])

    try:
        import duckdb
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("DuckDB is required for --language-mode sample-text sampling.") from exc

    con = duckdb.connect()
    try:
        query = f"""
        WITH pool AS (
          {_pool_values(pool)}
        ),
        comment_texts AS (
          SELECT
            c.subreddit,
            CAST(b.body AS VARCHAR) AS text,
            CAST(c.id AS VARCHAR) AS item_id,
            'comment' AS source
          FROM {_read_parquet_list(inv.comments_core)} c
          INNER JOIN {_read_parquet_list(inv.comments_body)} b
            ON c.id = b.id
          INNER JOIN pool p
            ON c.subreddit = p.subreddit
        ),
        submission_texts AS (
          SELECT
            s.subreddit,
            TRIM(
              COALESCE(CAST(t.title AS VARCHAR), '') || ' ' ||
              CASE
                WHEN lower(trim(COALESCE(CAST(t.selftext AS VARCHAR), ''))) IN ('[deleted]', '[removed]')
                THEN ''
                ELSE COALESCE(CAST(t.selftext AS VARCHAR), '')
              END
            ) AS text,
            CAST(s.id AS VARCHAR) AS item_id,
            'submission' AS source
          FROM {_read_parquet_list(inv.submissions_core)} s
          INNER JOIN {_read_parquet_list(inv.submissions_selftext)} t
            ON s.id = t.id
          INNER JOIN pool p
            ON s.subreddit = p.subreddit
        ),
        cleaned AS (
          SELECT subreddit, text, item_id, source
          FROM (
            SELECT * FROM comment_texts
            UNION ALL
            SELECT * FROM submission_texts
          ) z
          WHERE text IS NOT NULL
            AND length(trim(text)) >= 20
            AND lower(trim(text)) NOT IN ('[deleted]', '[removed]')
        ),
        ranked AS (
          SELECT
            subreddit,
            text,
            row_number() OVER (
              PARTITION BY subreddit
              ORDER BY hash(CAST({int(random_seed)} AS VARCHAR) || source || subreddit || item_id || text), source, item_id
            ) AS rn
          FROM cleaned
        )
        SELECT subreddit, text
        FROM ranked
        WHERE rn <= {int(sample_size_per_subreddit)}
        ORDER BY subreddit, rn
        """
        return con.execute(query).df()
    finally:
        con.close()


def _normalise_language_code(code: Optional[str]) -> Optional[str]:
    if code is None:
        return None
    text = str(code).strip().lower()
    if not text:
        return None
    if text in {"english", "en_us", "en-gb", "en-us"}:
        return "en"
    if len(text) >= 2:
        return text[:2]
    return text


def find_language_detector() -> Optional[LanguageDetector]:
    """Return the first available local language detector."""
    if importlib.util.find_spec("langid") is not None:
        import langid  # type: ignore

        return LanguageDetector("langid", lambda text: _normalise_language_code(langid.classify(text)[0]))

    if importlib.util.find_spec("lingua") is not None:
        try:
            from lingua import LanguageDetectorBuilder  # type: ignore

            detector = LanguageDetectorBuilder.from_all_languages().build()

            def detect_lingua(text: str) -> Optional[str]:
                language = detector.detect_language_of(text)
                if language is None:
                    return None
                iso = getattr(language, "iso_code_639_1", None)
                name = getattr(iso, "name", None)
                if name:
                    return _normalise_language_code(name)
                if "english" in str(language).lower():
                    return "en"
                return _normalise_language_code(str(language))

            return LanguageDetector("lingua", detect_lingua)
        except Exception as exc:  # pragma: no cover - optional dependency API variance
            logger.warning("Could not initialise lingua language detector: %s", exc)

    if importlib.util.find_spec("langdetect") is not None:
        from langdetect import DetectorFactory, detect  # type: ignore

        DetectorFactory.seed = 0
        return LanguageDetector("langdetect", lambda text: _normalise_language_code(detect(text)))

    return None


def summarise_detected_languages(
    df: pd.DataFrame,
    pool: Sequence[str],
    detector: LanguageDetector,
    min_english_share: float,
    strict_min_english_share: float,
) -> pd.DataFrame:
    """Summarise sampled text classifications by subreddit."""
    rows: List[Dict[str, object]] = []
    grouped = {str(k): v for k, v in df.groupby("subreddit")}
    for subreddit in pool:
        sample = grouped.get(str(subreddit))
        if sample is None or sample.empty:
            rows.append(
                {
                    "subreddit": subreddit,
                    "language_scan_status": "insufficient_text",
                    "n_language_texts_sampled": 0,
                    "english_share": pd.NA,
                    "non_english_share": pd.NA,
                    "dominant_language": pd.NA,
                    "language_filter_passed": pd.NA,
                    "strict_language_filter_passed": pd.NA,
                }
            )
            continue

        detected: List[Optional[str]] = []
        status = "scanned"
        for text in sample["text"].astype(str).tolist():
            try:
                detected.append(detector.detect(text))
            except Exception:
                logger.exception("Language detection failed for subreddit %s", subreddit)
                status = "error"
                detected.append(None)
        clean = [code for code in detected if code]
        n = len(detected)
        if status == "error" and not clean:
            english_share = pd.NA
            non_english_share = pd.NA
            dominant = pd.NA
            passed = pd.NA
            strict_passed = pd.NA
        else:
            english_count = sum(1 for code in clean if code == "en")
            denom = len(clean)
            english_share = float(english_count / denom) if denom else pd.NA
            non_english_share = float(1.0 - english_share) if denom else pd.NA
            dominant = pd.Series(clean).value_counts().sort_values(ascending=False).index[0] if clean else pd.NA
            passed = bool(english_share >= float(min_english_share)) if denom else pd.NA
            strict_passed = bool(english_share >= float(strict_min_english_share)) if denom else pd.NA
        rows.append(
            {
                "subreddit": subreddit,
                "language_scan_status": status,
                "n_language_texts_sampled": n,
                "english_share": english_share,
                "non_english_share": non_english_share,
                "dominant_language": dominant,
                "language_filter_passed": passed,
                "strict_language_filter_passed": strict_passed,
            }
        )
    return pd.DataFrame(rows)


def apply_language_summary(df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Merge language summary values onto the candidate frame."""
    out = initialize_language_columns(df, status="not_run")
    if summary.empty:
        return out
    by_subreddit = summary.set_index("subreddit")
    mask = out["subreddit"].astype(str).isin(by_subreddit.index.astype(str))
    for column in LANGUAGE_COLUMNS:
        out.loc[mask, column] = out.loc[mask, "subreddit"].astype(str).map(by_subreddit[column])
    return out
