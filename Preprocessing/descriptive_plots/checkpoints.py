from __future__ import annotations

import json
import logging
import os
import tempfile
from glob import glob
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import duckdb

from .config import (
    CHECKPOINT_MANIFEST_NAME,
    CHECKPOINT_PARTIAL_MANIFEST_NAME,
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_TABLES,
    MONTHLY_RECEIVED_ENGAGEMENT_AUDIT_CHECKPOINT_DIR,
    MONTHLY_RECEIVED_ENGAGEMENT_CHECKPOINT_MANIFEST_NAME,
    MONTHLY_USER_MONTH_PANEL_CHECKPOINT_DIR,
    PlotConfig,
    RECEIVED_ENGAGEMENT_DEFINITION_VERSION,
    RECEIVED_ENGAGEMENT_NORMALISATION_RULES,
    RECEIVED_ENGAGEMENT_SELF_ENGAGEMENT_SETTING,
)
from .output_io import (
    ensure_dir,
    log_table_count,
    normalize_metadata_path,
    quote_sql_string,
    utc_now_iso,
)


logger = logging.getLogger(__name__)

REUSABLE_RUN_STATUSES = {"partial", "started", "complete"}
MONTHLY_RECEIVED_ENGAGEMENT_LAYOUT_VERSION = "monthwise_parquet_v1"
DISABLED_MONOLITHIC_USER_MONTH_CHECKPOINTS = {
    "user_month_panel",
    "received_engagement_definition_audit",
}


def checkpoint_path(checkpoint_dir: str, table_name: str) -> str:
    """Return a checkpoint Parquet path for a table."""
    return os.path.join(checkpoint_dir, f"{table_name}.parquet")


def monthly_received_engagement_manifest_path(checkpoint_dir: str) -> str:
    """Return the manifest path for month-wise received-engagement checkpoints."""
    return os.path.join(checkpoint_dir, MONTHLY_RECEIVED_ENGAGEMENT_CHECKPOINT_MANIFEST_NAME)


def monthly_received_engagement_table_dir(checkpoint_dir: str, table_name: str) -> str:
    """Return the internal directory for one month-wise checkpoint table."""
    if table_name == "user_month_panel":
        dirname = MONTHLY_USER_MONTH_PANEL_CHECKPOINT_DIR
    elif table_name == "received_engagement_definition_audit":
        dirname = MONTHLY_RECEIVED_ENGAGEMENT_AUDIT_CHECKPOINT_DIR
    else:
        raise ValueError(f"Unsupported month-wise received-engagement checkpoint table: {table_name}")
    return os.path.join(checkpoint_dir, dirname)


def monthly_received_engagement_part_path(checkpoint_dir: str, table_name: str, month: str) -> str:
    """Return the Parquet path for one monthly received-engagement checkpoint part."""
    safe_month = str(month).replace("/", "_").replace("\\", "_")
    return os.path.join(
        monthly_received_engagement_table_dir(checkpoint_dir, table_name),
        f"{table_name}_{safe_month}.parquet",
    )


def checkpoint_manifest_path(checkpoint_dir: str) -> str:
    """Return the checkpoint manifest path."""
    return os.path.join(checkpoint_dir, CHECKPOINT_MANIFEST_NAME)


def checkpoint_partial_manifest_path(checkpoint_dir: str) -> str:
    """Return the legacy partial checkpoint manifest path."""
    return os.path.join(checkpoint_dir, CHECKPOINT_PARTIAL_MANIFEST_NAME)


def checkpoint_file_exists(checkpoint_dir: str, table_name: str) -> bool:
    """Return whether a checkpoint Parquet exists and is safe to load."""
    if table_name in DISABLED_MONOLITHIC_USER_MONTH_CHECKPOINTS:
        logger.warning(
            "Refusing legacy monolithic checkpoint reuse for %s; use month-wise received-engagement checkpoints.",
            table_name,
        )
        return False
    path = checkpoint_path(checkpoint_dir, table_name)
    if not os.path.exists(path):
        return False

    manifest = _read_reuse_manifest(checkpoint_dir)
    if manifest is None:
        logger.warning(
            "Checkpoint file exists for %s at %s, but no checkpoint manifest could be read; "
            "reusing only for legacy compatibility after the metadata gate.",
            table_name,
            path,
        )
        return True

    explicit_completed = _explicit_completed_table_entries(manifest)
    if table_name in explicit_completed:
        run_status = _manifest_run_status(manifest)
        if run_status in {"partial", "started"}:
            logger.info(
                "Reusing checkpoint table %s from %s run manifest: %s",
                table_name,
                run_status,
                path,
            )
        else:
            logger.info("Reusing checkpoint table %s from checkpoint manifest: %s", table_name, path)
        return True

    if _manifest_has_per_table_completion(manifest):
        logger.warning(
            "Checkpoint file exists for %s at %s, but the compatible manifest does not mark it complete; rebuilding.",
            table_name,
            path,
        )
        return False

    logger.warning(
        "Checkpoint file exists for %s at %s, but the compatible manifest has no per-table completion list; "
        "reusing for backwards compatibility.",
        table_name,
        path,
    )
    return True


def write_checkpoint_table(
    con: duckdb.DuckDBPyConnection,
    checkpoint_dir: str,
    table_name: str,
) -> None:
    """Persist a compact checkpoint table as zstd Parquet."""
    ensure_dir(checkpoint_dir)
    out_path = checkpoint_path(checkpoint_dir, table_name)
    if os.path.exists(out_path):
        os.remove(out_path)
    con.execute(
        f"""
        COPY (SELECT * FROM {table_name})
        TO {quote_sql_string(out_path)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    row_count = _try_table_row_count(con, table_name)
    logger.info("Wrote checkpoint table %s -> %s", table_name, out_path)
    mark_checkpoint_table_complete(checkpoint_dir, table_name, row_count=row_count)


def load_checkpoint_table(
    con: duckdb.DuckDBPyConnection,
    checkpoint_dir: str,
    table_name: str,
) -> None:
    """Load a checkpoint Parquet file into a temp table with the same name."""
    in_path = checkpoint_path(checkpoint_dir, table_name)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {table_name} AS
        SELECT *
        FROM read_parquet({quote_sql_string(in_path)})
        """
    )
    logger.info("Loaded checkpoint table %s <- %s", table_name, in_path)
    log_table_count(con, table_name)


def build_checkpoint_metadata(
    cfg: "PlotConfig",
    months: Sequence[str],
    author_col: str,
) -> Dict[str, object]:
    """Build the metadata used to decide whether checkpoints are reusable."""
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "selected_months": list(months),
        "year": str(cfg.year),
        "author_col": str(author_col),
        "smoke": bool(cfg.smoke),
        "smoke_max_rows": int(cfg.smoke_max_rows),
        "in_root": normalize_metadata_path(cfg.in_root),
        "min_monthly_active_users_for_subset": int(cfg.min_monthly_active_users_for_subset),
        "nsfw_submission_share_threshold_for_subset": float(cfg.nsfw_submission_share_threshold_for_subset),
        "checkpoint_tables": list(CHECKPOINT_TABLES),
        "received_engagement_definition_version": RECEIVED_ENGAGEMENT_DEFINITION_VERSION,
        "received_engagement_self_engagement_setting": RECEIVED_ENGAGEMENT_SELF_ENGAGEMENT_SETTING,
        "received_engagement_normalisation_rules": dict(RECEIVED_ENGAGEMENT_NORMALISATION_RULES),
        "received_engagement_checkpoint_layout": MONTHLY_RECEIVED_ENGAGEMENT_LAYOUT_VERSION,
        "script_metadata": {
            "script_name": "Preprocessing/06_0_descriptive_plots.py",
            "helper_module": "Preprocessing.descriptive_plots",
            "checkpoint_implementation": "monthwise_received_engagement",
        },
    }


def build_monthly_received_engagement_metadata(
    expected: Mapping[str, object],
    months: Sequence[str],
    author_col: str,
) -> Dict[str, object]:
    """Build strict metadata for month-wise user-month received-engagement checkpoints."""
    selected_months = list(months)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_kind": "monthwise_user_month_received_engagement",
        "checkpoint_layout": MONTHLY_RECEIVED_ENGAGEMENT_LAYOUT_VERSION,
        "input_root": expected.get("in_root"),
        "year": str(expected.get("year")),
        "selected_months": selected_months,
        "received_engagement_definition_version": RECEIVED_ENGAGEMENT_DEFINITION_VERSION,
        "self_engagement_setting": RECEIVED_ENGAGEMENT_SELF_ENGAGEMENT_SETTING,
        "author_col": str(author_col),
        "normalisation_rules": dict(RECEIVED_ENGAGEMENT_NORMALISATION_RULES),
        "script_metadata": {
            "script_name": "Preprocessing/06_0_descriptive_plots.py",
            "helper_module": "Preprocessing.descriptive_plots.derived_tables",
            "checkpoint_implementation": "monthwise_received_engagement",
        },
        "smoke": bool(expected.get("smoke")),
        "smoke_max_rows": int(expected.get("smoke_max_rows", 0)),
    }


def _atomic_json_dump(obj: Mapping[str, object], path: str) -> None:
    """Write JSON atomically to reduce corrupt manifests from killed jobs."""
    ensure_dir(os.path.dirname(path))
    tmp_path = ""
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=os.path.dirname(path),
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        delete=False,
    ) as f:
        tmp_path = f.name
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    try:
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _read_json_file(path: str) -> Optional[Dict[str, object]]:
    """Read a JSON object, returning None if it is absent or invalid."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning(
            "Checkpoint metadata could not be read from %s; ignoring it and rebuilding as needed: %s",
            path,
            exc,
        )
        return None
    if not isinstance(data, dict):
        logger.warning("Checkpoint metadata in %s is not a JSON object; ignoring it.", path)
        return None
    return data


def _metadata_equal(expected: Mapping[str, object], observed: Mapping[str, object]) -> Tuple[bool, List[str]]:
    """Compare metadata dictionaries, returning human-readable mismatch strings."""
    mismatches: List[str] = []
    for key, expected_value in expected.items():
        observed_value = observed.get(key)
        if observed_value != expected_value:
            mismatches.append(f"{key}: expected {expected_value!r}, found {observed_value!r}")
    return not mismatches, mismatches


def read_monthly_received_engagement_manifest(checkpoint_dir: str) -> Optional[Dict[str, object]]:
    """Read the month-wise received-engagement checkpoint manifest, if present."""
    return _read_json_file(monthly_received_engagement_manifest_path(checkpoint_dir))


def monthly_received_engagement_checkpoint_status(
    checkpoint_dir: str,
    expected_metadata: Mapping[str, object],
) -> Dict[str, object]:
    """Return metadata/coverage status for month-wise received-engagement checkpoints."""
    manifest_path = monthly_received_engagement_manifest_path(checkpoint_dir)
    manifest = read_monthly_received_engagement_manifest(checkpoint_dir)
    if manifest is None:
        return {
            "allowed": False,
            "source": manifest_path,
            "completed_months": [],
            "reason": "month-wise received-engagement checkpoint manifest was absent or unreadable",
        }

    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        return {
            "allowed": False,
            "source": manifest_path,
            "completed_months": [],
            "reason": "month-wise received-engagement checkpoint manifest has no metadata object",
        }

    matches, mismatches = _metadata_equal(expected_metadata, metadata)
    if not matches:
        _log_metadata_mismatches(manifest_path, mismatches)
        return {
            "allowed": False,
            "source": manifest_path,
            "completed_months": [],
            "reason": "month-wise received-engagement checkpoint metadata did not match",
        }

    completed_raw = manifest.get("completed_months")
    completed_months = sorted(str(month) for month in completed_raw) if isinstance(completed_raw, list) else []
    return {
        "allowed": True,
        "source": manifest_path,
        "completed_months": completed_months,
        "reason": "month-wise received-engagement checkpoint metadata matched",
        "run_status": str(manifest.get("run_status") or "partial"),
    }


def write_monthly_received_engagement_manifest(
    checkpoint_dir: str,
    metadata: Mapping[str, object],
    *,
    completed_months: Sequence[str],
    run_status: str,
) -> None:
    """Write the month-wise received-engagement checkpoint manifest atomically."""
    ensure_dir(checkpoint_dir)
    path = monthly_received_engagement_manifest_path(checkpoint_dir)
    existing = read_monthly_received_engagement_manifest(checkpoint_dir) or {}
    created_at = existing.get("created_at_utc") or utc_now_iso()
    payload = {
        "manifest_type": "monthwise_received_engagement_checkpoint",
        "run_status": run_status,
        "created_at_utc": created_at,
        "updated_at_utc": utc_now_iso(),
        "metadata": dict(metadata),
        "completed_months": sorted(dict.fromkeys(str(month) for month in completed_months)),
    }
    _atomic_json_dump(payload, path)
    logger.info("Wrote month-wise received-engagement checkpoint manifest: %s", path)


def mark_monthly_received_engagement_month_complete(
    checkpoint_dir: str,
    metadata: Mapping[str, object],
    month: str,
) -> None:
    """Mark one month complete in the month-wise received-engagement checkpoint manifest."""
    status = monthly_received_engagement_checkpoint_status(checkpoint_dir, metadata)
    completed = set(status["completed_months"]) if status["allowed"] else set()
    completed.add(str(month))
    write_monthly_received_engagement_manifest(
        checkpoint_dir,
        metadata,
        completed_months=sorted(completed),
        run_status="partial",
    )


def existing_monthly_received_engagement_part_months(checkpoint_dir: str, table_name: str) -> List[str]:
    """Return months with existing Parquet parts for one month-wise checkpoint table."""
    table_dir = monthly_received_engagement_table_dir(checkpoint_dir, table_name)
    prefix = f"{table_name}_"
    months: List[str] = []
    for path in sorted(glob(os.path.join(table_dir, f"{prefix}*.parquet"))):
        base = os.path.basename(path)
        if not base.startswith(prefix) or not base.endswith(".parquet"):
            continue
        months.append(base[len(prefix) : -len(".parquet")])
    return months


def validate_monthly_received_engagement_checkpoint_parts(
    con: duckdb.DuckDBPyConnection,
    checkpoint_dir: str,
    months: Sequence[str],
    *,
    allow_empty: bool,
) -> None:
    """Validate exact month coverage and non-empty monthly received-engagement checkpoint parts."""
    expected = list(months)
    expected_set = set(expected)
    for table_name in ("user_month_panel", "received_engagement_definition_audit"):
        observed = existing_monthly_received_engagement_part_months(checkpoint_dir, table_name)
        observed_set = set(observed)
        if observed_set != expected_set:
            missing = sorted(expected_set - observed_set)
            extra = sorted(observed_set - expected_set)
            parts: List[str] = []
            if missing:
                parts.append("missing months: " + ", ".join(missing))
            if extra:
                parts.append("unexpected months: " + ", ".join(extra))
            raise RuntimeError(
                f"Month-wise checkpoint parts for {table_name} do not exactly cover the requested month set ("
                + "; ".join(parts)
                + ")."
            )
        for month in expected:
            path = monthly_received_engagement_part_path(checkpoint_dir, table_name, month)
            try:
                n_rows = int(
                    con.execute(
                        f"SELECT COUNT(*) FROM read_parquet({quote_sql_string(path)})"
                    ).fetchone()[0]
                )
            except Exception as exc:
                raise RuntimeError(f"Could not validate monthly checkpoint part {path}: {exc}") from exc
            if n_rows == 0 and not allow_empty:
                raise RuntimeError(
                    f"Monthly checkpoint part is unexpectedly empty for {table_name} {month}: {path}. "
                    "Use smoke/tiny mode only for intentionally sparse fixtures."
                )


def _read_reuse_manifest(checkpoint_dir: str) -> Optional[Dict[str, object]]:
    """Read the canonical manifest, falling back to the old partial manifest."""
    manifest = _read_json_file(checkpoint_manifest_path(checkpoint_dir))
    if manifest is not None:
        return manifest
    return _read_json_file(checkpoint_partial_manifest_path(checkpoint_dir))


def _metadata_mismatches(expected: Dict[str, object], observed: Dict[str, object]) -> List[str]:
    """Return human-readable mismatches between expected and observed metadata."""
    mismatches: List[str] = []
    for key, expected_value in expected.items():
        observed_value = observed.get(key)
        if observed_value != expected_value:
            mismatches.append(f"{key}: expected {expected_value!r}, found {observed_value!r}")
    return mismatches


def _log_metadata_mismatches(source: str, mismatches: Sequence[str]) -> None:
    """Log checkpoint metadata mismatches in a compact reusable form."""
    if mismatches:
        logger.warning(
            "Refusing checkpoint reuse from %s because metadata do not match:\n  - %s",
            source,
            "\n  - ".join(mismatches),
        )


def _manifest_run_status(manifest: Dict[str, object]) -> str:
    """Return normalized manifest run status."""
    status = manifest.get("run_status")
    if isinstance(status, str) and status:
        return status
    manifest_type = manifest.get("manifest_type")
    if manifest_type == "final":
        return "complete"
    if manifest_type == "partial":
        return "partial"
    return "complete"


def _manifest_has_per_table_completion(manifest: Dict[str, object]) -> bool:
    """Return whether a manifest explicitly tracks table completion."""
    return any(key in manifest for key in ("completed_tables", "completed_checkpoint_tables", "available_tables"))


def _table_name_from_completed_item(default_name: str, item: Mapping[str, object]) -> str:
    """Extract a checkpoint table name from a completed-table entry."""
    raw_name = item.get("table_name") or item.get("name") or default_name
    return str(raw_name)


def _explicit_completed_table_entries(manifest: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    """Return table-completion entries explicitly recorded in a manifest."""
    completed: Dict[str, Dict[str, object]] = {}

    value = manifest.get("completed_tables")
    if isinstance(value, dict):
        for default_name, raw_entry in value.items():
            if isinstance(raw_entry, dict):
                status = str(raw_entry.get("status", "complete"))
                if status != "complete":
                    continue
                table_name = _table_name_from_completed_item(str(default_name), raw_entry)
                completed[table_name] = {**raw_entry, "table_name": table_name, "status": "complete"}
            else:
                table_name = str(default_name)
                completed[table_name] = {"table_name": table_name, "status": "complete"}
    elif isinstance(value, list):
        for raw_entry in value:
            if isinstance(raw_entry, dict):
                status = str(raw_entry.get("status", "complete"))
                if status != "complete":
                    continue
                table_name = _table_name_from_completed_item("", raw_entry)
                if table_name:
                    completed[table_name] = {**raw_entry, "table_name": table_name, "status": "complete"}
            else:
                table_name = str(raw_entry)
                completed[table_name] = {"table_name": table_name, "status": "complete"}

    for key in ("completed_checkpoint_tables", "available_tables"):
        legacy_value = manifest.get(key)
        if isinstance(legacy_value, list):
            for raw_name in legacy_value:
                table_name = str(raw_name)
                completed.setdefault(table_name, {"table_name": table_name, "status": "complete"})

    disabled = DISABLED_MONOLITHIC_USER_MONTH_CHECKPOINTS
    return {name: entry for name, entry in completed.items() if name in set(CHECKPOINT_TABLES) and name not in disabled}


def _completed_tables_from_manifest(manifest: Dict[str, object]) -> Set[str]:
    """Return table names marked or inferred complete in a checkpoint manifest."""
    explicit = _explicit_completed_table_entries(manifest)
    if explicit:
        return set(explicit)
    value = manifest.get("checkpoint_tables")
    if isinstance(value, list) and _manifest_run_status(manifest) == "complete":
        return {str(name) for name in value}
    return set()


def _completion_entry(
    checkpoint_dir: str,
    table_name: str,
    *,
    row_count: Optional[int] = None,
    completed_at_utc: Optional[str] = None,
    path: Optional[str] = None,
) -> Dict[str, object]:
    """Build a manifest entry for one completed checkpoint table."""
    entry: Dict[str, object] = {
        "table_name": table_name,
        "path": path or os.path.basename(checkpoint_path(checkpoint_dir, table_name)),
        "status": "complete",
        "completed_at_utc": completed_at_utc or utc_now_iso(),
    }
    if row_count is not None:
        entry["row_count"] = int(row_count)
    return entry


def _normalise_completed_table_inputs(
    checkpoint_dir: str,
    completed_tables: Optional[object],
) -> Dict[str, Dict[str, object]]:
    """Normalise caller-supplied completed-table names or entries."""
    if completed_tables is None:
        return {}

    items: Iterable[object]
    if isinstance(completed_tables, dict):
        items = [
            {**entry, "table_name": name} if isinstance(entry, dict) else {"table_name": name}
            for name, entry in completed_tables.items()
        ]
    elif isinstance(completed_tables, (list, tuple, set)):
        items = list(completed_tables)
    else:
        items = [completed_tables]

    completed: Dict[str, Dict[str, object]] = {}
    for item in items:
        if isinstance(item, dict):
            table_name = _table_name_from_completed_item("", item)
            if not table_name or table_name not in set(CHECKPOINT_TABLES):
                continue
            completed[table_name] = _completion_entry(
                checkpoint_dir,
                table_name,
                row_count=int(item["row_count"]) if item.get("row_count") is not None else None,
                completed_at_utc=str(item.get("completed_at_utc")) if item.get("completed_at_utc") else None,
                path=str(item.get("path")) if item.get("path") else None,
            )
        else:
            table_name = str(item)
            if table_name not in set(CHECKPOINT_TABLES):
                continue
            completed[table_name] = _completion_entry(checkpoint_dir, table_name)
    return completed


def _manifest_payload(
    checkpoint_dir: str,
    metadata: Dict[str, object],
    *,
    run_status: str,
    completed_tables: Optional[object] = None,
) -> Dict[str, object]:
    """Build the canonical checkpoint manifest payload."""
    completed = _normalise_completed_table_inputs(checkpoint_dir, completed_tables)
    return {
        **metadata,
        "manifest_type": "checkpoint",
        "run_status": run_status,
        "updated_at_utc": utc_now_iso(),
        "completed_tables": completed,
        "completed_checkpoint_tables": sorted(completed),
    }


def _existing_checkpoint_tables(checkpoint_dir: str) -> List[str]:
    """Return checkpoint table names whose Parquet files currently exist."""
    return [
        table_name
        for table_name in CHECKPOINT_TABLES
        if table_name not in DISABLED_MONOLITHIC_USER_MONTH_CHECKPOINTS
        if os.path.exists(checkpoint_path(checkpoint_dir, table_name))
    ]


def _try_table_row_count(con: duckdb.DuckDBPyConnection, table_name: str) -> Optional[int]:
    """Return a table row count if available without failing checkpoint writes."""
    try:
        return int(con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
    except Exception as exc:
        logger.warning("Could not record row count for checkpoint table %s: %s", table_name, exc)
        return None


def _checkpoint_months_match(checkpoint_dir: str, expected_months: Sequence[str]) -> Tuple[bool, str]:
    """Validate selected months against a legacy checkpoint Parquet file."""
    expected = list(expected_months)
    for table_name in ("user_month_panel", "subreddit_month_summary"):
        path = checkpoint_path(checkpoint_dir, table_name)
        if not os.path.exists(path):
            continue
        try:
            con = duckdb.connect()
            try:
                rows = con.execute(
                    f"""
                    SELECT DISTINCT CAST(month AS VARCHAR) AS month
                    FROM read_parquet({quote_sql_string(path)})
                    ORDER BY 1
                    """
                ).fetchall()
            finally:
                con.close()
        except Exception as exc:
            return False, f"could not inspect months in {path}: {exc}"
        observed = [str(row[0]) for row in rows]
        if observed != expected:
            return False, f"selected_months: expected {expected!r}, found {observed!r} in {table_name}"
        return True, f"validated selected_months from {table_name}"
    return False, "no checkpoint table with a month column was available for legacy month validation"


def _legacy_config_candidates(checkpoint_dir: str) -> List[str]:
    """Return likely run-config JSON files adjacent to a checkpoint directory."""
    out_root = os.path.dirname(os.path.abspath(checkpoint_dir))
    audit_dir = os.path.join(out_root, "audit")
    paths = glob(os.path.join(audit_dir, "descriptive_plots_config_*.json"))
    return sorted(paths, key=lambda p: os.path.getmtime(p), reverse=True)


def _legacy_config_mismatches(expected: Dict[str, object], payload: Dict[str, object]) -> List[str]:
    """Compare a start-of-run config JSON with expected checkpoint metadata."""
    cfg_obj = payload.get("config")
    if not isinstance(cfg_obj, dict):
        return ["legacy config has no top-level config object"]

    mismatches: List[str] = []
    scalar_checks: List[Tuple[str, object]] = [
        ("year", str(expected["year"])),
        ("smoke", bool(expected["smoke"])),
        ("smoke_max_rows", int(expected["smoke_max_rows"])),
        ("min_monthly_active_users_for_subset", int(expected["min_monthly_active_users_for_subset"])),
    ]
    for key, expected_value in scalar_checks:
        observed_raw = cfg_obj.get(key)
        try:
            if isinstance(expected_value, bool):
                observed_value = bool(observed_raw)
            elif isinstance(expected_value, int):
                observed_value = int(observed_raw)
            else:
                observed_value = str(observed_raw)
        except (TypeError, ValueError):
            observed_value = observed_raw
        if observed_value != expected_value:
            mismatches.append(f"{key}: expected {expected_value!r}, found {observed_raw!r}")

    observed_nsfw = cfg_obj.get("nsfw_submission_share_threshold_for_subset")
    try:
        observed_nsfw_value = float(observed_nsfw)
    except (TypeError, ValueError):
        observed_nsfw_value = None
    if observed_nsfw_value is None or observed_nsfw_value != float(expected["nsfw_submission_share_threshold_for_subset"]):
        mismatches.append(
            "nsfw_submission_share_threshold_for_subset: "
            f"expected {expected['nsfw_submission_share_threshold_for_subset']!r}, found {observed_nsfw!r}"
        )

    observed_in_root = cfg_obj.get("in_root")
    expected_in_root = expected.get("in_root")
    if observed_in_root is None or normalize_metadata_path(str(observed_in_root)) != expected_in_root:
        mismatches.append(f"in_root: expected {expected_in_root!r}, found {observed_in_root!r}")

    requested_author_col = str(cfg_obj.get("author_col"))
    expected_author_col = str(expected["author_col"])
    if requested_author_col not in {expected_author_col, "auto"}:
        mismatches.append(f"author_col: expected {expected_author_col!r} or 'auto', found {requested_author_col!r}")

    return mismatches


def _legacy_checkpoint_status(checkpoint_dir: str, expected: Dict[str, object]) -> Dict[str, object]:
    """Validate old failed-run checkpoints using adjacent start config JSON."""
    candidates = _legacy_config_candidates(checkpoint_dir)
    if not candidates:
        return {
            "allowed": False,
            "source": "none",
            "completed_tables": [],
            "reason": "no checkpoint manifest or adjacent start-of-run checkpoint metadata was found",
        }
    for path in candidates:
        payload = _read_json_file(path)
        if payload is None:
            continue
        mismatches = _legacy_config_mismatches(expected, payload)
        if mismatches:
            logger.info("Legacy checkpoint config did not match current run metadata: %s", path)
            logger.debug("Legacy checkpoint config mismatches: %s", "; ".join(mismatches))
            continue
        months_ok, month_reason = _checkpoint_months_match(checkpoint_dir, expected["selected_months"])
        if not months_ok:
            logger.warning("Refusing legacy checkpoint reuse from %s: %s", path, month_reason)
            continue
        completed_tables = _existing_checkpoint_tables(checkpoint_dir)
        if not completed_tables:
            return {
                "allowed": True,
                "source": path,
                "completed_tables": [],
                "reason": "legacy metadata matched but no checkpoint Parquet files were present",
                "run_status": "legacy",
            }
        logger.warning(
            "Legacy checkpoint metadata validated from %s (%s); reusing existing checkpoint files without "
            "per-table completion records: %s",
            path,
            month_reason,
            ", ".join(completed_tables),
        )
        return {
            "allowed": True,
            "source": path,
            "completed_tables": completed_tables,
            "reason": "legacy start-of-run config and checkpoint months matched",
            "run_status": "legacy",
        }
    return {
        "allowed": False,
        "source": "legacy_config",
        "completed_tables": [],
        "reason": "adjacent legacy config files were present but none matched current metadata",
    }


def _status_from_manifest(
    checkpoint_dir: str,
    path: str,
    manifest: Dict[str, object],
    expected: Dict[str, object],
) -> Dict[str, object]:
    """Return checkpoint reuse status for one manifest object."""
    mismatches = _metadata_mismatches(expected, manifest)
    if mismatches:
        _log_metadata_mismatches(path, mismatches)
        return {
            "allowed": False,
            "source": path,
            "completed_tables": [],
            "reason": "manifest metadata did not match",
        }

    run_status = _manifest_run_status(manifest)
    if manifest.get("run_status") is not None and run_status not in REUSABLE_RUN_STATUSES:
        logger.warning(
            "Refusing checkpoint reuse from %s because run_status=%r is not reusable.",
            path,
            run_status,
        )
        return {
            "allowed": False,
            "source": path,
            "completed_tables": [],
            "reason": f"manifest run_status={run_status!r} is not reusable",
        }

    explicit_completed = sorted(_explicit_completed_table_entries(manifest))
    if explicit_completed:
        if run_status in {"partial", "started"}:
            logger.info(
                "Partial checkpoint manifest validated: %s; completed tables: %s",
                path,
                ", ".join(explicit_completed),
            )
        else:
            logger.info("Checkpoint manifest validated: %s; completed tables: %s", path, ", ".join(explicit_completed))
        return {
            "allowed": True,
            "source": path,
            "completed_tables": explicit_completed,
            "reason": f"{run_status} manifest matched",
            "run_status": run_status,
        }

    if _manifest_has_per_table_completion(manifest):
        logger.info("Checkpoint manifest matched but no tables are marked complete: %s", path)
        return {
            "allowed": True,
            "source": path,
            "completed_tables": [],
            "reason": f"{run_status} manifest matched but no tables were marked complete",
            "run_status": run_status,
        }

    legacy_completed = _existing_checkpoint_tables(checkpoint_dir)
    if legacy_completed:
        logger.warning(
            "Checkpoint manifest %s has compatible metadata but no per-table completion list; "
            "existing checkpoint files may be reused for backwards compatibility: %s",
            path,
            ", ".join(legacy_completed),
        )
    else:
        logger.info("Checkpoint manifest matched but no checkpoint files are present: %s", path)
    return {
        "allowed": True,
        "source": path,
        "completed_tables": legacy_completed,
        "reason": f"{run_status} manifest matched without per-table completion records",
        "run_status": run_status,
    }


def checkpoint_reuse_status(checkpoint_dir: str, expected: Dict[str, object]) -> Dict[str, object]:
    """Return whether checkpoint metadata allow safe reuse and which tables exist."""
    manifest_path = checkpoint_manifest_path(checkpoint_dir)
    manifest_exists = os.path.exists(manifest_path)
    manifest = _read_json_file(manifest_path)
    if manifest is not None:
        return _status_from_manifest(checkpoint_dir, manifest_path, manifest, expected)
    if manifest_exists:
        return {
            "allowed": False,
            "source": manifest_path,
            "completed_tables": [],
            "reason": "checkpoint manifest was unreadable or corrupt",
        }

    partial_path = checkpoint_partial_manifest_path(checkpoint_dir)
    partial = _read_json_file(partial_path)
    if partial is not None:
        return _status_from_manifest(checkpoint_dir, partial_path, partial, expected)

    status = _legacy_checkpoint_status(checkpoint_dir, expected)
    if not bool(status["allowed"]):
        logger.info("Checkpoint reuse unavailable: %s", status["reason"])
    return status


def checkpoint_manifest_matches(checkpoint_dir: str, expected: Dict[str, object]) -> bool:
    """Validate checkpoint metadata against expected run metadata."""
    return bool(checkpoint_reuse_status(checkpoint_dir, expected)["allowed"])


def write_checkpoint_started_manifest(
    checkpoint_dir: str,
    metadata: Dict[str, object],
    completed_tables: Optional[object] = None,
) -> None:
    """Write the canonical checkpoint manifest at run start."""
    ensure_dir(checkpoint_dir)
    payload = _manifest_payload(
        checkpoint_dir,
        metadata,
        run_status="partial",
        completed_tables=completed_tables,
    )
    _atomic_json_dump(payload, checkpoint_manifest_path(checkpoint_dir))
    logger.info("Wrote partial checkpoint manifest: %s", checkpoint_manifest_path(checkpoint_dir))


def write_checkpoint_partial_manifest(
    checkpoint_dir: str,
    metadata: Dict[str, object],
    completed_tables: Optional[object] = None,
) -> None:
    """Backward-compatible alias for writing a started checkpoint manifest."""
    write_checkpoint_started_manifest(checkpoint_dir, metadata, completed_tables=completed_tables)


def mark_checkpoint_table_complete(
    checkpoint_dir: str,
    table_name: str,
    row_count: Optional[int] = None,
) -> None:
    """Mark a checkpoint table complete in the canonical manifest."""
    manifest_path = checkpoint_manifest_path(checkpoint_dir)
    manifest = _read_json_file(manifest_path)
    if manifest is None:
        logger.warning(
            "Cannot mark checkpoint table %s complete because %s is missing or unreadable.",
            table_name,
            manifest_path,
        )
        return

    completed = _explicit_completed_table_entries(manifest)
    completed[table_name] = _completion_entry(checkpoint_dir, table_name, row_count=row_count)
    manifest["manifest_type"] = "checkpoint"
    manifest["run_status"] = _manifest_run_status(manifest)
    manifest["updated_at_utc"] = utc_now_iso()
    manifest["completed_tables"] = completed
    manifest["completed_checkpoint_tables"] = sorted(completed)
    _atomic_json_dump(manifest, manifest_path)
    logger.info("Marked checkpoint table %s complete in manifest.", table_name)


def write_checkpoint_manifest(checkpoint_dir: str, metadata: Dict[str, object]) -> None:
    """Mark the checkpoint manifest complete after a successful full run."""
    ensure_dir(checkpoint_dir)
    existing_manifest = _read_json_file(checkpoint_manifest_path(checkpoint_dir)) or {}
    completed = _explicit_completed_table_entries(existing_manifest)
    for table_name in _existing_checkpoint_tables(checkpoint_dir):
        completed.setdefault(table_name, _completion_entry(checkpoint_dir, table_name))
    payload = _manifest_payload(
        checkpoint_dir,
        metadata,
        run_status="complete",
        completed_tables=completed,
    )
    _atomic_json_dump(payload, checkpoint_manifest_path(checkpoint_dir))
    logger.info("Marked checkpoint manifest complete: %s", checkpoint_manifest_path(checkpoint_dir))
