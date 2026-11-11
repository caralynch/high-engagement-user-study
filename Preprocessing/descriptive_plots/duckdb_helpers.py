from __future__ import annotations

from typing import List, Sequence

import duckdb

from .config import DuckDBConfig, Inventory
from .output_io import ensure_dir, quote_sql_string


def configure_duckdb(con: duckdb.DuckDBPyConnection, cfg: DuckDBConfig) -> None:
    """Apply DuckDB runtime settings."""
    ensure_dir(cfg.temp_dir)
    con.execute(f"SET memory_limit={quote_sql_string(cfg.memory_limit)}")
    con.execute(f"SET threads={int(cfg.threads)}")
    con.execute(f"SET temp_directory={quote_sql_string(cfg.temp_dir)}")
    if cfg.max_temp_directory_size:
        con.execute(f"SET max_temp_directory_size={quote_sql_string(cfg.max_temp_directory_size)}")
    con.execute("SET preserve_insertion_order=false")
    con.execute("PRAGMA disable_object_cache")

def build_relation_union(paths: Sequence[str], smoke_max_rows: int = 0) -> str:
    """Build a SQL fragment that unions one or more Parquet files."""
    if not paths:
        raise ValueError("No paths supplied to build_relation_union().")
    parts: List[str] = []
    for path in paths:
        lit = quote_sql_string(path)
        if smoke_max_rows > 0:
            parts.append(f"SELECT * FROM (SELECT * FROM read_parquet({lit}) LIMIT {int(smoke_max_rows)})")
        else:
            parts.append(f"SELECT * FROM read_parquet({lit})")
    return "\nUNION ALL\n".join(parts)

def get_table_columns(con: duckdb.DuckDBPyConnection, table_name: str) -> List[str]:
    """Return column names for a DuckDB table/view."""
    df = con.execute(f"DESCRIBE {table_name}").df()
    return df["column_name"].tolist()

def resolve_author_col(comments_cols: Sequence[str], submissions_cols: Sequence[str], requested: str) -> str:
    """Resolve the author identifier column."""
    comments_set = set(comments_cols)
    submissions_set = set(submissions_cols)
    if requested == "auto":
        if "author_pid" in comments_set and "author_pid" in submissions_set:
            return "author_pid"
        if "author" in comments_set and "author" in submissions_set:
            return "author"
        raise RuntimeError(
            "Could not resolve author column automatically. Neither 'author_pid' nor 'author' is present in both comments_core and submissions_core."
        )
    if requested not in comments_set or requested not in submissions_set:
        raise RuntimeError(
            f"Requested --author-col {requested!r} is not present in both comments_core and submissions_core."
        )
    return requested

def build_base_views(con: duckdb.DuckDBPyConnection, inv: Inventory, smoke_max_rows: int) -> None:
    """Create temp views for comments_core_all and submissions_core_all."""
    comments_paths = [mf.comments_core for mf in inv.values() if mf.comments_core]
    submissions_paths = [mf.submissions_core for mf in inv.values() if mf.submissions_core]
    con.execute(f"CREATE OR REPLACE TEMP VIEW comments_core_all AS {build_relation_union(comments_paths, smoke_max_rows)}")
    con.execute(f"CREATE OR REPLACE TEMP VIEW submissions_core_all AS {build_relation_union(submissions_paths, smoke_max_rows)}")

def build_user_level_views(con: duckdb.DuckDBPyConnection, author_col: str) -> None:
    """Create filtered views for user-level analysis excluding deleted authors."""
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW submissions_core_userlevel AS
        SELECT *
        FROM submissions_core_all
        WHERE COALESCE(is_deleted_author, FALSE) = FALSE
          AND {author_col} IS NOT NULL
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW comments_core_userlevel AS
        SELECT *
        FROM comments_core_all
        WHERE COALESCE(is_deleted_author, FALSE) = FALSE
          AND {author_col} IS NOT NULL
        """
    )
