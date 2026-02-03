"""PostgreSQL connection helpers for stock queries."""

from __future__ import annotations

import os
from typing import Any, Iterable

import psycopg
from psycopg import rows

DEFAULT_DB_HOST = os.getenv("PGHOST", "localhost")
DEFAULT_DB_PORT = int(os.getenv("PGPORT", "5431"))
DEFAULT_DB_NAME = os.getenv("PGDATABASE", "ai-grading-uat")
DEFAULT_DB_USER = os.getenv("PGUSER", "szekiat")
DEFAULT_DB_PASSWORD = os.getenv("PGPASSWORD", "")


def get_postgres_connection(
    *,
    host: str | None = None,
    port: int | None = None,
    dbname: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> psycopg.Connection:
    """Create a PostgreSQL connection using explicit params or environment defaults."""

    return psycopg.connect(
        host=host or DEFAULT_DB_HOST,
        port=port or DEFAULT_DB_PORT,
        dbname=dbname or DEFAULT_DB_NAME,
        user=user or DEFAULT_DB_USER,
        password=password if password is not None else DEFAULT_DB_PASSWORD,
    )


def fetch_stock_rows(
    sql: str,
    params: Iterable[Any] | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    dbname: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> list[dict[str, Any]]:
    """Run a read-only stock query and return rows as dictionaries."""

    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL query must be a non-empty string.")

    with get_postgres_connection(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
    ) as connection:
        with connection.cursor(row_factory=rows.dict_row) as cursor:
            cursor.execute(sql, params)
            return list(cursor.fetchall())
