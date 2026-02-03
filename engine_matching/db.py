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


def _qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _fetch_all(
    cursor: psycopg.Cursor, sql: str, params: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    cursor.execute(sql, params or {})
    cols = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    return [dict(zip(cols, row)) for row in rows]


def fetch_llm_table_profile(
    table: str,
    *,
    schema: str = "public",
    top_n_values: int = 10,
    sample_rows: int = 10,
    host: str | None = None,
    port: int | None = None,
    dbname: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-friendly profile of a table for LLM prompting."""

    qualified_table = f"{_qident(schema)}.{_qident(table)}"

    with get_postgres_connection(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
    ) as connection:
        with connection.cursor() as cursor:
            columns = _fetch_all(
                cursor,
                """
                SELECT
                  column_name,
                  data_type,
                  is_nullable
                FROM information_schema.columns
                WHERE table_schema = %(schema)s
                  AND table_name = %(table)s
                ORDER BY ordinal_position;
                """.strip(),
                {"schema": schema, "table": table},
            )
            if not columns:
                raise ValueError(f"No columns found for {schema}.{table}.")

            column_names = [column["column_name"] for column in columns]
            cursor.execute(f"SELECT COUNT(*)::bigint AS cnt FROM {qualified_table};")
            row_count = cursor.fetchone()[0]

            profile: list[dict[str, Any]] = []
            for column in columns:
                name = column["column_name"]
                dtype = column["data_type"]

                stats = _fetch_all(
                    cursor,
                    f"""
                    SELECT
                      COUNT(*)::bigint AS total_rows,
                      SUM(CASE WHEN {_qident(name)} IS NULL THEN 1 ELSE 0 END)::bigint AS null_rows,
                      COUNT(DISTINCT {_qident(name)})::bigint AS distinct_count
                    FROM {qualified_table};
                    """.strip(),
                )[0]

                col_info: dict[str, Any] = {
                    "name": name,
                    "type": dtype,
                    "nullable": column["is_nullable"] == "YES",
                    "null_pct": round((stats["null_rows"] / row_count) * 100, 2)
                    if row_count
                    else 0,
                    "distinct_count": stats["distinct_count"],
                }

                if dtype in {
                    "integer",
                    "bigint",
                    "numeric",
                    "real",
                    "double precision",
                    "smallint",
                    "date",
                    "timestamp without time zone",
                    "timestamp with time zone",
                }:
                    mm = _fetch_all(
                        cursor,
                        f"""
                        SELECT
                          MIN({_qident(name)}) AS min,
                          MAX({_qident(name)}) AS max
                        FROM {qualified_table};
                        """.strip(),
                    )[0]
                    col_info.update(mm)

                if dtype in {"text", "character varying", "character", "uuid"}:
                    top_vals = _fetch_all(
                        cursor,
                        f"""
                        SELECT
                          {_qident(name)} AS value,
                          COUNT(*)::bigint AS cnt
                        FROM {qualified_table}
                        WHERE {_qident(name)} IS NOT NULL
                        GROUP BY {_qident(name)}
                        ORDER BY cnt DESC
                        LIMIT {top_n_values};
                        """.strip(),
                    )
                    col_info["top_values"] = top_vals

                profile.append(col_info)

            samples: list[dict[str, Any]] = []
            if sample_rows > 0:
                cols_sql = ", ".join(_qident(name) for name in column_names)
                samples = _fetch_all(
                    cursor,
                    f"""
                    SELECT {cols_sql}
                    FROM {qualified_table}
                    LIMIT {sample_rows};
                    """.strip(),
                )

            return {
                "table": f"{schema}.{table}",
                "row_count": row_count,
                "columns": profile,
                "samples": samples,
            }
