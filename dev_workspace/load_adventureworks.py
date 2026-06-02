"""
Load AdventureWorks Excel export into a local PostgreSQL database.

Usage:
    uv run --with "xlrd,psycopg2-binary" dev_workspace/load_adventureworks.py
"""

from __future__ import annotations

import os
import re
import sys

import psycopg2
import xlrd

XLS_PATH = os.environ.get(
    "AW_XLS",
    os.path.expanduser("~/Downloads/AdventureWorks2019Export.xls"),
)
DB_NAME = os.environ.get("PGDATABASE", "adventureworks")
DB_USER = os.environ.get("PGUSER", os.environ.get("USER", "sai"))
DB_HOST = os.environ.get("PGHOST", "localhost")
DB_PORT = os.environ.get("PGPORT", "5432")
DB_PASSWORD = os.environ.get("PGPASSWORD", "")


def to_snake(name: str) -> str:
    """Convert CamelCase/PascalCase to snake_case, keeping 'ID' → 'id'."""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
    return s.lower()


def infer_pg_type(sheet: xlrd.sheet.Sheet, col_idx: int) -> str:
    for row in range(1, min(sheet.nrows, 30)):
        ctype = sheet.cell_type(row, col_idx)
        val = sheet.cell_value(row, col_idx)
        if ctype == xlrd.XL_CELL_DATE:
            return "TIMESTAMP"
        if ctype == xlrd.XL_CELL_NUMBER:
            if isinstance(val, float) and val != int(val):
                return "NUMERIC"
            return "BIGINT"
    return "TEXT"


def load(conn: psycopg2.extensions.connection, wb: xlrd.Book) -> None:
    cur = conn.cursor()
    for sheet_name in wb.sheet_names():
        ws = wb.sheet_by_name(sheet_name)
        table = to_snake(sheet_name)
        raw_headers = [str(ws.cell_value(0, c)) for c in range(ws.ncols)]
        headers = [to_snake(h) for h in raw_headers]
        col_types = [infer_pg_type(ws, c) for c in range(ws.ncols)]

        col_defs = ", ".join(f'"{h}" {t}' for h, t in zip(headers, col_types))
        cur.execute(f'DROP TABLE IF EXISTS "{table}"')
        cur.execute(f'CREATE TABLE "{table}" ({col_defs})')

        rows: list[tuple] = []
        for r in range(1, ws.nrows):
            row = []
            for c in range(ws.ncols):
                ctype = ws.cell_type(r, c)
                val = ws.cell_value(r, c)
                if ctype == xlrd.XL_CELL_DATE:
                    try:
                        row.append(xlrd.xldate_as_datetime(val, wb.datemode).isoformat())
                    except Exception:
                        row.append(None)
                elif ctype == xlrd.XL_CELL_EMPTY:
                    row.append(None)
                elif col_types[c] == "BIGINT" and isinstance(val, float):
                    row.append(int(val))
                else:
                    row.append(val if val != "" else None)
            rows.append(tuple(row))

        col_list = ", ".join(f'"{h}"' for h in headers)
        placeholders = ", ".join(["%s"] * ws.ncols)
        cur.executemany(
            f'INSERT INTO "{table}" ({col_list}) VALUES ({placeholders})', rows
        )
        conn.commit()
        print(f"  {table}: {len(rows)} rows  cols={headers}")

    cur.close()


def main() -> None:
    dsn_parts = [f"dbname={DB_NAME}", f"user={DB_USER}", f"host={DB_HOST}", f"port={DB_PORT}"]
    if DB_PASSWORD:
        dsn_parts.append(f"password={DB_PASSWORD}")
    conn = psycopg2.connect(" ".join(dsn_parts))
    try:
        wb = xlrd.open_workbook(XLS_PATH)
        print(f"Loading {XLS_PATH} → {DB_NAME} ...")
        load(conn, wb)
        print("Done.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
