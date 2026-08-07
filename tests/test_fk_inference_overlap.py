"""Tests for FK overlap containment validation."""

from __future__ import annotations

from unittest.mock import MagicMock

from aetherdialect._config import PolicyConfig
from aetherdialect._contracts_base import (
    InferenceTag,
)
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    FKEdge,
    SchemaGraph,
    TableMetadata,
)
from aetherdialect._schema_build import (
    _merge_ddl_column_constraints_into_schema_graph,
    _merge_ddl_primary_keys_into_schema_graph,
    _merge_svv_foreign_keys_into_tables_meta,
    _reflect_duckdb_catalog,
    _reflect_sqlite_catalog,
    merge_ddl_foreign_keys_into_schema_graph,
)
from aetherdialect._schema_catalog import _parse_sql_file_sqlglot
from aetherdialect._schema_graph import (
    _fk_containment_validates,
    _infer_missing_fks_composite,
    fk_overlap_validates,
    infer_missing_fks,
    promote_cross_component_semantic_edges,
    revalidate_named_fks_with_overlap,
)


def _col(name: str, sample: list[str], *, pk: bool = False) -> ColumnMetadata:
    return ColumnMetadata(
        name=name,
        data_type="varchar",
        value_type="string",
        is_primary_key=pk,
        value_overlap_sample=list(sample),
    )


def test_containment_accepts_child_inside_parent() -> None:
    child = _col("customer_id", ["1", "2", "3", "4", "5"])
    parent = _col("id", ["1", "2", "3", "4", "5", "6", "7"], pk=True)
    assert _fk_containment_validates(child, parent) is True


def test_containment_rejects_child_mostly_outside_parent() -> None:
    child = _col("customer_id", ["9", "8", "7", "6", "5"])
    parent = _col("id", ["1", "2", "3", "4", "5"], pk=True)
    assert _fk_containment_validates(child, parent) is False


_SHARED_COMPOSITE_SAMPLE = ["1", "2", "3", "4", "5"]
_DISJOINT_CHILD_SAMPLE = ["9", "8", "7", "6", "5"]
_DISJOINT_PARENT_SAMPLE = ["1", "2", "3", "4", "0"]


def _composite_overlap_tables(
    *,
    child_order_sample: list[str],
    child_line_sample: list[str],
    parent_order_sample: list[str],
    parent_line_sample: list[str],
) -> dict[str, TableMetadata]:
    return {
        "order_lines": TableMetadata(
            name="order_lines",
            columns={
                "order_id": _col("order_id", parent_order_sample, pk=True),
                "line_no": _col("line_no", parent_line_sample, pk=True),
            },
            primary_key=["order_id", "line_no"],
            foreign_keys=[],
        ),
        "shipments": TableMetadata(
            name="shipments",
            columns={
                "shipment_id": _col("shipment_id", ["s1"], pk=True),
                "order_id": _col("order_id", child_order_sample),
                "line_no": _col("line_no", child_line_sample),
            },
            primary_key=["shipment_id"],
            foreign_keys=[],
        ),
    }


def test_composite_fk_inference_rejects_disjoint_overlap_samples() -> None:
    tables = _composite_overlap_tables(
        child_order_sample=_DISJOINT_CHILD_SAMPLE,
        child_line_sample=_SHARED_COMPOSITE_SAMPLE,
        parent_order_sample=_DISJOINT_PARENT_SAMPLE,
        parent_line_sample=_SHARED_COMPOSITE_SAMPLE,
    )
    edges = infer_missing_fks(tables)
    composite = [edge for edge in edges if edge.inference_tag == InferenceTag.COMPOSITE]
    assert composite == []


def test_composite_fk_inference_accepts_overlapping_samples_per_column() -> None:
    tables = _composite_overlap_tables(
        child_order_sample=_SHARED_COMPOSITE_SAMPLE,
        child_line_sample=_SHARED_COMPOSITE_SAMPLE,
        parent_order_sample=_SHARED_COMPOSITE_SAMPLE,
        parent_line_sample=_SHARED_COMPOSITE_SAMPLE,
    )
    edges = infer_missing_fks(tables)
    composite = [edge for edge in edges if edge.inference_tag == InferenceTag.COMPOSITE]
    assert len(composite) == 1
    assert composite[0].src_table == "shipments"
    assert composite[0].dst_table == "order_lines"


def test_infer_missing_fks_composite_applies_overlap_gate_directly() -> None:
    tables = _composite_overlap_tables(
        child_order_sample=_DISJOINT_CHILD_SAMPLE,
        child_line_sample=_SHARED_COMPOSITE_SAMPLE,
        parent_order_sample=_DISJOINT_PARENT_SAMPLE,
        parent_line_sample=_SHARED_COMPOSITE_SAMPLE,
    )
    inferred = _infer_missing_fks_composite(tables, {name.lower(): name for name in tables}, existing=[])
    assert inferred == []


def test_containment_refuses_under_min_sample() -> None:
    child = _col("customer_id", ["1", "2"])
    parent = _col("id", ["9"], pk=True)
    assert _fk_containment_validates(child, parent) is False


def test_overlap_refuses_under_min_sample() -> None:
    src = _col("customer_id", ["1", "2"])
    dst = _col("id", ["9"], pk=True)
    assert fk_overlap_validates(src, dst) is False


def test_revalidate_removes_suffix_fk_failing_containment() -> None:
    orders = TableMetadata(
        name="orders",
        columns={"customer_id": _col("customer_id", ["9", "8", "7", "6", "5"])},
        primary_key=[],
        foreign_keys=[
            FKEdge(
                src_table="orders",
                src_cols=["customer_id"],
                dst_table="customers",
                dst_cols=["id"],
                inference_tag=InferenceTag.SUFFIX,
            )
        ],
    )
    customers = TableMetadata(
        name="customers",
        columns={"id": _col("id", ["1", "2", "3", "4", "5"], pk=True)},
        primary_key=["id"],
        foreign_keys=[],
    )
    sg = SchemaGraph(
        tables={"orders": orders, "customers": customers},
        join_paths_multi={},
        created_at="",
    )
    removed = revalidate_named_fks_with_overlap(sg)
    assert removed == 1
    assert orders.foreign_keys == []


def test_revalidate_removes_composite_fk_with_failing_column() -> None:
    line = TableMetadata(
        name="line_items",
        columns={
            "order_id": _col("order_id", ["9", "8", "7", "6", "5"]),
            "sku": _col("sku", ["a", "b", "c", "d", "e"]),
        },
        primary_key=[],
        foreign_keys=[
            FKEdge(
                src_table="line_items",
                src_cols=["order_id", "sku"],
                dst_table="orders",
                dst_cols=["order_id", "sku"],
                inference_tag=InferenceTag.COMPOSITE,
            )
        ],
    )
    orders = TableMetadata(
        name="orders",
        columns={
            "order_id": _col("order_id", ["1", "2", "3", "4", "5"], pk=True),
            "sku": _col("sku", ["a", "b", "c", "d", "e"], pk=True),
        },
        primary_key=["order_id", "sku"],
        foreign_keys=[],
    )
    sg = SchemaGraph(
        tables={"line_items": line, "orders": orders},
        join_paths_multi={},
        created_at="",
    )
    removed = revalidate_named_fks_with_overlap(sg)
    assert removed == 1
    assert line.foreign_keys == []


def test_semantic_promotion_uses_containment_and_child_direction() -> None:
    left = TableMetadata(
        name="orders",
        columns={
            "status_code": _col(
                "status_code",
                ["open", "closed", "pending", "done", "hold"],
            )
        },
        primary_key=[],
        foreign_keys=[],
    )
    right = TableMetadata(
        name="statuses",
        columns={
            "code": _col(
                "code",
                ["open", "closed", "pending", "done", "hold", "archived"],
                pk=True,
            )
        },
        primary_key=["code"],
        foreign_keys=[],
    )
    left.columns["status_code"].distinct_count = 10
    right.columns["code"].distinct_count = 10
    left.columns["status_code"].semantic_join_neighbors = [("statuses", "code")]
    right.columns["code"].semantic_join_neighbors = [("orders", "status_code")]
    sg = SchemaGraph(tables={"orders": left, "statuses": right}, join_paths_multi={}, created_at="")
    promoted = promote_cross_component_semantic_edges(sg)
    assert promoted == 1
    edge = left.foreign_keys[0]
    assert edge.src_table == "orders"
    assert edge.src_cols == ["status_code"]
    assert edge.dst_table == "statuses"
    assert edge.inference_tag == InferenceTag.SEMANTIC_PROMOTED


def test_containment_ratio_threshold_from_config() -> None:
    child = _col("x", ["1", "2", "3", "4", "5"])
    parent = _col("y", ["1", "2", "3", "6", "7"], pk=True)
    assert _fk_containment_validates(child, parent) is (3 / 5 >= PolicyConfig.FK_INFER_CONTAINMENT_MIN_RATIO)


def test_merge_svv_foreign_keys_dedupes_information_schema_edges() -> None:
    tables_meta = {
        "orders": {
            "foreign_keys": [
                {
                    "src_cols": ["customer_id"],
                    "dst_table": "customers",
                    "dst_cols": ["customer_id"],
                }
            ],
        },
        "customers": {"foreign_keys": []},
    }
    svv_rows = [
        ("fk_orders", "public", "orders", "customer_id", "public", "customers", "customer_id"),
        ("fk_lines", "public", "line_items", "order_id", "public", "orders", "order_id"),
    ]
    tables_meta["line_items"] = {"foreign_keys": []}
    _merge_svv_foreign_keys_into_tables_meta(tables_meta, svv_rows)
    assert len(tables_meta["orders"]["foreign_keys"]) == 1
    assert tables_meta["line_items"]["foreign_keys"] == [
        {"src_cols": ["order_id"], "dst_table": "orders", "dst_cols": ["order_id"]},
    ]


def test_reflect_sqlite_catalog_uses_pragma_foreign_key_list_when_enabled() -> None:
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn

    def fake_execute(sql: str, *_args: object, **_kwargs: object) -> MagicMock:
        stmt = str(sql)
        result = MagicMock()
        if "PRAGMA foreign_keys" in stmt:
            result.scalar.return_value = 1
        elif "sqlite_master" in stmt:
            result.fetchall.return_value = [("orders", "table"), ("customers", "table")]
        elif 'PRAGMA table_info("orders")' in stmt:
            result.fetchall.return_value = [
                (0, "order_id", "INTEGER", 1, None, 1),
                (1, "customer_id", "INTEGER", 0, None, 0),
            ]
        elif 'PRAGMA table_info("customers")' in stmt:
            result.fetchall.return_value = [(0, "customer_id", "INTEGER", 1, None, 1)]
        elif 'PRAGMA foreign_key_list("orders")' in stmt:
            result.fetchall.return_value = [
                (0, 0, "customers", "customer_id", "customer_id", "NO ACTION", "NO ACTION", "NONE")
            ]
        elif 'PRAGMA foreign_key_list("customers")' in stmt:
            result.fetchall.return_value = []
        else:
            result.fetchall.return_value = []
        return result

    conn.execute.side_effect = fake_execute
    sg = _reflect_sqlite_catalog(engine, include="tables", allow_objects=None)
    assert "orders" in sg.tables
    edge = sg.tables["orders"].foreign_keys[0]
    assert edge.src_cols == ["customer_id"]
    assert edge.dst_table == "customers"
    assert edge.dst_cols == ["customer_id"]


def test_reflect_sqlite_catalog_skips_pragma_fks_when_disabled() -> None:
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn

    def fake_execute(sql: str, *_args: object, **_kwargs: object) -> MagicMock:
        stmt = str(sql)
        result = MagicMock()
        if "PRAGMA foreign_keys" in stmt:
            result.scalar.return_value = 0
        elif "sqlite_master" in stmt:
            result.fetchall.return_value = [("orders", "table")]
        elif 'PRAGMA table_info("orders")' in stmt:
            result.fetchall.return_value = [(0, "customer_id", "INTEGER", 0, None, 0)]
        else:
            result.fetchall.return_value = []
        return result

    conn.execute.side_effect = fake_execute
    sg = _reflect_sqlite_catalog(engine, include="tables", allow_objects=None)
    assert sg.tables["orders"].foreign_keys == []


def test_reflect_duckdb_catalog_reads_key_column_usage_foreign_keys() -> None:
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn

    def fake_execute(sql: str, *_args: object, **_kwargs: object) -> MagicMock:
        stmt = str(sql).lower()
        result = MagicMock()
        if "from information_schema.tables" in stmt:
            result.fetchall.return_value = [("orders", "BASE TABLE"), ("customers", "BASE TABLE")]
        elif "from information_schema.columns" in stmt:
            result.fetchall.return_value = [
                ("orders", "order_id", 1, "INTEGER", "NO"),
                ("orders", "customer_id", 2, "INTEGER", "NO"),
                ("customers", "customer_id", 1, "INTEGER", "NO"),
            ]
        elif "constraint_type = 'primary key'" in stmt:
            result.fetchall.return_value = [
                ("orders", "order_id"),
                ("customers", "customer_id"),
            ]
        elif "constraint_type = 'unique'" in stmt:
            result.fetchall.return_value = []
        elif "constraint_type = 'foreign key'" in stmt:
            result.fetchall.return_value = [
                ("orders", "orders_customer_id_fkey", 1, "customer_id", "customers", "customer_id"),
            ]
        else:
            result.fetchall.return_value = []
        return result

    conn.execute.side_effect = fake_execute
    sg = _reflect_duckdb_catalog(engine, "main", include="tables", allow_objects=None)
    assert sg.tables["orders"].primary_key == ["order_id"]
    edge = sg.tables["orders"].foreign_keys[0]
    assert edge.dst_table == "customers"
    assert edge.src_cols == ["customer_id"]


def test_merge_ddl_primary_keys_from_alter_statement() -> None:
    ddl = """
    CREATE TABLE category (
        category_id INTEGER NOT NULL,
        name VARCHAR(25) NOT NULL
    );
    ALTER TABLE category ADD CONSTRAINT category_pkey PRIMARY KEY (category_id);
    """
    tables = _parse_sql_file_sqlglot(ddl, "duckdb")
    assert tables["category"]["primary_keys"] == ["category_id"]
    sg = SchemaGraph(
        tables={
            "category": TableMetadata(
                name="category",
                columns={
                    "category_id": ColumnMetadata(
                        name="category_id",
                        data_type="integer",
                        distinct_count=0,
                    ),
                    "name": ColumnMetadata(name="name", data_type="varchar", distinct_count=0),
                },
                primary_key=[],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
        created_at="",
    )
    _merge_ddl_primary_keys_into_schema_graph(sg, tables)
    cat = sg.tables["category"]
    assert cat.primary_key == ["category_id"]
    assert cat.columns["category_id"].is_primary_key
    assert cat.columns["category_id"].is_visible


def test_merge_ddl_column_constraints_applies_unique_and_not_null() -> None:
    ddl = {
        "staff": {
            "column_names_original": ["staff_id", "username"],
            "column_is_nullable": [False, True],
            "unique_columns": ["username"],
            "primary_keys": [],
            "foreign_keys": [],
        },
    }
    sg = SchemaGraph(
        tables={
            "staff": TableMetadata(
                name="staff",
                columns={
                    "staff_id": ColumnMetadata(name="staff_id", data_type="integer", is_nullable=True),
                    "username": ColumnMetadata(name="username", data_type="varchar", is_nullable=True),
                },
                primary_key=[],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
        created_at="",
    )
    _merge_ddl_column_constraints_into_schema_graph(sg, ddl)
    staff = sg.tables["staff"]
    assert staff.columns["staff_id"].is_nullable is False
    assert staff.columns["username"].is_unique is True


def test_bigquery_sql_file_fk_merge_adds_documented_edge() -> None:
    ddl = """
    CREATE TABLE `proj.ds.orders` (
      order_id INT64 NOT NULL,
      customer_id INT64 NOT NULL,
      FOREIGN KEY (customer_id) REFERENCES `proj.ds.customers` (customer_id) NOT ENFORCED
    );
    """
    tables = _parse_sql_file_sqlglot(ddl, "bigquery")
    assert "orders" in tables
    assert tables["orders"]["foreign_keys"]
    sg = SchemaGraph(
        tables={
            "orders": TableMetadata(
                name="orders",
                columns={
                    "order_id": _col("order_id", ["1"]),
                    "customer_id": _col("customer_id", ["1"]),
                },
                primary_key=["order_id"],
                foreign_keys=[],
            ),
            "customers": TableMetadata(
                name="customers",
                columns={"customer_id": _col("customer_id", ["1"], pk=True)},
                primary_key=["customer_id"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
        created_at="",
    )
    merge_ddl_foreign_keys_into_schema_graph(sg, tables)
    assert len(sg.tables["orders"].foreign_keys) == 1
    assert sg.tables["orders"].foreign_keys[0].dst_table == "customers"
