"""Tests for schema module: schema reflection, graph building, and utility functions."""

from __future__ import annotations

from aetherdialect._config import SCHEMA_OVERRIDES_VERSION, EngineConfig, PolicyConfig
from aetherdialect._contracts_base import (
    ColumnMetadata,
    ColumnRole,
    FKEdge,
    SchemaAccessError,
    SchemaContext,
    SchemaGraph,
    SchemaLimits,
    SensitivityClassification,
    TableMetadata,
    TableRole,
)
from aetherdialect._core_utils import schema_hash_fp
from aetherdialect._schema import (
    _analyze_fk_path_topology,
    _apply_schema_context_allow_columns,
    _edge_key,
    _fingerprint_tables_after_document_round_trip,
    _first_table_where_stable_json_differs,
    _infer_missing_fks,
    _infer_missing_pks_from_profile,
    _mark_canonical_duplicates,
    _normalize_fk_path,
    _notes_content_sha256,
    _promote_semantic_edges_to_fks,
    _recompute_join_paths_multi,
    _resolve_graph_table_name,
    _reverse_fk_path,
    _schema_cache_json_blob,
    _select_inferred_pk_candidate,
    _strip_schema_context_denied_columns,
    _table_from_dict,
    _table_to_dict,
    _tables_meta_to_schema_graph,
    _tables_payload_through_model_round_trip,
    _validate_scope_against_graph,
    assign_schema_graph_hashes,
    compute_schema_limits,
    compute_schema_stats,
    load_or_create_schema_postgresql,
    merge_ddl_foreign_keys_into_schema_graph,
)


def _col(**overrides) -> ColumnMetadata:
    """Build a ColumnMetadata with sensible defaults."""
    defaults = dict(
        name="col",
        data_type="varchar",
        value_type="",
        is_primary_key=False,
        is_foreign_key=False,
        fk_target=None,
        role=ColumnRole.CATEGORICAL.value,
        distinct_count=10,
        distinct_ratio=0.5,
        row_count=20,
    )
    defaults.update(overrides)
    return ColumnMetadata(**defaults)


def _table(name, columns, **overrides) -> TableMetadata:
    """Build a TableMetadata with sensible defaults."""
    defaults = dict(
        name=name,
        columns=columns,
        primary_key=[],
        foreign_keys=[],
        role=TableRole.DIMENSION.value,
        row_count=100,
    )
    defaults.update(overrides)
    return TableMetadata(**defaults)


def _ov_doc(**kwargs) -> dict:
    """Minimal valid v4 overrides document for :func:`apply_schema_overrides_to_graph` tests."""

    base: dict = {
        "version": SCHEMA_OVERRIDES_VERSION,
        "tables": {},
        "foreign_keys_add": [],
        "foreign_keys_remove": [],
        "primary_keys_add": [],
        "primary_keys_remove": [],
    }
    base.update(kwargs)
    return base


def _odesc(text: str) -> str:
    """Editable description value (bare string)."""

    return text


def _orole(value: str | None) -> str | None:
    """Editable role value (bare token or null)."""

    return value


class TestComputeSchemaStats:
    """Tests for compute_schema_stats."""

    def test_basic_stats(self, schema_graph):
        """Should compute correct stats for the 3-table schema."""
        for t in schema_graph.tables.values():
            for c in t.columns.values():
                c.is_filterable_override = c.role in (
                    ColumnRole.CATEGORICAL.value,
                    ColumnRole.TEMPORAL.value,
                    ColumnRole.BOOLEAN.value,
                    ColumnRole.NUMERIC_MEASURE.value,
                )
                c.is_groupable_override = c.role in (ColumnRole.CATEGORICAL.value,)
                c.is_aggregatable_override = c.role in (ColumnRole.NUMERIC_MEASURE.value,)

        stats = compute_schema_stats(schema_graph)
        assert stats["table_count"] == 3
        assert stats["total_filterable"] > 0
        assert stats["total_groupable"] > 0
        assert stats["total_aggregatable"] > 0

    def test_empty_schema(self):
        """Empty schema should produce zero counts."""
        sg = SchemaGraph(tables={}, join_paths_multi={}, effective_structural_hash="empty")
        stats = compute_schema_stats(sg)
        assert stats["table_count"] == 0
        assert stats["total_filterable"] == 0
        assert stats["min_filterable_per_table"] == 0

    def test_filterable_per_table_populated(self, schema_graph):
        """filterable_per_table should have one entry per table with filterable cols."""
        for t in schema_graph.tables.values():
            for c in t.columns.values():
                c.is_filterable_override = True
                c.is_groupable_override = False
                c.is_aggregatable_override = False
        stats = compute_schema_stats(schema_graph)
        assert len(stats["filterable_per_table"]) == 3


class TestComputeSchemaLimits:
    """Tests for compute_schema_limits."""

    def test_small_schema(self):
        """Small schema (<=3 tables) should set max_tables = table_count."""
        stats = {
            "table_count": 3,
            "total_filterable": 12,
            "total_groupable": 6,
        }
        limits = compute_schema_limits(stats)
        assert isinstance(limits, SchemaLimits)
        assert limits.max_tables == 3
        assert limits.max_filters >= 1
        assert limits.max_groupby >= 1

    def test_medium_schema(self):
        """Medium schema (4-10 tables) should set max_tables = 3."""
        stats = {"table_count": 7, "total_filterable": 28, "total_groupable": 14}
        limits = compute_schema_limits(stats)
        assert limits.max_tables == 3

    def test_large_schema(self):
        """Large schema (>10 tables) should set max_tables = 4."""
        stats = {"table_count": 15, "total_filterable": 60, "total_groupable": 30}
        limits = compute_schema_limits(stats)
        assert limits.max_tables == 4

    def test_zero_tables(self):
        """Zero tables should produce valid limits without division error."""
        stats = {"table_count": 0, "total_filterable": 0, "total_groupable": 0}
        limits = compute_schema_limits(stats)
        assert limits.max_filters == 1
        assert limits.max_groupby == 1

    def test_missing_keys_use_defaults(self):
        """Missing table_count / totals default so limits stay valid."""
        limits = compute_schema_limits({})
        assert limits.max_filters == 1
        assert limits.max_groupby == 1
        assert limits.max_tables == 1


class TestNotesContentSha256:
    """Tests for _notes_content_sha256."""

    def test_none_is_empty_digest(self):
        """None notes hash the same as empty string."""
        assert _notes_content_sha256(None) == _notes_content_sha256("")

    def test_utf8_bytes(self):
        """Unicode notes use UTF-8 encoding for the digest."""
        h = _notes_content_sha256("café")
        assert len(h) == 64
        assert h != _notes_content_sha256("cafe")


class TestTableDictRoundtrip:
    """Tests for _table_to_dict / _table_from_dict roundtrip."""

    def test_roundtrip_preserves_name(self):
        """Table name should survive serialization roundtrip."""
        t = _table("orders", {"id": _col(name="id", is_primary_key=True)}, primary_key=["id"])
        d = _table_to_dict(t)
        restored = _table_from_dict(d)
        assert restored.name == "orders"

    def test_roundtrip_preserves_columns(self):
        """All columns should survive roundtrip."""
        t = _table(
            "t",
            {
                "a": _col(name="a", data_type="integer"),
                "b": _col(name="b", data_type="varchar"),
            },
        )
        d = _table_to_dict(t)
        restored = _table_from_dict(d)
        assert set(restored.columns.keys()) == {"a", "b"}
        assert restored.columns["a"].data_type == "integer"

    def test_roundtrip_preserves_fk(self):
        """Foreign keys should survive roundtrip."""
        fk = FKEdge(src_table="o", src_cols=["uid"], dst_table="u", dst_cols=["id"])
        t = _table("o", {"uid": _col(name="uid")}, foreign_keys=[fk])
        d = _table_to_dict(t)
        restored = _table_from_dict(d)
        assert len(restored.foreign_keys) == 1
        assert restored.foreign_keys[0].dst_table == "u"

    def test_roundtrip_preserves_role(self):
        """Table role should survive roundtrip."""
        t = _table("t", {"c": _col(name="c")}, role=TableRole.FACT.value)
        d = _table_to_dict(t)
        restored = _table_from_dict(d)
        assert restored.role == TableRole.FACT.value

    def test_roundtrip_preserves_row_count(self):
        """Row count should survive roundtrip."""
        t = _table("t", {"c": _col(name="c")}, row_count=42)
        d = _table_to_dict(t)
        restored = _table_from_dict(d)
        assert restored.row_count == 42


class TestReverseFkPath:
    """Tests for _reverse_fk_path."""

    def test_single_edge_reversal(self):
        """Single edge should have src/dst swapped."""
        path = [
            {
                "src_table": "A",
                "src_cols": ["a_id"],
                "dst_table": "B",
                "dst_cols": ["b_id"],
            }
        ]
        result = _reverse_fk_path(path)
        assert len(result) == 1
        assert result[0]["src_table"] == "B"
        assert result[0]["dst_table"] == "A"
        assert result[0]["src_cols"] == ["b_id"]
        assert result[0]["dst_cols"] == ["a_id"]

    def test_multi_edge_reversal(self):
        """Multi-edge path should reverse order and swap each edge."""
        path = [
            {"src_table": "A", "src_cols": ["x"], "dst_table": "B", "dst_cols": ["y"]},
            {"src_table": "B", "src_cols": ["z"], "dst_table": "C", "dst_cols": ["w"]},
        ]
        result = _reverse_fk_path(path)
        assert len(result) == 2
        assert result[0]["src_table"] == "C"
        assert result[0]["dst_table"] == "B"
        assert result[1]["src_table"] == "B"
        assert result[1]["dst_table"] == "A"

    def test_empty_path(self):
        """Empty path should return empty list."""
        assert _reverse_fk_path([]) == []


class TestAnalyzeFkPathTopology:
    """Tests for _analyze_fk_path_topology."""

    def test_empty_path(self):
        """Empty path should produce 'none' topology."""
        topo, anchor, leaves = _analyze_fk_path_topology([])
        assert topo == "none"
        assert anchor == ""
        assert leaves == []

    def test_linear_path(self):
        """A->B->C is a linear path."""
        path = [
            {"src_table": "A", "src_cols": ["x"], "dst_table": "B", "dst_cols": ["y"]},
            {"src_table": "B", "src_cols": ["z"], "dst_table": "C", "dst_cols": ["w"]},
        ]
        topo, anchor, leaves = _analyze_fk_path_topology(path)
        assert topo == "linear"
        assert set(leaves) == {"A", "C"}

    def test_single_edge_is_linear_with_lexicographic_anchor(self):
        """Two endpoints, no hub: treat as linear anchored at min(leaf)."""
        path = [{"src_table": "Z", "src_cols": ["x"], "dst_table": "A", "dst_cols": ["y"]}]
        topo, anchor, leaves = _analyze_fk_path_topology(path)
        assert topo == "linear"
        assert anchor == "A"
        assert set(leaves) == {"A", "Z"}

    def test_star_topology(self):
        """Hub table connected to multiple leaves is a star."""
        path = [
            {
                "src_table": "Hub",
                "src_cols": ["x"],
                "dst_table": "L1",
                "dst_cols": ["y"],
            },
            {
                "src_table": "Hub",
                "src_cols": ["a"],
                "dst_table": "L2",
                "dst_cols": ["b"],
            },
            {
                "src_table": "Hub",
                "src_cols": ["c"],
                "dst_table": "L3",
                "dst_cols": ["d"],
            },
        ]
        topo, anchor, leaves = _analyze_fk_path_topology(path)
        assert topo == "star"
        assert anchor == "Hub"
        assert set(leaves) == {"L1", "L2", "L3"}

    def test_tree_topology_two_hubs(self):
        """Branching graph with two hubs classifies as tree; anchor is highest-degree hub."""
        path = [
            {
                "src_table": "H1",
                "src_cols": ["x"],
                "dst_table": "L1",
                "dst_cols": ["y"],
            },
            {
                "src_table": "H1",
                "src_cols": ["a"],
                "dst_table": "L2",
                "dst_cols": ["b"],
            },
            {
                "src_table": "H1",
                "src_cols": ["c"],
                "dst_table": "H2",
                "dst_cols": ["d"],
            },
            {
                "src_table": "H2",
                "src_cols": ["e"],
                "dst_table": "L3",
                "dst_cols": ["f"],
            },
        ]
        topo, anchor, leaves = _analyze_fk_path_topology(path)
        assert topo == "tree"
        assert anchor == "H1"
        assert set(leaves) == {"L1", "L2", "L3"}


class TestNormalizeFkPath:
    """Tests for _normalize_fk_path."""

    def test_empty_path_unchanged(self):
        """Empty path should return empty."""
        assert _normalize_fk_path([]) == []

    def test_linear_path_oriented_from_smallest_leaf(self):
        """Linear path should start from lexicographically smallest leaf."""
        path = [
            {"src_table": "B", "src_cols": ["x"], "dst_table": "A", "dst_cols": ["y"]},
        ]
        result = _normalize_fk_path(path)
        assert result[0]["src_table"] in ("A", "B")

    def test_single_edge_when_already_from_anchor_unchanged(self):
        """Linear path already starting at anchor is unchanged."""
        path = [
            {"src_table": "A", "src_cols": ["x"], "dst_table": "B", "dst_cols": ["y"]},
        ]
        assert _normalize_fk_path(path) == path


class TestInferMissingFks:
    """Tests for infer_missing_fks."""

    def test_infers_from_column_suffix(self):
        """Column ending in _id matching a table name should infer FK."""
        tables = {
            "orders": _table(
                "orders",
                {
                    "order_id": _col(name="order_id", is_primary_key=True),
                    "customer_id": _col(name="customer_id"),
                },
                primary_key=["order_id"],
            ),
            "customer": _table(
                "customer",
                {
                    "customer_id": _col(name="customer_id", is_primary_key=True),
                },
                primary_key=["customer_id"],
            ),
        }
        inferred = _infer_missing_fks(tables)
        assert len(inferred) == 1
        assert inferred[0].src_table == "orders"
        assert inferred[0].dst_table == "customer"

    def test_skips_existing_fk(self):
        """Column already marked as FK should not be inferred again."""
        tables = {
            "orders": _table(
                "orders",
                {
                    "customer_id": _col(
                        name="customer_id",
                        is_foreign_key=True,
                        fk_target=("customer", "customer_id"),
                    ),
                },
            ),
            "customer": _table(
                "customer",
                {
                    "customer_id": _col(name="customer_id", is_primary_key=True),
                },
                primary_key=["customer_id"],
            ),
        }
        inferred = _infer_missing_fks(tables)
        assert len(inferred) == 0

    def test_skips_pk_column(self):
        """Primary key columns should not be inferred as FK."""
        tables = {
            "customer": _table(
                "customer",
                {
                    "customer_id": _col(name="customer_id", is_primary_key=True),
                },
                primary_key=["customer_id"],
            ),
        }
        inferred = _infer_missing_fks(tables)
        assert len(inferred) == 0

    def test_no_match_returns_empty(self):
        """When no column suffix matches a table, no FKs should be inferred."""
        tables = {
            "alpha": _table("alpha", {"foo": _col(name="foo")}),
            "beta": _table("beta", {"bar": _col(name="bar")}),
        }
        inferred = _infer_missing_fks(tables)
        assert inferred == []

    def test_suffix_infer_self_referential_parent_node_id(self):
        """Suffix inference may link a column to the same table's primary key."""
        tables = {
            "node": _table(
                "node",
                {
                    "node_id": _col(name="node_id", is_primary_key=True),
                    "parent_node_id": _col(name="parent_node_id"),
                },
                primary_key=["node_id"],
            ),
        }
        inferred = _infer_missing_fks(tables)
        assert len(inferred) == 1
        assert inferred[0].src_table == "node"
        assert inferred[0].dst_table == "node"
        assert inferred[0].inference_tag == "self"

    def test_infers_via_key_suffix(self):
        """Column ending in _key matching a table infers FK when PK naming aligns."""
        tables = {
            "session": _table(
                "session",
                {
                    "session_key": _col(name="session_key", is_primary_key=True),
                },
                primary_key=["session_key"],
            ),
            "event": _table(
                "event",
                {
                    "id": _col(name="id", is_primary_key=True),
                    "session_key": _col(name="session_key"),
                },
                primary_key=["id"],
            ),
        }
        inferred = _infer_missing_fks(tables)
        assert any(e.src_table == "event" and e.dst_table == "session" for e in inferred)

    def test_skips_when_target_has_no_primary_key(self):
        """No inference when referenced table has empty primary_key."""
        tables = {
            "orphan": _table("orphan", {"x": _col(name="x")}, primary_key=[]),
            "child": _table(
                "child",
                {"orphan_id": _col(name="orphan_id")},
                primary_key=[],
            ),
        }
        assert _infer_missing_fks(tables) == []

    def test_suffix_match_case_insensitive_table_and_column(self):
        """Mixed-case table keys and column names still resolve suffix FKs."""
        tables = {
            "Orders": _table(
                "Orders",
                {
                    "OrderID": _col(name="OrderID", is_primary_key=True),
                    "CustomerID": _col(name="CustomerID"),
                },
                primary_key=["OrderID"],
            ),
            "Customer": _table(
                "Customer",
                {
                    "CustomerID": _col(name="CustomerID", is_primary_key=True),
                },
                primary_key=["CustomerID"],
            ),
        }
        inferred = _infer_missing_fks(tables)
        assert len(inferred) == 1
        assert inferred[0].src_table == "Orders"
        assert inferred[0].src_cols == ["CustomerID"]
        assert inferred[0].dst_table == "Customer"
        assert inferred[0].dst_cols == ["CustomerID"]

    def test_skips_suffix_infer_when_target_table_has_composite_pk(self):
        """Suffix FK inference requires a single-column primary key on the target."""
        tables = {
            "line_item": _table(
                "line_item",
                {
                    "id": _col(name="id", is_primary_key=True),
                    "order_id": _col(name="order_id"),
                },
                primary_key=["id"],
            ),
            "orders": _table(
                "orders",
                {
                    "order_id": _col(name="order_id", is_primary_key=True),
                    "line_no": _col(name="line_no", is_primary_key=True),
                },
                primary_key=["order_id", "line_no"],
            ),
        }
        assert _infer_missing_fks(tables) == []

    def test_view_as_fk_target_with_single_column_pk(self):
        """Views participate in suffix FK inference like tables when they expose a single PK column."""
        customer_v = _table(
            "customer_v",
            {"customer_v_id": _col(name="customer_v_id", is_primary_key=True)},
            primary_key=["customer_v_id"],
            kind="view",
        )
        order_line = _table(
            "order_line",
            {
                "id": _col(name="id", is_primary_key=True),
                "customer_v_id": _col(name="customer_v_id"),
            },
            primary_key=["id"],
        )
        tables = {"customer_v": customer_v, "order_line": order_line}
        inferred = _infer_missing_fks(tables)
        assert any(e.src_table == "order_line" and e.dst_table == "customer_v" for e in inferred)

    def test_same_name_column_without_suffix_does_not_infer(self):
        """Same-name-as-PK inference was removed; only suffix heuristics apply."""
        tables = {
            "product": _table(
                "product",
                {
                    "product_sku": _col(name="product_sku", is_primary_key=True),
                },
                primary_key=["product_sku"],
            ),
            "line_item": _table(
                "line_item",
                {
                    "id": _col(name="id", is_primary_key=True),
                    "product_sku": _col(name="product_sku"),
                },
                primary_key=["id"],
            ),
        }
        assert _infer_missing_fks(tables) == []

    def test_suffix_skips_when_profiled_types_incompatible(self):
        """Suffix heuristic skips when both columns have incompatible value_type."""
        tables = {
            "orders": _table(
                "orders",
                {
                    "order_id": _col(name="order_id", is_primary_key=True),
                    "customer_id": _col(
                        name="customer_id",
                        data_type="varchar",
                        value_type="string",
                    ),
                },
                primary_key=["order_id"],
            ),
            "customer": _table(
                "customer",
                {
                    "customer_id": _col(
                        name="customer_id",
                        data_type="integer",
                        value_type="integer",
                        is_primary_key=True,
                    ),
                },
                primary_key=["customer_id"],
            ),
        }
        assert _infer_missing_fks(tables) == []


class TestInferMissingFksLongestPrefixAndOverlap:
    """Tests for Phase 13 longest-prefix matching and value-overlap validation."""

    def test_prefers_plural_table_over_singular_when_both_exist(self):
        """`customer_id` should target the longer-named table when singular and plural both exist."""
        tables = {
            "customer": _table(
                "customer",
                {"customer_id": _col(name="customer_id", is_primary_key=True)},
                primary_key=["customer_id"],
            ),
            "customers": _table(
                "customers",
                {"customers_id": _col(name="customers_id", is_primary_key=True)},
                primary_key=["customers_id"],
            ),
            "orders": _table(
                "orders",
                {
                    "order_id": _col(name="order_id", is_primary_key=True),
                    "customer_id": _col(name="customer_id"),
                },
                primary_key=["order_id"],
            ),
        }
        inferred = _infer_missing_fks(tables)
        assert len(inferred) == 1
        assert inferred[0].dst_table == "customers"

    def test_pluralized_target_inferred_when_only_plural_table_exists(self):
        """`customer_id` should still link to `customers.customers_id` if no singular table exists."""
        tables = {
            "customers": _table(
                "customers",
                {"customers_id": _col(name="customers_id", is_primary_key=True)},
                primary_key=["customers_id"],
            ),
            "orders": _table(
                "orders",
                {
                    "order_id": _col(name="order_id", is_primary_key=True),
                    "customer_id": _col(name="customer_id"),
                },
                primary_key=["order_id"],
            ),
        }
        inferred = _infer_missing_fks(tables)
        assert len(inferred) == 1
        assert inferred[0].dst_table == "customers"

    def test_overlap_validate_rejects_disjoint_samples(self):
        """Inferred FK should be rejected when both sides expose disjoint sampled values."""
        src_samples = [str(i) for i in range(900, 910)]
        dst_samples = [str(i) for i in range(1, 11)]
        tables = {
            "customer": _table(
                "customer",
                {
                    "customer_id": _col(
                        name="customer_id",
                        is_primary_key=True,
                        top_k_values=dst_samples,
                    )
                },
                primary_key=["customer_id"],
            ),
            "orders": _table(
                "orders",
                {
                    "order_id": _col(name="order_id", is_primary_key=True),
                    "customer_id": _col(name="customer_id", top_k_values=src_samples),
                },
                primary_key=["order_id"],
            ),
        }
        assert _infer_missing_fks(tables) == []

    def test_overlap_validate_accepts_overlapping_samples(self):
        """Inferred FK should be retained when sampled values overlap above the threshold."""
        shared = [str(i) for i in range(1, 11)]
        tables = {
            "customer": _table(
                "customer",
                {
                    "customer_id": _col(
                        name="customer_id",
                        is_primary_key=True,
                        top_k_values=shared,
                    )
                },
                primary_key=["customer_id"],
            ),
            "orders": _table(
                "orders",
                {
                    "order_id": _col(name="order_id", is_primary_key=True),
                    "customer_id": _col(name="customer_id", top_k_values=shared),
                },
                primary_key=["order_id"],
            ),
        }
        inferred = _infer_missing_fks(tables)
        assert len(inferred) == 1
        assert inferred[0].dst_table == "customer"

    def test_overlap_falls_open_when_samples_missing(self):
        """When neither side has populated samples, overlap-validate should not block inference."""
        tables = {
            "customer": _table(
                "customer",
                {"customer_id": _col(name="customer_id", is_primary_key=True)},
                primary_key=["customer_id"],
            ),
            "orders": _table(
                "orders",
                {
                    "order_id": _col(name="order_id", is_primary_key=True),
                    "customer_id": _col(name="customer_id"),
                },
                primary_key=["order_id"],
            ),
        }
        inferred = _infer_missing_fks(tables)
        assert len(inferred) == 1

    def test_string_int_compatible_when_string_samples_are_digits(self):
        """String↔integer FK inference should be allowed when string-side samples are digit strings."""
        digit_strs = [str(i) for i in range(1, 11)]
        tables = {
            "customer": _table(
                "customer",
                {
                    "customer_id": _col(
                        name="customer_id",
                        data_type="integer",
                        value_type="integer",
                        is_primary_key=True,
                        top_k_values=digit_strs,
                    )
                },
                primary_key=["customer_id"],
            ),
            "orders": _table(
                "orders",
                {
                    "order_id": _col(name="order_id", is_primary_key=True),
                    "customer_id": _col(
                        name="customer_id",
                        data_type="varchar",
                        value_type="string",
                        top_k_values=digit_strs,
                    ),
                },
                primary_key=["order_id"],
            ),
        }
        inferred = _infer_missing_fks(tables)
        assert len(inferred) == 1
        assert inferred[0].dst_table == "customer"

    def test_string_int_rejected_when_string_samples_are_non_digits(self):
        """String↔integer FK inference should remain blocked when string-side samples contain non-digits."""
        tables = {
            "customer": _table(
                "customer",
                {
                    "customer_id": _col(
                        name="customer_id",
                        data_type="integer",
                        value_type="integer",
                        is_primary_key=True,
                        top_k_values=["1", "2", "3"],
                    )
                },
                primary_key=["customer_id"],
            ),
            "orders": _table(
                "orders",
                {
                    "order_id": _col(name="order_id", is_primary_key=True),
                    "customer_id": _col(
                        name="customer_id",
                        data_type="varchar",
                        value_type="string",
                        top_k_values=["abc", "def", "ghi"],
                    ),
                },
                primary_key=["order_id"],
            ),
        }
        assert _infer_missing_fks(tables) == []


class TestInferMissingFksComposite:
    """Tests for Phase 15 composite (multi-column) FK inference."""

    def test_infers_composite_fk_when_all_pk_columns_present(self):
        """When src table contains every column of dst's composite PK, infer the multi-column FK."""
        tables = {
            "order_lines": _table(
                "order_lines",
                {
                    "order_id": _col(name="order_id", is_primary_key=True),
                    "line_no": _col(name="line_no", is_primary_key=True),
                    "qty": _col(name="qty"),
                },
                primary_key=["order_id", "line_no"],
            ),
            "shipments": _table(
                "shipments",
                {
                    "shipment_id": _col(name="shipment_id", is_primary_key=True),
                    "order_id": _col(name="order_id"),
                    "line_no": _col(name="line_no"),
                },
                primary_key=["shipment_id"],
            ),
        }
        inferred = _infer_missing_fks(tables)
        composite = [e for e in inferred if e.inference_tag == "composite"]
        assert len(composite) == 1
        e = composite[0]
        assert e.src_table == "shipments"
        assert e.dst_table == "order_lines"
        assert sorted(e.src_cols) == ["line_no", "order_id"]
        assert sorted(e.dst_cols) == ["line_no", "order_id"]

    def test_skips_when_src_missing_any_pk_column(self):
        """Composite FK should not infer when src lacks any of the target's PK columns."""
        tables = {
            "order_lines": _table(
                "order_lines",
                {
                    "order_id": _col(name="order_id", is_primary_key=True),
                    "line_no": _col(name="line_no", is_primary_key=True),
                },
                primary_key=["order_id", "line_no"],
            ),
            "shipments": _table(
                "shipments",
                {
                    "shipment_id": _col(name="shipment_id", is_primary_key=True),
                    "order_id": _col(name="order_id"),
                },
                primary_key=["shipment_id"],
            ),
        }
        composite = [e for e in _infer_missing_fks(tables) if e.inference_tag == "composite"]
        assert composite == []

    def test_skips_when_per_column_types_incompatible(self):
        """Composite FK rejected when any per-column value_type pair is incompatible."""
        tables = {
            "order_lines": _table(
                "order_lines",
                {
                    "order_id": _col(
                        name="order_id",
                        data_type="integer",
                        value_type="integer",
                        is_primary_key=True,
                    ),
                    "line_no": _col(
                        name="line_no",
                        data_type="integer",
                        value_type="integer",
                        is_primary_key=True,
                    ),
                },
                primary_key=["order_id", "line_no"],
            ),
            "shipments": _table(
                "shipments",
                {
                    "shipment_id": _col(name="shipment_id", is_primary_key=True),
                    "order_id": _col(
                        name="order_id",
                        data_type="integer",
                        value_type="integer",
                    ),
                    "line_no": _col(
                        name="line_no",
                        data_type="varchar",
                        value_type="string",
                        top_k_values=["alpha", "beta"],
                    ),
                },
                primary_key=["shipment_id"],
            ),
        }
        composite = [e for e in _infer_missing_fks(tables) if e.inference_tag == "composite"]
        assert composite == []

    def test_skips_when_column_already_used_by_suffix_inference(self):
        """A column already promoted by suffix inference must not double-bind into a composite FK."""
        tables = {
            "order_lines": _table(
                "order_lines",
                {
                    "order_id": _col(name="order_id", is_primary_key=True),
                    "line_no": _col(name="line_no", is_primary_key=True),
                },
                primary_key=["order_id", "line_no"],
            ),
            "order": _table(
                "order",
                {"order_id": _col(name="order_id", is_primary_key=True)},
                primary_key=["order_id"],
            ),
            "shipments": _table(
                "shipments",
                {
                    "shipment_id": _col(name="shipment_id", is_primary_key=True),
                    "order_id": _col(name="order_id"),
                    "line_no": _col(name="line_no"),
                },
                primary_key=["shipment_id"],
            ),
        }
        edges = _infer_missing_fks(tables)
        composite = [e for e in edges if e.inference_tag == "composite"]
        assert composite == []
        suffix = [e for e in edges if e.inference_tag == "suffix"]
        assert any(e.src_table == "shipments" and e.dst_table == "order" for e in suffix)

    def test_skips_self_referential_composite(self):
        """Composite FK from a table to itself is rejected even when columns line up."""
        tables = {
            "node": _table(
                "node",
                {
                    "left_id": _col(name="left_id", is_primary_key=True),
                    "right_id": _col(name="right_id", is_primary_key=True),
                    "weight": _col(name="weight"),
                },
                primary_key=["left_id", "right_id"],
            ),
        }
        composite = [e for e in _infer_missing_fks(tables) if e.inference_tag == "composite"]
        assert composite == []


class TestTablesMetaToSchemaGraph:
    """Tests for _tables_meta_to_schema_graph."""

    def test_builds_tables(self):
        """Should create TableMetadata for each entry."""
        meta = {
            "users": {
                "column_names_original": ["user_id", "name"],
                "column_types": ["integer", "varchar"],
                "primary_keys": ["user_id"],
                "foreign_keys": [],
            },
        }
        sg = _tables_meta_to_schema_graph(meta)
        assert "users" in sg.tables
        assert "user_id" in sg.tables["users"].columns
        assert sg.tables["users"].columns["user_id"].is_primary_key is True
        assert sg.tables["users"].columns["user_id"].is_nullable is False
        assert sg.tables["users"].columns["name"].is_nullable is True

    def test_column_is_nullable_parallel_list(self):
        """Explicit ``column_is_nullable`` should map to ColumnMetadata.is_nullable (PK still wins)."""
        meta = {
            "t": {
                "column_names_original": ["id", "note"],
                "column_types": ["integer", "varchar"],
                "primary_keys": ["id"],
                "foreign_keys": [],
                "column_is_nullable": [True, False],
            },
        }
        sg = _tables_meta_to_schema_graph(meta)
        assert sg.tables["t"].columns["id"].is_nullable is False
        assert sg.tables["t"].columns["note"].is_nullable is False

    def test_non_pk_columns_default_nullable_without_parallel_list(self):
        """When ``column_is_nullable`` is absent, PK columns are treated as NOT NULL."""
        meta = {
            "t": {
                "column_names_original": ["id", "name"],
                "column_types": ["integer", "varchar"],
                "primary_keys": ["id"],
                "foreign_keys": [],
            },
        }
        sg = _tables_meta_to_schema_graph(meta)
        assert sg.tables["t"].columns["id"].is_nullable is False
        assert sg.tables["t"].columns["name"].is_nullable is True

    def test_join_paths_populated(self):
        """Two tables with FK should have join paths."""
        meta = {
            "orders": {
                "column_names_original": ["order_id", "user_id"],
                "column_types": ["integer", "integer"],
                "primary_keys": ["order_id"],
                "foreign_keys": [
                    {
                        "src_cols": ["user_id"],
                        "dst_table": "users",
                        "dst_cols": ["user_id"],
                    },
                ],
            },
            "users": {
                "column_names_original": ["user_id", "name"],
                "column_types": ["integer", "varchar"],
                "primary_keys": ["user_id"],
                "foreign_keys": [],
            },
        }
        sg = _tables_meta_to_schema_graph(meta)
        assert "orders" in sg.join_paths_multi
        assert "users" in sg.join_paths_multi["orders"]
        assert len(sg.join_paths_multi["orders"]["users"]) >= 1

    def test_self_join_path_is_empty_list(self):
        """Join path from a table to itself should be [[]]."""
        meta = {
            "t": {
                "column_names_original": ["id"],
                "column_types": ["integer"],
                "primary_keys": ["id"],
                "foreign_keys": [],
            },
        }
        sg = _tables_meta_to_schema_graph(meta)
        assert sg.join_paths_multi["t"]["t"] == [[]]

    def test_schema_hash_is_set(self):
        """Generated graph should have a non-empty schema_hash."""
        meta = {
            "t": {
                "column_names_original": ["id"],
                "column_types": ["integer"],
                "primary_keys": ["id"],
                "foreign_keys": [],
            },
        }
        sg = _tables_meta_to_schema_graph(meta)
        assert sg.schema_hash
        assert isinstance(sg.schema_hash, str)

    def test_fk_marks_column_as_foreign_key(self):
        """FK columns should have is_foreign_key=True and fk_target set."""
        meta = {
            "orders": {
                "column_names_original": ["order_id", "user_id"],
                "column_types": ["integer", "integer"],
                "primary_keys": ["order_id"],
                "foreign_keys": [
                    {
                        "src_cols": ["user_id"],
                        "dst_table": "users",
                        "dst_cols": ["uid"],
                    },
                ],
            },
            "users": {
                "column_names_original": ["uid"],
                "column_types": ["integer"],
                "primary_keys": ["uid"],
                "foreign_keys": [],
            },
        }
        sg = _tables_meta_to_schema_graph(meta)
        assert sg.tables["orders"].columns["user_id"].is_foreign_key is True
        assert sg.tables["orders"].columns["user_id"].fk_target == ("users", "uid")

    def test_skips_fk_when_dst_table_missing(self):
        """FK edges referencing missing tables are skipped without KeyError."""
        meta = {
            "film": {
                "column_names_original": ["film_id", "language_id"],
                "column_types": ["integer", "integer"],
                "primary_keys": ["film_id"],
                "foreign_keys": [
                    {
                        "src_cols": ["language_id"],
                        "dst_table": "language",
                        "dst_cols": ["language_id"],
                    },
                ],
            },
        }
        sg = _tables_meta_to_schema_graph(meta)
        assert "film" in sg.tables
        assert sg.join_paths_multi["film"]["film"] == [[]]

    def test_partition_and_unique_columns(self):
        """partition_columns and unique_columns flow into TableMetadata and columns."""
        meta = {
            "parted": {
                "column_names_original": ["id", "dt"],
                "column_types": ["bigint", "date"],
                "primary_keys": ["id"],
                "foreign_keys": [],
                "unique_columns": ["dt"],
                "partition_columns": ["dt"],
            },
        }
        sg = _tables_meta_to_schema_graph(meta)
        t = sg.tables["parted"]
        assert t.partition_columns == ["dt"]
        assert t.columns["dt"].is_unique is True


class TestEdgeKey:
    """Tests for _edge_key."""

    def test_produces_stable_tuple(self):
        """_edge_key returns stable sortable tuple."""
        e = FKEdge(
            src_table="orders",
            src_cols=["customer_id"],
            dst_table="customers",
            dst_cols=["id"],
        )
        key = _edge_key(e)
        assert key == ("orders", ("customer_id",), "customers", ("id",))

    def test_composite_key(self):
        """_edge_key handles composite src and dst cols."""
        e = FKEdge(
            src_table="t1",
            src_cols=["a", "b"],
            dst_table="t2",
            dst_cols=["x", "y"],
        )
        key = _edge_key(e)
        assert key[1] == ("a", "b")
        assert key[3] == ("x", "y")


class TestComputeSchemaStatsEdgeCases:
    """Additional edge cases for compute_schema_stats."""

    def test_no_filterable_columns_anywhere(self):
        """When no column is filterable, min_filterable stays 0 and list is empty."""
        t = _table("t", {"c": _col(name="user_password", role=ColumnRole.CATEGORICAL.value)})
        sg = SchemaGraph(tables={"t": t}, join_paths_multi={}, effective_structural_hash="x")
        stats = compute_schema_stats(sg)
        assert stats["total_filterable"] == 0
        assert stats["min_filterable_per_table"] == 0
        assert stats["filterable_per_table"] == []

    def test_column_types_longer_than_names_uses_unknown_type(self):
        """Short column_types list yields UNKNOWN for trailing columns."""
        meta = {
            "wide": {
                "column_names_original": ["a", "b"],
                "column_types": ["int"],
                "primary_keys": [],
                "foreign_keys": [],
            },
        }
        sg = _tables_meta_to_schema_graph(meta)
        assert sg.tables["wide"].columns["b"].data_type == "UNKNOWN"


class TestComputeSchemaLimitsEdgeCases:
    """Edge cases for compute_schema_limits."""

    def test_table_count_over_10_max_tables_four(self):
        """table_count > 10 yields max_tables=4."""
        stats = {"table_count": 15, "total_filterable": 50, "total_groupable": 30}
        limits = compute_schema_limits(stats)
        assert limits.max_tables == 4

    def test_single_table(self):
        """Single table should have max_tables = 1."""
        stats = {"table_count": 1, "total_filterable": 3, "total_groupable": 2}
        limits = compute_schema_limits(stats)
        assert limits.max_tables == 1


class TestMergeDdlForeignKeys:
    """Tests for merge_ddl_foreign_keys_into_schema_graph."""

    def test_adds_edge_and_join_path(self):
        """DDL FK merge connects tables missing catalog FKs."""
        tables_meta = {
            "child": {
                "column_names_original": ["id", "parent_id"],
                "column_types": ["INTEGER", "INTEGER"],
                "primary_keys": ["id"],
                "foreign_keys": [],
            },
            "parent": {
                "column_names_original": ["id"],
                "column_types": ["INTEGER"],
                "primary_keys": ["id"],
                "foreign_keys": [],
            },
        }
        sg = _tables_meta_to_schema_graph(tables_meta)
        assert sg.join_paths_multi["child"]["parent"] == []
        ddl = {
            "child": {
                "foreign_keys": [
                    {
                        "src_cols": ["parent_id"],
                        "dst_table": "parent",
                        "dst_cols": ["id"],
                    }
                ]
            }
        }
        merge_ddl_foreign_keys_into_schema_graph(sg, ddl)
        assert len(sg.tables["child"].foreign_keys) == 1
        paths = sg.join_paths_multi["child"]["parent"]
        assert paths and len(paths[0]) > 0

    def test_empty_ddl_or_empty_graph_noop(self):
        """Early return when ddl or graph tables are empty."""
        meta = {
            "t": {
                "column_names_original": ["id"],
                "column_types": ["int"],
                "primary_keys": ["id"],
                "foreign_keys": [],
            },
        }
        sg = _tables_meta_to_schema_graph(meta)
        h_before = sg.schema_hash
        merge_ddl_foreign_keys_into_schema_graph(sg, {})
        assert sg.schema_hash == h_before
        merge_ddl_foreign_keys_into_schema_graph(
            SchemaGraph(tables={}, join_paths_multi={}, effective_structural_hash="z"),
            {"t": {"foreign_keys": [{"src_cols": ["id"], "dst_table": "t", "dst_cols": ["id"]}]}},
        )

    def test_resolves_table_name_case_insensitive(self):
        """DDL table names differing only in case map onto graph keys."""
        meta = {
            "Child": {
                "column_names_original": ["id", "parent_id"],
                "column_types": ["INT", "INT"],
                "primary_keys": ["id"],
                "foreign_keys": [],
            },
            "Parent": {
                "column_names_original": ["id"],
                "column_types": ["INT"],
                "primary_keys": ["id"],
                "foreign_keys": [],
            },
        }
        sg = _tables_meta_to_schema_graph(meta)
        ddl = {
            "child": {
                "foreign_keys": [
                    {
                        "src_cols": ["parent_id"],
                        "dst_table": "PARENT",
                        "dst_cols": ["id"],
                    },
                ]
            }
        }
        merge_ddl_foreign_keys_into_schema_graph(sg, ddl)
        assert len(sg.tables["Child"].foreign_keys) == 1

    def test_skips_duplicate_edge(self):
        """Identical FK edge from DDL is not appended twice."""
        meta = {
            "child": {
                "column_names_original": ["id", "parent_id"],
                "column_types": ["INT", "INT"],
                "primary_keys": ["id"],
                "foreign_keys": [
                    {
                        "src_cols": ["parent_id"],
                        "dst_table": "parent",
                        "dst_cols": ["id"],
                    },
                ],
            },
            "parent": {
                "column_names_original": ["id"],
                "column_types": ["INT"],
                "primary_keys": ["id"],
                "foreign_keys": [],
            },
        }
        sg = _tables_meta_to_schema_graph(meta)
        n = len(sg.tables["child"].foreign_keys)
        merge_ddl_foreign_keys_into_schema_graph(
            sg,
            {
                "child": {
                    "foreign_keys": [
                        {
                            "src_cols": ["parent_id"],
                            "dst_table": "parent",
                            "dst_cols": ["id"],
                        },
                    ]
                }
            },
        )
        assert len(sg.tables["child"].foreign_keys) == n

    def test_skips_mismatched_column_counts(self):
        """DDL FK with len(src_cols) != len(dst_cols) is ignored."""
        meta = {
            "child": {
                "column_names_original": ["id", "a", "b"],
                "column_types": ["INT", "INT", "INT"],
                "primary_keys": ["id"],
                "foreign_keys": [],
            },
            "parent": {
                "column_names_original": ["id"],
                "column_types": ["INT"],
                "primary_keys": ["id"],
                "foreign_keys": [],
            },
        }
        sg = _tables_meta_to_schema_graph(meta)
        merge_ddl_foreign_keys_into_schema_graph(
            sg,
            {
                "child": {
                    "foreign_keys": [
                        {
                            "src_cols": ["a", "b"],
                            "dst_table": "parent",
                            "dst_cols": ["id"],
                        },
                    ]
                }
            },
        )
        assert sg.tables["child"].foreign_keys == []

    def test_skips_when_dst_column_missing(self):
        """FK referencing non-existent column on destination table is skipped."""
        meta = {
            "child": {
                "column_names_original": ["id", "parent_id"],
                "column_types": ["INT", "INT"],
                "primary_keys": ["id"],
                "foreign_keys": [],
            },
            "parent": {
                "column_names_original": ["id"],
                "column_types": ["INT"],
                "primary_keys": ["id"],
                "foreign_keys": [],
            },
        }
        sg = _tables_meta_to_schema_graph(meta)
        merge_ddl_foreign_keys_into_schema_graph(
            sg,
            {
                "child": {
                    "foreign_keys": [
                        {
                            "src_cols": ["parent_id"],
                            "dst_table": "parent",
                            "dst_cols": ["missing"],
                        },
                    ]
                }
            },
        )
        assert sg.tables["child"].foreign_keys == []


class TestLoadOrCreateSchemaPostgresqlDdlMerge:
    """``load_or_create_schema_postgresql`` merges parsed DDL FKs after successful reflection."""

    def test_merges_ddl_after_successful_reflect(self, monkeypatch, tmp_path):
        sql_path = tmp_path / "schema.sql"
        sql_path.write_text("-- stub\n", encoding="utf-8")
        monkeypatch.setattr(EngineConfig.RUNTIME, "SQL_FILE_PATH", str(sql_path), raising=False)

        tables_meta = {
            "child": {
                "column_names_original": ["id", "parent_id"],
                "column_types": ["INTEGER", "INTEGER"],
                "primary_keys": ["id"],
                "foreign_keys": [],
            },
            "parent": {
                "column_names_original": ["id"],
                "column_types": ["INTEGER"],
                "primary_keys": ["id"],
                "foreign_keys": [],
            },
        }
        baseline = _tables_meta_to_schema_graph(tables_meta)

        monkeypatch.setattr(
            "aetherdialect._schema._reflect_schema",
            lambda *a, **k: baseline,
        )

        ddl = {
            "child": {
                "foreign_keys": [
                    {
                        "src_cols": ["parent_id"],
                        "dst_table": "parent",
                        "dst_cols": ["id"],
                    }
                ]
            }
        }
        monkeypatch.setattr("aetherdialect._schema.parse_sql_file", lambda _p, **_kw: ddl)

        sg = load_or_create_schema_postgresql(None, include="tables")
        assert len(sg.tables["child"].foreign_keys) == 1
        assert sg.tables["child"].foreign_keys[0].dst_table == "parent"


class TestResolveGraphTableName:
    """Tests for _resolve_graph_table_name."""

    def test_exact_match(self):
        assert _resolve_graph_table_name("orders", {"orders", "users"}) == "orders"

    def test_case_insensitive_fallback(self):
        assert _resolve_graph_table_name("ORDERS", {"orders"}) == "orders"

    def test_unknown_returns_none(self):
        assert _resolve_graph_table_name("missing", {"a"}) is None


class TestRecomputeJoinPathsMulti:
    """Tests for _recompute_join_paths_multi."""

    def test_matches_tables_meta_graph_paths(self):
        """Recompute from the same FK set yields consistent reachability."""
        meta = {
            "a": {
                "column_names_original": ["id", "b_id"],
                "column_types": ["int", "int"],
                "primary_keys": ["id"],
                "foreign_keys": [{"src_cols": ["b_id"], "dst_table": "b", "dst_cols": ["id"]}],
            },
            "b": {
                "column_names_original": ["id"],
                "column_types": ["int"],
                "primary_keys": ["id"],
                "foreign_keys": [],
            },
        }
        sg = _tables_meta_to_schema_graph(meta)
        jp = _recompute_join_paths_multi(sg.tables)
        assert jp["a"]["b"]
        assert jp["b"]["a"]


class TestSchemaCacheHelpers:
    """Pure JSON / fingerprint helpers used when writing schema cache."""

    def test_schema_cache_json_blob_sorted_keys(self):
        blob = _schema_cache_json_blob({"z": 1, "a": 2})
        assert blob.index('"a"') < blob.index('"z"')

    def test_tables_payload_round_trip_stable(self):
        tables = {
            "t": _table_to_dict(_table("t", {"c": _col(name="c")})),
        }
        out = _tables_payload_through_model_round_trip(tables)
        assert out["t"]["name"] == "t"
        assert "c" in out["t"]["columns"]

    def test_first_table_where_stable_json_differs(self):
        left = {"t": {"name": "t", "columns": {}}}
        right = {"t": {"name": "t", "columns": {"x": {}}}}
        assert _first_table_where_stable_json_differs(left, right) == "t"

    def test_first_table_where_stable_json_differs_none(self):
        t = {"name": "x"}
        assert _first_table_where_stable_json_differs({"a": t}, {"a": t}) is None

    def test_fingerprint_tables_matches_direct_fp_when_json_stable(self):
        """Full-document JSON round-trip should not change tables fingerprint."""
        tables = {
            "t": _table_to_dict(
                _table(
                    "t",
                    {"id": _col(name="id", is_primary_key=True)},
                    primary_key=["id"],
                )
            )
        }
        cache_data = {
            "tables": tables,
            "join_paths_multi": {},
            "schema_hash": "unused",
            "created_at": "",
            "enum_values": {},
            "schema_stats": {},
            "notes_sha256": "",
        }
        assert _fingerprint_tables_after_document_round_trip(cache_data) == schema_hash_fp(tables)

    def test_first_table_differs_when_only_one_side_has_table(self):
        """Missing table on one side yields a slot mismatch for that name."""
        only_left = {"name": "only", "columns": {}}
        assert _first_table_where_stable_json_differs({"t": only_left}, {}) == "t"


def _pk_col(
    name: str,
    *,
    distinct: int = 100,
    distinct_from_sample: bool = False,
    value_type: str = "integer",
    is_nullable: bool = False,
    null_ratio: float = 0.0,
) -> ColumnMetadata:
    """Build a ColumnMetadata that can qualify as an inferred primary key."""
    return _col(
        name=name,
        data_type="int",
        value_type=value_type,
        distinct_count=distinct,
        distinct_ratio=1.0,
        row_count=distinct,
        distinct_from_sample=distinct_from_sample,
        is_nullable=is_nullable,
        null_ratio=null_ratio,
    )


class TestSelectInferredPkCandidate:
    """Tests for _select_inferred_pk_candidate name preference rules."""

    def test_empty_returns_none(self):
        assert _select_inferred_pk_candidate("t", []) is None

    def test_single_returned_regardless_of_name(self):
        assert _select_inferred_pk_candidate("companies", ["weird_uid"]) == "weird_uid"

    def test_prefers_id_over_others(self):
        assert _select_inferred_pk_candidate("companies", ["weird_uid", "id", "companies_id"]) == "id"

    def test_prefers_table_name_id_when_no_id(self):
        assert _select_inferred_pk_candidate("companies", ["weird_uid", "companies_id"]) == "companies_id"

    def test_falls_through_to_suffix(self):
        assert _select_inferred_pk_candidate("t", ["xyz", "abc_key"]) == "abc_key"

    def test_falls_through_to_sorted_first(self):
        assert _select_inferred_pk_candidate("t", ["zzz", "aaa"]) == "aaa"


class TestInferMissingPksFromProfile:
    """Tests for _infer_missing_pks_from_profile."""

    def test_single_candidate_with_arbitrary_name(self):
        cols = {"weird_uid": _pk_col("weird_uid")}
        tbl = _table("companies", cols, row_count=100)
        result = _infer_missing_pks_from_profile({"companies": tbl})
        assert result == [("companies", "weird_uid")]
        assert tbl.primary_key == ["weird_uid"]
        assert tbl.columns["weird_uid"].is_primary_key is True

    def test_multiple_candidates_picks_id(self):
        cols = {"weird_uid": _pk_col("weird_uid"), "id": _pk_col("id")}
        tbl = _table("companies", cols, row_count=100)
        result = _infer_missing_pks_from_profile({"companies": tbl})
        assert result == [("companies", "id")]
        assert tbl.primary_key == ["id"]

    def test_row_count_below_threshold_skips(self):
        cols = {"id": _pk_col("id", distinct=10)}
        tbl = _table("t", cols, row_count=10)
        assert int(PolicyConfig.INFERRED_PK_MIN_ROW_COUNT) > 10
        result = _infer_missing_pks_from_profile({"t": tbl})
        assert result == []
        assert tbl.primary_key == []

    def test_declared_pk_preserved(self):
        cols = {"x": _pk_col("x"), "id": _pk_col("id")}
        tbl = _table("t", cols, primary_key=["x"], row_count=100)
        result = _infer_missing_pks_from_profile({"t": tbl})
        assert result == []
        assert tbl.primary_key == ["x"]
        assert cols["id"].is_primary_key is False

    def test_nullable_column_with_nulls_rejected(self):
        cols = {"id": _pk_col("id", is_nullable=True, null_ratio=0.1)}
        tbl = _table("t", cols, row_count=100)
        result = _infer_missing_pks_from_profile({"t": tbl})
        assert result == []

    def test_value_type_filter_rejects_boolean(self):
        cols = {"id": _pk_col("id", value_type="boolean")}
        tbl = _table("t", cols, row_count=100)
        result = _infer_missing_pks_from_profile({"t": tbl})
        assert result == []

    def test_distinct_count_must_equal_row_count(self):
        cols = {"id": _pk_col("id", distinct=99)}
        tbl = _table("t", cols, row_count=100)
        result = _infer_missing_pks_from_profile({"t": tbl})
        assert result == []

    def test_sample_distinct_without_dialect_skips_inference(self):
        cols = {"id": _pk_col("id", distinct_from_sample=True)}
        tbl = _table("t", cols, row_count=100)
        assert _infer_missing_pks_from_profile({"t": tbl}) == []

    def test_sample_distinct_confirmed_by_dialect_infers_pk(self):
        cols = {"id": _pk_col("id", distinct_from_sample=True)}
        tbl = _table("t", cols, row_count=100)

        class _D:
            def refresh_full_table_distinct_for_pk_inference(self, *_a, **_k):
                return (100, 100, 0.0)

        assert _infer_missing_pks_from_profile({"t": tbl}, dialect=_D()) == [("t", "id")]


class TestInferMissingFksAfterPkInference:
    """The FK suffix inferer must produce edges against PKs created by _infer_missing_pks_from_profile."""

    def test_suffix_fk_inferred_after_pk_inference(self):
        parent_cols = {"customer_id": _pk_col("customer_id")}
        parent = _table("customer", parent_cols, row_count=100)
        child_cols = {
            "order_id": _pk_col("order_id"),
            "customer_id": _col(name="customer_id", data_type="int", value_type="integer"),
        }
        child = _table("orders", child_cols, row_count=100)
        tables = {"customer": parent, "orders": child}
        before = _infer_missing_fks(tables)
        assert before == []
        _infer_missing_pks_from_profile(tables)
        after = _infer_missing_fks(tables)
        assert len(after) == 1
        edge = after[0]
        assert edge.src_table == "orders"
        assert edge.src_cols == ["customer_id"]
        assert edge.dst_table == "customer"
        assert edge.dst_cols == ["customer_id"]


class TestPromoteSemanticEdgesToFks:
    """Tests for _promote_semantic_edges_to_fks."""

    def _graph(self, tables: dict[str, TableMetadata]) -> SchemaGraph:
        return SchemaGraph(tables=tables, join_paths_multi={}, effective_structural_hash="x")

    def test_one_side_pk_promotes_correctly(self):
        a_cols = {
            "b_id": _col(
                name="b_id",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.IDENTIFIER.value,
                distinct_count=10,
            )
        }
        b_cols = {"id": _pk_col("id", value_type="string")}
        a_cols["b_id"].semantic_join_neighbors = [("b", "id")]
        b_cols["id"].semantic_join_neighbors = [("a", "b_id")]
        a = _table("a", a_cols)
        b = _table("b", b_cols, primary_key=["id"])
        sg = self._graph({"a": a, "b": b})
        promoted = _promote_semantic_edges_to_fks(sg)
        assert promoted == 1
        assert len(a.foreign_keys) == 1
        edge = a.foreign_keys[0]
        assert (edge.src_table, edge.src_cols, edge.dst_table, edge.dst_cols) == (
            "a",
            ["b_id"],
            "b",
            ["id"],
        )
        assert a_cols["b_id"].is_foreign_key is True
        assert a_cols["b_id"].fk_target == ("b", "id")
        assert a_cols["b_id"].semantic_join_neighbors == []
        assert b_cols["id"].semantic_join_neighbors == []

    def test_neither_side_pk_skipped(self):
        a_cols = {"x": _col(name="x", data_type="varchar", value_type="string")}
        b_cols = {"y": _col(name="y", data_type="varchar", value_type="string")}
        a_cols["x"].semantic_join_neighbors = [("b", "y")]
        b_cols["y"].semantic_join_neighbors = [("a", "x")]
        a = _table("a", a_cols)
        b = _table("b", b_cols)
        sg = self._graph({"a": a, "b": b})
        assert _promote_semantic_edges_to_fks(sg) == 0
        assert a.foreign_keys == []
        assert b.foreign_keys == []
        assert a_cols["x"].semantic_join_neighbors == [("b", "y")]

    def test_both_pk_uses_lexicographically_smaller_as_src(self):
        a_cols = {
            "b_id": _col(
                name="b_id",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.IDENTIFIER.value,
                distinct_count=10,
                is_primary_key=True,
            )
        }
        b_cols = {
            "a_id": _col(
                name="a_id",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.IDENTIFIER.value,
                distinct_count=10,
                is_primary_key=True,
            )
        }
        a_cols["b_id"].semantic_join_neighbors = [("b", "a_id")]
        b_cols["a_id"].semantic_join_neighbors = [("a", "b_id")]
        a = _table("a", a_cols, primary_key=["b_id"])
        b = _table("b", b_cols, primary_key=["a_id"])
        sg = self._graph({"a": a, "b": b})
        promoted = _promote_semantic_edges_to_fks(sg)
        assert promoted == 1
        assert len(a.foreign_keys) == 1
        edge = a.foreign_keys[0]
        assert edge.src_table == "a"
        assert edge.dst_table == "b"

    def test_type_incompatible_skipped(self):
        a_cols = {"x": _col(name="x", data_type="varchar", value_type="string")}
        b_cols = {"id": _pk_col("id", value_type="integer")}
        a_cols["x"].semantic_join_neighbors = [("b", "id")]
        b_cols["id"].semantic_join_neighbors = [("a", "x")]
        a = _table("a", a_cols)
        b = _table("b", b_cols, primary_key=["id"])
        sg = self._graph({"a": a, "b": b})
        assert _promote_semantic_edges_to_fks(sg) == 0
        assert a.foreign_keys == []
        assert a_cols["x"].semantic_join_neighbors == [("b", "id")]

    def test_promotion_fires_when_name_shape_mismatches(self):
        a_cols = {
            "b_external": _col(
                name="b_external",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.IDENTIFIER.value,
                distinct_count=10,
            )
        }
        b_cols = {"id": _pk_col("id", value_type="string")}
        a_cols["b_external"].semantic_join_neighbors = [("b", "id")]
        b_cols["id"].semantic_join_neighbors = [("a", "b_external")]
        a = _table("a", a_cols)
        b = _table("b", b_cols, primary_key=["id"])
        sg = self._graph({"a": a, "b": b})
        promoted = _promote_semantic_edges_to_fks(sg)
        assert promoted == 1
        assert len(a.foreign_keys) == 1
        edge = a.foreign_keys[0]
        assert (edge.src_table, edge.src_cols, edge.dst_table, edge.dst_cols) == (
            "a",
            ["b_external"],
            "b",
            ["id"],
        )

    def test_promotion_fires_when_source_role_is_categorical(self):
        a_cols = {
            "b_id": _col(
                name="b_id",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.CATEGORICAL.value,
                distinct_count=10,
            )
        }
        b_cols = {"id": _pk_col("id", value_type="string")}
        a_cols["b_id"].semantic_join_neighbors = [("b", "id")]
        b_cols["id"].semantic_join_neighbors = [("a", "b_id")]
        a = _table("a", a_cols)
        b = _table("b", b_cols, primary_key=["id"])
        sg = self._graph({"a": a, "b": b})
        promoted = _promote_semantic_edges_to_fks(sg)
        assert promoted == 1
        assert len(a.foreign_keys) == 1
        assert a.foreign_keys[0].src_cols == ["b_id"]


class TestSchemaHashInvarianceForInferredPk:
    """An inferred PK and a declared PK must yield identical effective_structural_hash."""

    def _build(self, declared: bool) -> SchemaGraph:
        cols = {"id": _pk_col("id")}
        if declared:
            tbl = _table("t", cols, primary_key=["id"], row_count=100)
        else:
            tbl = _table("t", cols, row_count=100)
        tables = {"t": tbl}
        if not declared:
            _infer_missing_pks_from_profile(tables)
        sg = SchemaGraph(tables=tables, join_paths_multi={}, effective_structural_hash="")
        ctx = SchemaContext()
        assign_schema_graph_hashes(sg, ctx, notes_sha256="")
        return sg

    def test_declared_and_inferred_pk_produce_same_hash(self):
        declared = self._build(declared=True)
        inferred = self._build(declared=False)
        assert inferred.tables["t"].primary_key == ["id"]
        assert inferred.tables["t"].columns["id"].is_primary_key is True
        assert declared.effective_structural_hash == inferred.effective_structural_hash
        assert declared.structural_hash == inferred.structural_hash


class TestStripSchemaContextDeniedColumns:
    """Tests for :func:`_strip_schema_context_denied_columns` removing denied columns from the graph."""

    def _build(self) -> SchemaGraph:
        contacts_cols = {
            "id": _col(name="id", is_primary_key=True, value_type="integer"),
            "email": _col(name="email", value_type="string"),
            "name": _col(name="name", value_type="string"),
        }
        companies_cols = {
            "id": _col(name="id", is_primary_key=True, value_type="integer"),
            "email": _col(name="email", value_type="string"),
            "name": _col(name="name", value_type="string"),
        }
        contacts = _table("contacts", contacts_cols, primary_key=["id"])
        companies = _table("companies", companies_cols, primary_key=["id"])
        return SchemaGraph(
            tables={"contacts": contacts, "companies": companies},
            join_paths_multi={},
            effective_structural_hash="x",
        )

    def test_qualified_removes_only_named_table_column(self):
        sg = self._build()
        ctx = SchemaContext(deny_columns=frozenset({"contacts.email"}))
        _strip_schema_context_denied_columns(sg, ctx)
        assert sg.deny_columns == {}
        assert "email" not in sg.tables["contacts"].columns
        assert "email" in sg.tables["companies"].columns

    def test_glob_removes_column_across_tables(self):
        sg = self._build()
        ctx = SchemaContext(deny_columns=frozenset({"*.email"}))
        _strip_schema_context_denied_columns(sg, ctx)
        assert "email" not in sg.tables["contacts"].columns
        assert "email" not in sg.tables["companies"].columns
        assert "name" in sg.tables["contacts"].columns

    def test_mixed_glob_and_qualified(self):
        sg = self._build()
        ctx = SchemaContext(deny_columns=frozenset({"*.email", "contacts.name"}))
        _strip_schema_context_denied_columns(sg, ctx)
        assert "email" not in sg.tables["contacts"].columns
        assert "name" not in sg.tables["contacts"].columns
        assert "name" in sg.tables["companies"].columns


class TestValidateScopeAgainstGraph:
    """Tests for _validate_scope_against_graph rejecting unknown deny entries."""

    def _build(self) -> SchemaGraph:
        cols = {
            "id": _col(name="id", is_primary_key=True, value_type="integer"),
            "email": _col(name="email", value_type="string"),
        }
        return SchemaGraph(
            tables={"contacts": _table("contacts", cols, primary_key=["id"])},
            join_paths_multi={},
            effective_structural_hash="x",
        )

    def test_qualified_unknown_table_raises(self):
        sg = self._build()
        ctx = SchemaContext(deny_columns=frozenset({"orders.id"}))
        try:
            _validate_scope_against_graph(sg, ctx)
        except SchemaAccessError as e:
            assert "unknown table" in str(e)
        else:
            raise AssertionError("expected SchemaAccessError")

    def test_qualified_unknown_column_raises(self):
        sg = self._build()
        ctx = SchemaContext(deny_columns=frozenset({"contacts.zzz"}))
        try:
            _validate_scope_against_graph(sg, ctx)
        except SchemaAccessError as e:
            assert "unknown column" in str(e)
        else:
            raise AssertionError("expected SchemaAccessError")

    def test_glob_no_match_raises(self):
        sg = self._build()
        ctx = SchemaContext(deny_columns=frozenset({"*.zzz"}))
        try:
            _validate_scope_against_graph(sg, ctx)
        except SchemaAccessError as e:
            assert "matches no column" in str(e)
        else:
            raise AssertionError("expected SchemaAccessError")

    def test_glob_match_accepted(self):
        sg = self._build()
        ctx = SchemaContext(deny_columns=frozenset({"*.email"}))
        _validate_scope_against_graph(sg, ctx)

    def test_qualified_match_accepted(self):
        sg = self._build()
        ctx = SchemaContext(deny_columns=frozenset({"contacts.email"}))
        _validate_scope_against_graph(sg, ctx)


class TestValidateScopeAgainstGraphAllowColumns:
    """allow_columns entries must reference existing tables/columns."""

    def _build(self) -> SchemaGraph:
        cols = {
            "id": _col(name="id", is_primary_key=True, value_type="integer"),
            "email": _col(name="email", value_type="string"),
        }
        return SchemaGraph(
            tables={"contacts": _table("contacts", cols, primary_key=["id"])},
            join_paths_multi={},
            effective_structural_hash="x",
        )

    def test_qualified_unknown_table_raises(self):
        sg = self._build()
        ctx = SchemaContext(allow_columns=frozenset({"orders.id"}))
        try:
            _validate_scope_against_graph(sg, ctx)
        except SchemaAccessError as e:
            assert "allow_columns references unknown table" in str(e)
        else:
            raise AssertionError("expected SchemaAccessError")

    def test_qualified_unknown_column_raises(self):
        sg = self._build()
        ctx = SchemaContext(allow_columns=frozenset({"contacts.zzz"}))
        try:
            _validate_scope_against_graph(sg, ctx)
        except SchemaAccessError as e:
            assert "allow_columns references unknown column" in str(e)
        else:
            raise AssertionError("expected SchemaAccessError")

    def test_glob_no_match_raises(self):
        sg = self._build()
        ctx = SchemaContext(allow_columns=frozenset({"*.zzz"}))
        try:
            _validate_scope_against_graph(sg, ctx)
        except SchemaAccessError as e:
            assert "allow_columns glob" in str(e)
        else:
            raise AssertionError("expected SchemaAccessError")

    def test_glob_match_accepted(self):
        sg = self._build()
        ctx = SchemaContext(allow_columns=frozenset({"*.email"}))
        _validate_scope_against_graph(sg, ctx)


class TestApplySchemaContextAllowColumns:
    """allow_columns drops columns outside the allow set; PK and FK columns are auto-included."""

    def _build(self) -> SchemaGraph:
        contacts_cols = {
            "id": _col(name="id", is_primary_key=True, value_type="integer"),
            "company_id": _col(name="company_id", is_foreign_key=True, value_type="integer"),
            "email": _col(name="email", value_type="string"),
            "notes": _col(name="notes", value_type="string"),
        }
        companies_cols = {
            "id": _col(name="id", is_primary_key=True, value_type="integer"),
            "name": _col(name="name", value_type="string"),
            "secret": _col(name="secret", value_type="string"),
        }
        contacts_fk = FKEdge(
            src_table="contacts",
            src_cols=["company_id"],
            dst_table="companies",
            dst_cols=["id"],
        )
        contacts = _table("contacts", contacts_cols, primary_key=["id"], foreign_keys=[contacts_fk])
        companies = _table("companies", companies_cols, primary_key=["id"])
        return SchemaGraph(
            tables={"contacts": contacts, "companies": companies},
            join_paths_multi={},
            effective_structural_hash="x",
        )

    def test_empty_allow_columns_is_noop(self):
        sg = self._build()
        ctx = SchemaContext(allow_columns=frozenset())
        _apply_schema_context_allow_columns(sg, ctx)
        assert set(sg.tables["contacts"].columns) == {
            "id",
            "company_id",
            "email",
            "notes",
        }
        assert set(sg.tables["companies"].columns) == {"id", "name", "secret"}

    def test_qualified_keeps_listed_and_pk_fk(self):
        sg = self._build()
        ctx = SchemaContext(allow_columns=frozenset({"contacts.email", "companies.name"}))
        _apply_schema_context_allow_columns(sg, ctx)
        assert set(sg.tables["contacts"].columns) == {"id", "company_id", "email"}
        assert set(sg.tables["companies"].columns) == {"id", "name"}

    def test_glob_applies_to_every_table_with_column(self):
        sg = self._build()
        ctx = SchemaContext(allow_columns=frozenset({"*.name"}))
        _apply_schema_context_allow_columns(sg, ctx)
        assert set(sg.tables["companies"].columns) == {"id", "name"}
        assert set(sg.tables["contacts"].columns) == {"id", "company_id"}

    def test_pk_and_fk_always_retained(self):
        sg = self._build()
        ctx = SchemaContext(allow_columns=frozenset({"contacts.notes"}))
        _apply_schema_context_allow_columns(sg, ctx)
        assert "id" in sg.tables["contacts"].columns
        assert "company_id" in sg.tables["contacts"].columns
        assert "notes" in sg.tables["contacts"].columns
        assert "email" not in sg.tables["contacts"].columns
        assert set(sg.tables["companies"].columns) == {"id"}

    def test_fk_destination_columns_retained(self):
        sg = self._build()
        ctx = SchemaContext(allow_columns=frozenset({"contacts.email"}))
        _apply_schema_context_allow_columns(sg, ctx)
        assert "id" in sg.tables["companies"].columns


class TestSchemaContextAllowColumnsParser:
    """SchemaContext normalizes and rejects malformed allow_columns entries."""

    def test_normalization_lowercases(self):
        ctx = SchemaContext(allow_columns=frozenset({"Contacts.Email", "*.name"}))
        assert ctx.allow_columns == frozenset({"contacts.email", "*.name"})

    def test_too_many_dots_raises(self):
        from aetherdialect._contracts_base import ConfigError

        try:
            SchemaContext(allow_columns=frozenset({"a.b.c"}))
        except ConfigError as e:
            assert "allow_columns" in str(e)
        else:
            raise AssertionError("expected ConfigError")

    def test_qualified_and_glob_helpers(self):
        ctx = SchemaContext(allow_columns=frozenset({"contacts.email", "*.name"}))
        assert ctx.qualified_allows() == frozenset({("contacts", "email")})
        assert ctx.glob_column_allows() == frozenset({"name"})
        assert ctx.bare_allows() == frozenset()


class TestColumnMetadataIsDeniedRoundTrip:
    """is_denied must round-trip through to_dict/from_dict so cache reload preserves the flag."""

    def test_round_trip_default_false(self):
        c = ColumnMetadata(name="x", data_type="int")
        assert c.is_denied is False
        round_trip = ColumnMetadata.from_dict(c.to_dict())
        assert round_trip.is_denied is False

    def test_round_trip_true(self):
        c = ColumnMetadata(name="x", data_type="int", is_denied=True)
        assert c.to_dict()["is_denied"] is True
        round_trip = ColumnMetadata.from_dict(c.to_dict())
        assert round_trip.is_denied is True


class TestMarkCanonicalDuplicates:
    """Tests for _mark_canonical_duplicates: PK > most-distinct > lex tie-break."""

    def _sg_two_tables(self, t1_cols, t2_cols, *, t1_pk=None, t2_pk=None):
        t1 = _table("alpha", t1_cols, primary_key=t1_pk or [])
        t2 = _table("beta", t2_cols, primary_key=t2_pk or [])
        return SchemaGraph(
            tables={"alpha": t1, "beta": t2},
            join_paths_multi={},
            effective_structural_hash="x",
        )

    def test_pk_wins_over_higher_distinct(self):
        a = {"id": _col(name="id", is_primary_key=True, distinct_count=10)}
        b = {"id": _col(name="id", is_primary_key=False, distinct_count=999)}
        sg = self._sg_two_tables(a, b, t1_pk=["id"])
        demoted = _mark_canonical_duplicates(sg)
        assert demoted == 1
        assert sg.tables["alpha"].columns["id"].is_canonical_duplicate is True
        assert sg.tables["beta"].columns["id"].is_canonical_duplicate is False

    def test_highest_distinct_wins_among_non_pk(self):
        a = {"email": _col(name="email", distinct_count=50)}
        b = {"email": _col(name="email", distinct_count=200)}
        sg = self._sg_two_tables(a, b)
        _mark_canonical_duplicates(sg)
        assert sg.tables["beta"].columns["email"].is_canonical_duplicate is True
        assert sg.tables["alpha"].columns["email"].is_canonical_duplicate is False

    def test_lex_tie_break_when_distinct_equal(self):
        a = {"name": _col(name="name", distinct_count=100)}
        b = {"name": _col(name="name", distinct_count=100)}
        sg = self._sg_two_tables(a, b)
        _mark_canonical_duplicates(sg)
        assert sg.tables["alpha"].columns["name"].is_canonical_duplicate is True
        assert sg.tables["beta"].columns["name"].is_canonical_duplicate is False

    def test_singleton_untouched(self):
        a = {"only_here": _col(name="only_here", distinct_count=1)}
        b = {"other": _col(name="other", distinct_count=1)}
        sg = self._sg_two_tables(a, b)
        demoted = _mark_canonical_duplicates(sg)
        assert demoted == 0
        assert sg.tables["alpha"].columns["only_here"].is_canonical_duplicate is True
        assert sg.tables["beta"].columns["other"].is_canonical_duplicate is True

    def test_round_trip_preserves_field(self):
        c = ColumnMetadata(name="x", data_type="int", is_canonical_duplicate=False)
        assert c.to_dict()["is_canonical_duplicate"] is False
        rt = ColumnMetadata.from_dict(c.to_dict())
        assert rt.is_canonical_duplicate is False

    def test_default_is_true(self):
        c = ColumnMetadata(name="x", data_type="int")
        assert c.is_canonical_duplicate is True

    def test_three_way_pk_then_distinct(self):
        a = {"customer_id": _col(name="customer_id", distinct_count=500)}
        b = {"customer_id": _col(name="customer_id", is_primary_key=True, distinct_count=50)}
        c_cols = {"customer_id": _col(name="customer_id", distinct_count=100)}
        t_a = _table("orders", a)
        t_b = _table("customers", b, primary_key=["customer_id"])
        t_c = _table("returns", c_cols)
        sg = SchemaGraph(
            tables={"orders": t_a, "customers": t_b, "returns": t_c},
            join_paths_multi={},
            effective_structural_hash="x",
        )
        _mark_canonical_duplicates(sg)
        assert sg.tables["customers"].columns["customer_id"].is_canonical_duplicate is True
        assert sg.tables["orders"].columns["customer_id"].is_canonical_duplicate is False
        assert sg.tables["returns"].columns["customer_id"].is_canonical_duplicate is False


class TestSchemaOverrideNullSkipAndPrune:
    """Null description/role keys are dropped from the overrides document; invalid roles notify and prune."""

    def test_null_column_description_and_role_removed_from_document(self, schema_graph, monkeypatch):
        from aetherdialect._schema import apply_schema_overrides_to_graph

        monkeypatch.setattr("aetherdialect._schema.llm_credentials_configured", lambda: False)
        doc = _ov_doc(
            tables={"orders": {"columns": {"amount": {"description": None, "role": None}}}},
        )
        apply_schema_overrides_to_graph(schema_graph, doc)
        assert "orders" not in (doc.get("tables") or {})

    def test_invalid_table_role_pruned(self, schema_graph, monkeypatch):
        from aetherdialect._schema import apply_schema_overrides_to_graph

        monkeypatch.setattr("aetherdialect._schema.llm_credentials_configured", lambda: False)
        prev = schema_graph.tables["orders"].role
        doc = _ov_doc(tables={"orders": {"role": "not_a_table_role"}})
        report = apply_schema_overrides_to_graph(schema_graph, doc)
        assert schema_graph.tables["orders"].role == prev
        assert "orders" not in (doc.get("tables") or {})
        assert any(s.code == "invalid_role_override" for s in report.skipped)

    def test_save_sidecar_reflects_pruned_document(self, schema_graph, monkeypatch, tmp_path):
        import json

        from aetherdialect._config import SCHEMA_OVERRIDES_SIDECAR_FILENAME
        from aetherdialect._schema import (
            apply_schema_overrides_to_graph,
            compute_metadata_hash,
            save_overrides_sidecar,
        )

        monkeypatch.setattr("aetherdialect._schema.llm_credentials_configured", lambda: False)
        cache_path = tmp_path / "schema.json.gz"
        cache_path.write_bytes(b"x")
        doc = _ov_doc(
            tables={"orders": {"columns": {"amount": {"role": None, "description": None}}}},
        )
        apply_schema_overrides_to_graph(schema_graph, doc)
        save_overrides_sidecar(
            cache_path,
            doc,
            source_schema_hash=schema_graph.effective_structural_hash,
            metadata_hash=compute_metadata_hash(schema_graph),
        )
        side_path = tmp_path / SCHEMA_OVERRIDES_SIDECAR_FILENAME
        assert side_path.is_file()
        raw = json.loads(side_path.read_text(encoding="utf-8"))
        assert "orders" not in (raw.get("tables") or {})


class TestSchemaOverrides:
    """JSON-roundtrip user editing of the built schema graph."""

    def test_dump_dict_round_trips_current_state(self, schema_graph):
        from aetherdialect._config import (
            SCHEMA_OVERRIDES_EXPORT_DEFAULT_OWNER,
            SCHEMA_OVERRIDES_VERSION,
        )
        from aetherdialect._schema import dump_schema_overrides_dict

        d = dump_schema_overrides_dict(schema_graph)
        assert d["version"] == SCHEMA_OVERRIDES_VERSION
        assert set(d["tables"].keys()) == {"customers", "orders", "products"}
        for _tname, tval in d["tables"].items():
            assert set(tval["description"].keys()) == {"value", "owner"}
            assert set(tval["role"].keys()) == {"value", "owner"}
            assert tval["description"]["owner"] == SCHEMA_OVERRIDES_EXPORT_DEFAULT_OWNER
            assert tval["role"]["owner"] == SCHEMA_OVERRIDES_EXPORT_DEFAULT_OWNER
            for _cname, cval in tval["columns"].items():
                assert "description" in cval
                assert "role" in cval
                assert "sensitivity" in cval
                assert set(cval["description"].keys()) == {"value", "owner"}
                assert set(cval["role"].keys()) == {"value", "owner"}
                assert cval["description"]["owner"] == SCHEMA_OVERRIDES_EXPORT_DEFAULT_OWNER
                assert cval["role"]["owner"] == SCHEMA_OVERRIDES_EXPORT_DEFAULT_OWNER
        assert d["foreign_keys_add"] == []

    def test_dump_to_path_replaces_existing_atomically(self, schema_graph, tmp_path):
        from aetherdialect._schema import dump_schema_overrides_to_path

        target = tmp_path / "ov.json"
        first = dump_schema_overrides_to_path(schema_graph, target)
        assert first.is_file()
        second = dump_schema_overrides_to_path(schema_graph, target)
        assert second == first
        assert second.is_file()

    def test_load_validates_version(self, tmp_path):
        import json as _json

        import pytest

        from aetherdialect._schema import load_schema_overrides_file

        path = tmp_path / "bad.json"
        path.write_text(_json.dumps({"version": 999, "tables": {}, "foreign_keys_add": []}))
        with pytest.raises(ValueError, match="version"):
            load_schema_overrides_file(path)

    def test_load_rejects_unknown_top_level_key(self, tmp_path):
        import json as _json

        import pytest

        from aetherdialect._config import SCHEMA_OVERRIDES_VERSION
        from aetherdialect._schema import load_schema_overrides_file

        path = tmp_path / "bad.json"
        path.write_text(_json.dumps({"version": SCHEMA_OVERRIDES_VERSION, "tables": {}, "synonyms": {}}))
        with pytest.raises(ValueError, match="unsupported top-level"):
            load_schema_overrides_file(path)

    def test_load_rejects_bad_role_enum(self, tmp_path):
        import json as _json

        import pytest

        from aetherdialect._config import SCHEMA_OVERRIDES_VERSION
        from aetherdialect._schema import load_schema_overrides_file

        path = tmp_path / "bad.json"
        path.write_text(
            _json.dumps(
                {
                    "version": SCHEMA_OVERRIDES_VERSION,
                    "tables": {
                        "orders": {
                            "columns": {
                                "amount": {
                                    "role": {"value": "not_a_role"},
                                }
                            }
                        }
                    },
                    "foreign_keys_add": [],
                }
            )
        )
        with pytest.raises(ValueError, match="role"):
            load_schema_overrides_file(path)

    def test_load_rejects_description_owner_key(self, tmp_path):
        import json as _json

        import pytest

        from aetherdialect._config import SCHEMA_OVERRIDES_VERSION
        from aetherdialect._schema import load_schema_overrides_file

        path = tmp_path / "bad.json"
        path.write_text(
            _json.dumps(
                {
                    "version": SCHEMA_OVERRIDES_VERSION,
                    "tables": {
                        "orders": {
                            "description": {
                                "value": "Hello",
                                "owner": "catalog",
                            }
                        }
                    },
                    "foreign_keys_add": [],
                }
            )
        )
        with pytest.raises(ValueError, match="engine-managed"):
            load_schema_overrides_file(path)

    def test_apply_accepts_bare_string_description_and_envelope_value_only(self, schema_graph, monkeypatch):
        from aetherdialect._contracts_base import DescriptionOwner
        from aetherdialect._schema import apply_schema_overrides_to_graph

        monkeypatch.setattr("aetherdialect._schema.llm_credentials_configured", lambda: False)

        apply_schema_overrides_to_graph(
            schema_graph,
            _ov_doc(tables={"orders": {"description": "Bare human text."}}),
        )
        assert schema_graph.tables["orders"].description == "Bare human text."
        assert schema_graph.tables["orders"].description_owner == DescriptionOwner.USER_OVERRIDE

        apply_schema_overrides_to_graph(
            schema_graph,
            _ov_doc(tables={"orders": {"description": {"value": "Envelope without owner."}}}),
        )
        assert schema_graph.tables["orders"].description == "Envelope without owner."

    def test_load_rejects_bad_sensitivity(self, tmp_path):
        import json as _json

        import pytest

        from aetherdialect._config import SCHEMA_OVERRIDES_VERSION
        from aetherdialect._schema import load_schema_overrides_file

        path = tmp_path / "bad.json"
        path.write_text(
            _json.dumps(
                {
                    "version": SCHEMA_OVERRIDES_VERSION,
                    "tables": {"customers": {"columns": {"email": {"sensitivity": "secret"}}}},
                    "foreign_keys_add": [],
                }
            )
        )
        with pytest.raises(ValueError, match="sensitivity"):
            load_schema_overrides_file(path)

    def test_apply_sensitivity_pii_marks_unselectable(self, schema_graph, monkeypatch):
        from aetherdialect._schema import apply_schema_overrides_to_graph

        monkeypatch.setattr("aetherdialect._schema.llm_credentials_configured", lambda: False)
        report = apply_schema_overrides_to_graph(
            schema_graph,
            _ov_doc(
                tables={"customers": {"columns": {"email": {"sensitivity": "pii"}}}},
            ),
        )
        assert report.column_edits == 1
        col = schema_graph.tables["customers"].columns["email"]
        assert col.sensitivity == SensitivityClassification.STRICT
        assert col.is_selectable is False
        assert len(report.skipped) == 1
        assert report.skipped[0].code == "hidden_column_override"

    def test_apply_role_override(self, schema_graph, monkeypatch):
        from aetherdialect._schema import apply_schema_overrides_to_graph

        monkeypatch.setattr("aetherdialect._schema.llm_credentials_configured", lambda: False)
        report = apply_schema_overrides_to_graph(
            schema_graph,
            _ov_doc(
                tables={
                    "orders": {"columns": {"amount": {"role": _orole("numeric_categorical")}}},
                },
            ),
        )
        assert report.column_edits == 1
        assert schema_graph.tables["orders"].columns["amount"].role == "numeric_categorical"

    def test_user_role_override_incompatible_with_value_type_is_discarded(self, schema_graph, monkeypatch):
        """Invalid role vs ``value_type`` is skipped with notify; graph and persisted doc omit the role key."""

        from aetherdialect._schema import apply_schema_overrides_to_graph

        monkeypatch.setattr("aetherdialect._schema.llm_credentials_configured", lambda: False)
        prev_role = schema_graph.tables["orders"].columns["amount"].role
        doc = _ov_doc(
            tables={
                "orders": {"columns": {"amount": {"role": _orole(ColumnRole.TEMPORAL.value)}}},
            },
        )
        report = apply_schema_overrides_to_graph(schema_graph, doc)
        assert schema_graph.tables["orders"].columns["amount"].role == prev_role
        assert any("incompatible" in s.reason for s in report.skipped)
        tbls = doc.get("tables") or {}
        assert "orders" not in tbls or "role" not in (tbls.get("orders", {}).get("columns", {}) or {}).get(
            "amount",
            {},
        )

    def test_apply_rejects_unselectable_override_key(self, schema_graph, monkeypatch, tmp_path):
        import json as _json

        import pytest

        from aetherdialect._config import SCHEMA_OVERRIDES_VERSION
        from aetherdialect._schema import load_schema_overrides_file

        monkeypatch.setattr("aetherdialect._schema.llm_credentials_configured", lambda: False)
        path = tmp_path / "bad.json"
        path.write_text(
            _json.dumps(
                {
                    "version": SCHEMA_OVERRIDES_VERSION,
                    "tables": {"customers": {"columns": {"customer_id": {"is_selectable": False}}}},
                    "foreign_keys_add": [],
                }
            )
        )
        with pytest.raises(ValueError, match="system-derived"):
            load_schema_overrides_file(path)

    def test_apply_unknown_table_and_column_skipped(self, schema_graph, monkeypatch):
        from aetherdialect._schema import apply_schema_overrides_to_graph

        monkeypatch.setattr("aetherdialect._schema.llm_credentials_configured", lambda: False)
        report = apply_schema_overrides_to_graph(
            schema_graph,
            _ov_doc(
                tables={
                    "ghost_table": {"description": _odesc("x")},
                    "orders": {"columns": {"ghost_col": {"description": _odesc("y")}}},
                },
            ),
        )
        reasons = {s.reason for s in report.skipped}
        assert "unknown table" in reasons
        assert "unknown column" in reasons

    def test_apply_fk_add_valid_creates_edge(self, schema_graph, monkeypatch):
        from aetherdialect._schema import apply_schema_overrides_to_graph

        monkeypatch.setattr("aetherdialect._schema.llm_credentials_configured", lambda: False)
        before = len(schema_graph.tables["customers"].foreign_keys)
        report = apply_schema_overrides_to_graph(
            schema_graph,
            _ov_doc(
                foreign_keys_add=[
                    {
                        "from": "customers.customer_id",
                        "to": "products.product_id",
                        "kind": "structural",
                    }
                ],
            ),
        )
        assert report.fks_added == 1
        assert len(schema_graph.tables["customers"].foreign_keys) == before + 1
        edge = schema_graph.tables["customers"].foreign_keys[-1]
        assert edge.dst_table == "products"
        assert edge.inference_tag == "user_override_structural"
        assert schema_graph.tables["customers"].columns["customer_id"].is_foreign_key is True

    def test_apply_fk_add_unknown_endpoint_skipped(self, schema_graph, monkeypatch):
        from aetherdialect._schema import apply_schema_overrides_to_graph

        monkeypatch.setattr("aetherdialect._schema.llm_credentials_configured", lambda: False)
        report = apply_schema_overrides_to_graph(
            schema_graph,
            _ov_doc(
                foreign_keys_add=[
                    {
                        "from": "customers.customer_id",
                        "to": "ghost.id",
                        "kind": "structural",
                    }
                ],
            ),
        )
        assert report.fks_added == 0
        assert any("unknown destination table" in s.reason for s in report.skipped)

    def test_apply_description_uses_llm_when_enabled(self, schema_graph, monkeypatch):
        import json as _json

        from aetherdialect._schema import apply_schema_overrides_to_graph

        monkeypatch.setattr("aetherdialect._schema.llm_credentials_configured", lambda: True)

        def _fake_llm_chat(*, system, user, task):
            assert task == "schema"
            return _json.dumps(
                {
                    "items": [
                        {
                            "path": "tables.orders.description",
                            "text": "REFINED text for orders.",
                        }
                    ]
                }
            )

        monkeypatch.setattr("aetherdialect._schema.llm_chat", _fake_llm_chat)
        report = apply_schema_overrides_to_graph(
            schema_graph,
            _ov_doc(
                tables={
                    "orders": {"description": _odesc("Orders placed by customers; raw human text.")},
                },
            ),
        )
        assert report.table_edits == 1
        assert report.descriptions_refined == 1
        assert schema_graph.tables["orders"].description == "REFINED text for orders."

    def test_round_trip_dump_and_apply_no_changes(self, schema_graph, monkeypatch, tmp_path):
        from aetherdialect._schema import (
            apply_schema_overrides_to_graph,
            dump_schema_overrides_dict,
            load_schema_overrides_file,
        )

        monkeypatch.setattr("aetherdialect._schema.llm_credentials_configured", lambda: False)
        path = tmp_path / "ov.json"
        import json as _json

        path.write_text(_json.dumps(dump_schema_overrides_dict(schema_graph)))
        loaded = load_schema_overrides_file(path)
        report = apply_schema_overrides_to_graph(schema_graph, loaded)
        assert report.table_edits == 0
        assert report.column_edits == 0
        assert report.fks_added == 0
        assert report.skipped == ()


class TestBundleI:
    """Bundle I — schema layer model: drift-safe FK merge, sidecar, denylists, readonly snapshot."""

    def test_dump_includes_readonly_envelope_and_denylists(self, schema_graph):
        from aetherdialect._schema import dump_schema_overrides_dict

        d = dump_schema_overrides_dict(schema_graph)
        assert "_readonly" in d
        assert set(d["_readonly"].keys()) == {
            "foreign_keys_current",
            "primary_keys_current",
            "tables_current",
            "columns_current",
        }
        for entry in d["_readonly"]["foreign_keys_current"]:
            assert "removable" in entry
            assert isinstance(entry["removable"], bool)
            if entry["inference_tag"] is None:
                assert entry["removable"] is False
            else:
                assert entry["removable"] is True
        assert "foreign_keys_remove" in d
        assert "primary_keys_remove" in d
        assert d["foreign_keys_remove"] == []
        assert d["primary_keys_remove"] == []

    def test_apply_diff_preserves_user_override_fk_when_catalog_changes(self, schema_graph, monkeypatch):
        import copy

        from aetherdialect._schema import (
            apply_diff,
            apply_schema_overrides_to_graph,
            diff_schemas,
        )

        monkeypatch.setattr("aetherdialect._schema.llm_credentials_configured", lambda: False)

        report = apply_schema_overrides_to_graph(
            schema_graph,
            _ov_doc(
                foreign_keys_add=[
                    {
                        "from": "customers.customer_id",
                        "to": "products.product_id",
                        "kind": "structural",
                    }
                ],
            ),
        )
        assert report.fks_added == 1

        new_sg = copy.deepcopy(schema_graph)

        new_sg.tables["customers"].foreign_keys = []

        diff = diff_schemas(schema_graph, new_sg)

        class _FakeDialect:
            name = "test"

            def reflect_only(self, *_a, **_k):
                return new_sg

            def profile_schema(self, *_a, **_k):
                pass

            def refresh_full_table_distinct_for_pk_inference(self, *_a, **_k):
                return None

        merged = apply_diff(schema_graph, new_sg, diff, _FakeDialect(), notes_content=None)

        user_edges = [
            e for e in merged.tables["customers"].foreign_keys if (e.inference_tag or "").startswith("user_override_")
        ]
        assert len(user_edges) == 1, "user-override FK was wiped by apply_diff"

    def test_apply_diff_drops_user_override_fk_when_endpoint_disappears(self, schema_graph, monkeypatch):
        import copy

        from aetherdialect._schema import (
            apply_diff,
            apply_schema_overrides_to_graph,
            diff_schemas,
        )

        monkeypatch.setattr("aetherdialect._schema.llm_credentials_configured", lambda: False)
        apply_schema_overrides_to_graph(
            schema_graph,
            _ov_doc(
                foreign_keys_add=[
                    {
                        "from": "customers.customer_id",
                        "to": "products.product_id",
                        "kind": "structural",
                    }
                ],
            ),
        )

        new_sg = copy.deepcopy(schema_graph)

        new_sg.tables["products"].columns.pop("product_id", None)
        new_sg.tables["products"].primary_key = []
        new_sg.tables["customers"].foreign_keys = []

        diff = diff_schemas(schema_graph, new_sg)

        class _FakeDialect:
            name = "test"

            def profile_schema(self, *_a, **_k):
                pass

            def refresh_full_table_distinct_for_pk_inference(self, *_a, **_k):
                return None

        merged = apply_diff(schema_graph, new_sg, diff, _FakeDialect(), notes_content=None)
        user_edges = [
            e for e in merged.tables["customers"].foreign_keys if (e.inference_tag or "").startswith("user_override_")
        ]
        assert user_edges == []

    def test_block_inferred_fk_removes_edge_and_persists(self, schema_graph, monkeypatch):
        from aetherdialect._contracts_base import FKEdge
        from aetherdialect._schema import apply_schema_overrides_to_graph

        monkeypatch.setattr("aetherdialect._schema.llm_credentials_configured", lambda: False)
        edge = FKEdge(
            src_table="orders",
            src_cols=["order_date"],
            dst_table="customers",
            dst_cols=["created_at"],
            inference_tag="suffix",
        )
        schema_graph.tables["orders"].foreign_keys.append(edge)

        before = len(schema_graph.tables["orders"].foreign_keys)
        report = apply_schema_overrides_to_graph(
            schema_graph,
            _ov_doc(
                foreign_keys_remove=[{"from": "orders.order_date", "to": "customers.created_at"}],
            ),
        )
        assert report.fks_removed == 1
        assert len(schema_graph.tables["orders"].foreign_keys) == before - 1

    def test_remove_inferred_fk_writes_denylist(self, schema_graph, monkeypatch):
        from aetherdialect._contracts_base import FKEdge
        from aetherdialect._schema import apply_schema_overrides_to_graph

        monkeypatch.setattr("aetherdialect._schema.llm_credentials_configured", lambda: False)
        edge = FKEdge(
            src_table="orders",
            src_cols=["order_date"],
            dst_table="customers",
            dst_cols=["created_at"],
            inference_tag="suffix",
        )
        schema_graph.tables["orders"].foreign_keys.append(edge)

        doc: dict[str, object] = _ov_doc(
            foreign_keys_remove=[{"from": "orders.order_date", "to": "customers.created_at"}],
        )
        report = apply_schema_overrides_to_graph(schema_graph, doc)
        assert report.fks_removed == 1

        assert len(doc["_internal"]["fk_block_inferred"]) == 1

    def test_block_inferred_pk_demotes_column(self, schema_graph, monkeypatch):
        from aetherdialect._schema import apply_schema_overrides_to_graph

        monkeypatch.setattr("aetherdialect._schema.llm_credentials_configured", lambda: False)

        col = schema_graph.tables["orders"].columns["amount"]
        col.pk_inference_tag = "profile"
        if "amount" not in schema_graph.tables["orders"].primary_key:
            schema_graph.tables["orders"].primary_key.append("amount")

        report = apply_schema_overrides_to_graph(
            schema_graph,
            _ov_doc(
                primary_keys_remove=[{"table": "orders", "column": "amount"}],
            ),
        )
        assert report.pks_blocked == 1
        assert schema_graph.tables["orders"].columns["amount"].is_primary_key is False
        assert "amount" not in schema_graph.tables["orders"].primary_key

    def test_block_catalog_pk_skipped(self, schema_graph, monkeypatch):
        from aetherdialect._schema import apply_schema_overrides_to_graph

        monkeypatch.setattr("aetherdialect._schema.llm_credentials_configured", lambda: False)

        report = apply_schema_overrides_to_graph(
            schema_graph,
            _ov_doc(
                primary_keys_remove=[{"table": "customers", "column": "customer_id"}],
            ),
        )
        assert report.pks_blocked == 0
        assert any("catalog PK" in s.reason for s in report.skipped)

    def test_primary_keys_add_user_promotes_unique_column(self, schema_graph, monkeypatch):
        from aetherdialect._contracts_base import PkInferenceTag
        from aetherdialect._schema import apply_schema_overrides_to_graph

        monkeypatch.setattr("aetherdialect._schema.llm_credentials_configured", lambda: False)

        col = schema_graph.tables["customers"].columns["email"]
        col.is_unique = True

        report = apply_schema_overrides_to_graph(
            schema_graph,
            _ov_doc(
                primary_keys_add=[{"table": "customers", "column": "email"}],
            ),
        )
        assert report.pks_added == 1
        assert col.is_primary_key is True
        assert col.pk_inference_tag == PkInferenceTag.USER_OVERRIDE
        assert "email" in schema_graph.tables["customers"].primary_key

    def test_primary_keys_add_rejects_non_unique_column(self, schema_graph, monkeypatch):
        from aetherdialect._schema import apply_schema_overrides_to_graph

        monkeypatch.setattr("aetherdialect._schema.llm_credentials_configured", lambda: False)

        report = apply_schema_overrides_to_graph(
            schema_graph,
            _ov_doc(
                primary_keys_add=[{"table": "orders", "column": "amount"}],
            ),
        )
        assert report.pks_added == 0
        assert any("not unique" in s.reason for s in report.skipped)

    def test_primary_keys_add_endorses_profile_inferred_pk(self, schema_graph, monkeypatch):
        from aetherdialect._contracts_base import PkInferenceTag
        from aetherdialect._schema import (
            apply_schema_overrides_to_graph,
            dump_schema_overrides_dict,
        )

        monkeypatch.setattr("aetherdialect._schema.llm_credentials_configured", lambda: False)

        col = schema_graph.tables["orders"].columns["amount"]
        col.pk_inference_tag = PkInferenceTag.PROFILE
        if "amount" not in schema_graph.tables["orders"].primary_key:
            schema_graph.tables["orders"].primary_key.append("amount")

        report = apply_schema_overrides_to_graph(
            schema_graph,
            _ov_doc(
                primary_keys_add=[{"table": "orders", "column": "amount"}],
            ),
        )
        assert report.pks_added == 0
        assert report.pks_endorsed == 1
        assert col.pk_inference_tag == PkInferenceTag.USER_OVERRIDE
        assert any("endorsement" in s.reason for s in report.skipped)
        doc = dump_schema_overrides_dict(schema_graph)
        assert {"table": "orders", "column": "amount"} in doc["primary_keys_add"]

    def test_primary_keys_add_endorses_catalog_pk(self, schema_graph, monkeypatch):
        from aetherdialect._contracts_base import PkInferenceTag
        from aetherdialect._schema import (
            apply_schema_overrides_to_graph,
            dump_schema_overrides_dict,
        )

        monkeypatch.setattr("aetherdialect._schema.llm_credentials_configured", lambda: False)

        col = schema_graph.tables["customers"].columns["customer_id"]
        assert col.pk_inference_tag is None

        report = apply_schema_overrides_to_graph(
            schema_graph,
            _ov_doc(
                primary_keys_add=[{"table": "customers", "column": "customer_id"}],
            ),
        )
        assert report.pks_added == 0
        assert report.pks_endorsed == 1
        assert col.pk_inference_tag == PkInferenceTag.USER_OVERRIDE
        assert any("endorsement" in s.reason for s in report.skipped)
        doc = dump_schema_overrides_dict(schema_graph)
        assert {"table": "customers", "column": "customer_id"} in doc["primary_keys_add"]

    def test_primary_keys_add_idempotent_when_already_user_override(self, schema_graph, monkeypatch):
        from aetherdialect._contracts_base import PkInferenceTag
        from aetherdialect._schema import apply_schema_overrides_to_graph

        monkeypatch.setattr("aetherdialect._schema.llm_credentials_configured", lambda: False)

        col = schema_graph.tables["customers"].columns["customer_id"]
        col.pk_inference_tag = PkInferenceTag.USER_OVERRIDE

        report = apply_schema_overrides_to_graph(
            schema_graph,
            _ov_doc(
                primary_keys_add=[{"table": "customers", "column": "customer_id"}],
            ),
        )
        assert report.pks_added == 0
        assert report.pks_endorsed == 0
        assert any(s.reason == "pk_already_user_override" for s in report.skipped)
        assert not any("endorsement" in s.reason for s in report.skipped)

    def test_primary_keys_remove_blocked_when_would_empty(self, schema_graph, monkeypatch):
        from aetherdialect._schema import apply_schema_overrides_to_graph

        monkeypatch.setattr("aetherdialect._schema.llm_credentials_configured", lambda: False)

        col = schema_graph.tables["orders"].columns["amount"]
        col.pk_inference_tag = "profile"
        schema_graph.tables["orders"].primary_key = ["amount"]

        report = apply_schema_overrides_to_graph(
            schema_graph,
            _ov_doc(
                primary_keys_remove=[{"table": "orders", "column": "amount"}],
            ),
        )
        assert report.pks_blocked == 0
        assert any("empty primary key" in s.reason for s in report.skipped)
        assert "amount" in schema_graph.tables["orders"].primary_key

    def test_sidecar_round_trip(self, schema_graph, tmp_path, monkeypatch):
        from aetherdialect._schema import (
            _overrides_sidecar_path,
            load_overrides_sidecar,
            save_overrides_sidecar,
        )

        cache_path = tmp_path / "schema.json.gz"
        cache_path.write_bytes(b"")
        doc = {
            **{
                "version": SCHEMA_OVERRIDES_VERSION,
                "tables": {
                    "orders": {
                        "description": _odesc("Orders."),
                        "role": _orole(None),
                        "columns": {},
                    }
                },
                "foreign_keys_add": [],
                "foreign_keys_remove": [],
                "primary_keys_add": [],
                "primary_keys_remove": [],
            },
            "_internal": {
                "fk_block_inferred": [{"from": "a.x", "to": "b.y"}],
                "pk_block_inferred": [{"table": "t", "column": "c"}],
            },
        }
        path = save_overrides_sidecar(
            cache_path,
            doc,
            source_schema_hash="abc123",
            metadata_hash="0" * 64,
        )
        assert path == _overrides_sidecar_path(cache_path)
        loaded = load_overrides_sidecar(cache_path)
        assert loaded is not None
        assert loaded["source_schema_hash"] == "abc123"
        assert loaded["metadata_hash"] == "0" * 64
        assert loaded["_internal"]["fk_block_inferred"] == [{"from": "a.x", "to": "b.y"}]

    def test_sidecar_corrupt_returns_none(self, tmp_path):
        from aetherdialect._schema import (
            _overrides_sidecar_path,
            load_overrides_sidecar,
        )

        cache_path = tmp_path / "schema.json.gz"
        cache_path.write_bytes(b"")
        sidecar = _overrides_sidecar_path(cache_path)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text("{not valid json")
        assert load_overrides_sidecar(cache_path) is None

    def test_clear_persisted_overrides(self, tmp_path):
        from aetherdialect._schema import (
            _overrides_sidecar_path,
            clear_persisted_overrides,
            save_overrides_sidecar,
        )

        cache_path = tmp_path / "schema.json.gz"
        cache_path.write_bytes(b"")
        save_overrides_sidecar(
            cache_path,
            _ov_doc(),
            source_schema_hash="x",
            metadata_hash="1" * 64,
        )
        assert _overrides_sidecar_path(cache_path).is_file()
        assert clear_persisted_overrides(cache_path) is True
        assert clear_persisted_overrides(cache_path) is False

    def test_load_validates_new_top_level_keys(self, tmp_path):
        import json as _json

        from aetherdialect._config import SCHEMA_OVERRIDES_VERSION
        from aetherdialect._schema import load_schema_overrides_file

        path = tmp_path / "ov.json"
        path.write_text(
            _json.dumps(
                {
                    "version": SCHEMA_OVERRIDES_VERSION,
                    "tables": {},
                    "foreign_keys_add": [],
                    "foreign_keys_remove": [{"from": "a.b", "to": "c.d"}],
                    "primary_keys_remove": [{"table": "t", "column": "c"}],
                    "_readonly": {"foreign_keys_current": []},
                }
            )
        )
        loaded = load_schema_overrides_file(path)
        assert loaded["foreign_keys_remove"][0]["from"] == "a.b"
        assert loaded["primary_keys_remove"][0]["table"] == "t"

    def test_inferred_pk_sets_provenance_tag(self):
        from aetherdialect._contracts_base import ColumnMetadata, TableMetadata
        from aetherdialect._schema import _infer_missing_pks_from_profile

        col = ColumnMetadata(
            name="id",
            data_type="integer",
            is_nullable=False,
            distinct_count=100,
            null_ratio=0.0,
            value_type="integer",
        )
        tbl = TableMetadata(
            name="t",
            columns={"id": col},
            row_count=100,
            primary_key=[],
            foreign_keys=[],
        )
        out = _infer_missing_pks_from_profile({"t": tbl})
        assert out == [("t", "id")]
        assert tbl.columns["id"].pk_inference_tag == "profile"

    def test_infer_pks_respects_blocked(self):
        from aetherdialect._contracts_base import ColumnMetadata, TableMetadata
        from aetherdialect._schema import _infer_missing_pks_from_profile

        col = ColumnMetadata(
            name="id",
            data_type="integer",
            is_nullable=False,
            distinct_count=100,
            null_ratio=0.0,
            value_type="integer",
        )
        tbl = TableMetadata(
            name="t",
            columns={"id": col},
            row_count=100,
            primary_key=[],
            foreign_keys=[],
        )
        out = _infer_missing_pks_from_profile({"t": tbl}, blocked=frozenset({("t", "id")}))
        assert out == []
        assert tbl.columns["id"].is_primary_key is False
        assert tbl.columns["id"].pk_inference_tag is None


class TestRelaxDvdrentalSelectability:
    """``_relax_dvdrental_selectability`` gates used by live dvdrental-shaped fixtures."""

    def test_restores_visibility_after_sentinel_profile(self):
        from live_tests.conftest import _relax_dvdrental_selectability

        col = ColumnMetadata(
            name="release_year",
            data_type="integer",
            value_type="integer",
            distinct_count=1,
            null_ratio=0.99,
            mode_frequency_ratio=0.95,
            row_count=500,
            role=ColumnRole.NUMERIC_MEASURE.value,
        )
        tbl = TableMetadata(
            name="film",
            columns={"release_year": col},
            primary_key=[],
            foreign_keys=[],
            row_count=500,
        )
        sg = SchemaGraph(tables={"film": tbl}, join_paths_multi={}, effective_structural_hash="h")
        assert col.is_visible is False
        _relax_dvdrental_selectability(sg, "dvdrental_new")
        assert col.distinct_count >= 2
        assert col.null_ratio == 0.0
        assert col.mode_frequency_ratio == 0.0
        assert col.is_visible is True
