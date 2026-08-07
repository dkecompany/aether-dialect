"""Tests for aetherdialect._qsim: structural skeleton helpers and value sampling."""

import os
from datetime import datetime
from types import SimpleNamespace

import pytest

from aetherdialect._config import QSimConfig
from aetherdialect._constants import (
    HAVING_COUNT_VALUES,
    HAVING_MIN_MAX_VALUES,
    HAVING_SUM_AVG_VALUES,
)
from aetherdialect._contracts_base import (
    ColumnRole,
)
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    FKEdge,
    QSimHaving,
    QSimIntent,
    QSimSkeleton,
    QSimWhereParam,
    SchemaGraph,
    TableMetadata,
    ValueDomain,
)
from aetherdialect._qsim import (
    _compute_intent_variance,
    _extract_date_part,
    _format_date,
    _identify_range_pairs,
    _instantiate_intent,
    _is_excluded_where_column,
    _is_integer_type,
    _parse_date,
    _sample_boolean,
    _sample_categorical,
    _sample_in_values,
    _sample_numeric,
    _sample_numeric_categorical,
    _sample_numeric_range,
    _sample_temporal,
    _sample_temporal_range,
    _skeleton_cache,
    build_fk_adjacency,
    build_schema_context,
    compute_intent_id,
    decompose_between_filter,
    deterministic_having_value,
    enumerate_table_sets,
    generate_all_skeletons,
    get_aggregatable_columns,
    get_comparable_column_pairs,
    get_filterable_columns,
    get_groupable_columns,
    instantiate_all,
    is_connected,
    load_or_create_skeletons,
    sample_coordinated_range,
    sample_value_from_domain,
    validate_column_exists,
)


@pytest.fixture
def three_table_schema():
    """Schema with customers, orders, products linked by FKs."""
    customers = TableMetadata(
        name="customers",
        columns={
            "id": ColumnMetadata(
                name="id",
                data_type="integer",
                value_type="integer",
                is_primary_key=True,
                role=ColumnRole.IDENTIFIER.value,
                distinct_count=100,
            ),
            "name": ColumnMetadata(
                name="name",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.CATEGORICAL.value,
                distinct_count=50,
            ),
        },
        foreign_keys=[],
        primary_key="",
    )
    orders = TableMetadata(
        name="orders",
        columns={
            "id": ColumnMetadata(
                name="id",
                data_type="integer",
                value_type="integer",
                is_primary_key=True,
                role=ColumnRole.IDENTIFIER.value,
                distinct_count=100,
            ),
            "customer_id": ColumnMetadata(
                name="customer_id",
                data_type="integer",
                value_type="integer",
                is_foreign_key=True,
                role=ColumnRole.IDENTIFIER.value,
                distinct_count=50,
            ),
            "product_id": ColumnMetadata(
                name="product_id",
                data_type="integer",
                value_type="integer",
                is_foreign_key=True,
                role=ColumnRole.IDENTIFIER.value,
                distinct_count=30,
            ),
            "amount": ColumnMetadata(
                name="amount",
                data_type="numeric",
                value_type="number",
                role=ColumnRole.NUMERIC_MEASURE.value,
                distinct_count=80,
            ),
            "status": ColumnMetadata(
                name="status",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.CATEGORICAL.value,
                distinct_count=5,
            ),
        },
        foreign_keys=[
            FKEdge(
                src_table="orders",
                src_cols=["customer_id"],
                dst_table="customers",
                dst_cols=["id"],
            ),
            FKEdge(
                src_table="orders",
                src_cols=["product_id"],
                dst_table="products",
                dst_cols=["id"],
            ),
        ],
        primary_key="",
    )
    products = TableMetadata(
        name="products",
        columns={
            "id": ColumnMetadata(
                name="id",
                data_type="integer",
                value_type="integer",
                is_primary_key=True,
                role=ColumnRole.IDENTIFIER.value,
                distinct_count=50,
            ),
            "price": ColumnMetadata(
                name="price",
                data_type="numeric",
                value_type="number",
                role=ColumnRole.NUMERIC_MEASURE.value,
                distinct_count=40,
            ),
            "category": ColumnMetadata(
                name="category",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.CATEGORICAL.value,
                distinct_count=10,
            ),
        },
        foreign_keys=[],
        primary_key="",
    )
    return SchemaGraph(
        join_paths_multi={},
        effective_structural_hash="",
        tables={"customers": customers, "orders": orders, "products": products},
    )


@pytest.fixture
def column_roles(three_table_schema):
    """Column roles dict matching the three_table_schema fixture."""
    roles = {}
    for tname, tmeta in three_table_schema.tables.items():
        for cname, cmeta in tmeta.columns.items():
            roles[f"{tname}.{cname}"] = cmeta.role
    return roles


class TestBuildFkAdjacency:
    """Tests for build_fk_adjacency."""

    def test_adjacency_map(self, three_table_schema):
        """Build adjacency map from FK edges."""
        adj = build_fk_adjacency(three_table_schema)
        assert "customers" in adj["orders"]
        assert "orders" in adj["customers"]
        assert "products" in adj["orders"]
        assert "orders" in adj["products"]

    def test_no_direct_link(self, three_table_schema):
        """Customers and products not directly linked."""
        adj = build_fk_adjacency(three_table_schema)
        assert "products" not in adj["customers"]
        assert "customers" not in adj["products"]


class TestIsConnected:
    """Tests for is_connected."""

    def test_single_table(self, three_table_schema):
        """Single table is always connected."""
        adj = build_fk_adjacency(three_table_schema)
        assert is_connected(["customers"], adj) is True

    def test_directly_linked(self, three_table_schema):
        """Directly linked tables are connected."""
        adj = build_fk_adjacency(three_table_schema)
        assert is_connected(["customers", "orders"], adj) is True

    def test_indirectly_linked(self, three_table_schema):
        """Tables linked through bridge table are connected."""
        adj = build_fk_adjacency(three_table_schema)
        assert is_connected(["customers", "orders", "products"], adj) is True

    def test_disconnected(self):
        """Disconnected tables return False."""
        adj = {"a": set(), "b": set()}
        assert is_connected(["a", "b"], adj) is False


class TestEnumerateTableSets:
    """Tests for enumerate_table_sets."""

    def test_includes_single_tables(self, three_table_schema):
        """All single tables included in results."""
        result = enumerate_table_sets(three_table_schema)
        assert ["customers"] in result
        assert ["orders"] in result
        assert ["products"] in result

    def test_includes_connected_pairs(self, three_table_schema):
        """Connected pairs included in results."""
        result = enumerate_table_sets(three_table_schema)
        pairs = [s for s in result if len(s) == 2]
        assert ["customers", "orders"] in pairs
        assert ["orders", "products"] in pairs

    def test_excludes_disconnected_pairs(self):
        """Disconnected pairs excluded from results."""
        t1 = TableMetadata(name="t1", columns={}, foreign_keys=[], primary_key="")
        t2 = TableMetadata(name="t2", columns={}, foreign_keys=[], primary_key="")
        schema = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={"t1": t1, "t2": t2},
        )
        result = enumerate_table_sets(schema)
        pairs = [s for s in result if len(s) == 2]
        assert len(pairs) == 0


class TestIsExcludedFilterColumn:
    """Tests for is_excluded_filter_column."""

    def test_normal_column(self):
        """Normal column not excluded."""
        assert _is_excluded_where_column("name") is False

    def test_audit_column(self):
        """Audit-pattern column excluded if pattern matches."""
        patterns = QSimConfig.EXCLUDED_WHERE_PATTERNS
        if patterns:
            for p in patterns:
                assert _is_excluded_where_column(p) is True
                break

    def test_empty_string_not_excluded(self):
        """Empty string matches no pattern."""
        assert _is_excluded_where_column("") is False

    def test_substring_match(self):
        """Column containing pattern substring is excluded."""
        assert _is_excluded_where_column("user_password") is True

    def test_case_insensitive(self):
        """Pattern match is case-insensitive."""
        assert _is_excluded_where_column("PASSWORD") is True


class TestGetFilterableColumns:
    """Tests for get_filterable_columns."""

    def test_returns_filterable(self, three_table_schema, column_roles):
        """Return filterable columns for table."""
        result = get_filterable_columns("orders", three_table_schema, column_roles)
        col_names = [c for c, _ in result]
        assert any("amount" in c for c in col_names)

    def test_empty_for_unknown_table(self, three_table_schema, column_roles):
        """Return empty for unknown table."""
        result = get_filterable_columns("unknown", three_table_schema, column_roles)
        assert result == []

    def test_uses_column_roles_when_provided(self, three_table_schema):
        """Uses column_roles map when col_key present, else col_meta.role."""
        column_roles = {"orders.status": "categorical"}
        result = get_filterable_columns("orders", three_table_schema, column_roles)
        col_keys = [c for c, _ in result]
        assert any("status" in c for c in col_keys)
        roles = {c: r for c, r in result}
        assert roles.get("orders.status") == "categorical"


class TestGetAggregatableColumns:
    """Tests for get_aggregatable_columns."""

    def test_numeric_measure_columns(self, three_table_schema, column_roles):
        """Return NUMERIC_MEASURE columns only."""
        result = get_aggregatable_columns("orders", three_table_schema, column_roles)
        assert "orders.amount" in result

    def test_identifier_not_aggregatable(self, three_table_schema, column_roles):
        """IDENTIFIER columns not in aggregatable list."""
        result = get_aggregatable_columns("orders", three_table_schema, column_roles)
        assert "orders.id" not in result


class TestGetGroupableColumns:
    """Tests for get_groupable_columns."""

    def test_categorical_columns(self, three_table_schema, column_roles):
        """Return CATEGORICAL columns."""
        result = get_groupable_columns("orders", three_table_schema, column_roles)
        assert "orders.status" in result

    def test_numeric_measure_not_groupable(self, three_table_schema, column_roles):
        """NUMERIC_MEASURE columns not groupable."""
        result = get_groupable_columns("orders", three_table_schema, column_roles)
        assert "orders.amount" not in result


class TestGetComparableColumnPairs:
    """Tests for get_comparable_column_pairs."""

    def test_numeric_pairs_across_tables(self, three_table_schema, column_roles):
        """Find numeric column pairs across different tables."""
        result = get_comparable_column_pairs(["orders", "products"], three_table_schema, column_roles)
        assert any(("orders" in (t1, t2) and "products" in (t1, t2)) for t1, _, t2, _, _ in result)

    def test_single_table_no_cross_pairs(self, three_table_schema, column_roles):
        """No cross-table pairs for single table."""
        result = get_comparable_column_pairs(["orders"], three_table_schema, column_roles)
        assert len(result) == 0


class TestComputeIntentId:
    """Tests for compute_intent_id."""

    def test_deterministic(self):
        """Same input produces same intent_id."""
        d = {"tables": ["t1"], "grain": "scalar", "select_cols": ["t1.a"]}
        id1 = compute_intent_id(d)
        id2 = compute_intent_id(d)
        assert id1 == id2

    def test_different_inputs(self):
        """Different inputs produce different intent_ids."""
        d1 = {"tables": ["t1"], "grain": "scalar", "select_cols": ["t1.a"]}
        d2 = {"tables": ["t2"], "grain": "scalar", "select_cols": ["t2.b"]}
        assert compute_intent_id(d1) != compute_intent_id(d2)

    def test_order_independent_tables(self):
        """Table order does not affect intent_id."""
        d1 = {"tables": ["t1", "t2"], "grain": "grouped", "select_cols": ["t1.a"]}
        d2 = {"tables": ["t2", "t1"], "grain": "grouped", "select_cols": ["t1.a"]}
        assert compute_intent_id(d1) == compute_intent_id(d2)


class TestDecomposeBetweenFilter:
    """Tests for decompose_between_filter."""

    def test_between_decomposed(self):
        """BETWEEN filter decomposes into >= and <=."""
        f = QSimWhereParam(column="t.a", op="between", value_type="integer")
        result = decompose_between_filter(f)
        assert len(result) == 2
        assert result[0].op == ">="
        assert result[1].op == "<="

    def test_non_between_unchanged(self):
        """Non-BETWEEN filter returns as-is."""
        f = QSimWhereParam(column="t.a", op="=", value_type="string")
        result = decompose_between_filter(f)
        assert len(result) == 1
        assert result[0].op == "="


class TestValidateColumnExists:
    """Tests for validate_column_exists."""

    def test_existing_column(self, three_table_schema):
        """Return True for existing column."""
        assert validate_column_exists("orders.amount", ["orders"], three_table_schema) is True

    def test_nonexistent_column(self, three_table_schema):
        """Return False for nonexistent column."""
        assert validate_column_exists("orders.nonexistent", ["orders"], three_table_schema) is False

    def test_table_not_in_list(self, three_table_schema):
        """Return False for table not in allowed list."""
        assert validate_column_exists("orders.amount", ["customers"], three_table_schema) is False

    def test_unqualified_column(self, three_table_schema):
        """Return False for unqualified column."""
        assert validate_column_exists("amount", ["orders"], three_table_schema) is False


class TestBuildSchemaContext:
    """Tests for build_schema_context."""

    def test_includes_table_name(self, three_table_schema):
        """Schema context includes table names."""
        ctx = build_schema_context(["orders"], three_table_schema)
        assert "orders" in ctx

    def test_includes_column_info(self, three_table_schema):
        """Schema context includes column names and types."""
        ctx = build_schema_context(["orders"], three_table_schema)
        assert "amount" in ctx
        assert "PK" in ctx

    def test_includes_fk_markers(self, three_table_schema):
        """Schema context includes FK markers."""
        ctx = build_schema_context(["orders"], three_table_schema)
        assert "FK" in ctx


class TestGenerateAllSkeletons:
    """Tests for generate_all_skeletons."""

    def test_returns_non_empty(self, three_table_schema, column_roles):
        """Generates at least one skeleton for valid table set."""
        _skeleton_cache.clear()
        result = generate_all_skeletons(["orders"], three_table_schema, column_roles)
        assert len(result) > 0

    def test_all_items_are_skeletons(self, three_table_schema, column_roles):
        """All returned items are QSimSkeleton instances."""
        _skeleton_cache.clear()
        result = generate_all_skeletons(["orders"], three_table_schema, column_roles)
        for s in result:
            assert isinstance(s, QSimSkeleton)

    def test_tables_preserved(self, three_table_schema, column_roles):
        """All skeletons preserve the input table list."""
        _skeleton_cache.clear()
        result = generate_all_skeletons(["orders", "products"], three_table_schema, column_roles)
        for s in result:
            assert s.tables == ["orders", "products"]

    def test_cache_hit(self, three_table_schema, column_roles):
        """Second call returns cached result."""
        _skeleton_cache.clear()
        first = generate_all_skeletons(["orders"], three_table_schema, column_roles)
        second = generate_all_skeletons(["orders"], three_table_schema, column_roles)
        assert first is second

    def test_both_agg_variants(self, three_table_schema, column_roles):
        """Generates skeletons with and without aggregation."""
        _skeleton_cache.clear()
        result = generate_all_skeletons(["orders"], three_table_schema, column_roles)
        has_agg_true = any(s.has_aggregation for s in result)
        has_agg_false = any(not s.has_aggregation for s in result)
        assert has_agg_true
        assert has_agg_false

    def test_non_agg_groupby_zero(self, three_table_schema, column_roles):
        """Non-aggregation skeletons always have num_groupby == 0."""
        _skeleton_cache.clear()
        result = generate_all_skeletons(["orders"], three_table_schema, column_roles)
        for s in result:
            if not s.has_aggregation:
                assert s.num_groupby == 0

    def test_having_requires_agg_and_groupby(self, three_table_schema, column_roles):
        """``num_having > 0`` only when aggregated and grouped."""
        _skeleton_cache.clear()
        result = generate_all_skeletons(["orders"], three_table_schema, column_roles)
        for s in result:
            if s.num_having > 0:
                assert s.has_aggregation
                assert s.num_groupby > 0

    def test_distinct_only_single_non_agg(self, three_table_schema, column_roles):
        """has_distinct only True for single-table non-aggregation skeletons."""
        _skeleton_cache.clear()
        result = generate_all_skeletons(["orders"], three_table_schema, column_roles)
        for s in result:
            if s.has_distinct:
                assert not s.has_aggregation
                assert len(s.tables) == 1

    def test_multi_table_no_distinct(self, three_table_schema, column_roles):
        """Multi-table skeletons never have has_distinct True."""
        _skeleton_cache.clear()
        result = generate_all_skeletons(["orders", "products"], three_table_schema, column_roles)
        for s in result:
            assert not s.has_distinct


class TestLoadOrCreateSkeletons:
    """Tests for load_or_create_skeletons."""

    def test_creates_cache_file(self, three_table_schema, column_roles, tmp_path, monkeypatch):
        """Creates cache gzip JSON file when none exists."""
        _skeleton_cache.clear()
        cache_file = str(tmp_path / "skeletons.json.gz")
        monkeypatch.setattr("aetherdialect._qsim.QSimConfig.SKELETONS_JSON_PATH", cache_file)
        result = load_or_create_skeletons(three_table_schema, column_roles)
        assert os.path.exists(cache_file)
        assert len(result) > 0

    def test_loads_from_existing_cache(self, three_table_schema, column_roles, tmp_path, monkeypatch):
        """Loads from existing cache file with matching schema hash."""
        _skeleton_cache.clear()
        cache_file = str(tmp_path / "skeletons.json.gz")
        monkeypatch.setattr("aetherdialect._qsim.QSimConfig.SKELETONS_JSON_PATH", cache_file)
        load_or_create_skeletons(three_table_schema, column_roles)
        first_count = len(_skeleton_cache)
        _skeleton_cache.clear()
        result = load_or_create_skeletons(three_table_schema, column_roles)
        assert len(result) == first_count

    def test_hash_mismatch_regenerates(self, three_table_schema, column_roles, tmp_path, monkeypatch):
        """Schema hash mismatch triggers regeneration."""
        from aetherdialect._core_utils import read_gzip_json, write_gzip_json_atomic

        _skeleton_cache.clear()
        cache_file = str(tmp_path / "skeletons.json.gz")
        monkeypatch.setattr("aetherdialect._qsim.QSimConfig.SKELETONS_JSON_PATH", cache_file)
        load_or_create_skeletons(three_table_schema, column_roles)
        data = read_gzip_json(cache_file)
        data["structural_hash"] = "wrong_hash"
        write_gzip_json_atomic(cache_file, data, sort_keys=True)
        _skeleton_cache.clear()
        result = load_or_create_skeletons(three_table_schema, column_roles)
        assert len(result) > 0

    def test_corrupt_cache_file(self, three_table_schema, column_roles, tmp_path, monkeypatch):
        """Corrupt cache file triggers regeneration instead of crash."""
        _skeleton_cache.clear()
        cache_file = str(tmp_path / "skeletons.json.gz")
        with open(cache_file, "wb") as f:
            f.write(b"NOT GZIP JSON")
        monkeypatch.setattr("aetherdialect._qsim.QSimConfig.SKELETONS_JSON_PATH", cache_file)
        result = load_or_create_skeletons(three_table_schema, column_roles)
        assert len(result) > 0

    def test_cache_file_contains_valid_json(self, three_table_schema, column_roles, tmp_path, monkeypatch):
        """Written cache file contains valid JSON with expected keys."""
        from aetherdialect._core_utils import read_gzip_json

        _skeleton_cache.clear()
        cache_file = str(tmp_path / "skeletons.json.gz")
        monkeypatch.setattr("aetherdialect._qsim.QSimConfig.SKELETONS_JSON_PATH", cache_file)
        load_or_create_skeletons(three_table_schema, column_roles)
        data = read_gzip_json(cache_file)
        assert "structural_hash" in data
        assert "skeletons" in data
        assert "num_table_sets" in data


class TestEnumerateTableSetsEdgeCases:
    """Edge case tests for enumerate_table_sets."""

    def test_single_table_schema(self):
        """Single-table schema returns only one set."""
        t = TableMetadata(name="t", columns={}, foreign_keys=[], primary_key="")
        schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"t": t})
        result = enumerate_table_sets(schema)
        assert result == [["t"]]

    def test_empty_schema(self):
        """Empty schema returns empty list."""
        schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={})
        result = enumerate_table_sets(schema)
        assert result == []

    def test_max_tables_one(self):
        """max_tables=1 returns only single-table sets."""
        t1 = TableMetadata(name="t1", columns={}, foreign_keys=[], primary_key="")
        t2 = TableMetadata(name="t2", columns={}, foreign_keys=[], primary_key="")
        schema = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={"t1": t1, "t2": t2},
        )
        result = enumerate_table_sets(schema, max_tables=1)
        assert all(len(s) == 1 for s in result)


class TestIsConnectedEdgeCases:
    """Edge case tests for is_connected."""

    def test_empty_table_list(self):
        """Empty table list returns True."""
        adj = {"a": {"b"}, "b": {"a"}}
        assert is_connected([], adj) is True

    def test_table_not_in_adjacency(self):
        """Table missing from adjacency map is unreachable."""
        adj = {"a": set()}
        assert is_connected(["a", "b"], adj) is False


class TestBuildFkAdjacencyEdgeCases:
    """Edge case tests for build_fk_adjacency."""

    def test_empty_schema(self):
        """Empty schema produces empty adjacency."""
        empty = SchemaGraph(tables={}, join_paths_multi={}, effective_structural_hash="")
        assert build_fk_adjacency(empty) == {}

    def test_no_fk_edges(self):
        """Schema with no FK edges returns empty adjacency sets."""
        t = TableMetadata(name="t", columns={}, foreign_keys=[], primary_key="")
        schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"t": t})
        adj = build_fk_adjacency(schema)
        assert adj["t"] == set()

    def test_self_referential_fk(self):
        """Self-referential FK creates self-loop in adjacency."""
        t = TableMetadata(
            name="t",
            columns={},
            foreign_keys=[
                FKEdge(
                    src_table="t",
                    src_cols=["parent_id"],
                    dst_table="t",
                    dst_cols=["id"],
                )
            ],
            primary_key="",
        )
        schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"t": t})
        adj = build_fk_adjacency(schema)
        assert "t" in adj["t"]


class TestComputeIntentIdEdgeCases:
    """Edge case tests for compute_intent_id."""

    def test_empty_dict(self):
        """Empty dict produces a valid intent ID."""
        result = compute_intent_id({})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_with_filters_and_having(self):
        """Intent ID accounts for filters and having params."""
        d1 = {
            "tables": ["t1"],
            "grain": "scalar",
            "select_cols": ["t1.a"],
            "where": [{"column": "t1.x", "op": "="}],
        }
        d2 = {
            "tables": ["t1"],
            "grain": "scalar",
            "select_cols": ["t1.a"],
            "where": [],
        }
        assert compute_intent_id(d1) != compute_intent_id(d2)

    def test_filter_order_independent(self):
        """Filter order does not affect intent ID."""
        f1 = {"column": "t1.a", "op": "="}
        f2 = {"column": "t1.b", "op": ">"}
        d1 = {"tables": ["t1"], "where": [f1, f2]}
        d2 = {"tables": ["t1"], "where": [f2, f1]}
        assert compute_intent_id(d1) == compute_intent_id(d2)


class TestGetAggregatableColumnsEdgeCases:
    """Edge case tests for get_aggregatable_columns."""

    def test_unknown_table(self, three_table_schema, column_roles):
        """Unknown table returns empty list."""
        result = get_aggregatable_columns("nonexistent", three_table_schema, column_roles)
        assert result == []

    def test_table_without_numeric(self, column_roles):
        """Table with no NUMERIC_MEASURE columns returns empty."""
        t = TableMetadata(
            name="t",
            columns={
                "name": ColumnMetadata(
                    name="name",
                    data_type="varchar",
                    value_type="string",
                    role=ColumnRole.CATEGORICAL.value,
                    distinct_count=10,
                )
            },
            foreign_keys=[],
            primary_key="",
        )
        schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"t": t})
        result = get_aggregatable_columns("t", schema, column_roles)
        assert result == []


class TestGetGroupableColumnsEdgeCases:
    """Edge case tests for get_groupable_columns."""

    def test_unknown_table(self, three_table_schema, column_roles):
        """Unknown table returns empty list."""
        result = get_groupable_columns("nonexistent", three_table_schema, column_roles)
        assert result == []

    def test_temporal_column_groupable(self, column_roles):
        """TEMPORAL columns are groupable."""
        t = TableMetadata(
            name="t",
            columns={
                "created_at": ColumnMetadata(
                    name="created_at",
                    data_type="timestamp",
                    value_type="datetime",
                    role=ColumnRole.TEMPORAL.value,
                    distinct_count=100,
                )
            },
            foreign_keys=[],
            primary_key="",
        )
        schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"t": t})
        result = get_groupable_columns("t", schema, column_roles)
        assert "t.created_at" in result


class TestGetComparableColumnPairsEdgeCases:
    """Edge case tests for get_comparable_column_pairs."""

    def test_empty_table_set(self, three_table_schema, column_roles):
        """Empty table list returns no pairs."""
        result = get_comparable_column_pairs([], three_table_schema, column_roles)
        assert result == []

    def test_same_role_required(self, column_roles):
        """Columns must share same role to be comparable."""
        t1 = TableMetadata(
            name="t1",
            columns={
                "x": ColumnMetadata(
                    name="x",
                    data_type="numeric",
                    value_type="number",
                    role=ColumnRole.NUMERIC_MEASURE.value,
                    distinct_count=10,
                )
            },
            foreign_keys=[],
            primary_key="",
        )
        t2 = TableMetadata(
            name="t2",
            columns={
                "y": ColumnMetadata(
                    name="y",
                    data_type="numeric",
                    value_type="number",
                    role=ColumnRole.NUMERIC_CATEGORICAL.value,
                    distinct_count=10,
                )
            },
            foreign_keys=[],
            primary_key="",
        )
        schema = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={"t1": t1, "t2": t2},
        )
        roles = {
            "t1.x": ColumnRole.NUMERIC_MEASURE.value,
            "t2.y": ColumnRole.NUMERIC_CATEGORICAL.value,
        }
        result = get_comparable_column_pairs(["t1", "t2"], schema, roles)
        assert result == []


class TestBuildSchemaContextEdgeCases:
    """Edge case tests for build_schema_context."""

    def test_unknown_table_skipped(self, three_table_schema):
        """Unknown table name produces no output."""
        ctx = build_schema_context(["nonexistent"], three_table_schema)
        assert ctx == ""

    def test_empty_table_list(self, three_table_schema):
        """Empty table list produces empty context."""
        ctx = build_schema_context([], three_table_schema)
        assert ctx == ""

    def test_multiple_tables_separated(self, three_table_schema):
        """Multiple tables separated by blank lines."""
        ctx = build_schema_context(["customers", "orders"], three_table_schema)
        assert "customers" in ctx
        assert "orders" in ctx
        assert "\n\n" in ctx


class TestValidateColumnExistsEdgeCases:
    """Edge case tests for validate_column_exists."""

    def test_empty_col_ref(self, three_table_schema):
        """Empty string returns False."""
        assert validate_column_exists("", ["orders"], three_table_schema) is False

    def test_multiple_dots(self, three_table_schema):
        """Column with multiple dots splits on first dot only."""
        assert validate_column_exists("orders.a.b", ["orders"], three_table_schema) is False

    def test_empty_tables_list(self, three_table_schema):
        """Empty tables list returns False for any column."""
        assert validate_column_exists("orders.amount", [], three_table_schema) is False


class TestDecomposeBetweenFilterEdgeCases:
    """Edge case tests for decompose_between_filter."""

    def test_preserves_column(self):
        """Decomposed filters preserve original column."""
        f = QSimWhereParam(column="t.date_col", op="between", value_type="date")
        result = decompose_between_filter(f)
        assert all(r.column == "t.date_col" for r in result)

    def test_preserves_value_type(self):
        """Decomposed filters preserve original value_type."""
        f = QSimWhereParam(column="t.x", op="between", value_type="integer")
        result = decompose_between_filter(f)
        assert all(r.value_type == "integer" for r in result)


class TestSampleValueFromDomain:
    """Tests for sample_value_from_domain."""

    def test_null_type_returns_none(self):
        """Null value type returns None."""
        domain = ValueDomain(values=["a", "b"], min_val=None, max_val=None, data_type="varchar")
        assert sample_value_from_domain(domain, "null") is None

    def test_is_null_op_returns_none(self):
        """IS NULL operator returns None regardless of value_type."""
        domain = ValueDomain(values=["a", "b"], min_val=None, max_val=None, data_type="varchar")
        assert sample_value_from_domain(domain, "categorical", op="is null") is None

    def test_is_not_null_op_returns_none(self):
        """IS NOT NULL operator returns None."""
        domain = ValueDomain(values=["a", "b"], min_val=None, max_val=None, data_type="varchar")
        assert sample_value_from_domain(domain, "categorical", op="is not null") is None

    def test_categorical_cycles(self):
        """Categorical sampling stays in-domain and repeats with period ``len(values)``."""
        domain = ValueDomain(values=["PG", "R", "G"], min_val=None, max_val=None, data_type="varchar")
        n = len(domain.values)
        for k in range(n):
            v = sample_value_from_domain(domain, "categorical", variant_idx=k)
            assert v in domain.values
        v0 = sample_value_from_domain(domain, "categorical", variant_idx=0)
        vn = sample_value_from_domain(domain, "categorical", variant_idx=n)
        assert vn == v0

    def test_numeric_range(self):
        """Numeric sampling produces value within domain range."""
        domain = ValueDomain(values=[], min_val="10", max_val="100", data_type="integer")
        val = sample_value_from_domain(domain, "numeric", op="=", variant_idx=0)
        assert val is not None
        assert 10 <= int(val) <= 100

    def test_boolean_cycles(self):
        """Boolean sampling alternates true/false."""
        domain = ValueDomain(values=[], min_val=None, max_val=None, data_type="boolean")
        v0 = sample_value_from_domain(domain, "boolean", variant_idx=0)
        v1 = sample_value_from_domain(domain, "boolean", variant_idx=1)
        assert v0 in ("true", "false")
        assert v1 in ("true", "false")
        assert v0 != v1

    def test_in_operator_categorical(self):
        """IN operator returns comma-separated quoted values."""
        domain = ValueDomain(
            values=["a", "b", "c", "d", "e"],
            min_val=None,
            max_val=None,
            data_type="varchar",
        )
        val = sample_value_from_domain(domain, "categorical", op="in", variant_idx=0)
        assert val is not None
        assert "'" in val

    def test_temporal_returns_date(self):
        """Temporal sampling returns date string."""
        domain = ValueDomain(values=[], min_val="2020-01-01", max_val="2023-12-31", data_type="date")
        val = sample_value_from_domain(domain, "temporal", op="=", variant_idx=0)
        assert val is not None
        assert "-" in val

    def test_numeric_categorical_from_values(self):
        """Numeric categorical sampling picks from values list."""
        domain = ValueDomain(values=[1, 2, 3, 4, 5], min_val=None, max_val=None, data_type="integer")
        val = sample_value_from_domain(domain, "numeric_categorical", variant_idx=0)
        assert val is not None
        assert val in ("1", "2", "3", "4", "5")


class TestIdentifyRangePairs:
    """Tests for identify_range_pairs."""

    def test_paired_range(self):
        """Paired >= and <= on same column detected."""
        filters = [
            QSimWhereParam(column="orders.amount", op=">=", value_type="numeric"),
            QSimWhereParam(column="orders.amount", op="<=", value_type="numeric"),
        ]
        pairs = _identify_range_pairs(filters)
        assert "orders.amount" in pairs
        assert pairs["orders.amount"]["lower_idx"] == 0
        assert pairs["orders.amount"]["upper_idx"] == 1

    def test_no_pair(self):
        """Single-sided range is not a pair."""
        filters = [
            QSimWhereParam(column="orders.amount", op=">=", value_type="numeric"),
        ]
        pairs = _identify_range_pairs(filters)
        assert pairs == {}

    def test_expr_comparison_skipped(self):
        """Expr comparisons are skipped."""
        filters = [
            QSimWhereParam(
                column="orders.amount",
                op=">=",
                value_type="numeric",
                right_column="products.price",
            ),
        ]
        pairs = _identify_range_pairs(filters)
        assert pairs == {}

    def test_different_columns(self):
        """Different columns with matching ops don't pair."""
        filters = [
            QSimWhereParam(column="orders.amount", op=">=", value_type="numeric"),
            QSimWhereParam(column="orders.quantity", op="<=", value_type="numeric"),
        ]
        pairs = _identify_range_pairs(filters)
        assert pairs == {}

    def test_mixed_operators(self):
        """Mixed > and < on same column detected."""
        filters = [
            QSimWhereParam(column="rental.date", op=">", value_type="temporal"),
            QSimWhereParam(column="rental.date", op="<", value_type="temporal"),
        ]
        pairs = _identify_range_pairs(filters)
        assert "rental.date" in pairs


class TestSampleCoordinatedRange:
    """Tests for sample_coordinated_range."""

    def test_numeric_range(self):
        """Numeric range returns lower < upper."""
        domain = ValueDomain(values=[], min_val="0", max_val="100", data_type="integer")
        lower, upper = sample_coordinated_range(domain, "numeric", variant_idx=0)
        assert lower is not None
        assert upper is not None
        assert float(lower) < float(upper)

    def test_temporal_range(self):
        """Temporal range returns lower < upper date."""
        domain = ValueDomain(values=[], min_val="2020-01-01", max_val="2023-12-31", data_type="date")
        lower, upper = sample_coordinated_range(domain, "temporal", variant_idx=0)
        assert lower is not None
        assert upper is not None
        assert lower < upper

    def test_categorical_unsupported(self):
        """Categorical value type returns (None, None)."""
        domain = ValueDomain(values=["a", "b"], min_val=None, max_val=None, data_type="varchar")
        lower, upper = sample_coordinated_range(domain, "categorical", variant_idx=0)
        assert lower is None
        assert upper is None

    def test_variant_variation(self):
        """Different variant_idx yields different range values."""
        domain = ValueDomain(values=[], min_val="0", max_val="1000", data_type="numeric")
        r0 = sample_coordinated_range(domain, "numeric", variant_idx=0)
        r1 = sample_coordinated_range(domain, "numeric", variant_idx=1)
        assert r0 != r1


class TestDeterministicHavingValue:
    """Tests for deterministic_having_value."""

    def test_count_from_pool(self):
        """Count values come from HAVING_COUNT_VALUES pool."""
        val = deterministic_having_value("count", variant_idx=0)
        assert int(val) in HAVING_COUNT_VALUES

    def test_sum_from_pool(self):
        """Sum values come from HAVING_SUM_AVG_VALUES pool."""
        val = deterministic_having_value("sum", variant_idx=0)
        assert float(val) in HAVING_SUM_AVG_VALUES

    def test_avg_from_pool(self):
        """Avg values come from HAVING_SUM_AVG_VALUES pool."""
        val = deterministic_having_value("avg", variant_idx=0)
        assert float(val) in HAVING_SUM_AVG_VALUES

    def test_min_from_pool(self):
        """Min values come from HAVING_MIN_MAX_VALUES pool."""
        val = deterministic_having_value("min", variant_idx=0)
        assert float(val) in HAVING_MIN_MAX_VALUES

    def test_max_from_pool(self):
        """Max values come from HAVING_MIN_MAX_VALUES pool."""
        val = deterministic_having_value("max", variant_idx=0)
        assert float(val) in HAVING_MIN_MAX_VALUES

    def test_unknown_agg_fallback(self):
        """Unknown aggregation falls back to COUNT pool."""
        val = deterministic_having_value("median", variant_idx=0)
        assert int(val) in HAVING_COUNT_VALUES

    def test_deterministic(self):
        """Same inputs always produce same output."""
        v1 = deterministic_having_value("count", variant_idx=5, having_idx=2)
        v2 = deterministic_having_value("count", variant_idx=5, having_idx=2)
        assert v1 == v2

    def test_variant_varies(self):
        """Different variant_idx produces different values (for sufficient pool)."""
        vals = {deterministic_having_value("count", variant_idx=i) for i in range(10)}
        assert len(vals) > 1


class TestComputeIntentVariance:
    """Tests for compute_intent_variance."""

    def test_no_filters_no_having(self):
        """Intent with no filters and no having has zero variance."""
        intent = QSimIntent(
            intent_id="test",
            tables=["orders"],
            grain="row_level",
            select_cols=["orders.order_id"],
            group_by_cols=[],
            order_by_cols=[],
            where=[],
            having_param=[],
            param_values={},
            question="",
            variant_idx=0,
        )
        score = _compute_intent_variance(intent, {})
        assert score == 0

    def test_filter_with_values(self):
        """Filter on column with values adds len(values) to variance."""
        intent = QSimIntent(
            intent_id="test",
            tables=["orders"],
            grain="row_level",
            select_cols=["orders.order_id"],
            group_by_cols=[],
            order_by_cols=[],
            where=[QSimWhereParam(column="orders.status", op="=", value_type="categorical")],
            having_param=[],
            param_values={},
            question="",
            variant_idx=0,
        )
        domains = {
            "orders.status": ValueDomain(
                values=["pending", "shipped", "delivered"],
                min_val=None,
                max_val=None,
                data_type="varchar",
            )
        }
        score = _compute_intent_variance(intent, domains)
        assert score >= 3

    def test_having_adds_variance(self):
        """Having param adds variance score."""
        intent = QSimIntent(
            intent_id="test",
            tables=["orders"],
            grain="grouped",
            select_cols=["COUNT(orders.order_id)"],
            group_by_cols=["orders.status"],
            order_by_cols=[],
            where=[QSimWhereParam(column="orders.status", op="=", value_type="categorical")],
            having_param=[QSimHaving(expression="COUNT(orders.order_id)", op=">", value_type="number")],
            param_values={},
            question="",
            variant_idx=0,
        )
        domains = {
            "orders.status": ValueDomain(values=["a", "b"], min_val=None, max_val=None, data_type="varchar"),
        }
        score_with_having = _compute_intent_variance(intent, domains)
        intent_no_having = QSimIntent(
            intent_id="test",
            tables=["orders"],
            grain="grouped",
            select_cols=["COUNT(orders.order_id)"],
            group_by_cols=["orders.status"],
            order_by_cols=[],
            where=[QSimWhereParam(column="orders.status", op="=", value_type="categorical")],
            having_param=[],
            param_values={},
            question="",
            variant_idx=0,
        )
        score_no_having = _compute_intent_variance(intent_no_having, domains)
        assert score_with_having > score_no_having

    def test_expr_comparison_skipped(self):
        """Expr comparison filters do not contribute to variance."""
        intent = QSimIntent(
            intent_id="test",
            tables=["orders"],
            grain="row_level",
            select_cols=["orders.order_id"],
            group_by_cols=[],
            order_by_cols=[],
            where=[
                QSimWhereParam(
                    column="orders.amount",
                    op=">",
                    value_type="numeric",
                    right_column="products.price",
                )
            ],
            having_param=[],
            param_values={},
            question="",
            variant_idx=0,
        )
        score = _compute_intent_variance(intent, {})
        assert score == 0


class TestIsIntegerType:
    """Tests for _is_integer_type helper."""

    def test_integer_types(self):
        """Standard integer type names recognised."""
        for dt in ("integer", "int", "bigint", "smallint", "tinyint", "long", "short"):
            assert _is_integer_type(dt) is True

    def test_mixed_case(self):
        """Case-insensitive matching."""
        assert _is_integer_type("INTEGER") is True
        assert _is_integer_type("BigInt") is True

    def test_non_integer_types(self):
        """Non-integer types rejected."""
        assert _is_integer_type("numeric") is False
        assert _is_integer_type("float") is False
        assert _is_integer_type("varchar") is False

    def test_interval_excluded(self):
        """Interval types not treated as integer despite containing 'int'."""
        assert _is_integer_type("interval") is False

    def test_int_substring_included(self):
        """Type containing 'int' but not 'interval' is integer."""
        assert _is_integer_type("myinteger") is True

    def test_none_and_empty(self):
        """None and empty string return False."""
        assert _is_integer_type(None) is False
        assert _is_integer_type("") is False


class TestParseDate:
    """Tests for _parse_date helper."""

    def test_iso_format(self):
        """Standard YYYY-MM-DD parsed."""
        result = _parse_date("2024-01-15")
        assert result == datetime(2024, 1, 15)

    def test_datetime_with_time(self):
        """Datetime string with T separator truncated to date."""
        result = _parse_date("2024-06-01T14:30:00")
        assert result == datetime(2024, 6, 1)

    def test_datetime_with_space(self):
        """Datetime string with space separator truncated to date."""
        result = _parse_date("2024-06-01 14:30:00")
        assert result == datetime(2024, 6, 1)

    def test_slash_format(self):
        """Slash-separated date parsed."""
        result = _parse_date("2024/03/20")
        assert result == datetime(2024, 3, 20)

    def test_dd_mm_yyyy_format(self):
        """Dd-mm-yyyy format parsed."""
        result = _parse_date("15-01-2024")
        assert result == datetime(2024, 1, 15)

    def test_dd_slash_mm_slash_yyyy_format(self):
        """Dd/mm/yyyy format parsed."""
        result = _parse_date("15/01/2024")
        assert result == datetime(2024, 1, 15)

    def test_unparseable_returns_none(self):
        """Unrecognised format returns None."""
        assert _parse_date("not-a-date") is None


class TestFormatDate:
    """Tests for _format_date helper."""

    def test_basic_format(self):
        """Datetime formatted as YYYY-MM-DD."""
        assert _format_date(datetime(2024, 1, 15)) == "2024-01-15"

    def test_padding(self):
        """Single-digit months/days zero-padded."""
        assert _format_date(datetime(2024, 3, 5)) == "2024-03-05"


class TestExtractDatePart:
    """Tests for _extract_date_part helper."""

    def test_with_t_separator(self):
        """Date extracted before T."""
        assert _extract_date_part("2024-01-15T10:00:00") == "2024-01-15"

    def test_with_space_separator(self):
        """Date extracted before space."""
        assert _extract_date_part("2024-01-15 10:00:00") == "2024-01-15"

    def test_date_only(self):
        """Plain date returned unchanged."""
        assert _extract_date_part("2024-01-15") == "2024-01-15"


class TestSampleCategorical:
    """Tests for _sample_categorical helper."""

    def test_rotates_through_values(self):
        """Values list is indexed deterministically with period ``len(values)``."""
        domain = ValueDomain(values=["red", "green", "blue"])
        n = len(domain.values)
        for k in range(n):
            assert _sample_categorical(domain, k) in domain.values
        assert _sample_categorical(domain, 0) == _sample_categorical(domain, n)

    def test_variant_indexes_cover_all_values_each_period(self):
        """Across ``variant_idx`` 0..n-1, each categorical value appears once per period (diversity)."""
        domain = ValueDomain(values=["red", "green", "blue"])
        n = len(domain.values)
        assert {_sample_categorical(domain, i) for i in range(n)} == set(domain.values)

    def test_min_max_fallback(self):
        """When no values, uses min/max range."""
        domain = ValueDomain(min_val="1", max_val="5")
        result = _sample_categorical(domain, 2)
        assert result == "3"

    def test_empty_returns_none(self):
        """No values and no range returns None."""
        domain = ValueDomain()
        assert _sample_categorical(domain, 0) is None


class TestSampleBoolean:
    """Tests for _sample_boolean helper."""

    def test_with_values(self):
        """Boolean values normalised and cycled."""
        domain = ValueDomain(values=[True, False])
        assert _sample_boolean(domain, 0) == "true"
        assert _sample_boolean(domain, 1) == "false"

    def test_default_without_values(self):
        """Defaults to ['true', 'false'] when no domain values."""
        domain = ValueDomain()
        assert _sample_boolean(domain, 0) == "true"
        assert _sample_boolean(domain, 1) == "false"


class TestSampleNumericCategorical:
    """Tests for _sample_numeric_categorical helper."""

    def test_with_values(self):
        """Discrete values cycled as integers."""
        domain = ValueDomain(values=[10, 20, 30])
        assert _sample_numeric_categorical(domain, 0) == "10"
        assert _sample_numeric_categorical(domain, 1) == "20"

    def test_min_max_fallback(self):
        """Range-based sampling when no values."""
        domain = ValueDomain(min_val="1", max_val="5")
        result = _sample_numeric_categorical(domain, 0)
        assert result == "1"

    def test_empty_returns_none(self):
        """No values and no range returns None."""
        domain = ValueDomain()
        assert _sample_numeric_categorical(domain, 0) is None


class TestSampleNumeric:
    """Tests for _sample_numeric helper."""

    def test_equals_integer(self):
        """Equal op on integer type produces integer string."""
        domain = ValueDomain(min_val="0", max_val="100", data_type="integer")
        result = _sample_numeric(domain, "=", 0)
        assert result == "0"

    def test_greater_than(self):
        """Greater-than op samples from lower portion."""
        domain = ValueDomain(min_val="0", max_val="100", data_type="integer")
        result = _sample_numeric(domain, ">", 0)
        assert result is not None
        assert int(result) >= 0

    def test_less_than(self):
        """Less-than op samples from upper portion."""
        domain = ValueDomain(min_val="0", max_val="100", data_type="integer")
        result = _sample_numeric(domain, "<", 0)
        assert result is not None
        assert int(result) <= 100

    def test_fallback_to_values(self):
        """Falls back to values list when no min/max."""
        domain = ValueDomain(values=["42", "99"])
        result = _sample_numeric(domain, "=", 0)
        assert result == "42"

    def test_no_domain_returns_none(self):
        """No values and no range returns None."""
        domain = ValueDomain()
        assert _sample_numeric(domain, "=", 0) is None


class TestSampleTemporal:
    """Tests for _sample_temporal helper."""

    def test_date_interpolation(self):
        """Samples date between min and max."""
        domain = ValueDomain(min_val="2020-01-01", max_val="2020-12-31")
        result = _sample_temporal(domain, "=", 0)
        assert result == "2020-01-01"

    def test_greater_than_segment(self):
        """Greater-than op uses lower segment range."""
        domain = ValueDomain(min_val="2020-01-01", max_val="2020-12-31")
        result = _sample_temporal(domain, ">=", 0)
        assert result is not None
        assert result >= "2020-01-01"

    def test_fallback_to_values(self):
        """Falls back to values list."""
        domain = ValueDomain(values=["2024-06-15T10:00:00"])
        result = _sample_temporal(domain, "=", 0)
        assert result == "2024-06-15"

    def test_unparseable_min(self):
        """Unparseable date falls back to extract_date_part."""
        domain = ValueDomain(min_val="not-a-date", max_val="also-bad")
        result = _sample_temporal(domain, "=", 0)
        assert result == "not-a-date"


class TestSampleInValues:
    """Tests for _sample_in_values helper."""

    def test_categorical_in(self):
        """Categorical IN produces quoted CSV."""
        domain = ValueDomain(values=["a", "b", "c", "d", "e"])
        result = _sample_in_values(domain, "categorical", 0)
        assert result.startswith("'")
        assert "," in result

    def test_numeric_categorical_in(self):
        """Numeric categorical IN produces comma-separated integers."""
        domain = ValueDomain(values=[1, 2, 3, 4, 5])
        result = _sample_in_values(domain, "numeric_categorical", 0)
        assert "," in result
        parts = result.split(",")
        for p in parts:
            int(p)

    def test_boolean_in(self):
        """Boolean IN returns 'true,false'."""
        domain = ValueDomain()
        result = _sample_in_values(domain, "boolean", 0)
        assert result == "true,false"

    def test_numeric_in_with_range(self):
        """Numeric IN with min/max range."""
        domain = ValueDomain(min_val="0", max_val="100", data_type="integer")
        result = _sample_in_values(domain, "numeric", 0)
        assert result is not None
        assert "," in result

    def test_unsupported_type_returns_none(self):
        """Unknown value_type returns None."""
        domain = ValueDomain()
        result = _sample_in_values(domain, "unknown_type", 0)
        assert result is None


class TestSampleNumericRange:
    """Tests for _sample_numeric_range helper."""

    def test_integer_range(self):
        """Integer range produces two ordered integer strings."""
        domain = ValueDomain(min_val="0", max_val="100", data_type="integer")
        lower, upper = _sample_numeric_range(domain, 0)
        assert lower is not None and upper is not None
        assert int(lower) < int(upper)

    def test_float_range(self):
        """Float range produces two ordered float strings."""
        domain = ValueDomain(min_val="0.0", max_val="100.0", data_type="numeric")
        lower, upper = _sample_numeric_range(domain, 0)
        assert lower is not None and upper is not None
        assert float(lower) < float(upper)

    def test_no_min_max_returns_nones(self):
        """Missing min/max returns (None, None)."""
        domain = ValueDomain()
        assert _sample_numeric_range(domain, 0) == (None, None)

    def test_zero_range_returns_nones(self):
        """Zero-width range returns (None, None)."""
        domain = ValueDomain(min_val="50", max_val="50", data_type="integer")
        assert _sample_numeric_range(domain, 0) == (None, None)


class TestSampleTemporalRange:
    """Tests for _sample_temporal_range helper."""

    def test_date_range(self):
        """Date range produces two ordered date strings."""
        domain = ValueDomain(min_val="2020-01-01", max_val="2020-12-31")
        lower, upper = _sample_temporal_range(domain, 0)
        assert lower is not None and upper is not None
        assert lower < upper

    def test_no_min_max_returns_nones(self):
        """Missing min/max returns (None, None)."""
        domain = ValueDomain()
        assert _sample_temporal_range(domain, 0) == (None, None)

    def test_unparseable_fallback(self):
        """Unparseable dates fall back to extract_date_part."""
        domain = ValueDomain(min_val="bad-date", max_val="also-bad")
        lower, upper = _sample_temporal_range(domain, 0)
        assert lower == "bad-date"
        assert upper == "also-bad"


class TestInstantiateIntent:
    """Tests for instantiate_intent."""

    def _make_intent(self, **overrides):
        defaults = dict(
            intent_id="test-1",
            tables=["orders"],
            grain="row_level",
            select_cols=["orders.order_id"],
            group_by_cols=[],
            order_by_cols=[],
            where=[],
            having_param=[],
            param_values={},
            question="",
            variant_idx=0,
        )
        defaults.update(overrides)
        return QSimIntent(**defaults)

    def test_simple_filter_populated(self):
        """Single categorical filter gets value in param_values."""
        intent = self._make_intent(
            where=[QSimWhereParam(column="orders.status", op="=", value_type="categorical")],
        )
        domains = {"orders.status": ValueDomain(values=["active", "closed"])}
        result = _instantiate_intent(intent, domains, 0)
        assert result is not None
        assert "f0" in result.param_values

    def test_missing_value_domain_skips_entire_variant(self):
        """When a non-null filter column has no domain entry, instantiation returns None."""
        intent = self._make_intent(
            where=[QSimWhereParam(column="orders.missing_col", op="=", value_type="categorical")],
        )
        assert _instantiate_intent(intent, {}, 0) is None

    def test_null_filter_skipped(self):
        """IS NULL filter does not generate param value."""
        intent = self._make_intent(
            where=[QSimWhereParam(column="orders.status", op="is null", value_type="null")],
        )
        result = _instantiate_intent(intent, {}, 0)
        assert result is not None
        assert len(result.param_values) == 0

    def test_expr_comparison_passthrough(self):
        """Expr comparison filters pass through without param value."""
        intent = self._make_intent(
            where=[
                QSimWhereParam(
                    column="orders.amount",
                    op=">",
                    value_type="numeric",
                    right_column="orders.min_amount",
                ),
            ],
        )
        result = _instantiate_intent(intent, {}, 0)
        assert result is not None
        assert len(result.param_values) == 0

    def test_having_populated(self):
        """Having param gets deterministic value."""
        intent = self._make_intent(
            having_param=[QSimHaving(expression="COUNT(orders.order_id)", op=">", value_type="numeric")],
        )
        result = _instantiate_intent(intent, {}, 0)
        assert result is not None
        assert "h0" in result.param_values

    def test_variant_idx_propagated(self):
        """Variant index carried forward to result."""
        intent = self._make_intent()
        result = _instantiate_intent(intent, {}, 5)
        assert result is not None
        assert result.variant_idx == 5


class TestInstantiateAll:
    """Tests for instantiate_all."""

    def _make_col(self, **overrides):
        defaults = dict(
            data_type="varchar",
            value_type="string",
            frequent_values=["a", "b", "c"],
            min_val=None,
            max_val=None,
            distinct_count=3,
            role="dimension",
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _make_schema(self):
        col = self._make_col()
        table = SimpleNamespace(columns={"status": col})
        return SimpleNamespace(tables={"orders": table})

    def test_produces_instances(self):
        """Generates at least one instantiated intent."""
        schema = self._make_schema()
        intent = QSimIntent(
            intent_id="i1",
            tables=["orders"],
            grain="row_level",
            select_cols=["orders.status"],
            group_by_cols=[],
            order_by_cols=[],
            where=[QSimWhereParam(column="orders.status", op="=", value_type="categorical")],
            having_param=[],
            param_values={},
            question="",
            variant_idx=0,
        )
        results = instantiate_all([intent], schema, num_questions=3)
        assert len(results) >= 1
        assert all(isinstance(r, QSimIntent) for r in results)

    def test_empty_intents(self):
        """Empty intent list returns empty."""
        schema = self._make_schema()
        results = instantiate_all([], schema, num_questions=5)
        assert results == []

    def test_truncation_when_over_budget(self):
        """Output truncated to num_questions when allocation exceeds budget."""
        schema = self._make_schema()
        intents = []
        for i in range(20):
            intents.append(
                QSimIntent(
                    intent_id=f"i{i}",
                    tables=["orders"],
                    grain="row_level",
                    select_cols=["orders.status"],
                    group_by_cols=[],
                    order_by_cols=[],
                    where=[QSimWhereParam(column="orders.status", op="=", value_type="categorical")],
                    having_param=[],
                    param_values={},
                    question="",
                    variant_idx=0,
                )
            )
        results = instantiate_all(intents, schema, num_questions=5)
        assert len(results) <= 5
