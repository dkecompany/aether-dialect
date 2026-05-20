"""Tests for intent_repair module."""

import pytest

from aetherdialect._contracts_base import (
    ColumnMetadata,
    ColumnRole,
    FKEdge,
    SchemaGraph,
    SqlDiagnostic,
    SqlDiagnosticCode,
    TableMetadata,
)
from aetherdialect._contracts_core import (
    CaseRegistryStep,
    CaseWhenBranch,
    CaseWhenExpr,
    FilterParam,
    HavingParam,
    MulGroup,
    NormalizedExpr,
    OrderByCol,
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
)
from aetherdialect._intent_expr import parse_expr_string
from aetherdialect._intent_repair import (
    DIAGNOSTIC_REPAIR_DISPATCH,
    _dedup_contradictory_filters_list,
    _dedup_value_vs_right_expr_filters,
    _is_impossible_having,
    _is_null_value,
    _is_pk_column,
    _match_enum_value,
    _qualify_term,
    _resolve_filter_list_cascade,
    _strip_distinct_prefix,
    _tables_from_columns,
    align_filter_value_type_to_exprs,
    apply_diagnostic_repairs,
    auto_repair_filter_having,
    best_descriptive_column,
    best_descriptive_columns,
    collect_referenced_tables,
    decompose_in_not_in_filters,
    dedup_contradictory_filters,
    dedup_value_vs_right_expr,
    drop_invalid_case_registry_entries,
    enforce_sensitivity_policy_intent,
    expand_fk_select_to_descriptive,
    infer_cte_output_columns,
    intent_text_has_leakable_placeholder,
    normalize_boolean_filter_values,
    normalize_in_filter_types,
    normalize_null_filter_values,
    normalize_pk_distinct,
    qualify_cte_output_columns,
    reconcile_tables,
    repair_array_filters_intent,
    repair_case_when_intent,
    repair_fk_filter_type_mismatch,
    repair_intent_placeholder_tokens,
    repair_null_equality_filters,
    resolve_filter_value_case,
    runtime_intent_has_instructional_placeholders,
    sanitize_table_names,
    strip_impossible_having,
    strip_join_conditions,
    strip_spurious_group_by,
)


class TestDropInvalidCaseRegistryEntries:
    """Tests for drop_invalid_case_registry_entries."""

    def test_removes_empty_case_registry_and_dangling_c_ref(self):
        schema = SchemaGraph(join_paths_multi={}, effective_structural_hash="h", tables={})
        cr_ok = CaseRegistryStep(
            registry_id="c02",
            case_when=CaseWhenExpr(branches=[CaseWhenBranch()]),
        )
        cr_bad = CaseRegistryStep(registry_id="c01", case_when=CaseWhenExpr(branches=[]))
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr(column_ref="c01")),
                SelectCol(expr=NormalizedExpr.from_column("orders.id")),
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            case_registry=[cr_bad, cr_ok],
        )
        out = drop_invalid_case_registry_entries(intent, schema)
        assert [c.registry_id for c in out.case_registry] == ["c02"]
        assert len(out.select_cols) == 1
        assert out.select_cols[0].expr.column_ref == "orders.id"


class TestEnforceSensitivityPolicyIntent:
    """Tests for enforce_sensitivity_policy_intent."""

    def test_keeps_aggregated_count_on_non_selectable_pk(self):
        """COUNT on a PII PK remains in select_cols."""
        schema = SchemaGraph(
            tables={
                "customer": TableMetadata(
                    name="customer",
                    columns={
                        "customer_id": ColumnMetadata(
                            name="customer_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                            sensitivity="pii",
                        ),
                    },
                    primary_key=["customer_id"],
                    foreign_keys=[],
                ),
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )
        agg_col = SelectCol(expr=NormalizedExpr.from_agg("count", "customer.customer_id"))
        intent = RuntimeIntent(
            tables=["customer"],
            grain="grouped",
            select_cols=[agg_col],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        out = enforce_sensitivity_policy_intent(intent, schema)
        assert len(out.select_cols or []) == 1

    def test_drops_bare_non_selectable_projection(self):
        """When every main select column resolves to a hidden-sensitivity field, the policy raises SENSITIVITY_ALL_SELECT_DROPPED."""
        schema = SchemaGraph(
            tables={
                "customer": TableMetadata(
                    name="customer",
                    columns={
                        "customer_id": ColumnMetadata(
                            name="customer_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                            sensitivity="pii",
                        ),
                    },
                    primary_key=["customer_id"],
                    foreign_keys=[],
                ),
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )
        bare = SelectCol(expr=NormalizedExpr.from_column("customer.customer_id"))
        intent = RuntimeIntent(
            tables=["customer"],
            grain="row_level",
            select_cols=[bare],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        with pytest.raises(ValueError, match="sensitivity_all_select_dropped"):
            enforce_sensitivity_policy_intent(intent, schema)

    def test_keeps_sum_on_non_selectable_column(self):
        schema = SchemaGraph(
            tables={
                "customer": TableMetadata(
                    name="customer",
                    columns={
                        "email": ColumnMetadata(
                            name="email",
                            data_type="varchar",
                            value_type="string",
                            sensitivity="pii",
                        ),
                    },
                    primary_key=[],
                    foreign_keys=[],
                ),
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )
        agg_col = SelectCol(expr=NormalizedExpr.from_agg("sum", "customer.email"))
        intent = RuntimeIntent(
            tables=["customer"],
            grain="grouped",
            select_cols=[agg_col],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        out = enforce_sensitivity_policy_intent(intent, schema)
        assert len(out.select_cols or []) == 1

    def test_keeps_count_distinct_on_non_selectable(self):
        schema = SchemaGraph(
            tables={
                "customer": TableMetadata(
                    name="customer",
                    columns={
                        "email": ColumnMetadata(
                            name="email",
                            data_type="varchar",
                            value_type="string",
                            sensitivity="pii",
                        ),
                    },
                    primary_key=[],
                    foreign_keys=[],
                ),
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )
        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["DISTINCT customer.email"], agg_func="count")],
        )
        agg_col = SelectCol(expr=expr)
        intent = RuntimeIntent(
            tables=["customer"],
            grain="grouped",
            select_cols=[agg_col],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        out = enforce_sensitivity_policy_intent(intent, schema)
        assert len(out.select_cols or []) == 1


class TestStripDistinctPrefix:
    """Tests for _strip_distinct_prefix."""

    def test_strips_distinct(self):
        """_strip_distinct_prefix removes DISTINCT prefix."""
        assert _strip_distinct_prefix("DISTINCT t.id") == "t.id"

    def test_leaves_non_distinct(self):
        """_strip_distinct_prefix leaves non-DISTINCT unchanged."""
        assert _strip_distinct_prefix("t.id") == "t.id"

    def test_case_insensitive(self):
        """_strip_distinct_prefix is case-insensitive."""
        assert _strip_distinct_prefix("distinct t.id") == "t.id"


class TestNormalizePkDistinct:
    """Tests for normalize_pk_distinct."""

    @pytest.fixture
    def pk_schema(self):
        """Schema with a PK column for DISTINCT tests."""
        customers = TableMetadata(
            name="customers",
            columns={
                "id": ColumnMetadata(
                    name="id",
                    data_type="integer",
                    value_type="integer",
                    is_primary_key=True,
                    role=ColumnRole.IDENTIFIER.value,
                ),
            },
            foreign_keys=[],
            primary_key="",
        )
        return SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={"customers": customers},
        )

    def test_strips_distinct_from_count_pk(self, pk_schema):
        """normalize_pk_distinct strips DISTINCT from COUNT on PK."""
        sc = SelectCol(
            expr=NormalizedExpr(
                add_groups=[
                    MulGroup(
                        coefficient=1.0,
                        distinct=True,
                        multiply=["customers.id"],
                        agg_func="count",
                    )
                ],
            ),
        )
        intent = RuntimeIntent(
            tables=["customers"],
            grain="scalar",
            select_cols=[sc],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        result = normalize_pk_distinct(intent, pk_schema)
        assert result.select_cols[0].expr.add_groups[0].distinct is False

    def test_no_strip_non_pk(self, pk_schema):
        """normalize_pk_distinct does not strip DISTINCT for non-PK."""
        non_pk_schema = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={
                "customers": TableMetadata(
                    name="customers",
                    columns={
                        "name": ColumnMetadata(
                            name="name",
                            data_type="varchar",
                            value_type="string",
                            role=ColumnRole.CATEGORICAL.value,
                        )
                    },
                    foreign_keys=[],
                    primary_key="",
                )
            },
        )
        sc = SelectCol(
            expr=NormalizedExpr(
                add_groups=[
                    MulGroup(
                        coefficient=1.0,
                        distinct=True,
                        multiply=["customers.name"],
                        agg_func="count",
                    )
                ],
            ),
        )
        intent = RuntimeIntent(
            tables=["customers"],
            grain="scalar",
            select_cols=[sc],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        result = normalize_pk_distinct(intent, non_pk_schema)
        assert result.select_cols[0].expr.add_groups[0].distinct is True


class TestPruneUnreferencedTables:
    """Tests for reconcile_tables."""

    def test_adds_missing_referenced_table(self):
        """reconcile_tables adds table referenced in columns."""
        intent = RuntimeIntent(
            tables=[],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        result = reconcile_tables(intent)
        assert "orders" in result.tables

    def test_removes_unreferenced_table(self):
        """reconcile_tables removes table not used in expressions."""
        intent = RuntimeIntent(
            tables=["customers", "orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("customers.name"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        result = reconcile_tables(intent)
        assert "customers" in result.tables
        assert "orders" not in result.tables


class TestStripJoinConditionsFromIntent:
    """Tests for strip_join_conditions."""

    @pytest.fixture
    def fk_schema(self):
        """Schema with FK relationship for join condition tests."""
        orders = TableMetadata(
            name="orders",
            columns={
                "id": ColumnMetadata(
                    name="id",
                    data_type="integer",
                    value_type="integer",
                    is_primary_key=True,
                    role=ColumnRole.IDENTIFIER.value,
                ),
                "customer_id": ColumnMetadata(
                    name="customer_id",
                    data_type="integer",
                    value_type="integer",
                    is_foreign_key=True,
                    role=ColumnRole.IDENTIFIER.value,
                ),
            },
            foreign_keys=[
                FKEdge(
                    src_table="orders",
                    src_cols=["customer_id"],
                    dst_table="customers",
                    dst_cols=["id"],
                )
            ],
            primary_key="",
        )
        customers = TableMetadata(
            name="customers",
            columns={
                "id": ColumnMetadata(
                    name="id",
                    data_type="integer",
                    value_type="integer",
                    is_primary_key=True,
                    role=ColumnRole.IDENTIFIER.value,
                ),
            },
            foreign_keys=[],
            primary_key="",
        )
        return SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={"orders": orders, "customers": customers},
        )

    def test_strips_fk_join_filter(self, fk_schema):
        """strip_join_conditions removes FK equi-join filter."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("orders.customer_id"),
            op="=",
            right_expr=NormalizedExpr.from_column("customers.id"),
            value_type="integer",
        )
        intent = RuntimeIntent(
            tables=["orders", "customers"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
        )
        result = strip_join_conditions(intent, fk_schema)
        assert len(result.filters_param) == 0

    def test_keeps_non_fk_filter(self, fk_schema):
        """strip_join_conditions keeps non-FK filter."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("orders.id"),
            op=">",
            value_type="number",
            param_key="p1",
        )
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
        )
        result = strip_join_conditions(intent, fk_schema)
        assert len(result.filters_param) == 1


class TestIntentTextHasLeakablePlaceholder:
    """Regression tests for instructional placeholder scans."""

    def test_none_text_returns_false(self):
        """Absent scan strings must not reach the angle-bracket regex."""
        assert intent_text_has_leakable_placeholder(None) is False

    def test_runtime_scan_tolerates_none_column_map_value(self):
        """Malformed column_map values must not raise during placeholder scans."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.a"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            column_map={"alias": None},
        )
        assert runtime_intent_has_instructional_placeholders(intent) is False


class TestBestDescriptiveColumn:
    """Tests for best_descriptive_column."""

    @pytest.fixture
    def desc_schema(self):
        """Schema with PK, FK, and descriptive columns."""
        return SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={
                "customers": TableMetadata(
                    name="customers",
                    columns={
                        "id": ColumnMetadata(
                            name="id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                            role=ColumnRole.IDENTIFIER.value,
                        ),
                        "email": ColumnMetadata(
                            name="email",
                            data_type="varchar",
                            value_type="string",
                            role=ColumnRole.IDENTIFIER.value,
                            distinct_count=100,
                            distinct_ratio=0.99,
                        ),
                        "name": ColumnMetadata(
                            name="name",
                            data_type="varchar",
                            value_type="string",
                            role=ColumnRole.CATEGORICAL.value,
                            distinct_count=50,
                            distinct_ratio=0.5,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="",
                )
            },
        )

    def test_returns_best_non_pk_column(self, desc_schema):
        """best_descriptive_column returns column with highest distinct_count meeting threshold."""
        result = best_descriptive_column("customers", desc_schema, set())
        assert result == "email"

    def test_excludes_specified_columns(self, desc_schema):
        """best_descriptive_column skips columns in exclude set."""
        result = best_descriptive_column("customers", desc_schema, {"customers.email"})
        assert result is None

    def test_unknown_table_returns_none(self, desc_schema):
        """best_descriptive_column returns None for unknown table."""
        result = best_descriptive_column("nonexistent", desc_schema, set())
        assert result is None

    def test_excludes_free_text_role_even_with_high_cardinality(self):
        """Long-text ``free_text`` columns must not win descriptive auto-picks."""
        schema = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={
                "film": TableMetadata(
                    name="film",
                    columns={
                        "film_id": ColumnMetadata(
                            name="film_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                            role=ColumnRole.IDENTIFIER.value,
                        ),
                        "description": ColumnMetadata(
                            name="description",
                            data_type="text",
                            value_type="string",
                            role=ColumnRole.FREE_TEXT.value,
                            distinct_count=900,
                            distinct_ratio=0.99,
                        ),
                        "title": ColumnMetadata(
                            name="title",
                            data_type="varchar",
                            value_type="string",
                            role=ColumnRole.CATEGORICAL.value,
                            distinct_count=900,
                            distinct_ratio=0.99,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="",
                )
            },
        )
        assert best_descriptive_column("film", schema, set()) == "title"

    def test_all_pk_returns_none(self):
        """best_descriptive_column returns None when only PK columns exist."""
        schema = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={
                "t": TableMetadata(
                    name="t",
                    columns={
                        "id": ColumnMetadata(
                            name="id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                            role=ColumnRole.IDENTIFIER.value,
                        )
                    },
                    foreign_keys=[],
                    primary_key="",
                )
            },
        )
        assert best_descriptive_column("t", schema, set()) is None


class TestIsPkColumn:
    """Tests for _is_pk_column."""

    @pytest.fixture
    def pk_schema(self):
        """Schema with a PK column."""
        return SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={
                "t": TableMetadata(
                    name="t",
                    columns={
                        "id": ColumnMetadata(
                            name="id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                            role=ColumnRole.IDENTIFIER.value,
                        )
                    },
                    foreign_keys=[],
                    primary_key="",
                )
            },
        )

    def test_pk_column_returns_true(self, pk_schema):
        """_is_pk_column returns True for PK column."""
        assert _is_pk_column("t.id", pk_schema) is True

    def test_non_pk_returns_false(self):
        """_is_pk_column returns False for non-PK column."""
        schema = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={
                "t": TableMetadata(
                    name="t",
                    columns={
                        "name": ColumnMetadata(
                            name="name",
                            data_type="varchar",
                            value_type="string",
                            role=ColumnRole.CATEGORICAL.value,
                        )
                    },
                    foreign_keys=[],
                    primary_key="",
                )
            },
        )
        assert _is_pk_column("t.name", schema) is False

    def test_bare_column_returns_false(self, pk_schema):
        """_is_pk_column returns False for unqualified column."""
        assert _is_pk_column("id", pk_schema) is False

    def test_unknown_table_returns_false(self, pk_schema):
        """_is_pk_column returns False for unknown table."""
        assert _is_pk_column("x.id", pk_schema) is False


class TestTablesFromColumns:
    """Tests for _tables_from_columns."""

    def test_extracts_table_names(self):
        """_tables_from_columns extracts unique table names."""
        result = _tables_from_columns(["orders.id", "customers.name", "orders.amount"])
        assert result == {"orders", "customers"}

    def test_skips_bare_columns(self):
        """_tables_from_columns skips bare column names."""
        result = _tables_from_columns(["id", "name"])
        assert result == set()

    def test_empty_input(self):
        """_tables_from_columns returns empty set for empty input."""
        assert _tables_from_columns([]) == set()

    def test_single_qualified_column(self):
        """_tables_from_columns with one table.col returns one table."""
        assert _tables_from_columns(["orders.amount"]) == {"orders"}

    def test_deduplicates_same_table(self):
        """_tables_from_columns deduplicates table names."""
        result = _tables_from_columns(["t.a", "t.b", "t.c"])
        assert result == {"t"}


class TestCollectReferencedTables:
    """Tests for collect_referenced_tables."""

    def test_collects_from_all_clauses(self):
        """collect_referenced_tables gathers tables from all clause types."""
        sc = [SelectCol(expr=NormalizedExpr.from_column("orders.id"))]
        obc = [OrderByCol(expr=NormalizedExpr.from_column("customers.name"))]
        gb = [NormalizedExpr.from_column("products.category")]
        fp = [FilterParam(left_expr=NormalizedExpr.from_column("inventory.qty"), op=">")]
        hp = [HavingParam(left_expr=NormalizedExpr.from_agg("sum", "sales.amount"), op=">")]
        result = collect_referenced_tables(sc, obc, gb, fp, hp)
        assert "orders" in result
        assert "customers" in result
        assert "products" in result
        assert "inventory" in result
        assert "sales" in result

    def test_empty_clauses(self):
        """collect_referenced_tables returns empty for empty clauses."""
        result = collect_referenced_tables([], [], [], [], [])
        assert result == set()


class TestStripJoinConditionsEdgeCases:
    """Edge-case tests for strip_join_conditions."""

    def test_strips_from_cte_steps(self):
        """strip_join_conditions also strips from CTE steps."""
        orders = TableMetadata(
            name="orders",
            columns={
                "customer_id": ColumnMetadata(
                    name="customer_id",
                    data_type="integer",
                    value_type="integer",
                    is_foreign_key=True,
                    role=ColumnRole.IDENTIFIER.value,
                )
            },
            foreign_keys=[
                FKEdge(
                    src_table="orders",
                    src_cols=["customer_id"],
                    dst_table="customers",
                    dst_cols=["id"],
                )
            ],
            primary_key="",
        )
        customers = TableMetadata(
            name="customers",
            columns={
                "id": ColumnMetadata(
                    name="id",
                    data_type="integer",
                    value_type="integer",
                    is_primary_key=True,
                    role=ColumnRole.IDENTIFIER.value,
                )
            },
            foreign_keys=[],
            primary_key="",
        )
        schema = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={"orders": orders, "customers": customers},
        )
        fk_filter = FilterParam(
            left_expr=NormalizedExpr.from_column("orders.customer_id"),
            op="=",
            right_expr=NormalizedExpr.from_column("customers.id"),
            value_type="integer",
        )
        cte = RuntimeCteStep(cte_name="cte1", filters_param=[fk_filter])
        intent = RuntimeIntent(
            tables=["orders", "customers"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        result = strip_join_conditions(intent, schema)
        assert len(result.cte_steps[0].filters_param) == 0

    def test_keeps_non_equality_join(self):
        """strip_join_conditions keeps non-equality operator."""
        orders = TableMetadata(
            name="orders",
            columns={
                "customer_id": ColumnMetadata(
                    name="customer_id",
                    data_type="integer",
                    value_type="integer",
                    is_foreign_key=True,
                    role=ColumnRole.IDENTIFIER.value,
                )
            },
            foreign_keys=[
                FKEdge(
                    src_table="orders",
                    src_cols=["customer_id"],
                    dst_table="customers",
                    dst_cols=["id"],
                )
            ],
            primary_key="",
        )
        customers = TableMetadata(
            name="customers",
            columns={
                "id": ColumnMetadata(
                    name="id",
                    data_type="integer",
                    value_type="integer",
                    is_primary_key=True,
                    role=ColumnRole.IDENTIFIER.value,
                )
            },
            foreign_keys=[],
            primary_key="",
        )
        schema = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={"orders": orders, "customers": customers},
        )
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("orders.customer_id"),
            op=">",
            right_expr=NormalizedExpr.from_column("customers.id"),
            value_type="integer",
        )
        intent = RuntimeIntent(
            tables=["orders", "customers"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
        )
        result = strip_join_conditions(intent, schema)
        assert len(result.filters_param) == 1


class TestPruneUnreferencedTablesEdgeCases:
    """Edge-case tests for reconcile_tables."""

    def test_preserves_cte_names_in_tables(self):
        """reconcile_tables keeps CTE names in tables list."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("t.x"))],
        )
        intent = RuntimeIntent(
            tables=["cte1"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("cte1.val"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        result = reconcile_tables(intent)
        assert "cte1" in result.tables

    def test_cte_tables_include_prior_cte_names_only(self):
        """Per-CTE tables list adds prior CTE names, not all CTE names."""
        cte1 = RuntimeCteStep(
            cte_name="cte1",
            tables=["rental"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("rental.rental_id"))],
        )
        cte2 = RuntimeCteStep(
            cte_name="cte2",
            tables=["inventory"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("cte1.x"))],
        )
        intent = RuntimeIntent(
            tables=["rental"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("cte2.y"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte1, cte2],
        )
        result = reconcile_tables(intent)
        assert result.cte_steps[0].tables == ["rental"]
        assert "cte1" in result.cte_steps[1].tables
        assert "cte2" not in result.cte_steps[0].tables

    def test_empty_intent(self):
        """reconcile_tables handles intent with no columns or tables."""
        intent = RuntimeIntent(
            tables=[],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        result = reconcile_tables(intent)
        assert result.tables == []


class TestNormalizePkDistinctEdgeCases:
    """Edge-case tests for normalize_pk_distinct."""

    def test_no_strip_non_count_agg(self):
        """normalize_pk_distinct does not strip DISTINCT for non-count aggregation."""
        schema = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={
                "t": TableMetadata(
                    name="t",
                    columns={
                        "id": ColumnMetadata(
                            name="id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                            role=ColumnRole.IDENTIFIER.value,
                        )
                    },
                    foreign_keys=[],
                    primary_key="",
                )
            },
        )
        sc = SelectCol(
            expr=NormalizedExpr(
                add_groups=[
                    MulGroup(
                        coefficient=1.0,
                        distinct=True,
                        multiply=["t.id"],
                        agg_func="sum",
                    )
                ]
            )
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="scalar",
            select_cols=[sc],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        result = normalize_pk_distinct(intent, schema)
        assert result.select_cols[0].expr.add_groups[0].distinct is True

    def test_strips_in_cte_steps(self):
        """normalize_pk_distinct strips DISTINCT from CTE step select cols."""
        schema = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={
                "t": TableMetadata(
                    name="t",
                    columns={
                        "id": ColumnMetadata(
                            name="id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                            role=ColumnRole.IDENTIFIER.value,
                        )
                    },
                    foreign_keys=[],
                    primary_key="",
                )
            },
        )
        sc = SelectCol(
            expr=NormalizedExpr(
                add_groups=[
                    MulGroup(
                        coefficient=1.0,
                        distinct=True,
                        multiply=["t.id"],
                        agg_func="count",
                    )
                ]
            )
        )
        cte = RuntimeCteStep(cte_name="cte1", select_cols=[sc])
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        result = normalize_pk_distinct(intent, schema)
        assert result.cte_steps[0].select_cols[0].expr.add_groups[0].distinct is False


class TestRepairFkFilterTypeMismatch:
    @pytest.fixture
    def fk_schema(self):
        orders = TableMetadata(
            name="orders",
            columns={
                "order_id": ColumnMetadata(
                    name="order_id",
                    data_type="integer",
                    value_type="integer",
                    is_primary_key=True,
                    role=ColumnRole.IDENTIFIER.value,
                    distinct_count=500,
                ),
                "category_id": ColumnMetadata(
                    name="category_id",
                    data_type="integer",
                    value_type="integer",
                    is_foreign_key=True,
                    fk_target=("categories", "category_id"),
                    role=ColumnRole.IDENTIFIER.value,
                    distinct_count=10,
                ),
                "amount": ColumnMetadata(
                    name="amount",
                    data_type="numeric",
                    value_type="number",
                    role=ColumnRole.NUMERIC_MEASURE.value,
                    distinct_count=200,
                ),
            },
            primary_key=["order_id"],
            foreign_keys=[
                FKEdge(
                    src_table="orders",
                    src_cols=["category_id"],
                    dst_table="categories",
                    dst_cols=["category_id"],
                ),
            ],
        )
        categories = TableMetadata(
            name="categories",
            columns={
                "category_id": ColumnMetadata(
                    name="category_id",
                    data_type="integer",
                    value_type="integer",
                    is_primary_key=True,
                    role=ColumnRole.IDENTIFIER.value,
                    distinct_count=10,
                    distinct_ratio=1.0,
                ),
                "name": ColumnMetadata(
                    name="name",
                    data_type="varchar",
                    value_type="string",
                    role=ColumnRole.CATEGORICAL.value,
                    distinct_count=10,
                    distinct_ratio=1.0,
                ),
            },
            primary_key=["category_id"],
            foreign_keys=[],
        )
        return SchemaGraph(
            tables={"orders": orders, "categories": categories},
            join_paths_multi={},
            effective_structural_hash="h",
        )

    def test_rewires_string_filter_on_fk_int(self, fk_schema):
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("orders.category_id"),
                    op="=",
                    value_type="string",
                    raw_value="Action",
                ),
            ],
        )
        result = repair_fk_filter_type_mismatch(intent, fk_schema)
        assert result.filters_param[0].left_expr.primary_column == "orders.category_id"
        assert result.tables == intent.tables

    def test_no_change_when_value_type_integer(self, fk_schema):
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("orders.category_id"),
                    op="=",
                    value_type="integer",
                    raw_value=5,
                ),
            ],
        )
        result = repair_fk_filter_type_mismatch(intent, fk_schema)
        assert result.filters_param[0].left_expr.primary_column == "orders.category_id"

    def test_no_change_when_column_not_fk(self, fk_schema):
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("orders.amount"),
                    op=">",
                    value_type="string",
                    raw_value="high",
                ),
            ],
        )
        result = repair_fk_filter_type_mismatch(intent, fk_schema)
        assert result.filters_param[0].left_expr.primary_column == "orders.amount"

    def test_no_change_when_no_descriptive_column(self):
        orders = TableMetadata(
            name="orders",
            columns={
                "order_id": ColumnMetadata(
                    name="order_id",
                    data_type="integer",
                    value_type="integer",
                    is_primary_key=True,
                    role=ColumnRole.IDENTIFIER.value,
                    distinct_count=500,
                ),
                "tag_id": ColumnMetadata(
                    name="tag_id",
                    data_type="integer",
                    value_type="integer",
                    is_foreign_key=True,
                    fk_target=("tags", "tag_id"),
                    role=ColumnRole.IDENTIFIER.value,
                    distinct_count=5,
                ),
            },
            primary_key=["order_id"],
            foreign_keys=[
                FKEdge(
                    src_table="orders",
                    src_cols=["tag_id"],
                    dst_table="tags",
                    dst_cols=["tag_id"],
                )
            ],
        )
        tags = TableMetadata(
            name="tags",
            columns={
                "tag_id": ColumnMetadata(
                    name="tag_id",
                    data_type="integer",
                    value_type="integer",
                    is_primary_key=True,
                    role=ColumnRole.IDENTIFIER.value,
                    distinct_count=5,
                    distinct_ratio=1.0,
                ),
            },
            primary_key=["tag_id"],
            foreign_keys=[],
        )
        sg = SchemaGraph(
            tables={"orders": orders, "tags": tags},
            join_paths_multi={},
            effective_structural_hash="h",
        )
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("orders.tag_id"),
                    op="=",
                    value_type="string",
                    raw_value="important",
                ),
            ],
        )
        result = repair_fk_filter_type_mismatch(intent, sg)
        assert result.filters_param[0].left_expr.primary_column == "orders.tag_id"

    def test_target_table_not_duplicated(self, fk_schema):
        intent = RuntimeIntent(
            tables=["categories", "orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("orders.category_id"),
                    op="=",
                    value_type="string",
                    raw_value="Action",
                ),
            ],
        )
        result = repair_fk_filter_type_mismatch(intent, fk_schema)
        assert result.tables.count("categories") == 1

    def test_mixed_filters_only_bad_repaired(self, fk_schema):
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("orders.status"),
                    op="=",
                    value_type="string",
                    raw_value="shipped",
                ),
                FilterParam(
                    left_expr=NormalizedExpr.from_column("orders.category_id"),
                    op="=",
                    value_type="string",
                    raw_value="Action",
                ),
            ],
        )
        result = repair_fk_filter_type_mismatch(intent, fk_schema)
        assert result.filters_param[0].left_expr.primary_column == "orders.status"
        assert result.filters_param[1].left_expr.primary_column == "orders.category_id"
        assert result.tables == intent.tables


class TestStripSpuriousGroupBy:
    """Tests for strip_spurious_group_by."""

    def test_no_group_by_unchanged(self):
        """Intent without group_by_cols is returned unchanged."""
        intent = RuntimeIntent(
            tables=["film"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.title"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        result = strip_spurious_group_by(intent)
        assert result.group_by_cols == []
        assert result.grain == "row_level"

    def test_strips_when_no_agg(self):
        """GROUP BY is stripped when no select column is aggregated."""
        intent = RuntimeIntent(
            tables=["film"],
            grain="grouped",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.title"))],
            group_by_cols=[NormalizedExpr.from_column("film.title")],
            order_by_cols=[],
            filters_param=[],
        )
        result = strip_spurious_group_by(intent)
        assert result.group_by_cols == []
        assert result.grain == "row_level"

    def test_keeps_when_agg_present(self):
        """GROUP BY is kept when an aggregated select column exists."""
        intent = RuntimeIntent(
            tables=["film"],
            grain="grouped",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("film.rating")),
                SelectCol(expr=NormalizedExpr.from_agg("count", "film.film_id")),
            ],
            group_by_cols=[NormalizedExpr.from_column("film.rating")],
            order_by_cols=[],
            filters_param=[],
        )
        result = strip_spurious_group_by(intent)
        assert len(result.group_by_cols) == 1
        assert result.grain == "grouped"

    def test_preserves_non_grouped_grain(self):
        """When stripping, grain that isn't 'grouped' stays unchanged."""
        intent = RuntimeIntent(
            tables=["film"],
            grain="scalar",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.title"))],
            group_by_cols=[NormalizedExpr.from_column("film.title")],
            order_by_cols=[],
            filters_param=[],
        )
        result = strip_spurious_group_by(intent)
        assert result.group_by_cols == []
        assert result.grain == "scalar"


class TestStripSpuriousGroupByCte:
    """CTE parity tests for strip_spurious_group_by."""

    def test_cte_spurious_group_by_stripped(self):
        """CTE step with group_by but no aggregation has group_by stripped."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["film"],
            grain="grouped",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.title"))],
            group_by_cols=[NormalizedExpr.from_column("film.title")],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            output_columns=[],
        )
        intent = RuntimeIntent(
            tables=["film"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.title"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        result = strip_spurious_group_by(intent)
        assert result.cte_steps[0].group_by_cols == []
        assert result.cte_steps[0].grain == "row_level"

    def test_cte_group_by_kept_with_agg(self):
        """CTE step with aggregation keeps group_by."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["film"],
            grain="grouped",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("film.rating")),
                SelectCol(expr=NormalizedExpr.from_agg("count", "film.film_id")),
            ],
            group_by_cols=[NormalizedExpr.from_column("film.rating")],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            output_columns=[],
        )
        intent = RuntimeIntent(
            tables=["film"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        result = strip_spurious_group_by(intent)
        assert len(result.cte_steps[0].group_by_cols) == 1
        assert result.cte_steps[0].grain == "grouped"

    def test_main_and_cte_both_stripped(self):
        """Both main and CTE have spurious GROUP BY stripped."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["film"],
            grain="grouped",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.title"))],
            group_by_cols=[NormalizedExpr.from_column("film.title")],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            output_columns=[],
        )
        intent = RuntimeIntent(
            tables=["film"],
            grain="grouped",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.rating"))],
            group_by_cols=[NormalizedExpr.from_column("film.rating")],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        result = strip_spurious_group_by(intent)
        assert result.group_by_cols == []
        assert result.grain == "row_level"
        assert result.cte_steps[0].group_by_cols == []
        assert result.cte_steps[0].grain == "row_level"

    def test_no_cte_steps_unchanged(self):
        """Intent without CTE steps processes main only and returns unchanged if fine."""
        intent = RuntimeIntent(
            tables=["film"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.title"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        result = strip_spurious_group_by(intent)
        assert result is intent


class TestRepairFkFilterTypeMismatchCte:
    """CTE parity tests for repair_fk_filter_type_mismatch."""

    @pytest.fixture
    def fk_schema(self):
        orders = TableMetadata(
            name="orders",
            columns={
                "order_id": ColumnMetadata(
                    name="order_id",
                    data_type="integer",
                    value_type="integer",
                    is_primary_key=True,
                    role=ColumnRole.IDENTIFIER.value,
                    distinct_count=500,
                ),
                "category_id": ColumnMetadata(
                    name="category_id",
                    data_type="integer",
                    value_type="integer",
                    is_foreign_key=True,
                    fk_target=("categories", "category_id"),
                    role=ColumnRole.IDENTIFIER.value,
                    distinct_count=10,
                ),
            },
            primary_key=["order_id"],
            foreign_keys=[
                FKEdge(
                    src_table="orders",
                    src_cols=["category_id"],
                    dst_table="categories",
                    dst_cols=["category_id"],
                ),
            ],
        )
        categories = TableMetadata(
            name="categories",
            columns={
                "category_id": ColumnMetadata(
                    name="category_id",
                    data_type="integer",
                    value_type="integer",
                    is_primary_key=True,
                    role=ColumnRole.IDENTIFIER.value,
                    distinct_count=10,
                    distinct_ratio=1.0,
                ),
                "name": ColumnMetadata(
                    name="name",
                    data_type="varchar",
                    value_type="string",
                    role=ColumnRole.CATEGORICAL.value,
                    distinct_count=10,
                    distinct_ratio=1.0,
                ),
            },
            primary_key=["category_id"],
            foreign_keys=[],
        )
        return SchemaGraph(
            tables={"orders": orders, "categories": categories},
            join_paths_multi={},
            effective_structural_hash="h",
        )

    def test_cte_fk_filter_rewired(self, fk_schema):
        """CTE step filter on FK integer column with string value is rewired."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["orders"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("orders.category_id"),
                    op="=",
                    value_type="string",
                    raw_value="Action",
                ),
            ],
            having_param=[],
            param_values={},
            output_columns=[],
        )
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        result = repair_fk_filter_type_mismatch(intent, fk_schema)
        assert result.cte_steps[0].filters_param[0].left_expr.primary_column == "orders.category_id"
        assert result.cte_steps[0].tables == cte.tables

    def test_cte_non_fk_filter_unchanged(self, fk_schema):
        """CTE step filter on non-FK column is not rewired."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["orders"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("orders.order_id"),
                    op="=",
                    value_type="integer",
                    raw_value=5,
                ),
            ],
            having_param=[],
            param_values={},
            output_columns=[],
        )
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        result = repair_fk_filter_type_mismatch(intent, fk_schema)
        assert result.cte_steps[0].filters_param[0].left_expr.primary_column == "orders.order_id"

    def test_main_and_cte_both_rewired(self, fk_schema):
        """Both main and CTE FK filters are rewired."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["orders"],
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("orders.category_id"),
                    op="=",
                    value_type="string",
                    raw_value="Drama",
                ),
            ],
            having_param=[],
            param_values={},
            output_columns=[],
        )
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("orders.category_id"),
                    op="=",
                    value_type="string",
                    raw_value="Action",
                ),
            ],
            cte_steps=[cte],
        )
        result = repair_fk_filter_type_mismatch(intent, fk_schema)
        assert result.filters_param[0].left_expr.primary_column == "orders.category_id"
        assert result.cte_steps[0].filters_param[0].left_expr.primary_column == "orders.category_id"
        assert result.tables == intent.tables
        assert result.cte_steps[0].tables == cte.tables


class TestMatchEnumValue:
    """Tests for _match_enum_value."""

    def test_exact_match(self):
        """Exact casing returns the enum member unchanged."""
        col_meta = ColumnMetadata(
            name="rating",
            data_type="mpaa_rating",
            value_type="string",
            role=ColumnRole.CATEGORICAL.value,
        )
        sg = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={},
            enum_values={"mpaa_rating": ["G", "PG", "PG-13", "R", "NC-17"]},
        )
        assert _match_enum_value("PG-13", col_meta, sg) == "PG-13"

    def test_case_insensitive_match(self):
        """Lower-case input matches the correctly-cased enum member."""
        col_meta = ColumnMetadata(
            name="rating",
            data_type="mpaa_rating",
            value_type="string",
            role=ColumnRole.CATEGORICAL.value,
        )
        sg = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={},
            enum_values={"mpaa_rating": ["G", "PG", "PG-13", "R", "NC-17"]},
        )
        assert _match_enum_value("pg-13", col_meta, sg) == "PG-13"

    def test_no_enum_values_returns_none(self):
        """Returns None when schema has no enum_values."""
        col_meta = ColumnMetadata(
            name="rating",
            data_type="mpaa_rating",
            value_type="string",
            role=ColumnRole.CATEGORICAL.value,
        )
        sg = SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={})
        assert _match_enum_value("PG", col_meta, sg) is None

    def test_no_match_returns_none(self):
        """Returns None when value does not match any enum member."""
        col_meta = ColumnMetadata(
            name="rating",
            data_type="mpaa_rating",
            value_type="string",
            role=ColumnRole.CATEGORICAL.value,
        )
        sg = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={},
            enum_values={"mpaa_rating": ["G", "PG", "R"]},
        )
        assert _match_enum_value("NC-17", col_meta, sg) is None


class TestResolveFilterListCascade:
    """Tests for _resolve_filter_list_cascade."""

    @pytest.fixture
    def rating_schema(self):
        """Schema with film.rating enum column."""
        film = TableMetadata(
            name="film",
            columns={
                "rating": ColumnMetadata(
                    name="rating",
                    data_type="mpaa_rating",
                    value_type="string",
                    role=ColumnRole.CATEGORICAL.value,
                    top_k_values=["G", "PG", "PG-13", "R", "NC-17"],
                ),
            },
            foreign_keys=[],
            primary_key="",
        )
        return SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={"film": film},
            enum_values={"mpaa_rating": ["G", "PG", "PG-13", "R", "NC-17"]},
        )

    def test_corrects_scalar_value(self, rating_schema):
        """Scalar string filter gets corrected via enum match."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("film.rating"),
            op="=",
            value_type="string",
            raw_value="pg",
        )
        result, changed = _resolve_filter_list_cascade([fp], rating_schema, "")
        assert changed
        assert result[0].raw_value == "PG"

    def test_corrects_list_values(self, rating_schema):
        """List filter values get corrected per element."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("film.rating"),
            op="in",
            value_type="string",
            raw_value=["pg", "r"],
        )
        result, changed = _resolve_filter_list_cascade([fp], rating_schema, "")
        assert changed
        assert result[0].raw_value == ["PG", "R"]

    def test_no_change_for_correct_casing(self, rating_schema):
        """Already correct casing returns unchanged."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("film.rating"),
            op="=",
            value_type="string",
            raw_value="PG",
        )
        result, changed = _resolve_filter_list_cascade([fp], rating_schema, "")
        assert not changed
        assert result[0].raw_value == "PG"

    def test_non_string_value_type_unchanged(self, rating_schema):
        """Non-string value_type filters are not touched."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("film.rating"),
            op="=",
            value_type="integer",
            raw_value=5,
        )
        result, changed = _resolve_filter_list_cascade([fp], rating_schema, "")
        assert not changed
        assert result[0].raw_value == 5


class TestResolveFilterValueCase:
    """Tests for resolve_filter_value_case."""

    @pytest.fixture
    def rating_schema(self):
        """Schema with film.rating enum column."""
        film = TableMetadata(
            name="film",
            columns={
                "rating": ColumnMetadata(
                    name="rating",
                    data_type="mpaa_rating",
                    value_type="string",
                    role=ColumnRole.CATEGORICAL.value,
                    top_k_values=["G", "PG", "PG-13", "R", "NC-17"],
                ),
            },
            foreign_keys=[],
            primary_key="",
        )
        return SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={"film": film},
            enum_values={"mpaa_rating": ["G", "PG", "PG-13", "R", "NC-17"]},
        )

    def test_main_query_corrected(self, rating_schema):
        """Main query filter value gets corrected."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("film.rating"),
            op="=",
            value_type="string",
            raw_value="pg-13",
        )
        intent = RuntimeIntent(
            tables=["film"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            having_param=[],
            param_values={},
            natural_language="Show PG-13 films",
        )
        result = resolve_filter_value_case(intent, rating_schema, "Show PG-13 films")
        assert result.filters_param[0].raw_value == "PG-13"

    def test_cte_step_corrected(self, rating_schema):
        """CTE step filter value gets corrected."""
        cte_fp = FilterParam(
            left_expr=NormalizedExpr.from_column("film.rating"),
            op="=",
            value_type="string",
            raw_value="r",
        )
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["film"],
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[cte_fp],
            having_param=[],
            param_values={},
            output_columns=[],
        )
        intent = RuntimeIntent(
            tables=["film"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            cte_steps=[cte],
            natural_language="test",
        )
        result = resolve_filter_value_case(intent, rating_schema, "test")
        assert result.cte_steps[0].filters_param[0].raw_value == "R"

    def test_no_change_returns_same_intent(self, rating_schema):
        """When no corrections needed the original intent is returned."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("film.rating"),
            op="=",
            value_type="string",
            raw_value="PG",
        )
        intent = RuntimeIntent(
            tables=["film"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            having_param=[],
            param_values={},
            natural_language="test",
        )
        result = resolve_filter_value_case(intent, rating_schema, "test")
        assert result is intent


class TestNormalizeInFilterTypes:
    """Tests for normalize_in_filter_types."""

    @pytest.fixture
    def int_col_schema(self):
        """Schema with an integer column."""
        t = TableMetadata(
            name="t",
            columns={
                "id": ColumnMetadata(
                    name="id",
                    data_type="integer",
                    value_type="integer",
                    is_primary_key=True,
                    role=ColumnRole.IDENTIFIER.value,
                ),
                "name": ColumnMetadata(
                    name="name",
                    data_type="varchar",
                    value_type="string",
                    role=ColumnRole.CATEGORICAL.value,
                ),
            },
            foreign_keys=[],
            primary_key="",
        )
        return SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"t": t})

    def test_coerces_string_elements_to_int(self, int_col_schema):
        """String elements on an integer column are coerced to ints."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.id"),
            op="in",
            value_type="integer",
            raw_value=["1", "2", "3"],
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            having_param=[],
            param_values={},
            natural_language="test",
        )
        result = normalize_in_filter_types(intent, int_col_schema)
        assert [fp.raw_value for fp in result.filters_param] == [1, 2, 3]
        assert all(fp.op == "=" for fp in result.filters_param)

    def test_consolidates_string_list(self, int_col_schema):
        """String list on a string column is consolidated with quotes."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.name"),
            op="in",
            value_type="string",
            raw_value=["Alice", "Bob"],
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            having_param=[],
            param_values={},
            natural_language="test",
        )
        result = normalize_in_filter_types(intent, int_col_schema)
        assert [fp.raw_value for fp in result.filters_param] == ["Alice", "Bob"]
        assert all(fp.op == "=" for fp in result.filters_param)

    def test_non_in_op_unchanged(self, int_col_schema):
        """Filters with non-IN ops are not touched."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.id"),
            op="=",
            value_type="integer",
            raw_value=5,
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
            having_param=[],
            param_values={},
            natural_language="test",
        )
        result = normalize_in_filter_types(intent, int_col_schema)
        assert result.filters_param == intent.filters_param

    def test_cte_step_coerced(self, int_col_schema):
        """CTE step IN filters are also coerced."""
        cte_fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.id"),
            op="in",
            value_type="integer",
            raw_value=["10", "20"],
        )
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["t"],
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[cte_fp],
            having_param=[],
            param_values={},
            output_columns=[],
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            param_values={},
            cte_steps=[cte],
            natural_language="test",
        )
        result = normalize_in_filter_types(intent, int_col_schema)
        assert [fp.raw_value for fp in result.cte_steps[0].filters_param] == [10, 20]
        assert all(fp.op == "=" for fp in result.cte_steps[0].filters_param)


class TestNormalizeBooleanFilterValues:
    """Tests for normalize_boolean_filter_values (main + CTE)."""

    @pytest.fixture
    def bool_schema(self):
        """Schema with a boolean column."""
        t = TableMetadata(
            name="customer",
            columns={
                "customer_id": ColumnMetadata(
                    name="customer_id",
                    data_type="integer",
                    value_type="integer",
                    is_primary_key=True,
                    role=ColumnRole.IDENTIFIER.value,
                ),
                "activebool": ColumnMetadata(
                    name="activebool",
                    data_type="boolean",
                    value_type="boolean",
                    role=ColumnRole.CATEGORICAL.value,
                ),
                "name": ColumnMetadata(
                    name="name",
                    data_type="varchar",
                    value_type="string",
                    role=ColumnRole.CATEGORICAL.value,
                ),
            },
            foreign_keys=[],
            primary_key="",
        )
        return SchemaGraph(join_paths_multi={}, effective_structural_hash="", tables={"customer": t})

    def test_int_1_normalised_to_true(self, bool_schema):
        """Integer 1 on boolean column becomes True."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("customer.activebool"),
            op="=",
            value_type="integer",
            raw_value=1,
        )
        intent = RuntimeIntent(
            tables=["customer"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
        )
        result = normalize_boolean_filter_values(intent, bool_schema)
        assert result.filters_param[0].raw_value is True
        assert result.filters_param[0].value_type == "boolean"

    def test_string_false_normalised(self, bool_schema):
        """String 'false' on boolean column becomes False."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("customer.activebool"),
            op="=",
            value_type="string",
            raw_value="false",
        )
        intent = RuntimeIntent(
            tables=["customer"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
        )
        result = normalize_boolean_filter_values(intent, bool_schema)
        assert result.filters_param[0].raw_value is False
        assert result.filters_param[0].value_type == "boolean"

    def test_already_bool_unchanged(self, bool_schema):
        """Python bool on boolean column is kept unchanged."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("customer.activebool"),
            op="=",
            value_type="boolean",
            raw_value=True,
        )
        intent = RuntimeIntent(
            tables=["customer"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
        )
        result = normalize_boolean_filter_values(intent, bool_schema)
        assert result.filters_param[0].raw_value is True
        assert result.filters_param[0].value_type == "boolean"

    def test_non_boolean_column_unchanged(self, bool_schema):
        """Non-boolean column filter is not touched."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("customer.name"),
            op="=",
            value_type="string",
            raw_value="Alice",
        )
        intent = RuntimeIntent(
            tables=["customer"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
        )
        result = normalize_boolean_filter_values(intent, bool_schema)
        assert result.filters_param[0].raw_value == "Alice"

    def test_no_change_returns_same_intent(self, bool_schema):
        """When nothing changes the original intent is returned."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("customer.name"),
            op="=",
            value_type="string",
            raw_value="Bob",
        )
        intent = RuntimeIntent(
            tables=["customer"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
        )
        result = normalize_boolean_filter_values(intent, bool_schema)
        assert result is intent

    def test_cte_boolean_normalised(self, bool_schema):
        """CTE step boolean filter is normalised."""
        cte_fp = FilterParam(
            left_expr=NormalizedExpr.from_column("customer.activebool"),
            op="=",
            value_type="string",
            raw_value="yes",
        )
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["customer"],
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[cte_fp],
            having_param=[],
            param_values={},
            output_columns=[],
        )
        intent = RuntimeIntent(
            tables=["customer"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        result = normalize_boolean_filter_values(intent, bool_schema)
        assert result.cte_steps[0].filters_param[0].raw_value is True
        assert result.cte_steps[0].filters_param[0].value_type == "boolean"

    def test_main_and_cte_both_normalised(self, bool_schema):
        """Both main and CTE boolean filters are normalised."""
        main_fp = FilterParam(
            left_expr=NormalizedExpr.from_column("customer.activebool"),
            op="=",
            value_type="integer",
            raw_value=0,
        )
        cte_fp = FilterParam(
            left_expr=NormalizedExpr.from_column("customer.activebool"),
            op="=",
            value_type="string",
            raw_value="true",
        )
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["customer"],
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[cte_fp],
            having_param=[],
            param_values={},
            output_columns=[],
        )
        intent = RuntimeIntent(
            tables=["customer"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[main_fp],
            cte_steps=[cte],
        )
        result = normalize_boolean_filter_values(intent, bool_schema)
        assert result.filters_param[0].raw_value is False
        assert result.cte_steps[0].filters_param[0].raw_value is True


class TestStripSpuriousGroupByHavingParam:
    """Tests for strip_spurious_group_by when having_param has aggregation."""

    def test_keeps_group_by_when_having_is_aggregated(self):
        """GROUP BY is preserved when having_param contains an aggregated expression even though no select column is aggregated."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "orders.order_id"),
            op=">",
            value_type="number",
            raw_value=5,
        )
        intent = RuntimeIntent(
            tables=["orders"],
            grain="grouped",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.status"))],
            group_by_cols=[NormalizedExpr.from_column("orders.status")],
            order_by_cols=[],
            filters_param=[],
            having_param=[hp],
        )
        result = strip_spurious_group_by(intent)
        assert len(result.group_by_cols) == 1
        assert result.grain == "grouped"

    def test_strips_when_having_not_aggregated(self):
        """GROUP BY is stripped when having_param exists but is not aggregated and no select col is aggregated either."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_column("orders.status"),
            op="=",
            value_type="string",
            raw_value="active",
        )
        intent = RuntimeIntent(
            tables=["orders"],
            grain="grouped",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.status"))],
            group_by_cols=[NormalizedExpr.from_column("orders.status")],
            order_by_cols=[],
            filters_param=[],
            having_param=[hp],
        )
        result = strip_spurious_group_by(intent)
        assert result.group_by_cols == []
        assert result.grain == "row_level"

    def test_cte_keeps_group_by_when_cte_having_is_aggregated(self):
        """CTE GROUP BY is preserved when CTE having_param has aggregated expression."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("sum", "items.amount"),
            op=">=",
            value_type="number",
            raw_value=100,
        )
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["items"],
            grain="grouped",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("items.category"))],
            group_by_cols=[NormalizedExpr.from_column("items.category")],
            order_by_cols=[],
            filters_param=[],
            having_param=[hp],
            param_values={},
            output_columns=[],
        )
        intent = RuntimeIntent(
            tables=["items"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        result = strip_spurious_group_by(intent)
        assert len(result.cte_steps[0].group_by_cols) == 1
        assert result.cte_steps[0].grain == "grouped"


class TestIsImpossibleHaving:
    """Tests for _is_impossible_having helper."""

    def test_count_less_than_zero(self):
        """COUNT(...) < 0 is impossible."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "orders.id"),
            op="<",
            raw_value=0,
        )
        assert _is_impossible_having(hp) is True

    def test_count_less_equal_zero(self):
        """COUNT(...) <= 0 is impossible."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "orders.id"),
            op="<=",
            raw_value=0,
        )
        assert _is_impossible_having(hp) is True

    def test_count_equal_negative(self):
        """COUNT(...) = -1 is impossible."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "orders.id"),
            op="=",
            raw_value=-1,
        )
        assert _is_impossible_having(hp) is True

    def test_count_greater_than_zero_possible(self):
        """COUNT(...) > 0 is possible and should not be flagged."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "orders.id"),
            op=">",
            raw_value=0,
        )
        assert _is_impossible_having(hp) is False

    def test_count_equal_positive_possible(self):
        """COUNT(...) = 5 is possible and should not be flagged."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "orders.id"),
            op="=",
            raw_value=5,
        )
        assert _is_impossible_having(hp) is False

    def test_sum_less_than_negative_possible(self):
        """SUM(...) < -10 is possible for SUM."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("sum", "orders.amount"),
            op="<",
            raw_value=-10,
        )
        assert _is_impossible_having(hp) is False

    def test_non_agg_expr_not_flagged(self):
        """Plain column expression is never flagged as impossible."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_column("orders.status"),
            op="<",
            raw_value=0,
        )
        assert _is_impossible_having(hp) is False

    def test_none_raw_value_not_flagged(self):
        """None raw_value is not flagged."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "orders.id"),
            op="<",
            raw_value=None,
        )
        assert _is_impossible_having(hp) is False

    def test_string_raw_value_zero(self):
        """String '0' is converted and detected as impossible for COUNT < '0'."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "orders.id"),
            op="<",
            raw_value="0",
        )
        assert _is_impossible_having(hp) is True


class TestStripImpossibleHaving:
    """Tests for strip_impossible_having."""

    def test_removes_count_less_than_zero(self):
        """Impossible HAVING COUNT < 0 is removed from intent."""
        hp_bad = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "orders.id"),
            op="<",
            raw_value=0,
        )
        hp_good = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "orders.id"),
            op=">",
            raw_value=5,
        )
        intent = RuntimeIntent(
            tables=["orders"],
            grain="grouped",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("orders.status")),
                SelectCol(expr=NormalizedExpr.from_agg("count", "orders.id")),
            ],
            group_by_cols=[NormalizedExpr.from_column("orders.status")],
            order_by_cols=[],
            filters_param=[],
            having_param=[hp_bad, hp_good],
        )
        result = strip_impossible_having(intent)
        assert len(result.having_param) == 1
        assert result.having_param[0].op == ">"

    def test_no_change_when_all_possible(self):
        """Intent returned unchanged when all HAVING conditions are possible."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "orders.id"),
            op=">",
            raw_value=5,
        )
        intent = RuntimeIntent(
            tables=["orders"],
            grain="grouped",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_agg("count", "orders.id")),
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[hp],
        )
        result = strip_impossible_having(intent)
        assert result is intent

    def test_empty_having_unchanged(self):
        """Intent with empty having_param is returned unchanged."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
        )
        result = strip_impossible_having(intent)
        assert result is intent

    def test_cte_impossible_having_removed(self):
        """Impossible HAVING in CTE is removed independently."""
        hp_bad = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "items.id"),
            op="<=",
            raw_value=0,
        )
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["items"],
            grain="grouped",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("items.category")),
                SelectCol(expr=NormalizedExpr.from_agg("count", "items.id")),
            ],
            group_by_cols=[NormalizedExpr.from_column("items.category")],
            order_by_cols=[],
            filters_param=[],
            having_param=[hp_bad],
            param_values={},
            output_columns=[],
        )
        intent = RuntimeIntent(
            tables=["items"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[cte],
        )
        result = strip_impossible_having(intent)
        assert result.cte_steps[0].having_param == []


class TestSanitizeTableNames:
    """Tests for sanitize_table_names."""

    @pytest.fixture
    def basic_schema(self):
        """Schema with two simple tables."""
        return SchemaGraph(
            tables={
                "orders": TableMetadata(
                    name="orders",
                    columns={
                        "id": ColumnMetadata(
                            name="id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                            role=ColumnRole.IDENTIFIER.value,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="",
                ),
                "customer": TableMetadata(
                    name="customer",
                    columns={
                        "id": ColumnMetadata(
                            name="id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                            role=ColumnRole.IDENTIFIER.value,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="",
                ),
            },
            join_paths_multi={},
            effective_structural_hash="",
        )

    def test_strips_distinct_prefix(self, basic_schema):
        """'DISTINCT customer' becomes 'customer'."""
        intent = RuntimeIntent(
            tables=["DISTINCT customer"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        result = sanitize_table_names(intent, basic_schema)
        assert result.tables == ["customer"]

    def test_strips_join_prefix(self, basic_schema):
        """'JOIN orders' becomes 'orders'."""
        intent = RuntimeIntent(
            tables=["JOIN orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        result = sanitize_table_names(intent, basic_schema)
        assert result.tables == ["orders"]

    def test_no_change_for_valid_name(self, basic_schema):
        """Already valid table name returns the same intent."""
        intent = RuntimeIntent(
            tables=["orders", "customer"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        result = sanitize_table_names(intent, basic_schema)
        assert result is intent

    def test_unknown_table_left_alone(self, basic_schema):
        """Table name that doesn't match schema even after stripping is left unchanged."""
        intent = RuntimeIntent(
            tables=["DISTINCT unknown_tbl"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        result = sanitize_table_names(intent, basic_schema)
        assert result.tables == ["DISTINCT unknown_tbl"]

    def test_multi_keyword_prefix(self, basic_schema):
        """'LEFT JOIN orders' strips both keywords."""
        intent = RuntimeIntent(
            tables=["LEFT JOIN orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        result = sanitize_table_names(intent, basic_schema)
        assert result.tables == ["orders"]


class TestPruneUnreferencedTablesWithSchemaGraph:
    """Tests for :func:`reconcile_tables` beside FK-focused prune schemas."""

    @pytest.fixture
    def prune_schema(self):
        """Schema with 'orders' (PK-only) and 'customer' (has descriptive 'name' column)."""
        return SchemaGraph(
            tables={
                "orders": TableMetadata(
                    name="orders",
                    columns={
                        "order_id": ColumnMetadata(
                            name="order_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                            role=ColumnRole.IDENTIFIER.value,
                        ),
                        "customer_id": ColumnMetadata(
                            name="customer_id",
                            data_type="integer",
                            value_type="integer",
                            is_foreign_key=True,
                            role=ColumnRole.IDENTIFIER.value,
                        ),
                    },
                    foreign_keys=[
                        FKEdge(
                            src_table="orders",
                            dst_table="customer",
                            src_cols=["customer_id"],
                            dst_cols=["id"],
                        ),
                    ],
                    primary_key="",
                ),
                "customer": TableMetadata(
                    name="customer",
                    columns={
                        "id": ColumnMetadata(
                            name="id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                            role=ColumnRole.IDENTIFIER.value,
                        ),
                        "name": ColumnMetadata(
                            name="name",
                            data_type="text",
                            value_type="string",
                            role=ColumnRole.CATEGORICAL.value,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="",
                ),
            },
            join_paths_multi={},
            effective_structural_hash="",
        )

    def test_cosmetic_table_kept_when_descriptive_but_no_fk_filter(self, prune_schema):
        """After removing cosmetic prune logic, a table with descriptive select columns is kept even without FK filter."""
        intent = RuntimeIntent(
            tables=["orders", "customer"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("orders.order_id")),
                SelectCol(expr=NormalizedExpr.from_column("customer.name")),
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("orders.order_id"),
                    op=">",
                    raw_value=0,
                ),
            ],
        )
        result = reconcile_tables(intent)
        assert "customer" in result.tables
        assert "orders" in result.tables

    def test_cosmetic_table_kept_when_only_pk(self, prune_schema):
        """After removing cosmetic prune logic, a table contributing only PK columns is kept (pruning is purely reference-based)."""
        intent = RuntimeIntent(
            tables=["orders", "customer"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("orders.order_id")),
                SelectCol(expr=NormalizedExpr.from_column("customer.id")),
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("orders.order_id"),
                    op=">",
                    raw_value=0,
                ),
            ],
        )
        result = reconcile_tables(intent)
        assert "orders" in result.tables
        assert "customer" in result.tables


class TestResolveFilterListCascadeLower:
    """Tests for LOWER-based casing in _resolve_filter_list_cascade."""

    @pytest.fixture
    def name_schema(self):
        """Schema with a non-enum string column."""
        t = TableMetadata(
            name="customer",
            columns={
                "name": ColumnMetadata(
                    name="name",
                    data_type="varchar",
                    value_type="string",
                    role=ColumnRole.CATEGORICAL.value,
                ),
            },
            foreign_keys=[],
            primary_key="",
        )
        return SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={"customer": t},
        )

    def test_lowercases_non_enum_string(self, name_schema):
        """Non-enum string value is lowercased."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("customer.name"),
            op="=",
            value_type="string",
            raw_value="Alice",
        )
        result, changed = _resolve_filter_list_cascade([fp], name_schema, "")
        assert changed
        assert result[0].raw_value == "alice"

    def test_already_lowercase_unchanged(self, name_schema):
        """Already lowercase value is not changed."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("customer.name"),
            op="=",
            value_type="string",
            raw_value="alice",
        )
        result, changed = _resolve_filter_list_cascade([fp], name_schema, "")
        assert not changed
        assert result[0].raw_value == "alice"

    def test_enum_match_preserves_casing(self):
        """Enum match preserves the database casing."""
        film = TableMetadata(
            name="film",
            columns={
                "rating": ColumnMetadata(
                    name="rating",
                    data_type="mpaa_rating",
                    value_type="string",
                    role=ColumnRole.CATEGORICAL.value,
                ),
            },
            foreign_keys=[],
            primary_key="",
        )
        sg = SchemaGraph(
            join_paths_multi={},
            effective_structural_hash="",
            tables={"film": film},
            enum_values={"mpaa_rating": ["G", "PG", "PG-13", "R", "NC-17"]},
        )
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("film.rating"),
            op="=",
            value_type="string",
            raw_value="pg",
        )
        result, changed = _resolve_filter_list_cascade([fp], sg, "")
        assert changed
        assert result[0].raw_value == "PG"

    def test_list_values_lowercased(self, name_schema):
        """List filter values are lowercased for non-enum columns."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("customer.name"),
            op="in",
            value_type="string",
            raw_value=["Alice", "Bob"],
        )
        result, changed = _resolve_filter_list_cascade([fp], name_schema, "")
        assert changed
        assert result[0].raw_value == ["alice", "bob"]


class TestPruneWithFkFilterTarget:
    """Tests for reconcile_tables with FK filter target logic."""

    @pytest.fixture
    def fk_prune_schema(self):
        """Schema with FK fk_target metadata on orders.customer_id."""
        return SchemaGraph(
            tables={
                "orders": TableMetadata(
                    name="orders",
                    columns={
                        "order_id": ColumnMetadata(
                            name="order_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                            role=ColumnRole.IDENTIFIER.value,
                        ),
                        "customer_id": ColumnMetadata(
                            name="customer_id",
                            data_type="integer",
                            value_type="integer",
                            is_foreign_key=True,
                            role=ColumnRole.IDENTIFIER.value,
                            fk_target=("customer", "id"),
                        ),
                    },
                    foreign_keys=[
                        FKEdge(
                            src_table="orders",
                            dst_table="customer",
                            src_cols=["customer_id"],
                            dst_cols=["id"],
                        ),
                    ],
                    primary_key="order_id",
                ),
                "customer": TableMetadata(
                    name="customer",
                    columns={
                        "id": ColumnMetadata(
                            name="id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                            role=ColumnRole.IDENTIFIER.value,
                        ),
                        "name": ColumnMetadata(
                            name="name",
                            data_type="text",
                            value_type="string",
                            role=ColumnRole.CATEGORICAL.value,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="id",
                ),
            },
            join_paths_multi={},
            effective_structural_hash="",
        )

    def test_kept_when_fk_filter_targets_table(self, fk_prune_schema):
        """Cosmetic table kept when FK filter targets its PK and it has descriptive select."""
        intent = RuntimeIntent(
            tables=["orders", "customer"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("orders.order_id")),
                SelectCol(expr=NormalizedExpr.from_column("customer.name")),
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("orders.customer_id"),
                    op="=",
                    raw_value=5,
                ),
            ],
        )
        result = reconcile_tables(intent)
        assert "customer" in result.tables
        assert "orders" in result.tables

    def test_kept_when_no_fk_filter_targets_table(self, fk_prune_schema):
        """After removing cosmetic prune logic, table is kept even without FK filter targeting its PK."""
        intent = RuntimeIntent(
            tables=["orders", "customer"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("orders.order_id")),
                SelectCol(expr=NormalizedExpr.from_column("customer.name")),
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("orders.order_id"),
                    op=">",
                    raw_value=0,
                ),
            ],
        )
        result = reconcile_tables(intent)
        assert "customer" in result.tables
        assert "orders" in result.tables


class TestInferCteOutputColumns:
    """Tests for infer_cte_output_columns."""

    def test_bare_columns_extracted(self):
        """Extracts trailing column name from qualified expressions."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["orders"],
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("orders.customer_id")),
                SelectCol(expr=NormalizedExpr.from_column("orders.total")),
            ],
        )
        result = infer_cte_output_columns(cte)
        assert result == ["customer_id", "total"]

    def test_aggregated_column_prefixed(self):
        """Aggregated columns get agg_func prefix."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["rental"],
            select_cols=[
                SelectCol(
                    expr=NormalizedExpr(
                        add_groups=[MulGroup(multiply=["rental.rental_id"])],
                        agg_func="count",
                    ),
                ),
            ],
        )
        result = infer_cte_output_columns(cte)
        assert result == ["count_rental_id"]

    def test_empty_select_cols(self):
        """Returns empty list when no select columns."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["orders"],
        )
        result = infer_cte_output_columns(cte)
        assert result == []

    def test_deduplicates_names(self):
        """Does not produce duplicate output column names."""
        cte = RuntimeCteStep(
            cte_name="cte1",
            tables=["orders"],
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("orders.total")),
                SelectCol(expr=NormalizedExpr.from_column("payment.total")),
            ],
        )
        result = infer_cte_output_columns(cte)
        assert result == ["total"]


class TestQualifyCteWithInferredOutputs:
    """Tests for qualify_cte_output_columns with inferred outputs."""

    def test_qualifies_main_query_from_inferred_outputs(self):
        """Main query references qualified when CTE output_columns is empty but select_cols can be inferred."""
        intent = RuntimeIntent(
            tables=[],
            grain="grouped",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("customer_id")),
                SelectCol(expr=NormalizedExpr.from_column("count_rental_id")),
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[
                RuntimeCteStep(
                    cte_name="cte1",
                    tables=["rental"],
                    select_cols=[
                        SelectCol(expr=NormalizedExpr.from_column("rental.customer_id")),
                        SelectCol(
                            expr=NormalizedExpr(
                                add_groups=[MulGroup(multiply=["rental.rental_id"])],
                                agg_func="count",
                            ),
                        ),
                    ],
                ),
            ],
        )
        result = qualify_cte_output_columns(intent)
        main_cols = [sc.expr.primary_column for sc in result.select_cols]
        assert any("cte1." in c for c in main_cols)


class TestIsNullValue:
    """Tests for _is_null_value."""

    def test_none_is_null(self):
        """Python None is considered null."""
        assert _is_null_value(None) is True

    def test_null_string_is_null(self):
        """String 'null' is considered null."""
        assert _is_null_value("null") is True

    def test_null_string_case_insensitive(self):
        """String 'NULL' is considered null."""
        assert _is_null_value("NULL") is True

    def test_null_string_with_whitespace(self):
        """String ' null ' with whitespace is considered null."""
        assert _is_null_value(" null ") is True

    def test_non_null_string(self):
        """Regular string value is not null."""
        assert _is_null_value("hello") is False

    def test_zero_is_not_null(self):
        """Integer zero is not null."""
        assert _is_null_value(0) is False

    def test_empty_string_is_not_null(self):
        """Empty string is not null."""
        assert _is_null_value("") is False


class TestRepairNullEqualityFilters:
    """Tests for repair_null_equality_filters."""

    def test_equality_null_becomes_is_null(self):
        """Filter with op='=' and value=None becomes op='is null'."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("orders.return_date"),
                    op="=",
                    value_type="date",
                    raw_value=None,
                ),
            ],
        )
        result = repair_null_equality_filters(intent)
        assert result.filters_param[0].op == "is null"
        assert result.filters_param[0].raw_value is None
        assert result.filters_param[0].value_type == "null"

    def test_equality_null_string_becomes_is_null(self):
        """Filter with op='=' and value='null' becomes op='is null'."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("orders.return_date"),
                    op="=",
                    value_type="string",
                    raw_value="null",
                ),
            ],
        )
        result = repair_null_equality_filters(intent)
        assert result.filters_param[0].op == "is null"
        assert result.filters_param[0].raw_value is None

    def test_not_equal_null_becomes_is_not_null(self):
        """Filter with op='!=' and value=None becomes op='is not null'."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("orders.return_date"),
                    op="!=",
                    value_type="date",
                    raw_value=None,
                ),
            ],
        )
        result = repair_null_equality_filters(intent)
        assert result.filters_param[0].op == "is not null"
        assert result.filters_param[0].raw_value is None
        assert result.filters_param[0].value_type == "null"

    def test_diamond_not_equal_null_becomes_is_not_null(self):
        """Filter with op='<>' and value=None becomes op='is not null'."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("orders.return_date"),
                    op="<>",
                    value_type="date",
                    raw_value=None,
                ),
            ],
        )
        result = repair_null_equality_filters(intent)
        assert result.filters_param[0].op == "is not null"

    def test_non_null_equality_unchanged(self):
        """Filter with op='=' and non-null value is unchanged."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("orders.status"),
                    op="=",
                    value_type="string",
                    raw_value="active",
                ),
            ],
        )
        result = repair_null_equality_filters(intent)
        assert result.filters_param[0].op == "="
        assert result.filters_param[0].raw_value == "active"

    def test_cte_filters_also_repaired(self):
        """Null equality filters inside CTE steps are also repaired."""
        intent = RuntimeIntent(
            tables=[],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[
                RuntimeCteStep(
                    cte_name="cte1",
                    tables=["rental"],
                    filters_param=[
                        FilterParam(
                            left_expr=NormalizedExpr.from_column("rental.return_date"),
                            op="=",
                            value_type="date",
                            raw_value=None,
                        ),
                    ],
                ),
            ],
        )
        result = repair_null_equality_filters(intent)
        assert result.cte_steps[0].filters_param[0].op == "is null"

    def test_having_equality_null_rewritten(self):
        """HAVING ``= NULL`` becomes ``is null`` like WHERE filters."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="aggregate",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[
                HavingParam(
                    left_expr=NormalizedExpr.from_column("orders.amount"),
                    op="=",
                    value_type="number",
                    raw_value=None,
                ),
            ],
        )
        result = repair_null_equality_filters(intent)
        assert result.having_param[0].op == "is null"
        assert result.having_param[0].value_type == "null"


class TestRepairNullEqualityLoopSafety:
    """Harvested filters with param_key must not be rewritten to IS NULL on repeat repairs."""

    def _base_intent(self, filters: list, having: list | None = None) -> RuntimeIntent:
        return RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=filters,
            having_param=having or [],
        )

    def test_filter_skips_when_param_key_set_string(self) -> None:
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("orders.rating"),
            op="=",
            value_type="string",
            param_key="p1",
            raw_value=None,
        )
        intent = self._base_intent([fp])
        once = repair_null_equality_filters(intent)
        twice = repair_null_equality_filters(once)
        assert once.filters_param[0].op == "="
        assert twice.filters_param[0].op == "="

    def test_filter_skips_when_param_key_set_integer(self) -> None:
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("orders.store_id"),
            op="=",
            value_type="integer",
            param_key="p1",
            raw_value=None,
        )
        intent = self._base_intent([fp])
        out = repair_null_equality_filters(intent)
        assert out.filters_param[0].op == "="

    def test_having_skips_when_param_key_set(self) -> None:
        hp = HavingParam(
            left_expr=NormalizedExpr.from_column("orders.amount"),
            op="=",
            value_type="number",
            param_key="p1",
            raw_value=None,
        )
        intent = self._base_intent([], [hp])
        out = repair_null_equality_filters(intent)
        assert out.having_param[0].op == "="

    def test_cte_filter_skips_when_param_key_set(self) -> None:
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("rental.rental_id"),
            op="=",
            value_type="integer",
            param_key="p1",
            raw_value=None,
        )
        intent = RuntimeIntent(
            tables=[],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[
                RuntimeCteStep(
                    cte_name="c1",
                    tables=["rental"],
                    filters_param=[fp],
                ),
            ],
        )
        out = repair_null_equality_filters(intent)
        assert out.cte_steps[0].filters_param[0].op == "="

    def test_empty_param_key_still_rewrites_null_literal(self) -> None:
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("orders.return_date"),
            op="=",
            value_type="date",
            param_key="",
            raw_value=None,
        )
        intent = self._base_intent([fp])
        out = repair_null_equality_filters(intent)
        assert out.filters_param[0].op == "is null"


class TestRepairNullEqualityFiltersContinued:
    """More repair_null_equality_filters cases grouped after loop-safety coverage."""

    def test_having_count_equal_drops_contradictory_range(self):
        """``HAVING COUNT(*) = 3 AND COUNT(*) > 10`` keeps only the equality."""
        count_expr = NormalizedExpr(
            add_groups=[MulGroup(coefficient=1.0, multiply=["orders.id"], agg_func="count")],
        )
        intent = RuntimeIntent(
            tables=["orders"],
            grain="aggregate",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[
                HavingParam(left_expr=count_expr, op="=", raw_value=3, value_type="number"),
                HavingParam(left_expr=count_expr, op=">", raw_value=10, value_type="number"),
            ],
        )
        result = dedup_contradictory_filters(intent)
        assert len(result.having_param) == 1
        assert result.having_param[0].op == "="

    def test_normalize_null_having_fixes_value_side(self):
        """``is null`` on HAVING gets ``value_type=null`` and cleared ``raw_value``."""
        intent = RuntimeIntent(
            tables=["t"],
            grain="aggregate",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[
                HavingParam(
                    left_expr=NormalizedExpr.from_column("t.x"),
                    op="is null",
                    value_type="number",
                    raw_value=0,
                ),
            ],
        )
        out = normalize_null_filter_values(intent)
        assert out.having_param[0].value_type == "null"
        assert out.having_param[0].raw_value is None


class TestPruneKeepsAggregationTables:
    """Tests for Fix X: reconcile_tables keeps tables that contribute aggregated SelectCols."""

    @pytest.fixture
    def revenue_schema(self):
        """Schema with country and payment tables for AJ-006 scenario."""
        return SchemaGraph(
            tables={
                "country": TableMetadata(
                    name="country",
                    columns={
                        "country_id": ColumnMetadata(
                            name="country_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                        ),
                        "country": ColumnMetadata(
                            name="country",
                            data_type="varchar",
                            value_type="string",
                        ),
                    },
                    foreign_keys=[],
                    primary_key="",
                ),
                "payment": TableMetadata(
                    name="payment",
                    columns={
                        "payment_id": ColumnMetadata(
                            name="payment_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                        ),
                        "amount": ColumnMetadata(
                            name="amount",
                            data_type="numeric",
                            value_type="number",
                        ),
                    },
                    foreign_keys=[],
                    primary_key="",
                ),
            },
            join_paths_multi={},
            effective_structural_hash="",
        )

    def test_payment_kept_for_total_revenue_per_country(self, revenue_schema):
        """payment table with SUM(amount) is not pruned even though 'payment' is not in the question."""
        intent = RuntimeIntent(
            tables=["country", "payment"],
            grain="grouped",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("country.country")),
                SelectCol(expr=NormalizedExpr.from_agg("sum", "payment.amount")),
            ],
            group_by_cols=[NormalizedExpr.from_column("country.country")],
            order_by_cols=[],
            filters_param=[],
        )
        result = reconcile_tables(intent)
        assert "payment" in result.tables
        assert "country" in result.tables
        assert any(sc.is_aggregated for sc in result.select_cols)


class TestPruneColumnComponentSuppression:
    """Tests for Fix Y1: column-component suppression in pruning."""

    @pytest.fixture
    def rental_schema(self):
        """Schema where film has rental_duration and rental table exists."""
        return SchemaGraph(
            tables={
                "film": TableMetadata(
                    name="film",
                    columns={
                        "film_id": ColumnMetadata(
                            name="film_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                        ),
                        "rating": ColumnMetadata(
                            name="rating",
                            data_type="varchar",
                            value_type="string",
                        ),
                        "rental_duration": ColumnMetadata(
                            name="rental_duration",
                            data_type="integer",
                            value_type="integer",
                        ),
                    },
                    foreign_keys=[],
                    primary_key="",
                ),
                "inventory": TableMetadata(
                    name="inventory",
                    columns={
                        "inventory_id": ColumnMetadata(
                            name="inventory_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                        ),
                        "film_id": ColumnMetadata(
                            name="film_id",
                            data_type="integer",
                            value_type="integer",
                            is_foreign_key=True,
                            fk_target=("film", "film_id"),
                        ),
                    },
                    foreign_keys=[],
                    primary_key="",
                ),
                "rental": TableMetadata(
                    name="rental",
                    columns={
                        "rental_id": ColumnMetadata(
                            name="rental_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                        ),
                        "inventory_id": ColumnMetadata(
                            name="inventory_id",
                            data_type="integer",
                            value_type="integer",
                            is_foreign_key=True,
                            fk_target=("inventory", "inventory_id"),
                        ),
                    },
                    foreign_keys=[],
                    primary_key="",
                ),
            },
            join_paths_multi={},
            effective_structural_hash="",
        )

    def test_rental_table_kept_when_referenced_in_select(self, rental_schema):
        """rental is kept because COUNT(rental.rental_id) references it; inventory pruned as genuinely unreferenced.  The old cosmetic column-component prune was removed (Fix 1)."""
        intent = RuntimeIntent(
            tables=["film", "inventory", "rental"],
            grain="grouped",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("film.rating")),
                SelectCol(expr=NormalizedExpr.from_agg("avg", "film.rental_duration")),
                SelectCol(expr=NormalizedExpr.from_agg("count", "rental.rental_id")),
            ],
            group_by_cols=[NormalizedExpr.from_column("film.rating")],
            order_by_cols=[],
            filters_param=[],
        )
        result = reconcile_tables(intent)
        assert "film" in result.tables
        assert "rental" in result.tables
        assert "inventory" not in result.tables


class TestPruneUnreferencedKeepsExplicitSelectCol:
    """Tests for Task 1 safety check: reconcile_tables keeps tables explicitly referenced as 'table.column' in select_cols."""

    @pytest.fixture
    def language_schema(self):
        """Minimal schema with film and language tables."""
        return SchemaGraph(
            tables={
                "film": TableMetadata(
                    name="film",
                    columns={
                        "film_id": ColumnMetadata(
                            name="film_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                        ),
                        "title": ColumnMetadata(
                            name="title",
                            data_type="varchar",
                            value_type="string",
                        ),
                        "language_id": ColumnMetadata(
                            name="language_id",
                            data_type="integer",
                            value_type="integer",
                            is_foreign_key=True,
                            fk_target=("language", "language_id"),
                        ),
                    },
                    foreign_keys=[],
                    primary_key="",
                ),
                "language": TableMetadata(
                    name="language",
                    columns={
                        "language_id": ColumnMetadata(
                            name="language_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                        ),
                        "name": ColumnMetadata(
                            name="name",
                            data_type="varchar",
                            value_type="string",
                        ),
                    },
                    foreign_keys=[],
                    primary_key="",
                ),
            },
            join_paths_multi={},
            effective_structural_hash="",
        )

    def test_language_kept_when_explicitly_in_select_cols(self, language_schema):
        """language table is NOT pruned when a non-aggregated language.name is in select_cols and language appears in the question."""
        intent = RuntimeIntent(
            tables=["film", "language"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("film.title")),
                SelectCol(expr=NormalizedExpr.from_column("language.name")),
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        result = reconcile_tables(intent)
        assert "film" in result.tables
        assert "language" in result.tables


class TestRepairIntentPlaceholderTokens:
    """Tests for repair_intent_placeholder_tokens and leak detection."""

    @staticmethod
    def _film_schema() -> SchemaGraph:
        """Minimal single-table graph for placeholder repair."""

        film = TableMetadata(
            name="film",
            columns={
                "film_id": ColumnMetadata(name="film_id", data_type="int", is_primary_key=True),
                "special_features": ColumnMetadata(name="special_features", data_type="ARRAY"),
            },
            primary_key=[],
            foreign_keys=[],
        )
        return SchemaGraph(
            tables={"film": film},
            join_paths_multi={},
            effective_structural_hash="x",
            created_at="",
        )

    def test_maps_table_1_prefix_for_single_table(self):
        """``table_1.col`` rewrites to the sole intent table."""

        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["table_1.special_features"], agg_func="count")],
        )
        intent = RuntimeIntent(
            tables=["film"],
            grain="row_level",
            select_cols=[SelectCol(expr=expr)],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("table_1.special_features"),
                    op="contains",
                    value_type="string",
                    raw_value="x",
                )
            ],
        )
        sg = self._film_schema()
        out = repair_intent_placeholder_tokens(intent, sg)
        assert out.select_cols[0].expr.primary_column == "film.special_features"
        assert out.filters_param[0].left_expr.primary_column == "film.special_features"

    def test_intent_text_detects_angle_placeholder(self):
        """Angle-bracket instructional tokens are detectable before repair."""

        assert intent_text_has_leakable_placeholder("<table_1>.col")
        assert not intent_text_has_leakable_placeholder("film.title")

    def test_runtime_intent_detects_table_n_in_select(self):
        """Structured scan finds instructional table_N tokens in select expr."""

        intent = RuntimeIntent(
            tables=["film"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("table_1.film_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        assert runtime_intent_has_instructional_placeholders(intent)

    def test_runtime_intent_clean_after_deterministic_placeholder_repair(self):
        """Deterministic placeholder repair clears instructional tokens for scan."""

        expr = NormalizedExpr(
            add_groups=[MulGroup(multiply=["table_1.film_id"], agg_func="count")],
        )
        intent = RuntimeIntent(
            tables=["film"],
            grain="row_level",
            select_cols=[SelectCol(expr=expr)],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        sg = self._film_schema()
        repaired = repair_intent_placeholder_tokens(intent, sg)
        assert not runtime_intent_has_instructional_placeholders(repaired)


class TestBestDescriptiveColumnsNullability:
    """Non-nullable descriptive columns rank ahead at equal cardinality."""

    def test_prefers_is_nullable_false(self):
        schema = SchemaGraph(
            tables={
                "t": TableMetadata(
                    name="t",
                    columns={
                        "a_name": ColumnMetadata(
                            name="a_name",
                            data_type="varchar",
                            value_type="string",
                            distinct_ratio=0.99,
                            is_nullable=True,
                        ),
                        "b_name": ColumnMetadata(
                            name="b_name",
                            data_type="varchar",
                            value_type="string",
                            distinct_ratio=0.99,
                            is_nullable=False,
                        ),
                    },
                    primary_key=[],
                    foreign_keys=[],
                ),
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )
        cols = best_descriptive_columns("t", schema, set(), max_count=1)
        assert cols == ["b_name"]


class TestDedupValueVsRightExpr:
    """Remove redundant ``right_expr`` when it matches the bound ``value``."""

    def test_drops_quoted_string_matching_raw_value(self):
        col = NormalizedExpr.from_column("orders.region")
        fp = FilterParam(
            left_expr=col,
            op="=",
            right_expr=parse_expr_string("'west'"),
            raw_value="west",
            value_type="string",
        )
        out, changed = _dedup_value_vs_right_expr_filters([fp])
        assert changed
        assert out[0].right_expr is None

    def test_drops_numeric_literal_matching_raw_value(self):
        col = NormalizedExpr.from_column("t.amount")
        fp = FilterParam(
            left_expr=col,
            op="=",
            right_expr=parse_expr_string("42"),
            raw_value=42,
            value_type="integer",
        )
        out, changed = _dedup_value_vs_right_expr_filters([fp])
        assert changed
        assert out[0].right_expr is None

    def test_keeps_qualified_column_right_even_when_value_matches_text(self):
        col = NormalizedExpr.from_column("t.amount")
        fp = FilterParam(
            left_expr=col,
            op="=",
            right_expr=NormalizedExpr.from_column("other.amount"),
            raw_value="other.amount",
            value_type="string",
        )
        out, changed = _dedup_value_vs_right_expr_filters([fp])
        assert not changed

    def test_end_to_end_main_and_cte(self):
        col = NormalizedExpr.from_column("t.x")
        fp_m = FilterParam(
            left_expr=col,
            op="=",
            right_expr=parse_expr_string("7"),
            raw_value=7,
            value_type="integer",
        )
        fp_c = FilterParam(
            left_expr=col,
            op="=",
            right_expr=parse_expr_string("8"),
            raw_value=8,
            value_type="integer",
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp_m],
            cte_steps=[
                RuntimeCteStep(
                    cte_name="c1",
                    tables=["t"],
                    filters_param=[fp_c],
                ),
            ],
        )
        result = dedup_value_vs_right_expr(intent)
        assert result.filters_param[0].right_expr is None
        assert result.cte_steps[0].filters_param[0].right_expr is None


class TestDedupContradictoryFilters:
    """Range filters on a column are dropped when an equality exists on that column."""

    def test_drops_range_when_equality_on_same_column(self):
        col = NormalizedExpr.from_column("orders.status")
        filters = [
            FilterParam(left_expr=col, op="=", raw_value="open", value_type="string"),
            FilterParam(left_expr=col, op=">", raw_value=0, value_type="integer"),
        ]
        out, changed = _dedup_contradictory_filters_list(filters)
        assert changed
        assert len(out) == 1
        assert out[0].op == "="

    def test_no_equality_leaves_ranges(self):
        col = NormalizedExpr.from_column("orders.amount")
        filters = [
            FilterParam(left_expr=col, op=">", raw_value=0, value_type="number"),
            FilterParam(left_expr=col, op="<", raw_value=100, value_type="number"),
        ]
        out, changed = _dedup_contradictory_filters_list(filters)
        assert not changed
        assert len(out) == 2

    def test_dedup_applies_to_main_and_cte(self):
        col = NormalizedExpr.from_column("t.x")
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(left_expr=col, op="=", raw_value=1, value_type="integer"),
                FilterParam(left_expr=col, op=">=", raw_value=0, value_type="integer"),
            ],
            cte_steps=[
                RuntimeCteStep(
                    cte_name="c1",
                    tables=["t"],
                    filters_param=[
                        FilterParam(left_expr=col, op="=", raw_value=2, value_type="integer"),
                        FilterParam(left_expr=col, op="<=", raw_value=9, value_type="integer"),
                    ],
                ),
            ],
        )
        result = dedup_contradictory_filters(intent)
        assert len(result.filters_param) == 1
        assert len(result.cte_steps[0].filters_param) == 1


class TestQualifyTermBoundaries:
    """_qualify_term must not match column names as substrings of identifiers."""

    def test_qualifies_standalone_token(self):
        assert _qualify_term("customer_id", {"customer_id": "cte1"}) == "cte1.customer_id"

    def test_does_not_match_inside_longer_identifier(self):
        assert _qualify_term("x_customer_id", {"customer_id": "cte1"}) == "x_customer_id"

    def test_case_insensitive_match(self):
        assert _qualify_term("Customer_ID * 2", {"customer_id": "cte1"}) == "cte1.customer_id * 2"


class TestQualifyCteOutputColumnsEdges:
    """Extra cases for qualify_cte_output_columns."""

    def test_no_cte_returns_same_object(self):
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        assert qualify_cte_output_columns(intent) is intent

    def test_skips_terms_already_prefixed_with_main_table(self):
        """Bare CTE output names are not forced onto main-qualified columns."""
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.customer_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[
                RuntimeCteStep(
                    cte_name="rollup",
                    tables=["orders"],
                    output_columns=["customer_id"],
                    select_cols=[],
                ),
            ],
        )
        out = qualify_cte_output_columns(intent)
        assert out.select_cols[0].expr.primary_column == "orders.customer_id"

    def test_qualifies_group_by_and_order_by(self):
        intent = RuntimeIntent(
            tables=[],
            grain="row_level",
            select_cols=[],
            group_by_cols=[NormalizedExpr.from_column("region_id")],
            order_by_cols=[OrderByCol(expr=NormalizedExpr.from_column("region_id"))],
            filters_param=[],
            cte_steps=[
                RuntimeCteStep(
                    cte_name="regions",
                    tables=["geo"],
                    select_cols=[SelectCol(expr=NormalizedExpr.from_column("geo.region_id"))],
                ),
            ],
        )
        out = qualify_cte_output_columns(intent)
        assert "regions." in out.group_by_cols[0].primary_column
        assert "regions." in out.order_by_cols[0].expr.primary_column

    def test_uses_explicit_output_columns_when_set(self):
        intent = RuntimeIntent(
            tables=[],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("sku"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[
                RuntimeCteStep(
                    cte_name="inv",
                    tables=["stock"],
                    output_columns=["SKU"],
                    select_cols=[SelectCol(expr=NormalizedExpr.from_column("stock.sku_code"))],
                ),
            ],
        )
        out = qualify_cte_output_columns(intent)
        assert "inv." in out.select_cols[0].expr.primary_column


class TestBestDescriptiveColumnsMaxCount:
    """best_descriptive_columns with max_count > 1 (non-composite path)."""

    def test_returns_two_top_columns_when_max_count_two(self):
        schema = SchemaGraph(
            tables={
                "person": TableMetadata(
                    name="person",
                    columns={
                        "id": ColumnMetadata(
                            name="id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                        ),
                        "email": ColumnMetadata(
                            name="email",
                            data_type="varchar",
                            value_type="string",
                            distinct_ratio=0.99,
                            distinct_count=500,
                            is_unique=True,
                        ),
                        "phone": ColumnMetadata(
                            name="phone",
                            data_type="varchar",
                            value_type="string",
                            distinct_ratio=0.99,
                            distinct_count=400,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="",
                ),
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )
        cols = best_descriptive_columns("person", schema, set(), max_count=2)
        assert cols == ["email", "phone"]


class TestExpandFkSelectToDescriptive:
    """expand_fk_select_to_descriptive replaces bare FK id selects."""

    @pytest.fixture
    def fk_customer_schema(self):
        return SchemaGraph(
            tables={
                "orders": TableMetadata(
                    name="orders",
                    columns={
                        "order_id": ColumnMetadata(
                            name="order_id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                        ),
                        "customer_id": ColumnMetadata(
                            name="customer_id",
                            data_type="integer",
                            value_type="integer",
                            is_foreign_key=True,
                            fk_target=("customer", "id"),
                        ),
                    },
                    foreign_keys=[],
                    primary_key="",
                ),
                "customer": TableMetadata(
                    name="customer",
                    columns={
                        "id": ColumnMetadata(
                            name="id",
                            data_type="integer",
                            value_type="integer",
                            is_primary_key=True,
                        ),
                        "full_name": ColumnMetadata(
                            name="full_name",
                            data_type="varchar",
                            value_type="string",
                            distinct_ratio=0.99,
                            distinct_count=200,
                        ),
                    },
                    foreign_keys=[],
                    primary_key="",
                ),
            },
            join_paths_multi={},
            effective_structural_hash="h",
        )

    def test_expands_fk_select_adds_descriptive_and_table(self, fk_customer_schema):
        intent = RuntimeIntent(
            tables=["orders"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.customer_id"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        out = expand_fk_select_to_descriptive(intent, fk_customer_schema)
        terms = [sc.expr.primary_column for sc in out.select_cols]
        assert "customer.full_name" in terms
        assert "orders.customer_id" not in terms
        assert "customer" in out.tables

    def test_keeps_aggregated_fk_unchanged(self, fk_customer_schema):
        agg = SelectCol(
            expr=NormalizedExpr(
                add_groups=[MulGroup(multiply=["orders.customer_id"], agg_func="count")],
            ),
        )
        intent = RuntimeIntent(
            tables=["orders"],
            grain="grouped",
            select_cols=[agg],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        out = expand_fk_select_to_descriptive(intent, fk_customer_schema)
        assert len(out.select_cols) == 1
        assert out.select_cols[0] is intent.select_cols[0]


class TestNormalizeNullFilterValues:
    """normalize_null_filter_values fixes value_type/raw_value for null operators."""

    def test_is_null_gets_null_type_and_none_value(self):
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[
                FilterParam(
                    left_expr=NormalizedExpr.from_column("t.x"),
                    op="is null",
                    value_type="string",
                    raw_value="unused",
                ),
            ],
        )
        out = normalize_null_filter_values(intent)
        fp = out.filters_param[0]
        assert fp.value_type == "null"
        assert fp.raw_value is None

    def test_cte_filters_normalized(self):
        intent = RuntimeIntent(
            tables=[],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            cte_steps=[
                RuntimeCteStep(
                    cte_name="c1",
                    tables=["t"],
                    filters_param=[
                        FilterParam(
                            left_expr=NormalizedExpr.from_column("t.x"),
                            op="is not null",
                            value_type="integer",
                            raw_value=0,
                        ),
                    ],
                ),
            ],
        )
        out = normalize_null_filter_values(intent)
        fp = out.cte_steps[0].filters_param[0]
        assert fp.value_type == "null"
        assert fp.raw_value is None


class TestRepairCaseWhenIntent:
    """repair_case_when_intent removes empty CASE registry rows."""

    def test_drops_empty_branches(self):
        cw = CaseWhenExpr(branches=[], else_result=NormalizedExpr.from_column("t.default"))
        step = CaseRegistryStep(registry_id="c01", case_when=cw)
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("c01"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            case_registry=[step],
        )
        sg = SchemaGraph(tables={}, join_paths_multi={}, effective_structural_hash="")
        out = repair_case_when_intent(intent, sg)
        assert out.case_registry == []

    def test_keeps_non_empty_case(self):
        branch = CaseWhenBranch(
            condition=FilterParam(
                left_expr=NormalizedExpr.from_column("t.flag"),
                op="=",
                raw_value=1,
                value_type="integer",
            ),
            result=NormalizedExpr.from_column("t.a"),
        )
        cw = CaseWhenExpr(branches=[branch], else_result=NormalizedExpr.from_column("t.b"))
        step = CaseRegistryStep(registry_id="c01", case_when=cw)
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("c01"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            case_registry=[step],
        )
        sg = SchemaGraph(tables={}, join_paths_multi={}, effective_structural_hash="")
        out = repair_case_when_intent(intent, sg)
        assert out.case_registry[0].case_when is not None
        assert len(out.case_registry[0].case_when.branches) == 1


class TestRepairArrayFiltersIntent:
    """repair_array_filters_intent normalises array-column predicates."""

    @pytest.fixture
    def array_schema(self):
        tags = ColumnMetadata(
            name="tags",
            data_type="text[]",
            value_type="string",
            element_type="text",
        )
        title = ColumnMetadata(name="title", data_type="varchar", value_type="string")
        t = TableMetadata(
            name="article",
            columns={
                "tags": tags,
                "title": title,
                "cnt": ColumnMetadata(name="cnt", data_type="integer", value_type="integer"),
            },
            foreign_keys=[],
            primary_key="",
        )
        return SchemaGraph(tables={"article": t}, join_paths_multi={}, effective_structural_hash="h")

    def test_rewrites_equals_to_contains_on_array_column(self, array_schema):
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("article.tags"),
            op="=",
            raw_value="x",
            value_type="string",
            param_key="p1",
        )
        intent = RuntimeIntent(
            tables=["article"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
        )
        out = repair_array_filters_intent(intent, array_schema)
        assert out.filters_param[0].op == "contains"
        assert out.filters_param[0].value_type == "string"

    def test_rewrites_contains_to_like_on_text_column(self, array_schema):
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("article.title"),
            op="contains",
            raw_value="hello",
            value_type="string",
            param_key="p2",
        )
        intent = RuntimeIntent(
            tables=["article"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
        )
        out = repair_array_filters_intent(intent, array_schema)
        assert len(out.filters_param) == 1
        assert out.filters_param[0].op == "like"
        assert out.filters_param[0].raw_value == "%hello%"

    def test_drops_contains_on_non_text_scalar(self, array_schema):
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("article.cnt"),
            op="contains",
            raw_value=1,
            value_type="integer",
            param_key="p3",
        )
        intent = RuntimeIntent(
            tables=["article"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
        )
        out = repair_array_filters_intent(intent, array_schema)
        assert out.filters_param == []

    def test_keeps_contains_on_array_column(self, array_schema):
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("article.tags"),
            op="contains",
            raw_value="x",
            value_type="string",
        )
        intent = RuntimeIntent(
            tables=["article"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
        )
        out = repair_array_filters_intent(intent, array_schema)
        assert len(out.filters_param) == 1
        assert out.filters_param[0].op == "contains"

    def test_ilike_rewrites_to_contains(self, array_schema):
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("article.tags"),
            op="ilike",
            raw_value="%x%",
            value_type="string",
        )
        intent = RuntimeIntent(
            tables=["article"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
        )
        out = repair_array_filters_intent(intent, array_schema)
        assert out.filters_param[0].op == "contains"


class TestDecomposeInNotInFilters:
    """decompose_in_not_in_filters expands short IN/NOT IN lists."""

    def test_in_list_becomes_or_chain(self):
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.id"),
            op="in",
            raw_value=[1, 2, 3],
            value_type="integer",
            bool_op="AND",
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
        )
        out = decompose_in_not_in_filters(intent)
        assert len(out.filters_param) == 3
        assert out.filters_param[0].op == "="
        assert out.filters_param[0].raw_value == 1
        assert out.filters_param[0].bool_op == "OR"
        assert out.filters_param[1].bool_op == "OR"
        assert out.filters_param[2].bool_op == "AND"

    def test_not_in_list_becomes_and_chain_of_neq(self):
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.code"),
            op="not in",
            raw_value=["a", "b"],
            value_type="string",
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
        )
        out = decompose_in_not_in_filters(intent)
        assert len(out.filters_param) == 2
        assert out.filters_param[0].op == "!="
        assert out.filters_param[1].op == "!="
        assert out.filters_param[0].bool_op == "AND"
        assert out.filters_param[1].bool_op == "AND"

    def test_in_two_elements_or_between_last_keeps_chain_bool_op(self):
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.id"),
            op="in",
            raw_value=[10, 20],
            value_type="integer",
            bool_op="AND",
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
        )
        out = decompose_in_not_in_filters(intent)
        assert len(out.filters_param) == 2
        assert out.filters_param[0].bool_op == "OR"
        assert out.filters_param[1].bool_op == "AND"

    def test_not_in_three_elements_and_chain_last_bool_op(self):
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.code"),
            op="not in",
            raw_value=["a", "b", "c"],
            value_type="string",
            bool_op="OR",
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
        )
        out = decompose_in_not_in_filters(intent)
        assert len(out.filters_param) == 3
        assert out.filters_param[0].bool_op == "AND"
        assert out.filters_param[1].bool_op == "AND"
        assert out.filters_param[2].bool_op == "OR"

    def test_long_list_not_expanded(self):
        vals = list(range(11))
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.id"),
            op="in",
            raw_value=vals,
            value_type="integer",
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[fp],
        )
        out = decompose_in_not_in_filters(intent)
        assert len(out.filters_param) == 1
        assert out.filters_param[0].raw_value == vals


class TestIntentTextHasLeakablePlaceholderPatterns:
    """Additional patterns for intent_text_has_leakable_placeholder."""

    def test_numeric_table_prefix_patterns(self):
        assert intent_text_has_leakable_placeholder("table_1.id")
        assert intent_text_has_leakable_placeholder("table2.name")

    def test_column_placeholder_tokens(self):
        assert intent_text_has_leakable_placeholder("foo column_1 bar")
        assert intent_text_has_leakable_placeholder("x col5 y")

    def test_clean_sql_like_text(self):
        assert not intent_text_has_leakable_placeholder("public.users.email")


class TestInferCteOutputColumnsEdges:
    """More infer_cte_output_columns edge cases."""

    def test_skips_rows_without_primary_column(self):
        cte = RuntimeCteStep(
            cte_name="c",
            tables=["t"],
            select_cols=[SelectCol(expr=NormalizedExpr())],
        )
        assert infer_cte_output_columns(cte) == []

    def test_is_aggregated_without_agg_func_uses_bare_name(self):
        cte = RuntimeCteStep(
            cte_name="c",
            tables=["t"],
            select_cols=[
                SelectCol(
                    expr=NormalizedExpr(add_groups=[MulGroup(multiply=["COUNT(t.amount)"])]),
                ),
            ],
        )
        assert infer_cte_output_columns(cte) == ["amount"]


class TestIsImpossibleHavingEdges:
    """Extra branches for _is_impossible_having."""

    def test_count_in_primary_term_triggers_numeric_rules(self):
        hp = HavingParam(
            left_expr=NormalizedExpr.from_column("count(*)"),
            op="<",
            raw_value=0,
            value_type="integer",
        )
        assert _is_impossible_having(hp) is True

    def test_invalid_numeric_string_not_impossible(self):
        hp = HavingParam(
            left_expr=NormalizedExpr(
                add_groups=[MulGroup(multiply=["t.id"], agg_func="COUNT")],
            ),
            op="<",
            raw_value="nope",
            value_type="string",
        )
        assert _is_impossible_having(hp) is False

    def test_less_than_positive_threshold_possible(self):
        hp = HavingParam(
            left_expr=NormalizedExpr(
                add_groups=[MulGroup(multiply=["t.id"], agg_func="COUNT")],
            ),
            op="<",
            raw_value=5,
            value_type="integer",
        )
        assert _is_impossible_having(hp) is False


class TestResolveFilterListCascadeEdges:
    """Branches in _resolve_filter_list_cascade."""

    def test_unqualified_column_left_unchanged(self):
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("id"),
            op="=",
            raw_value="X",
            value_type="string",
        )
        sg = SchemaGraph(
            tables={
                "t": TableMetadata(
                    name="t",
                    columns={
                        "id": ColumnMetadata(
                            name="id",
                            data_type="varchar",
                            value_type="string",
                        ),
                    },
                    foreign_keys=[],
                    primary_key="",
                ),
            },
            join_paths_multi={},
            effective_structural_hash="",
        )
        out, changed = _resolve_filter_list_cascade([fp], sg, "")
        assert not changed
        assert out[0].raw_value == "X"

    def test_non_string_scalar_skipped(self):
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("t.id"),
            op="=",
            raw_value=42,
            value_type="integer",
        )
        t = TableMetadata(
            name="t",
            columns={
                "id": ColumnMetadata(
                    name="id",
                    data_type="integer",
                    value_type="integer",
                ),
            },
            foreign_keys=[],
            primary_key="",
        )
        sg = SchemaGraph(tables={"t": t}, join_paths_multi={}, effective_structural_hash="")
        out, changed = _resolve_filter_list_cascade([fp], sg, "")
        assert not changed

    def test_list_mixed_enum_and_lower(self):
        film = TableMetadata(
            name="film",
            columns={
                "rating": ColumnMetadata(
                    name="rating",
                    data_type="mpaa_rating",
                    value_type="string",
                ),
            },
            foreign_keys=[],
            primary_key="",
        )
        sg = SchemaGraph(
            tables={"film": film},
            join_paths_multi={},
            effective_structural_hash="",
            enum_values={"mpaa_rating": ["G", "PG"]},
        )
        fp = FilterParam(
            left_expr=NormalizedExpr.from_column("film.rating"),
            op="in",
            value_type="string",
            raw_value=["pg", "G", "Other"],
        )
        out, changed = _resolve_filter_list_cascade([fp], sg, "")
        assert changed
        assert out[0].raw_value == ["PG", "G", "other"]


class TestFlipComparisonOp:
    """Tests for _flip_comparison_op."""

    def test_gt_flipped_to_lt(self):
        from aetherdialect._intent_repair import _flip_comparison_op

        assert _flip_comparison_op(">") == "<"

    def test_le_flipped_to_ge(self):
        from aetherdialect._intent_repair import _flip_comparison_op

        assert _flip_comparison_op("<=") == ">="

    def test_unmapped_op_returned_unchanged(self):
        from aetherdialect._intent_repair import _flip_comparison_op

        assert _flip_comparison_op("like") == "like"


class TestAutoRepairFilterHaving:
    """Tests for auto_repair_filter_having."""

    def test_move_agg_filter_to_having(self):
        """Move aggregated filter to having list."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_agg("count", "t.a"),
            op=">",
            value_type="integer",
            param_key="p1",
        )
        gb = [NormalizedExpr.from_column("t.a")]
        repaired_fp, repaired_hp = auto_repair_filter_having([fp], [], group_by_cols=gb)
        assert len(repaired_fp) == 0
        assert len(repaired_hp) == 1
        assert repaired_hp[0].param_key == "p1"

    def test_move_nonagg_having_to_filter(self):
        """Move non-aggregated having to filter list."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_column("t.a"),
            op="=",
            value_type="string",
            param_key="p1",
        )
        repaired_fp, repaired_hp = auto_repair_filter_having([], [hp], group_by_cols=[])
        assert len(repaired_fp) == 1
        assert len(repaired_hp) == 0
        assert repaired_fp[0].param_key == "p1"

    def test_agg_filter_not_promoted_without_group_by(self):
        """Aggregated WHERE filter stays in WHERE when there is no GROUP BY."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_agg("count", "t.a"),
            op=">",
            value_type="integer",
            param_key="p1",
        )
        repaired_fp, repaired_hp = auto_repair_filter_having([fp], [], group_by_cols=[])
        assert len(repaired_fp) == 1
        assert len(repaired_hp) == 0

    def test_agg_filter_not_promoted_with_non_numeric_op(self):
        """Aggregated filter with LIKE is not promoted to HAVING."""
        fp = FilterParam(
            left_expr=NormalizedExpr.from_agg("count", "t.a"),
            op="like",
            value_type="string",
            param_key="p1",
        )
        gb = [NormalizedExpr.from_column("t.a")]
        repaired_fp, repaired_hp = auto_repair_filter_having([fp], [], group_by_cols=gb)
        assert len(repaired_fp) == 1
        assert len(repaired_hp) == 0

    def test_keep_correct_placement(self):
        """Keep correctly placed filter and having."""
        fp = FilterParam(left_expr=NormalizedExpr.from_column("t.a"), op="=", value_type="string")
        hp = HavingParam(
            left_expr=NormalizedExpr.from_agg("count", "t.a"),
            op=">",
            value_type="integer",
        )
        gb = [NormalizedExpr.from_column("t.a")]
        repaired_fp, repaired_hp = auto_repair_filter_having([fp], [hp], group_by_cols=gb)
        assert len(repaired_fp) == 1
        assert len(repaired_hp) == 1

    def test_empty_inputs(self):
        """Handle empty input lists."""
        repaired_fp, repaired_hp = auto_repair_filter_having([], [])
        assert repaired_fp == []
        assert repaired_hp == []

    def test_flip_having_when_right_has_agg(self):
        """Flip and keep in HAVING when left lacks agg but right has it."""
        hp = HavingParam(
            left_expr=NormalizedExpr.from_column("cte.avg_col"),
            op=">",
            right_expr=NormalizedExpr.from_agg("avg", "t.col"),
            param_key="p1",
        )
        gb = [NormalizedExpr.from_column("cte.avg_col")]
        repaired_fp, repaired_hp = auto_repair_filter_having([], [hp], group_by_cols=gb)
        assert len(repaired_fp) == 0
        assert len(repaired_hp) == 1
        has_agg_left = repaired_hp[0].left_expr.has_aggregation
        has_agg_right = repaired_hp[0].right_expr is not None and repaired_hp[0].right_expr.has_aggregation
        assert has_agg_left or has_agg_right
        assert not (has_agg_left and has_agg_right)


class TestCaseBranchRepairCoverage:
    """Verify repairs walk into CASE WHEN branch conditions in registry layouts."""

    def _make_intent_with_registry_case_between(self) -> RuntimeIntent:
        cond = FilterParam(
            left_expr=NormalizedExpr.from_column("t.amount"),
            op="between",
            raw_value=[10, 20],
            value_type="integer",
            bool_op="AND",
        )
        cw = CaseWhenExpr(
            branches=[
                CaseWhenBranch(
                    condition=cond,
                    result=NormalizedExpr.from_column("t.id"),
                ),
            ],
            else_result=NormalizedExpr.from_column("t.id"),
            condition_scope="filter",
        )
        step = CaseRegistryStep(registry_id="c01", case_when=cw)
        sc = SelectCol(expr=NormalizedExpr.from_column("c01"))
        return RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[sc],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            case_registry=[step],
        )

    def test_decompose_between_normalises_registry_case_branch_bounds(self):
        from aetherdialect._intent_expr import decompose_between_params

        intent = self._make_intent_with_registry_case_between()
        intent.case_registry[0].case_when.branches[0].condition.raw_value = "10 AND 20"
        out = decompose_between_params(intent)
        rv = out.case_registry[0].case_when.branches[0].condition.raw_value
        assert isinstance(rv, list) and rv == [10, 20] or rv == ["10", "20"]

    def test_apply_filters_to_main_and_ctes_walks_registry_case_filter_branch(self):
        from aetherdialect._intent_repair import _apply_filters_to_main_and_ctes

        intent = self._make_intent_with_registry_case_between()
        intent.case_registry[0].case_when.branches[0].condition.raw_value = " HELLO "

        def lower_strip(filters: list[FilterParam]) -> tuple[list[FilterParam], bool]:
            changed = False
            out: list[FilterParam] = []
            for f in filters:
                if isinstance(f.raw_value, str) and f.raw_value != f.raw_value.strip().lower():
                    changed = True
                    from dataclasses import replace as dc_replace

                    out.append(dc_replace(f, raw_value=f.raw_value.strip().lower()))
                else:
                    out.append(f)
            return out, changed

        out = _apply_filters_to_main_and_ctes(intent, lower_strip)
        assert out.case_registry[0].case_when.branches[0].condition.raw_value == "hello"

    def test_apply_having_to_main_and_ctes_walks_having_scope_branch_only(self):
        from aetherdialect._intent_repair import _apply_having_to_main_and_ctes

        cond = FilterParam(
            left_expr=NormalizedExpr.from_agg("count", "t.id"),
            op=">",
            raw_value=5,
            value_type="integer",
            bool_op="AND",
        )
        cw_having = CaseWhenExpr(
            branches=[CaseWhenBranch(condition=cond, result=NormalizedExpr.from_column("t.id"))],
            else_result=NormalizedExpr.from_column("t.id"),
            condition_scope="having",
        )
        cw_filter = CaseWhenExpr(
            branches=[CaseWhenBranch(condition=cond, result=NormalizedExpr.from_column("t.id"))],
            else_result=NormalizedExpr.from_column("t.id"),
            condition_scope="filter",
        )
        intent = RuntimeIntent(
            tables=["t"],
            grain="row_level",
            select_cols=[
                SelectCol(expr=NormalizedExpr.from_column("c01")),
                SelectCol(expr=NormalizedExpr.from_column("c02")),
            ],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            having_param=[],
            case_registry=[
                CaseRegistryStep(registry_id="c01", case_when=cw_having),
                CaseRegistryStep(registry_id="c02", case_when=cw_filter),
            ],
        )
        seen: list[int] = []

        def bump_value(hp_list: list[HavingParam]) -> tuple[list[HavingParam], bool]:
            from dataclasses import replace as dc_replace

            new = []
            changed = False
            for h in hp_list:
                seen.append(int(h.raw_value))
                new.append(dc_replace(h, raw_value=int(h.raw_value) + 1))
                changed = True
            return new, changed

        out = _apply_having_to_main_and_ctes(intent, bump_value)
        having_branch_val = out.case_registry[0].case_when.branches[0].condition.raw_value
        filter_branch_val = out.case_registry[1].case_when.branches[0].condition.raw_value
        assert having_branch_val == 6
        assert filter_branch_val == 5


def _two_table_schema() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "customer": TableMetadata(
                name="customer",
                columns={
                    "customer_id": ColumnMetadata(
                        name="customer_id",
                        data_type="integer",
                        value_type="integer",
                        is_primary_key=True,
                    ),
                    "email": ColumnMetadata(
                        name="email",
                        data_type="varchar",
                        value_type="string",
                    ),
                    "first_name": ColumnMetadata(
                        name="first_name",
                        data_type="varchar",
                        value_type="string",
                    ),
                },
                primary_key=["customer_id"],
                foreign_keys=[],
            ),
            "rental": TableMetadata(
                name="rental",
                columns={
                    "rental_id": ColumnMetadata(
                        name="rental_id",
                        data_type="integer",
                        value_type="integer",
                        is_primary_key=True,
                    ),
                    "email": ColumnMetadata(
                        name="email",
                        data_type="varchar",
                        value_type="string",
                    ),
                    "rental_date": ColumnMetadata(
                        name="rental_date",
                        data_type="timestamp",
                        value_type="datetime",
                    ),
                },
                primary_key=["rental_id"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
        effective_structural_hash="h",
    )


class TestDiagnosticRepairDispatch:
    """Tests for B.3 diagnostic-driven structural repair primitives."""

    def test_dispatch_table_covers_actionable_codes(self) -> None:
        expected = {
            SqlDiagnosticCode.UNKNOWN_COLUMN,
            SqlDiagnosticCode.AMBIGUOUS_COLUMN,
            SqlDiagnosticCode.UNKNOWN_TABLE,
            SqlDiagnosticCode.NON_GROUPED_SELECT_COL,
            SqlDiagnosticCode.AGG_IN_WHERE,
            SqlDiagnosticCode.EXPLAIN_CARTESIAN_JOIN,
            SqlDiagnosticCode.EXPLAIN_ZERO_ESTIMATE,
            SqlDiagnosticCode.PARAM_UNBOUND,
        }
        assert set(DIAGNOSTIC_REPAIR_DISPATCH.keys()) == expected

    def test_repair_unknown_column_qualified_fuzzy_swap(self) -> None:
        schema = _two_table_schema()
        intent = RuntimeIntent(
            tables=["customer"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("customer.emial"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        diag = SqlDiagnostic(
            code=SqlDiagnosticCode.UNKNOWN_COLUMN,
            message="unknown column 'customer.emial'",
            offending_identifier="customer.emial",
        )
        out, changed = apply_diagnostic_repairs(intent, schema, [diag])
        assert changed is True
        assert out.select_cols[0].expr.primary_column == "customer.email"

    def test_repair_unknown_column_bare_qualifies_to_owner(self) -> None:
        schema = _two_table_schema()
        intent = RuntimeIntent(
            tables=["customer"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("frist_name"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        diag = SqlDiagnostic(
            code=SqlDiagnosticCode.UNKNOWN_COLUMN,
            message="unknown column 'frist_name'",
            offending_identifier="frist_name",
        )
        out, changed = apply_diagnostic_repairs(intent, schema, [diag])
        assert changed is True
        assert out.select_cols[0].expr.primary_column == "customer.first_name"

    def test_repair_ambiguous_column_qualifies_with_first_owner_in_intent(self) -> None:
        schema = _two_table_schema()
        intent = RuntimeIntent(
            tables=["rental", "customer"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("email"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        diag = SqlDiagnostic(
            code=SqlDiagnosticCode.AMBIGUOUS_COLUMN,
            message="ambiguous column 'email'",
            offending_identifier="email",
            details={"owners": "customer,rental"},
        )
        out, changed = apply_diagnostic_repairs(intent, schema, [diag])
        assert changed is True
        assert out.select_cols[0].expr.primary_column == "rental.email"

    def test_repair_unknown_table_fuzzy_swap_and_retarget(self) -> None:
        schema = _two_table_schema()
        intent = RuntimeIntent(
            tables=["custmer"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("custmer.email"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        diag = SqlDiagnostic(
            code=SqlDiagnosticCode.UNKNOWN_TABLE,
            message="unknown table 'custmer'",
            offending_identifier="custmer",
        )
        out, changed = apply_diagnostic_repairs(intent, schema, [diag])
        assert changed is True
        assert out.tables == ["customer"]
        assert out.select_cols[0].expr.primary_column == "customer.email"

    def test_repair_grain_consistency_adds_to_group_by(self) -> None:
        schema = _two_table_schema()
        intent = RuntimeIntent(
            tables=["customer"],
            grain="grouped",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("customer.email"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        diag = SqlDiagnostic(
            code=SqlDiagnosticCode.NON_GROUPED_SELECT_COL,
            message="non-grouped select col 'customer.email'",
            offending_identifier="customer.email",
        )
        out, changed = apply_diagnostic_repairs(intent, schema, [diag])
        assert changed is True
        assert len(out.group_by_cols) == 1
        assert out.group_by_cols[0].primary_column == "customer.email"

    def test_repair_grain_consistency_no_op_when_already_grouped(self) -> None:
        schema = _two_table_schema()
        existing = NormalizedExpr.from_column("customer.email")
        intent = RuntimeIntent(
            tables=["customer"],
            grain="grouped",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("customer.email"))],
            group_by_cols=[existing],
            order_by_cols=[],
            filters_param=[],
        )
        diag = SqlDiagnostic(
            code=SqlDiagnosticCode.NON_GROUPED_SELECT_COL,
            message="non-grouped",
            offending_identifier="customer.email",
        )
        out, changed = apply_diagnostic_repairs(intent, schema, [diag])
        assert changed is False
        assert len(out.group_by_cols) == 1

    def test_repair_agg_in_where_promotes_to_having(self) -> None:
        schema = _two_table_schema()
        agg_filter = FilterParam(
            left_expr=NormalizedExpr.from_agg("count", "customer.customer_id"),
            op=">",
            value_type="number",
            param_key="cnt",
            raw_value=10,
        )
        intent = RuntimeIntent(
            tables=["customer"],
            grain="grouped",
            select_cols=[SelectCol(expr=NormalizedExpr.from_agg("count", "customer.customer_id"))],
            group_by_cols=[NormalizedExpr.from_column("customer.email")],
            order_by_cols=[],
            filters_param=[agg_filter],
            having_param=[],
        )
        diag = SqlDiagnostic(
            code=SqlDiagnosticCode.AGG_IN_WHERE,
            message="aggregate in WHERE",
            offending_identifier=None,
        )
        out, changed = apply_diagnostic_repairs(intent, schema, [diag])
        assert changed is True
        assert out.filters_param == []
        assert len(out.having_param) == 1
        assert out.having_param[0].param_key == "cnt"

    def test_repair_cartesian_clears_chosen_join(self) -> None:
        schema = _two_table_schema()
        intent = RuntimeIntent(
            tables=["customer", "rental"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("customer.email"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
            chosen_join_candidate_id="J01",
            chosen_join_path_signature=["customer:rental"],
        )
        diag = SqlDiagnostic(
            code=SqlDiagnosticCode.EXPLAIN_CARTESIAN_JOIN,
            message="cartesian join",
            offending_identifier=None,
        )
        out, changed = apply_diagnostic_repairs(intent, schema, [diag])
        assert changed is True
        assert out.chosen_join_candidate_id == ""
        assert out.chosen_join_path_signature == []

    def test_repair_param_binding_drops_unbound_filter(self) -> None:
        schema = _two_table_schema()
        bound_filter = FilterParam(
            left_expr=NormalizedExpr.from_column("customer.email"),
            op="=",
            value_type="string",
            param_key="email_v",
            raw_value="alice@example.com",
        )
        unbound_filter = FilterParam(
            left_expr=NormalizedExpr.from_column("customer.first_name"),
            op="=",
            value_type="string",
            param_key="name_v",
            raw_value=None,
        )
        intent = RuntimeIntent(
            tables=["customer"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("customer.email"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[bound_filter, unbound_filter],
        )
        diag = SqlDiagnostic(
            code=SqlDiagnosticCode.PARAM_UNBOUND,
            message="unbound parameter",
            offending_identifier="name_v",
        )
        out, changed = apply_diagnostic_repairs(intent, schema, [diag])
        assert changed is True
        assert len(out.filters_param) == 1
        assert out.filters_param[0].param_key == "email_v"

    def test_apply_diagnostic_repairs_skips_soft_codes(self) -> None:
        schema = _two_table_schema()
        intent = RuntimeIntent(
            tables=["customer"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("customer.email"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        diag = SqlDiagnostic(
            code=SqlDiagnosticCode.EXPLAIN_SEQ_SCAN_INDEXED,
            message="seq scan on indexed",
            offending_identifier=None,
        )
        out, changed = apply_diagnostic_repairs(intent, schema, [diag])
        assert changed is False
        assert out is intent

    def test_apply_diagnostic_repairs_per_code_cap(self) -> None:
        schema = _two_table_schema()
        intent = RuntimeIntent(
            tables=["customer"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("customer.emial"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        diag = SqlDiagnostic(
            code=SqlDiagnosticCode.UNKNOWN_COLUMN,
            message="unknown column",
            offending_identifier="customer.emial",
        )
        out, changed = apply_diagnostic_repairs(intent, schema, [diag, diag, diag])
        assert changed is True
        assert out.select_cols[0].expr.primary_column == "customer.email"

    def test_apply_diagnostic_repairs_unknown_code_is_no_op(self) -> None:
        schema = _two_table_schema()
        intent = RuntimeIntent(
            tables=["customer"],
            grain="row_level",
            select_cols=[SelectCol(expr=NormalizedExpr.from_column("customer.email"))],
            group_by_cols=[],
            order_by_cols=[],
            filters_param=[],
        )
        diag = SqlDiagnostic(
            code=SqlDiagnosticCode.SUBQUERY_NOT_ALLOWED,
            message="subquery present",
            offending_identifier=None,
        )
        out, changed = apply_diagnostic_repairs(intent, schema, [diag])
        assert changed is False
        assert out is intent


def test_reconcile_tables_keeps_table_refs_from_raw_sql(
    schema_graph: SchemaGraph,
) -> None:
    intent = RuntimeIntent(
        tables=["orders"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr(raw_sql="DISTINCT customers.name"))],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
    )
    result = reconcile_tables(intent)
    assert "customers" in result.tables
    assert "orders" not in result.tables


def test_align_filter_coerces_boolean_bindings_to_int_for_integer_column() -> None:
    schema = SchemaGraph(
        tables={
            "customer": TableMetadata(
                name="customer",
                columns={
                    "active": ColumnMetadata(
                        name="active",
                        data_type="integer",
                        value_type="integer",
                        is_primary_key=False,
                        sensitivity="public",
                    ),
                },
                primary_key=[],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
        effective_structural_hash="h",
    )
    fp = FilterParam(
        left_expr=NormalizedExpr.from_column("customer.active"),
        op="=",
        value_type="boolean",
        param_key="p1",
    )
    intent = RuntimeIntent(
        tables=["customer"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("customer.customer_id"))],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[fp],
        param_values={"p1": True},
    )
    out = align_filter_value_type_to_exprs(intent, schema)
    assert out.filters_param[0].value_type == "number"
    assert out.param_values["p1"] == 1
