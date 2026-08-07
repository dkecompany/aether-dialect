"""Shared fixtures for aetherdialect test suite."""

import os
from pathlib import Path
from typing import Any

import pytest

from aetherdialect._config import DuckDBRuntimeConfig
from aetherdialect._contracts_base import (
    ColumnRole,
    EngineIdentity,
    NormalizedExpr,
    PredicateGroup,
    TableRole,
    WhereParam,
)
from aetherdialect._contracts_core import (
    RuntimeIntent,
    SelectCol,
    Template,
    ValueHistory,
)
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    FKEdge,
    SchemaGraph,
    SQLShape,
    TableMetadata,
    TemplateStats,
)
from aetherdialect._schema_catalog import assign_column_ops


def duckdb_engine_identity() -> EngineIdentity:
    """DuckDB identity for federation runtime binding tests."""
    return EngineIdentity("duckdb", DuckDBRuntimeConfig)


@pytest.fixture(autouse=True)
def _default_engine_identity() -> Any:
    """Bind a default engine identity for tests that call pipeline helpers directly."""
    from aetherdialect._config import EngineConfig
    from aetherdialect._core_utils import pop_engine_identity, push_engine_identity

    token = push_engine_identity(EngineIdentity(EngineConfig.TYPE, EngineConfig.RUNTIME))
    yield
    pop_engine_identity(token)


@pytest.fixture
def unbound_engine_identity() -> Any:
    """Clear the autouse engine identity so unbound-path assertions can run."""
    from aetherdialect._core_utils import _ACTIVE_ENGINE_IDENTITY

    token = _ACTIVE_ENGINE_IDENTITY.set(None)
    yield
    _ACTIVE_ENGINE_IDENTITY.reset(token)


def _term_str(term: Any) -> str:
    """Render a multiply/divide leaf as the legacy display string ('*', 'COUNT(*)', 'tbl.col', etc.)."""
    if isinstance(term, str):
        return term
    if not isinstance(term, NormalizedExpr):
        return str(term)
    if term.column_ref:
        return term.column_ref
    if term.star:
        return "*"
    if term.keyword:
        return term.keyword.upper()
    if term.raw_sql:
        return term.raw_sql
    from aetherdialect._sql_gen import render_expr_sql

    return render_expr_sql(term)


def term_strs(terms: Any) -> list[str]:
    """Map a sequence of multiply/divide leaves to legacy display strings."""
    return [_term_str(t) for t in (terms or [])]


_SLOW_SANDBOX_TESTS = frozenset(
    {
        "test_tour_question",
        "test_playground_question",
        "test_paraphrase_question",
        "test_catalog_paraphrase_hits_direct_reuse_after_accept",
        "test_rental_shop_category_film_emits_non_empty_join_path",
    }
)

_SLOW_SANDBOX_MODULES = frozenset(
    {
        "test_sandbox_scenarios.py",
        "test_sandbox_recording_isolation.py",
    }
)

_SLOW_SANDBOX_OFFLINE_TESTS = frozenset(
    {
        "test_sandbox_question",
        "test_validation_failure_question",
        "test_practice_questions_have_mock_fixture_coverage",
        "test_fixture_corpus_covers_consumer_reader_practice",
        "test_assert_sandbox_complete_with_corpus",
        "test_feedback_samples_rejection_flow",
        "test_rank_films_question",
        "test_direct_reuse_2025_2026_pair",
        "test_recipe_chat_basics_produces_sql",
        "test_recipe_rejections_completes",
        "test_template_reuse_second_question",
        "test_reader_writer_queue",
        "test_recipe_validation_failures",
        "test_recipe_full_session_completes",
        "test_aetherspace_demo",
        "test_restricted_consumer_permission_denied",
        "test_deny_columns_column_security",
        "test_consumer_reader_preset_first_question",
        "test_owner_bundled_overrides",
        "test_consumer_override_proposal_only",
        "test_session_active_error",
    }
)

_FAST_TEST_MAX_SECS = 25.0

_SANDBOX_BUNDLE_MODULES = frozenset(
    {
        "test_sandbox_offline.py",
        "test_sandbox_scenarios.py",
        "test_sandbox_corpus_helpers.py",
        "test_sandbox_corpus_validate.py",
        "test_sandbox_recording_isolation.py",
        "test_sandbox_recording_toml.py",
        "test_sandbox_recording_trace.py",
        "test_sandbox_tail_capture.py",
        "test_aetherspace.py",
        "test_doc_examples_smoke.py",
    }
)

_EXECUTE_STEP_SANDBOX_TESTS = frozenset(
    {
        "test_happy_path_returns_rows",
        "test_forbidden_statement_rejected_before_execute",
        "test_consumer_out_of_scope_blocked",
    }
)

_SANDBOX_AUTHORING_NO_CORPUS_TESTS = frozenset(
    {
        "test_unrecorded_question_in_recorded_corpus_mode_names_mode",
        "test_sandbox_guide_documents_engine_session_mode_as_primary_entry",
        "test_api_reference_documents_sandbox_authoring_surface",
    }
)


def _is_slow_sandbox_test(item: pytest.Item) -> bool:
    nodeid = item.nodeid.replace("\\", "/")
    for mod in _SLOW_SANDBOX_MODULES:
        if mod in nodeid:
            return True
    base_name = item.name.split("[", 1)[0]
    if base_name in _SLOW_SANDBOX_TESTS:
        return True
    fspath = getattr(item, "path", None) or getattr(item, "fspath", None)
    mod = fspath.name if fspath is not None else ""
    if mod == "test_sandbox_offline.py":
        return base_name in _SLOW_SANDBOX_OFFLINE_TESTS
    return False


def _is_live_test(item: pytest.Item) -> bool:
    """Return True when the test lives under ``live_tests/`` (real engine connections)."""
    fspath = getattr(item, "path", None) or getattr(item, "fspath", None)
    if fspath is None:
        return "live_tests/" in item.nodeid.replace("\\", "/")
    parts = fspath.parts
    return "live_tests" in parts


def _requires_corpus_bundle(item: pytest.Item) -> bool:
    """Return True when the test needs bundled ``sandbox/data.zip`` (``offline_sandbox`` corpus)."""
    if item.get_closest_marker("needs_corpus") is not None:
        return True
    if item.get_closest_marker("requires_sandbox") is not None:
        return True
    fspath = getattr(item, "path", None) or getattr(item, "fspath", None)
    mod = fspath.name if fspath is not None else ""
    if mod in _SANDBOX_BUNDLE_MODULES:
        return True
    base_name = item.name.split("[", 1)[0]
    if mod == "test_sandbox_authoring.py" and base_name not in _SANDBOX_AUTHORING_NO_CORPUS_TESTS:
        return True
    if mod == "test_execute_step.py" and base_name in _EXECUTE_STEP_SANDBOX_TESTS:
        return True
    return mod == "test_sql_gen.py" and base_name == "test_rental_shop_category_film_emits_non_empty_join_path"


def _sandbox_bundle_ready() -> bool:
    """Return True when bundled ``sandbox/data.zip`` is present and passes ``Sandbox.sandbox_doctor()``."""
    from aetherdialect._sandbox import Sandbox

    if not Sandbox.data_zip_path().is_file():
        return False
    try:
        return Sandbox.sandbox_doctor() == []
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _cleanup_cwd_schema_migration_map() -> Any:
    """Remove migration map files left in the repo cwd by ``apply_migration_map`` demos."""
    cwd = Path.cwd()
    paths = (
        cwd / "schema_migration_map.json",
        cwd / "schema_migration_map.applied.json",
        cwd / "federation_migration_map.json",
        cwd / "federation_migration_map.applied.json",
    )
    for path in paths:
        path.unlink(missing_ok=True)
    for path in cwd.glob("federation_migration_map.applied.*.json"):
        path.unlink(missing_ok=True)
    yield
    for path in paths:
        path.unlink(missing_ok=True)
    for path in cwd.glob("federation_migration_map.applied.*.json"):
        path.unlink(missing_ok=True)


@pytest.fixture
def stub_schema_llm_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    """No-op ``apply_column_roles_llm`` for schema diff and partial- rebuild unit tests."""

    def _noop(schema: Any, notes_content: str | None = None, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr("aetherdialect._schema_overrides.apply_column_roles_llm", _noop)
    monkeypatch.setattr("aetherdialect._schema_graph.apply_column_roles_llm", _noop)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    sandbox_ready = _sandbox_bundle_ready()
    skip_corpus = pytest.mark.skip(reason="needs_corpus: bundled sandbox data.zip absent")
    for item in items:
        if _is_live_test(item):
            continue
        if _requires_corpus_bundle(item):
            item.add_marker(pytest.mark.needs_corpus)
            if not sandbox_ready:
                item.add_marker(skip_corpus)
            elif not _is_slow_sandbox_test(item):
                item.add_marker(pytest.mark.fast)
        elif not _is_slow_sandbox_test(item):
            item.add_marker(pytest.mark.fast)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Warn when a ``fast`` test exceeds the per-test fast budget (marker drift guard)."""
    if report.when != "call" or not report.passed:
        return
    duration = getattr(report, "duration", None)
    if duration is None or duration <= _FAST_TEST_MAX_SECS:
        return
    keywords = getattr(report, "keywords", {})
    if "fast" not in keywords:
        return
    print(
        f"\nNOT FAST ({duration:.1f}s > {_FAST_TEST_MAX_SECS}s): {report.nodeid}",
        flush=True,
    )


@pytest.fixture(autouse=True)
def _reset_engine_config_after_test() -> None:
    """Sandbox and init paths mutate class-level engine storage; restore defaults after each test."""
    yield
    from aetherdialect._config import (
        EngineConfig,
        PostgresRuntimeConfig,
        QSimConfig,
    )
    from aetherdialect._constants import ENGINE_STORAGE_PLACEHOLDER_DIR, TEMPLATE_STORE_SEGMENT
    from aetherdialect._llm_provider import MockProvider

    EngineConfig.TYPE = "postgresql"
    EngineConfig.RUNTIME = PostgresRuntimeConfig
    EngineConfig.SCHEMA_JSON_PATH = os.path.join(ENGINE_STORAGE_PLACEHOLDER_DIR, "schema_graph.json.gz")
    EngineConfig.TEMPLATE_STORE_DIR = os.path.join(ENGINE_STORAGE_PLACEHOLDER_DIR, TEMPLATE_STORE_SEGMENT)
    EngineConfig.LLM_PROVIDER = "openai"
    EngineConfig.MOCK_FIXTURES_FILE = ""
    # Credential ClassVars are mutated by tests/init; restore from the live process env
    # after monkeypatch teardown so a prior test's sk-test token cannot force live LLM calls.
    EngineConfig.API_TOKEN = os.environ.get("OPENAI_API_KEY")
    EngineConfig.AZURE_API_TOKEN = os.environ.get("AZURE_OPENAI_API_KEY")
    EngineConfig.AZURE_OPENAI_BASE_URL = os.environ.get("AZURE_OPENAI_BASE_URL")
    EngineConfig.AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
    EngineConfig.AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION")
    MockProvider.reset_mock_provider()
    QSimConfig.SKELETONS_JSON_PATH = os.path.join(ENGINE_STORAGE_PLACEHOLDER_DIR, "qsim_skeletons.json.gz")


@pytest.fixture
def schema_graph() -> SchemaGraph:
    """Three-table schema: customers, orders, products with FK edges."""
    customers = TableMetadata(
        name="customers",
        columns={
            "customer_id": ColumnMetadata(
                name="customer_id",
                data_type="integer",
                value_type="integer",
                is_primary_key=True,
                role=ColumnRole.IDENTIFIER.value,
                distinct_count=100,
                distinct_ratio=1.0,
                row_count=100,
            ),
            "name": ColumnMetadata(
                name="name",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.CATEGORICAL.value,
                distinct_count=95,
                distinct_ratio=0.95,
                row_count=100,
            ),
            "email": ColumnMetadata(
                name="email",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.FREE_TEXT.value,
                distinct_count=100,
                distinct_ratio=1.0,
                row_count=100,
            ),
            "active": ColumnMetadata(
                name="active",
                data_type="boolean",
                value_type="boolean",
                role=ColumnRole.BOOLEAN.value,
                distinct_count=2,
                distinct_ratio=0.02,
                row_count=100,
                frequent_values=["true", "false"],
            ),
            "created_at": ColumnMetadata(
                name="created_at",
                data_type="timestamp",
                value_type="date",
                role=ColumnRole.AUDIT.value,
                distinct_count=100,
                distinct_ratio=1.0,
                row_count=100,
            ),
        },
        primary_key=["customer_id"],
        foreign_keys=[],
        role=TableRole.DIMENSION.value,
        row_count=100,
    )
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
                distinct_ratio=1.0,
                row_count=500,
            ),
            "customer_id": ColumnMetadata(
                name="customer_id",
                data_type="integer",
                value_type="integer",
                is_foreign_key=True,
                fk_target=("customers", "customer_id"),
                role=ColumnRole.IDENTIFIER.value,
                distinct_count=100,
                distinct_ratio=0.2,
                row_count=500,
            ),
            "amount": ColumnMetadata(
                name="amount",
                data_type="numeric",
                value_type="number",
                role=ColumnRole.NUMERIC_MEASURE.value,
                distinct_count=300,
                distinct_ratio=0.6,
                row_count=500,
            ),
            "status": ColumnMetadata(
                name="status",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.CATEGORICAL.value,
                distinct_count=5,
                distinct_ratio=0.01,
                row_count=500,
                frequent_values=[
                    "pending",
                    "shipped",
                    "delivered",
                    "cancelled",
                    "returned",
                ],
            ),
            "order_date": ColumnMetadata(
                name="order_date",
                data_type="date",
                value_type="date",
                role=ColumnRole.TEMPORAL.value,
                distinct_count=365,
                distinct_ratio=0.73,
                row_count=500,
            ),
            "product_id": ColumnMetadata(
                name="product_id",
                data_type="integer",
                value_type="integer",
                is_foreign_key=True,
                fk_target=("products", "product_id"),
                role=ColumnRole.IDENTIFIER.value,
                distinct_count=50,
                distinct_ratio=0.1,
                row_count=500,
            ),
        },
        primary_key=["order_id"],
        foreign_keys=[
            FKEdge(
                src_table="orders",
                src_cols=["customer_id"],
                dst_table="customers",
                dst_cols=["customer_id"],
            ),
            FKEdge(
                src_table="orders",
                src_cols=["product_id"],
                dst_table="products",
                dst_cols=["product_id"],
            ),
        ],
        role=TableRole.FACT.value,
        row_count=500,
    )
    products = TableMetadata(
        name="products",
        columns={
            "product_id": ColumnMetadata(
                name="product_id",
                data_type="integer",
                value_type="integer",
                is_primary_key=True,
                role=ColumnRole.IDENTIFIER.value,
                distinct_count=50,
                distinct_ratio=1.0,
                row_count=50,
            ),
            "title": ColumnMetadata(
                name="title",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.CATEGORICAL.value,
                distinct_count=50,
                distinct_ratio=1.0,
                row_count=50,
            ),
            "price": ColumnMetadata(
                name="price",
                data_type="numeric",
                value_type="number",
                role=ColumnRole.NUMERIC_MEASURE.value,
                distinct_count=40,
                distinct_ratio=0.8,
                row_count=50,
            ),
            "category": ColumnMetadata(
                name="category",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.CATEGORICAL.value,
                distinct_count=10,
                distinct_ratio=0.2,
                row_count=50,
                frequent_values=["electronics", "books", "clothing", "food", "toys"],
            ),
        },
        primary_key=["product_id"],
        foreign_keys=[],
        role=TableRole.DIMENSION.value,
        row_count=50,
    )
    sg = SchemaGraph(
        tables={"customers": customers, "orders": orders, "products": products},
        join_paths_multi={
            "customers": {
                "orders": [[{"src": "orders.customer_id", "dst": "customers.customer_id"}]],
            },
            "orders": {
                "customers": [[{"src": "orders.customer_id", "dst": "customers.customer_id"}]],
                "products": [[{"src": "orders.product_id", "dst": "products.product_id"}]],
            },
            "products": {
                "orders": [[{"src": "orders.product_id", "dst": "products.product_id"}]],
            },
        },
        effective_structural_hash="test_hash_abc123",
    )
    assign_column_ops(sg)
    return sg


@pytest.fixture
def minimal_intent() -> RuntimeIntent:
    """Minimal RuntimeIntent for orders table."""
    return RuntimeIntent(
        tables=["orders"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.order_id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )


@pytest.fixture
def grouped_intent() -> RuntimeIntent:
    """Grouped RuntimeIntent with aggregation."""
    return RuntimeIntent(
        tables=["orders", "customers"],
        grain="grouped",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("customers.name")),
            SelectCol(expr=NormalizedExpr.from_agg("sum", "orders.amount")),
        ],
        group_by_cols=[NormalizedExpr.from_column("customers.name")],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("orders.status"),
                    op="=",
                    value_type="string",
                    param_key="p1",
                    raw_value="shipped",
                ),
            ]
        ),
        having=None,
        param_values={"p1": "shipped"},
        column_map={"name": "customers", "amount": "orders", "status": "orders"},
    )


@pytest.fixture
def sample_template(grouped_intent) -> Template:
    """Sample Template wrapping grouped_intent."""
    return Template(
        id="T0001",
        effective_structural_hash="test_hash_abc123",
        intent_signature=grouped_intent.to_concrete("test_id"),
        intent_key="test_intent_key_hash",
        tables_used=["customers", "orders"],
        sql_param="SELECT customers.name, SUM(orders.amount) FROM orders JOIN customers ON orders.customer_id = customers.customer_id WHERE orders.status = :p1 GROUP BY customers.name",
        sql_fp="test_sql_fp",
        shape=SQLShape(num_joins=1, has_group_by=True, has_agg=True),
        colmap_sig="test_colmap_sig",
        value_history=ValueHistory(
            param_values=[{"p1": "shipped"}],
            questions=["total amount by customer for shipped orders"],
            natural_language=["total amount by customer for shipped orders"],
        ),
        stats=TemplateStats(accept=1, reject=0),
        trust_level=1,
    )


@pytest.fixture
def simple_schema() -> SchemaGraph:
    """Schema with customers and orders tables for validation tests."""
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
                distinct_count=80,
            ),
            "email": ColumnMetadata(
                name="email",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.FREE_TEXT.value,
                distinct_count=100,
            ),
            "balance": ColumnMetadata(
                name="balance",
                data_type="numeric",
                value_type="number",
                role=ColumnRole.NUMERIC_MEASURE.value,
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
                distinct_count=200,
            ),
            "customer_id": ColumnMetadata(
                name="customer_id",
                data_type="integer",
                value_type="integer",
                is_foreign_key=True,
                role=ColumnRole.IDENTIFIER.value,
                distinct_count=100,
            ),
            "amount": ColumnMetadata(
                name="amount",
                data_type="numeric",
                value_type="number",
                role=ColumnRole.NUMERIC_MEASURE.value,
                distinct_count=150,
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
            )
        ],
        primary_key="",
    )
    sg = SchemaGraph(
        join_paths_multi={},
        effective_structural_hash="",
        tables={"customers": customers, "orders": orders},
    )
    assign_column_ops(sg)
    return sg


@pytest.fixture
def typed_schema() -> SchemaGraph:
    """Schema with typed columns for validation tests."""
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
            "name": ColumnMetadata(
                name="name",
                data_type="varchar",
                value_type="string",
                role=ColumnRole.CATEGORICAL.value,
            ),
            "balance": ColumnMetadata(
                name="balance",
                data_type="numeric",
                value_type="number",
                role=ColumnRole.NUMERIC_MEASURE.value,
            ),
            "created_at": ColumnMetadata(
                name="created_at",
                data_type="timestamp",
                value_type="date",
                role=ColumnRole.TEMPORAL.value,
            ),
            "description": ColumnMetadata(
                name="description",
                data_type="text",
                value_type="string",
                role=ColumnRole.FREE_TEXT.value,
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
            ),
            "customer_id": ColumnMetadata(
                name="customer_id",
                data_type="integer",
                value_type="integer",
                is_foreign_key=True,
                role=ColumnRole.IDENTIFIER.value,
            ),
            "amount": ColumnMetadata(
                name="amount",
                data_type="numeric",
                value_type="number",
                role=ColumnRole.NUMERIC_MEASURE.value,
            ),
            "order_date": ColumnMetadata(
                name="order_date",
                data_type="date",
                value_type="date",
                role=ColumnRole.TEMPORAL.value,
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
    sg = SchemaGraph(
        join_paths_multi={},
        effective_structural_hash="",
        tables={"customers": customers, "orders": orders},
    )
    assign_column_ops(sg)
    return sg


@pytest.fixture
def two_member_federation():
    """Two-member duckdb federation with a declared cross-source join on ``id``."""
    from tests.federation_helpers import build_two_member_federation

    return build_two_member_federation()
