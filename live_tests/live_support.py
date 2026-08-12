"""Shared live-test harness: invoice artifacts, engine bootstrap, federation loaders, and seed helpers. Database-specific constants and oracles live in ``mydb_profile.py``; question catalogs in ``mydb_scenarios.py``."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from sqlalchemy import create_engine, text

from aetherdialect import AetherEngine, AetherFederation
from aetherdialect._config import (
    EngineConfig,
    MariaDBRuntimeConfig,
    MySQLRuntimeConfig,
    PostgresRuntimeConfig,
    QSimConfig,
    SeedWarmupConfig,
)
from aetherdialect._contracts_base import (
    EngineContext,
    FederationConfigError,
    NormalizedExpr,
    PredicateGroup,
    SchemaInclude,
    WhereParam,
)
from aetherdialect._contracts_core import (
    FeedbackKind,
    LiveTestRunner,
    RuntimeIntent,
    SelectCol,
    Template,
    ValueHistory,
)
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    ColumnRole,
    SchemaGraph,
    TableMetadata,
    TemplateStats,
)
from aetherdialect._dialect import DialectRegistry
from aetherdialect._federation_manifest import parse_federation_declaration, parse_federation_manifest
from aetherdialect._qsim import get_aggregatable_columns, get_groupable_columns
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils import (
    is_date_type,
    is_numeric_type,
    llm_usage_build_scope,
)
from aetherdialect._utils_artifacts import write_gzip_json_atomic
from aetherdialect._utils_intent import intent_key
from sandbox_recording import (
    set_invoice_path,
    write_live_env_file_to_temp_config_toml,
)

from .mydb_profile import (
    PROFILE_FEDERATION_DECLARATION,
    PROFILE_SQL_DEFAULT,
    PROFILE_VIEWS_SQL,
    apply_synthetic_rental_partition_metadata,
)

_LIVE_ARTIFACTS_ROOT = Path(__file__).parent / "_run_artifacts"
set_invoice_path(Path(__file__).parent / "invoice.txt")

# --- federation equivalence questions ---


@dataclass(frozen=True)
class FederationEquivalenceQuestion:
    """One schema-derived natural-language question for equivalence checking."""

    question_id: str
    question: str
    category: str


_AGG_FUNCS: tuple[tuple[str, str], ...] = (
    ("sum", "total"),
    ("avg", "average"),
    ("min", "minimum"),
    ("max", "maximum"),
    ("count", "count"),
)


def _column_roles(schema: SchemaGraph) -> dict[str, str]:
    roles: dict[str, str] = {}
    for table_name, table in sorted(schema.tables.items()):
        for col_name, col in sorted(table.columns.items()):
            key = f"{table_name}.{col_name}"
            role = (col.role or "").strip()
            if role:
                roles[key] = role
            elif is_numeric_type(str(col.data_type or "")):
                roles[key] = ColumnRole.NUMERIC_MEASURE.value
            elif is_date_type(str(col.data_type or "")):
                roles[key] = ColumnRole.TEMPORAL.value
            else:
                roles[key] = ColumnRole.CATEGORICAL.value
    return roles


def _joinable_table_pairs(schema: SchemaGraph) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    join_paths = schema.join_paths_multi or {}
    tables = sorted(join_paths.keys())
    for left_index, left in enumerate(tables):
        left_row = join_paths.get(left, {})
        for right in tables[left_index + 1 :]:
            right_paths = left_row.get(right) or join_paths.get(right, {}).get(left) or []
            if right_paths:
                pairs.append((left, right))
    return pairs


def _date_columns(schema: SchemaGraph, column_roles: dict[str, str]) -> list[str]:
    found: list[str] = []
    for table_name, table in sorted(schema.tables.items()):
        for col_name, col in sorted(table.columns.items()):
            key = f"{table_name}.{col_name}"
            role = column_roles.get(key, "")
            if role == ColumnRole.TEMPORAL.value or is_date_type(str(col.data_type or "")):
                found.append(key)
    return found


def _humanize_table(table_name: str) -> str:
    return table_name.replace("_", " ")


def _humanize_column(column_key: str) -> str:
    return column_key.split(".", 1)[-1].replace("_", " ")


def generate_federation_equivalence_questions(schema: SchemaGraph) -> list[FederationEquivalenceQuestion]:
    """Build a deterministically ordered corpus question set from schema shapes."""
    column_roles = _column_roles(schema)
    questions: list[FederationEquivalenceQuestion] = []

    for left, right in _joinable_table_pairs(schema):
        left_label = _humanize_table(left)
        right_label = _humanize_table(right)
        questions.append(
            FederationEquivalenceQuestion(
                question_id=f"join_count:{left}:{right}",
                question=f"how many {left_label} rows are linked to {right_label}",
                category="join_pair",
            )
        )
        questions.append(
            FederationEquivalenceQuestion(
                question_id=f"join_distinct_left:{left}:{right}",
                question=f"how many distinct {left_label} rows are linked to {right_label}",
                category="join_pair",
            )
        )

    for table_name in sorted(schema.tables.keys()):
        table_label = _humanize_table(table_name)
        aggregatable = get_aggregatable_columns(table_name, schema, column_roles)
        groupable = get_groupable_columns(table_name, schema, column_roles)
        for column_key in aggregatable:
            column_label = _humanize_column(column_key)
            for agg_func, phrase in _AGG_FUNCS:
                if agg_func == "count":
                    question_text = f"what is the {phrase} of {table_label} rows"
                else:
                    question_text = f"what is the {phrase} {column_label} for {table_label}"
                questions.append(
                    FederationEquivalenceQuestion(
                        question_id=f"aggregate:{agg_func}:{column_key}",
                        question=question_text,
                        category="aggregate",
                    )
                )
        for group_key in groupable:
            group_label = _humanize_column(group_key)
            for measure_key in aggregatable:
                measure_label = _humanize_column(measure_key)
                questions.append(
                    FederationEquivalenceQuestion(
                        question_id=f"grouping:sum:{measure_key}:by:{group_key}",
                        question=f"what is the total {measure_label} grouped by {group_label} for {table_label}",
                        category="grouping",
                    )
                )
                questions.append(
                    FederationEquivalenceQuestion(
                        question_id=f"grouping:count:{group_key}",
                        question=f"how many {table_label} rows are there grouped by {group_label}",
                        category="grouping",
                    )
                )

    for date_key in _date_columns(schema, column_roles):
        table_name, column_name = date_key.split(".", 1)
        table_label = _humanize_table(table_name)
        column_label = _humanize_column(date_key)
        for preset in SeedWarmupConfig.DATE_WINDOW_EXPANSION_PRESETS:
            unit = str(preset["unit"])
            amount = int(preset["amount"])
            unit_label = unit if amount == 1 else f"{unit}s"
            questions.append(
                FederationEquivalenceQuestion(
                    question_id=f"date_window:{date_key}:{unit}:{amount}",
                    question=(f"how many {table_label} rows have {column_label} in the last {amount} {unit_label}"),
                    category="date_window",
                )
            )

    questions.sort(key=lambda row: (row.category, row.question_id, row.question))
    return questions


# --- federation live helpers ---


_REPO = Path(__file__).resolve().parents[1]
_ENV_FILE = _REPO / "env.env"
_DECLARATION_PATH = PROFILE_FEDERATION_DECLARATION


def _scripts_on_path() -> None:
    scripts = str(_REPO / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


def missing_federation_partition_engines() -> list[str]:
    """Return engine families that do not respond for federation partition loading."""
    _scripts_on_path()
    from load_rental_shop_engines import load_env_file

    missing: list[str] = []
    if not _ENV_FILE.is_file():
        return ["postgresql", "mysql", "mariadb"]
    load_env_file(_ENV_FILE, override=True)
    try:
        pg_engine = create_engine(PostgresRuntimeConfig.db_url(), future=True)
        with pg_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        pg_engine.dispose()
    except Exception:
        missing.append("postgresql")
    try:
        MySQLRuntimeConfig.apply_environment(os.environ)
        mysql_engine = create_engine(
            MySQLRuntimeConfig.db_url(),
            connect_args=MySQLRuntimeConfig.connect_args(),
            future=True,
        )
        with mysql_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        mysql_engine.dispose()
    except Exception:
        missing.append("mysql")
    try:
        MariaDBRuntimeConfig.apply_environment(os.environ)
        mariadb_engine = create_engine(
            MariaDBRuntimeConfig.db_url(),
            connect_args=MariaDBRuntimeConfig.connect_args(),
            future=True,
        )
        with mariadb_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        mariadb_engine.dispose()
    except Exception:
        missing.append("mariadb")
    return missing


def federation_partitions_available() -> bool:
    """Return True when postgres, mysql and mariadb federation partition targets respond."""
    return not missing_federation_partition_engines()


def ensure_federation_partitions_loaded() -> None:
    """Load federation partition databases without touching full rental_shop targets."""
    loader = _REPO / "scripts" / "load_rental_shop_engines.py"
    subprocess.run(
        [
            sys.executable,
            str(loader),
            "--federation-load",
            "all",
            "--drop-first",
            "--env-file",
            str(_ENV_FILE),
        ],
        check=True,
        cwd=str(_REPO),
    )


def _stamp_source_id(graph: SchemaGraph, source_id: str) -> SchemaGraph:
    tables = {name: replace(table, source_id=source_id) for name, table in graph.tables.items()}
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_stats=graph.schema_stats,
    )


def _reflect_member_graph(engine_type: str, sa_engine: Any, source_id: str) -> SchemaGraph:
    _scripts_on_path()
    from sandbox_corpus import federation_partition_tables

    tables = federation_partition_tables(source_id)
    if engine_type == "postgresql":
        PostgresRuntimeConfig.apply_environment(os.environ)
        runtime_cls = PostgresRuntimeConfig
    elif engine_type == "mariadb":
        MariaDBRuntimeConfig.apply_environment(os.environ)
        runtime_cls = MariaDBRuntimeConfig
    else:
        MySQLRuntimeConfig.apply_environment(os.environ)
        runtime_cls = MySQLRuntimeConfig
    dialect = DialectRegistry.get(engine_type, runtime_cls, sqlalchemy_engine=sa_engine)
    ctx = EngineContext(include="tables", allow_objects=frozenset(tables))
    graph = dialect.reflect_schema_graph(ctx)
    return _stamp_source_id(graph, source_id)


def _postgres_engine_for_schema(schema: str):
    env = dict(os.environ)
    env["PGSCHEMA"] = schema
    env["POSTGRESQL_SCHEMA"] = schema
    PostgresRuntimeConfig.apply_environment(env)
    return create_engine(PostgresRuntimeConfig.db_url(), future=True)


def build_federation_live_engine() -> AetherFederation:
    """Construct a federated owner scope over four live partition members."""
    _scripts_on_path()
    from load_rental_shop_engines import load_env_file
    from sandbox_corpus import (
        FEDERATION_CATALOG_MYSQL_DATABASE,
        FEDERATION_CRM_MARIADB_DATABASE,
        FEDERATION_LOGISTICS_PG_SCHEMA,
        FEDERATION_STOREFRONT_PG_SCHEMA,
    )

    load_env_file(_ENV_FILE, override=True)
    os.environ["MYSQL_DATABASE"] = FEDERATION_CATALOG_MYSQL_DATABASE
    os.environ["MARIADB_DATABASE"] = FEDERATION_CRM_MARIADB_DATABASE
    pg_storefront = _postgres_engine_for_schema(FEDERATION_STOREFRONT_PG_SCHEMA)
    pg_logistics = _postgres_engine_for_schema(FEDERATION_LOGISTICS_PG_SCHEMA)
    MySQLRuntimeConfig.apply_environment(os.environ)
    mysql_engine = create_engine(
        MySQLRuntimeConfig.db_url(),
        connect_args=MySQLRuntimeConfig.connect_args(),
        future=True,
    )
    MariaDBRuntimeConfig.apply_environment(os.environ)
    mariadb_engine = create_engine(
        MariaDBRuntimeConfig.db_url(),
        connect_args=MariaDBRuntimeConfig.connect_args(),
        future=True,
    )
    manifest, _ = parse_federation_declaration(json.loads(_DECLARATION_PATH.read_text(encoding="utf-8")))
    notes = _REPO / "scripts" / "data" / "rental_shop_notes.txt"
    sql_file = _REPO / "scripts" / "data" / "rental_shop.sql"
    cfg_path = _write_federation_toml()
    _LIVE_ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    artifacts_root = tempfile.mkdtemp(prefix="live_fed_artifacts_", dir=str(_LIVE_ARTIFACTS_ROOT))
    master_ctx = EngineContext(
        notes_file=str(notes) if notes.is_file() else None,
        sql_file=str(sql_file) if sql_file.is_file() else None,
    )
    try:
        with llm_usage_build_scope():
            members = {
                "storefront": AetherEngine(
                    master_ctx,
                    artifacts_dir=artifacts_root,
                    config_file=cfg_path,
                    execution_engine=pg_storefront,
                ),
                "catalog": AetherEngine(
                    master_ctx,
                    artifacts_dir=artifacts_root,
                    config_file=cfg_path,
                    execution_engine=mysql_engine,
                ),
                "logistics": AetherEngine(
                    master_ctx,
                    artifacts_dir=artifacts_root,
                    config_file=cfg_path,
                    execution_engine=pg_logistics,
                ),
                "crm": AetherEngine(
                    master_ctx,
                    artifacts_dir=artifacts_root,
                    config_file=cfg_path,
                    execution_engine=mariadb_engine,
                ),
            }
            return AetherFederation(
                manifest.federation_id,
                members=members,
                declaration_file=str(_DECLARATION_PATH),
                artifacts_dir=artifacts_root,
            )
    finally:
        Path(cfg_path).unlink(missing_ok=True)


def _write_federation_toml() -> str:
    return write_live_env_file_to_temp_config_toml(
        _ENV_FILE,
        {
            "AETHERDIALECT_ENGINE": "postgresql",
            "PGSCHEMA": os.environ.get("PGSCHEMA", "rental_shop_fed_storefront"),
            "MYSQL_DATABASE": os.environ.get("MYSQL_DATABASE", "rental_shop_fed_catalog"),
            "MARIADB_DATABASE": os.environ.get("MARIADB_DATABASE", "rental_shop_fed_crm"),
        },
    )


def sensitive_manifest_rejects_ssn_key() -> bool:
    """Return True when a sensitive cross-source join is rejected at manifest parse."""
    manifest_payload = json.loads(_DECLARATION_PATH.read_text(encoding="utf-8"))
    bad_manifest = dict(manifest_payload)
    bad_manifest["cross_source_joins"] = [
        {
            "left": "staff.ssn",
            "right": "customer.email",
            "kind": "inner",
            "logical_key": "ssn",
        }
    ]
    try:
        parse_federation_manifest(bad_manifest)
    except FederationConfigError:
        return True
    return False


# --- seed helpers ---


def _question_feedback_entry_count(store: dict[str, Any]) -> int:
    qf = store.get("question_feedback")
    if not isinstance(qf, dict):
        return 0
    n = 0
    for rows in qf.values():
        if isinstance(rows, list):
            n += len(rows)
    return n


def empty_store(effective_structural_hash: str):
    """Return a fresh empty partitioned template store (alias for :func:`empty_template_store`)."""
    return TemplateOps.empty_template_store(effective_structural_hash)


@contextmanager
def isolated_runner(schema: Any, schema_terms: set[str], t2s: Any, *, label: str) -> Iterator[LiveTestRunner]:
    """Yield an instrumented ``LiveTestRunner`` with an isolated on- disk. store. ``EngineConfig.TEMPLATE_STORE_DIR`` is redirected to a unique directory inside ``t2s._artifacts_dir`` and removed on teardown so residual state cannot leak. Args: schema: Profiled ``SchemaGraph`` shared across live tests. schema_terms: Schema term tokens for the runner. t2s: Session ``AetherEngine`` instance for the artifacts directory. label: Short identifier embedded in the isolated directory name. Yields: A ``LiveTestRunner`` bound to the fresh store."""
    from .conftest import _instrument_runner

    original_dir = EngineConfig.TEMPLATE_STORE_DIR
    isolated_dir = os.path.join(
        str(t2s._artifacts_dir),
        f"seed_{label}_{uuid.uuid4().hex[:8]}_tmpl",
    )
    if os.path.isdir(isolated_dir):
        shutil.rmtree(isolated_dir, ignore_errors=True)
    os.makedirs(isolated_dir, exist_ok=True)
    EngineConfig.TEMPLATE_STORE_DIR = isolated_dir
    try:
        store = TemplateOps.empty_template_store(schema.effective_structural_hash)
        runner = LiveTestRunner(
            schema=schema,
            store=store,
            templates=TemplateOps.store_to_templates(store),
            rejected={},
            schema_terms=set(schema_terms),
            csv_dir=t2s._artifacts_dir,
        )
        _instrument_runner(runner)
        yield runner
    finally:
        EngineConfig.TEMPLATE_STORE_DIR = original_dir
        if os.path.isdir(isolated_dir):
            shutil.rmtree(isolated_dir, ignore_errors=True)


def run_isolated_scenario(
    schema: Any,
    schema_terms: set[str],
    t2s: Any,
    scenario: Any,
    *,
    label: str,
) -> None:
    """Run one scenario under ``isolated_runner`` and assert its expectations."""
    from aetherdialect._live_testing import run_and_assert

    with isolated_runner(schema, schema_terms, t2s, label=label) as runner:
        run_and_assert(
            runner,
            scenario,
            header=f"[isolated:{scenario.id}] {scenario.question}",
        )


def intent_rental_count_by_store() -> RuntimeIntent:
    """Grouped rental count per ``inventory.store_id``."""
    return RuntimeIntent(
        tables=["rental", "inventory"],
        grain="grouped",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("inventory.store_id")),
            SelectCol(expr=NormalizedExpr.from_agg("count", "rental.rental_id")),
        ],
        group_by_cols=[NormalizedExpr.from_column("inventory.store_id")],
        order_by_cols=[],
        where=None,
        having=None,
        natural_language="rentals per store",
    )


def intent_payment_sum_by_staff() -> RuntimeIntent:
    """Grouped sum of ``payment.amount`` per ``payment.staff_id``."""
    return RuntimeIntent(
        tables=["payment"],
        grain="grouped",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("payment.staff_id")),
            SelectCol(expr=NormalizedExpr.from_agg("sum", "payment.amount")),
        ],
        group_by_cols=[NormalizedExpr.from_column("payment.staff_id")],
        order_by_cols=[],
        where=None,
        having=None,
        natural_language="total payments per staff",
    )


def intent_customer_first_names() -> RuntimeIntent:
    """Row-level select of ``customer.customer_id`` and ``customer.first_name``."""
    return RuntimeIntent(
        tables=["customer"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("customer.customer_id")),
            SelectCol(expr=NormalizedExpr.from_column("customer.first_name")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
        natural_language="list customer ids and first names",
    )


def intent_customer_full_names() -> RuntimeIntent:
    """Row-level select of customer id, first name, and last name."""
    return RuntimeIntent(
        tables=["customer"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("customer.customer_id")),
            SelectCol(expr=NormalizedExpr.from_column("customer.first_name")),
            SelectCol(expr=NormalizedExpr.from_column("customer.last_name")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
        natural_language="list customer ids first names and last names",
    )


def intent_customer_emails_only() -> RuntimeIntent:
    """Row-level select of ``customer.customer_id`` and ``customer.email``."""
    return RuntimeIntent(
        tables=["customer"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("customer.customer_id")),
            SelectCol(expr=NormalizedExpr.from_column("customer.email")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
        natural_language="list customer ids and emails",
    )


def intent_store_staff_by_work() -> RuntimeIntent:
    """Row-level ``store`` + ``staff`` join on the ``staff.store_id`` FK edge."""
    return RuntimeIntent(
        tables=["store", "staff"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("store.store_id")),
            SelectCol(expr=NormalizedExpr.from_column("store.last_update")),
            SelectCol(expr=NormalizedExpr.from_column("staff.first_name")),
            SelectCol(expr=NormalizedExpr.from_column("staff.last_name")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
        natural_language=(
            "list every staff first and last name together with the last update time of the store where they work"
        ),
    )


def intent_store_manager() -> RuntimeIntent:
    """Row-level ``store`` + ``staff`` join on the ``store.manager_staff_id`` FK edge."""
    return RuntimeIntent(
        tables=["store", "staff"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("store.store_id")),
            SelectCol(expr=NormalizedExpr.from_column("staff.first_name")),
            SelectCol(expr=NormalizedExpr.from_column("staff.last_name")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        having=None,
        natural_language="who manages each store",
    )


def intent_film_in_category(category_name: str) -> RuntimeIntent:
    """Row-level ``film`` title filtered by ``item_category``/``category`` join (via ``item``)."""
    return RuntimeIntent(
        tables=["film", "item", "item_category", "category"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("item.title")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("category.name"),
                    op="=",
                    value_type="text",
                    bool_op="AND",
                    raw_value=category_name,
                ),
            ]
        ),
        having=None,
        natural_language=f"films in the {category_name} category",
    )


def seed_template(
    runner: LiveTestRunner,
    *,
    q_norm: str,
    intent: RuntimeIntent,
    sql: str,
    trust_level: int = 1,
    stats: TemplateStats | None = None,
    value_history: ValueHistory | None = None,
    structural_override: dict[str, Any] | None = None,
    template_source: str = "human",
) -> Template:
    """Insert one template into *runner*'s store. When. *structural_override* is non-empty, each key replaces the matching entry in ``Template.structural_defaults`` after insertion so the stored value_history row differs from the structural default (the lever that routes fuzzy reuse through ``GenerationPath.FUZZY_REUSE_FULL_PARAMS``). Args: runner: Target ``LiveTestRunner`` whose store/templates are mutated. q_norm: Normalised seed question stored in ``value_history.questions``. intent: Seed intent whose ``param_values`` feed structural defaults. sql: Seed SQL (canonicalised + parameterised by ``insert_template``). trust_level: Stored ``Template.trust_level``; default is 1. stats: Optional ``TemplateStats``; default is ``accept=3, reject=0``. value_history: Optional pre-built history; default is a single row. structural_override: Optional ``s*`` overrides applied post-insert. template_source: Stored ``Template.source`` marker. Returns: The newly inserted or merged ``Template``."""
    if stats is None:
        stats = TemplateStats(accept=3, reject=0)
    template = TemplateOps.insert_template(
        runner.store,
        runner.templates,
        runner.schema,
        q_norm,
        intent,
        sql,
        template_source=template_source,
        template_trust_level=trust_level,
        template_initial_stats=stats,
        template_value_history=value_history,
    )
    if structural_override:
        for key, value in structural_override.items():
            template.structural_defaults[key] = value
    return template


def seed_rejected(
    runner: LiveTestRunner,
    *,
    q_norm: str,
    intent: RuntimeIntent,
    sql: str,
    reason: str = "seeded rejection",
) -> SimpleNamespace:
    """Insert one ``INTENT_REJECTED`` question-feedback row into *runner*'s store."""
    ent = TemplateOps.summarize_failure_for_memory(
        question=q_norm,
        intent=intent,
        kind=FeedbackKind.INTENT_REJECTED,
        schema_hash=runner.schema.effective_structural_hash,
        user_reason=reason,
    )
    TemplateOps.record_question_feedback(runner.store, q_norm, ent)
    return SimpleNamespace(
        id=f"qf:{q_norm}",
        intent_key=intent_key(intent),
        value_history=SimpleNamespace(rejection_reasons=[reason]),
    )


def seed_negative_memory(
    runner: LiveTestRunner,
    *,
    intent: RuntimeIntent,
    sql: str,
    reason: str,
    repeats: int = 1,
    q_norm: str | None = None,
) -> dict[str, str]:
    """Seed validation-failure feedback rows for *intent* (penalty / hint paths). Each repeat appends one ``question_feedback`` row scoped to the runner schema so :func:`aetherdialect._templates_ops.TemplateOps.compute_question_feedback_penalty` observes the seed. Returns the computed keys (``ikey``, ``sql_fp``, ``colmap_sig``, ``q_norm``) for assertions."""
    from aetherdialect._dialect import Dialect
    from aetherdialect._utils import (
        canonicalize_sql,
        colmap_signature,
        normalize_sql,
    )

    sql_canon = canonicalize_sql(sql)
    sql_norm = normalize_sql(sql_canon)
    sql_param, _ = Dialect.parameter_abstract(sql_norm, sqlglot_dialect=Dialect.active_sqlglot_dialect())
    ikey = intent_key(intent)
    sql_fp = Dialect.compute_sql_fp(sql_param, sqlglot_dialect=Dialect.active_sqlglot_dialect())
    cmap_sig = colmap_signature(intent.column_map)
    qn = q_norm or intent.natural_language or f"seed-negative-memory::{ikey}"
    eff = runner.schema.effective_structural_hash
    for _ in range(max(1, repeats)):
        ent = TemplateOps.summarize_failure_for_memory(
            question=qn,
            intent=intent,
            kind=FeedbackKind.VALIDATION_FAILURE,
            schema_hash=eff,
            validator_errors=[reason],
        )
        TemplateOps.record_question_feedback(runner.store, qn, ent)
    return {"ikey": ikey, "sql_fp": sql_fp, "colmap_sig": cmap_sig, "q_norm": qn}


@contextmanager
def capture_parse_prompt() -> Iterator[list[dict[str, Any]]]:
    """Record calls into intent-parse prompt construction and forward them through. The captured list includes entries from ``build_intent_parse_prompt`` (with ``prior_question_feedback``) and a slim record for each ``full_intent_parse`` / ``invoke_intent_parse_with_hints`` call (``store`` / ``in_turn_seed`` / ``budget``)."""
    import aetherdialect._intent_loop

    calls: list[dict[str, Any]] = []
    original_parse = aetherdialect._intent_loop.full_intent_parse
    original_invoke = aetherdialect._intent_loop.invoke_intent_parse_with_hints
    original_build = aetherdialect._intent_loop.build_intent_parse_prompt

    def _recording_build(
        question: str,
        schema_literal_json: str,
        table_list: list[str],
        prior_question_feedback: list[dict[str, str]] | None = None,
    ) -> tuple[str, str]:
        calls.append(
            {
                "via": "build_intent_parse_prompt",
                "question": question,
                "prior_question_feedback": (list(prior_question_feedback) if prior_question_feedback else None),
            }
        )
        return original_build(question, schema_literal_json, table_list, prior_question_feedback)

    def _recording_parse(
        question: str,
        schema_graph: Any,
        max_retries: int = 3,
        *,
        store: Any | None = None,
        in_turn_seed: list[dict[str, str]] | None = None,
        budget: Any | None = None,
    ) -> Any:
        calls.append(
            {
                "via": "full_intent_parse",
                "question": question,
                "store": store is not None,
                "in_turn_seed": (list(in_turn_seed) if in_turn_seed else None),
                "budget": budget is not None,
            }
        )
        return original_parse(
            question,
            schema_graph,
            max_retries=max_retries,
            store=store,
            in_turn_seed=in_turn_seed,
            budget=budget,
        )

    def _recording_invoke(
        question: str,
        schema_graph: Any,
        *,
        max_retries: int = 3,
        store: dict[str, Any] | None = None,
        in_turn_seed: list[dict[str, str]] | None = None,
        budget: Any | None = None,
    ) -> Any:
        calls.append(
            {
                "via": "invoke_intent_parse_with_hints",
                "question": question,
                "store": store is not None,
                "in_turn_seed": (list(in_turn_seed) if in_turn_seed else None),
                "budget": budget is not None,
            }
        )
        return original_invoke(
            question,
            schema_graph,
            max_retries=max_retries,
            store=store,
            in_turn_seed=in_turn_seed,
            budget=budget,
        )

    try:
        aetherdialect._intent_loop.full_intent_parse = _recording_parse
        aetherdialect._intent_loop.invoke_intent_parse_with_hints = _recording_invoke
        aetherdialect._intent_loop.build_intent_parse_prompt = _recording_build
        yield calls
    finally:
        aetherdialect._intent_loop.full_intent_parse = original_parse
        aetherdialect._intent_loop.invoke_intent_parse_with_hints = original_invoke
        aetherdialect._intent_loop.build_intent_parse_prompt = original_build


def deterministic_join_choice_patch() -> Any:
    """Patch ``aetherdialect._sql_gen.get_join_choice_from_llm`` to pick the first candidate. Use this inside seeded generation-path tests where the specific join edge is irrelevant and the assertion is about the generation path code. Matches the keyword-only join-choice API and returns a scope- to-id dict merged with any preset choices."""

    def _first_candidate(
        q_norm: str,
        deterministic_sql: str,
        *,
        llm_scopes: list[dict[str, Any]],
        preset_choices: dict[str, str] | None = None,
        accept_na_by_scope: dict[str, bool] | None = None,
        require_final: bool = False,
    ) -> dict[str, str]:
        out = dict(preset_choices or {})
        for block in llm_scopes:
            sk = str(block.get("scope") or "")
            cands = list(block.get("candidates") or [])
            if not cands:
                out[sk] = "J00"
                continue
            out[sk] = str(cands[0].get("candidate_id") or "J00")
        return out

    return patch("aetherdialect._sql_gen.get_join_choice_from_llm", side_effect=_first_candidate)


def forced_join_choice_patch(
    predicate: Callable[[list[dict[str, Any]]], str],
) -> Any:
    """Patch ``aetherdialect._sql_gen.get_join_choice_from_llm`` to choose by *predicate*. *predicate* receives the raw main-query candidate list (as produced by ``join_hints_multi``) and must return one of the ``id`` strings in that list. Raises ``RuntimeError`` when *predicate* picks an id that is not valid so tests fail fast on fixture mistakes."""

    def _forced(
        q_norm: str,
        deterministic_sql: str,
        *,
        llm_scopes: list[dict[str, Any]],
        preset_choices: dict[str, str] | None = None,
        accept_na_by_scope: dict[str, bool] | None = None,
        require_final: bool = False,
    ) -> dict[str, str]:
        out = dict(preset_choices or {})
        for block in llm_scopes:
            sk = str(block.get("scope") or "")
            candidates = list(block.get("candidates") or [])
            valid = {str(c.get("candidate_id")) for c in candidates if c.get("candidate_id")}
            if not valid:
                out[sk] = "J00"
                continue
            chosen = predicate(candidates)
            if chosen not in valid:
                raise RuntimeError(f"forced_join_choice_patch picked {chosen!r} which is not in {sorted(valid)!r}")
            out[sk] = chosen
        return out

    return patch("aetherdialect._sql_gen.get_join_choice_from_llm", side_effect=_forced)


def capture_join_candidates() -> Any:
    """Capture arguments passed to ``aetherdialect._sql_gen.get_join_choice_from_llm``. Returns a ``patch`` context whose ``side_effect`` records ``q_norm`` and ``llm_scopes`` on the shared ``calls`` list. The patched callable still forwards to the real implementation so join choice proceeds as-is."""
    import aetherdialect._sql_gen

    calls: list[dict[str, Any]] = []
    original = aetherdialect._sql_gen.get_join_choice_from_llm

    def _recording(
        q_norm: str,
        deterministic_sql: str,
        *,
        llm_scopes: list[dict[str, Any]],
        preset_choices: dict[str, str] | None = None,
        accept_na_by_scope: dict[str, bool] | None = None,
        require_final: bool = False,
    ) -> dict[str, str]:
        calls.append({"q_norm": q_norm, "llm_scopes": list(llm_scopes)})
        return original(
            q_norm,
            deterministic_sql,
            llm_scopes=llm_scopes,
            preset_choices=preset_choices,
            accept_na_by_scope=accept_na_by_scope,
            require_final=require_final,
        )

    ctx = patch("aetherdialect._sql_gen.get_join_choice_from_llm", side_effect=_recording)
    ctx._calls = calls
    return ctx


def _kit_baseline_templates(runner: LiveTestRunner) -> dict[str, str]:
    """Seed four trust=2 templates with realistic per-pair feedback counts. Returns a dict mapping a short alias to the inserted template id so callers can reference seeded rows in assertions."""
    from aetherdialect._contracts_core import FeedbackCounts

    out: dict[str, str] = {}
    for alias, q_norm, intent, sql in (
        (
            "first_names",
            "list customer first names",
            intent_customer_first_names(),
            "SELECT customer.customer_id, customer.first_name FROM customer",
        ),
        (
            "full_names",
            "list customer first and last names",
            intent_customer_full_names(),
            "SELECT customer.customer_id, customer.first_name, customer.last_name FROM customer",
        ),
        (
            "rentals_per_store",
            "rental count per store baseline",
            intent_rental_count_by_store(),
            "SELECT inventory.store_id, COUNT(rental.rental_id) FROM rental "
            "JOIN inventory ON rental.inventory_id = inventory.inventory_id "
            "GROUP BY inventory.store_id",
        ),
        (
            "payments_per_staff",
            "total payments per staff baseline",
            intent_payment_sum_by_staff(),
            "SELECT staff_id, SUM(amount) FROM payment GROUP BY staff_id",
        ),
    ):
        tmpl = seed_template(
            runner,
            q_norm=q_norm,
            intent=intent,
            sql=sql,
            trust_level=2,
            stats=TemplateStats(accept=4, reject=1),
        )
        tmpl.feedback_by_question[q_norm] = FeedbackCounts(accepts=4, rejects=1, last_path=1)
        out[alias] = tmpl.id
    return out


def _kit_cold_templates(runner: LiveTestRunner) -> dict[str, str]:
    """Seed two trust=1 templates with stats=(1, 0) for promotion-gate tests."""
    out: dict[str, str] = {}
    for alias, q_norm, intent, sql in (
        (
            "first_names_cold",
            "list customer first names",
            intent_customer_first_names(),
            "SELECT customer.customer_id, customer.first_name FROM customer",
        ),
        (
            "full_names_cold",
            "list customer first and last names",
            intent_customer_full_names(),
            "SELECT customer.customer_id, customer.first_name, customer.last_name FROM customer",
        ),
    ):
        tmpl = seed_template(
            runner,
            q_norm=q_norm,
            intent=intent,
            sql=sql,
            trust_level=1,
            stats=TemplateStats(accept=1, reject=0),
        )
        out[alias] = tmpl.id
    return out


def _kit_rejected_aggregations(runner: LiveTestRunner) -> dict[str, str]:
    """Seed two ``INTENT_REJECTED`` feedback rows representing wrong- aggregation feedback."""
    out: dict[str, str] = {}
    rt_a = seed_rejected(
        runner,
        q_norm="rentals per store wrong agg",
        intent=intent_rental_count_by_store(),
        sql="SELECT inventory.store_id, SUM(rental.rental_id) FROM rental "
        "JOIN inventory ON rental.inventory_id = inventory.inventory_id "
        "GROUP BY inventory.store_id",
        reason="seeded wrong aggregation: SUM where COUNT was expected",
    )
    out["rentals_wrong_agg"] = str(rt_a.id)
    rt_b = seed_rejected(
        runner,
        q_norm="payments per staff wrong agg",
        intent=intent_payment_sum_by_staff(),
        sql="SELECT staff_id, COUNT(amount) FROM payment GROUP BY staff_id",
        reason="seeded wrong aggregation: COUNT where SUM was expected",
    )
    out["payments_wrong_agg"] = str(rt_b.id)
    return out


def _kit_rejected_join_paths(runner: LiveTestRunner) -> dict[str, str]:
    """Seed two ``INTENT_REJECTED`` feedback rows representing wrong- join-edge feedback."""
    work_intent = intent_store_staff_by_work()
    work_intent.chosen_join_candidate_id = "J99"
    work_intent.chosen_join_path_signature = ["store.manager_staff_id->staff.staff_id"]
    rt_work = seed_rejected(
        runner,
        q_norm="staff working at each store wrong edge",
        intent=work_intent,
        sql="SELECT staff.first_name, staff.last_name, store.store_id, store.last_update "
        "FROM store JOIN staff ON store.manager_staff_id = staff.staff_id",
        reason="seeded wrong join edge: chose manager FK for work-assignment question",
    )
    manager_intent = intent_store_manager()
    manager_intent.chosen_join_candidate_id = "J98"
    manager_intent.chosen_join_path_signature = ["staff.store_id->store.store_id"]
    rt_manager = seed_rejected(
        runner,
        q_norm="store manager wrong edge",
        intent=manager_intent,
        sql="SELECT store.store_id, staff.first_name, staff.last_name "
        "FROM staff JOIN store ON staff.store_id = store.store_id",
        reason="seeded wrong join edge: chose work FK for manager question",
    )
    return {
        "work_wrong_edge": str(rt_work.id),
        "manager_wrong_edge": str(rt_manager.id),
    }


def _kit_intent_failures(runner: LiveTestRunner) -> dict[str, str]:
    """Seed distinct structural validation rows scoped to the runner's schema."""
    from aetherdialect._utils import normalize_question

    seed_negative_memory(
        runner,
        intent=intent_customer_first_names(),
        sql="SELECT customer.customer_id FROM customer",
        reason="intent parse failed: unknown column 'thing'",
        q_norm=normalize_question("show me a thing that does not parse"),
    )
    seed_negative_memory(
        runner,
        intent=intent_payment_sum_by_staff(),
        sql="SELECT staff_id, SUM(amount) FROM payment",
        reason="validation failed: missing group_by_cols for grouped grain",
        q_norm=normalize_question("grouped query missing group by column"),
    )
    seed_negative_memory(
        runner,
        intent=intent_rental_count_by_store(),
        sql=(
            "SELECT inventory.store_id, COUNT(rental.rental_id) FROM rental "
            "JOIN inventory ON rental.inventory_id = inventory.inventory_id "
            "WHERE film.film_id = 1 GROUP BY inventory.store_id"
        ),
        reason="validation failed: filter references table not in tables list",
        q_norm=normalize_question("filter on unrelated table"),
    )
    seed_negative_memory(
        runner,
        intent=intent_customer_full_names(),
        sql="SELECT customer.customer_id, customer.first_name FROM customer",
        reason="validation failed: select_cols missing customer.last_name",
        q_norm=normalize_question("list customer first and last names with hints"),
    )
    return {}


def _kit_negative_memory_full(runner: LiveTestRunner) -> dict[str, str]:
    """Seed all three negative-memory sources for one canonical template/intent. Inserts the accepted template, a matching intent-level rejection for the same shape, and one validation row keyed to the same q_norm so ``compute_question_feedback_penalty`` observes stacked feedback."""
    intent = intent_rental_count_by_store()
    sql = (
        "SELECT inventory.store_id, COUNT(rental.rental_id) FROM rental "
        "JOIN inventory ON rental.inventory_id = inventory.inventory_id "
        "GROUP BY inventory.store_id"
    )
    q_norm = "rentals per store negative memory full"
    tmpl = seed_template(
        runner,
        q_norm=q_norm,
        intent=intent,
        sql=sql,
        trust_level=2,
        stats=TemplateStats(accept=4, reject=1),
    )
    rt = seed_rejected(
        runner,
        q_norm=q_norm,
        intent=intent,
        sql=sql,
        reason="seeded rejection co-located with accepted template",
    )
    pay = intent_payment_sum_by_staff()
    seed_negative_memory(
        runner,
        intent=pay,
        sql="SELECT staff_id, COUNT(amount) FROM payment GROUP BY staff_id",
        reason="seeded failure-log row co-located with accepted template",
        q_norm=q_norm,
    )
    return {"template": tmpl.id, "rejected": str(rt.id), "q_norm": q_norm}


def _kit_multi_pair_template(runner: LiveTestRunner) -> dict[str, str]:
    """Seed one trust=1 template with three distinct ``feedback_by_question`` pairs. Pair A: accepts=2, rejects=0 (eligible for promotion). Pair B: accepts=0, rejects=1 (mid-rejection). Pair C: accepts=1, rejects=0 (single accept)."""
    from aetherdialect._contracts_core import FeedbackCounts

    tmpl = seed_template(
        runner,
        q_norm="multi pair template seed",
        intent=intent_customer_first_names(),
        sql="SELECT customer.customer_id, customer.first_name FROM customer",
        trust_level=1,
        stats=TemplateStats(accept=3, reject=1),
    )
    tmpl.feedback_by_question["pair a list customer first names"] = FeedbackCounts(accepts=2, rejects=0, last_path=1)
    tmpl.feedback_by_question["pair b show customer first names"] = FeedbackCounts(accepts=0, rejects=1, last_path=2)
    tmpl.feedback_by_question["pair c give me customer first names"] = FeedbackCounts(accepts=1, rejects=0, last_path=3)
    return {"template": tmpl.id}


_KITS: dict[str, Callable[[LiveTestRunner], dict[str, str]]] = {
    "baseline_templates": _kit_baseline_templates,
    "cold_templates": _kit_cold_templates,
    "rejected_aggregations": _kit_rejected_aggregations,
    "rejected_join_paths": _kit_rejected_join_paths,
    "intent_failures": _kit_intent_failures,
    "negative_memory_full": _kit_negative_memory_full,
    "multi_pair_template": _kit_multi_pair_template,
}


@contextmanager
def seeded_runner(
    schema: Any,
    schema_terms: set[str],
    t2s: Any,
    *,
    label: str,
    kits: tuple[str, ...] = (),
) -> Iterator[LiveTestRunner]:
    """Yield an isolated ``LiveTestRunner`` pre-populated by the requested *kits*. Each kit name in ``kits`` is applied in order against the fresh runner. Unknown kit names raise ``KeyError`` immediately so fixture typos surface at test setup time. The mapping ``runner.seeded_ids`` is attached so tests can look up artifact ids by alias (e.g. ``runner.seeded_ids["baseline_templates"]["first_names"]``)."""
    with isolated_runner(schema, schema_terms, t2s, label=label) as runner:
        seeded_ids: dict[str, dict[str, str]] = {}
        for kit_name in kits:
            if kit_name not in _KITS:
                raise KeyError(f"unknown kit name: {kit_name!r}; available: {sorted(_KITS)!r}")
            seeded_ids[kit_name] = _KITS[kit_name](runner)
        runner.seeded_ids = seeded_ids
        yield runner


def snapshot_store(runner: LiveTestRunner) -> dict[str, Any]:
    """Return a structural snapshot of *runner*'s store sections for before/after diffing. The snapshot captures id sets, per-template trust level, per- template stats, and per-template feedback_by_question dict so tests can pinpoint which row changed."""
    return {
        "template_ids": frozenset(runner.templates),
        "rejected_ids": frozenset(),
        "rejected_intent_ids": frozenset(),
        "question_feedback_keys": frozenset((runner.store.get("question_feedback") or {}).keys()),
        "question_feedback_total_entries": _question_feedback_entry_count(runner.store),
        "intent_failure_count": _question_feedback_entry_count(runner.store),
        "trust_by_id": {tid: t.trust_level for tid, t in runner.templates.items()},
        "stats_by_id": {tid: (t.stats.accept, t.stats.reject) for tid, t in runner.templates.items()},
        "feedback_by_question_by_id": {
            tid: {qn: (fc.accepts, fc.rejects, fc.last_path) for qn, fc in (t.feedback_by_question or {}).items()}
            for tid, t in runner.templates.items()
        },
    }


def assert_template_unchanged(before: dict[str, Any], after: dict[str, Any], template_id: str) -> None:
    """Assert *template_id*'s trust_level, stats, and feedback_by_question are unchanged."""
    _before_missing = f"[assert_template_unchanged] template {template_id!r} missing from before snapshot"
    assert template_id in before["trust_by_id"], _before_missing
    _after_missing = f"[assert_template_unchanged] template {template_id!r} was deleted between snapshots"
    assert template_id in after["trust_by_id"], _after_missing
    assert before["trust_by_id"][template_id] == after["trust_by_id"][template_id], (
        f"[assert_template_unchanged] trust changed for {template_id!r}: "
        f"{before['trust_by_id'][template_id]} -> {after['trust_by_id'][template_id]}"
    )
    assert before["stats_by_id"][template_id] == after["stats_by_id"][template_id], (
        f"[assert_template_unchanged] stats changed for {template_id!r}: "
        f"{before['stats_by_id'][template_id]} -> {after['stats_by_id'][template_id]}"
    )
    assert before["feedback_by_question_by_id"][template_id] == after["feedback_by_question_by_id"][template_id], (
        f"[assert_template_unchanged] feedback_by_question changed for {template_id!r}: "
        f"{before['feedback_by_question_by_id'][template_id]} -> "
        f"{after['feedback_by_question_by_id'][template_id]}"
    )


def assert_new_template_forked(before: dict[str, Any], after: dict[str, Any]) -> set[str]:
    """Assert at least one new template id appeared; return the set of new ids."""
    new_ids = set(after["template_ids"]) - set(before["template_ids"])
    assert new_ids, (
        f"[assert_new_template_forked] no new templates inserted; "
        f"before={sorted(before['template_ids'])!r} after={sorted(after['template_ids'])!r}"
    )
    return new_ids


def assert_new_rejected_template(before: dict[str, Any], after: dict[str, Any]) -> set[str]:
    """Assert at least one new ``question_feedback`` row appeared (compat name for rejection tests)."""
    b = int(before.get("question_feedback_total_entries", 0))
    a = int(after.get("question_feedback_total_entries", 0))
    assert a > b, f"[assert_new_rejected_template] no new question_feedback rows; before_count={b!r} after_count={a!r}"
    return set()


__all__ = [
    "assert_new_rejected_template",
    "assert_new_template_forked",
    "assert_template_unchanged",
    "capture_join_candidates",
    "capture_parse_prompt",
    "deterministic_join_choice_patch",
    "empty_store",
    "forced_join_choice_patch",
    "intent_customer_emails_only",
    "intent_customer_first_names",
    "intent_customer_full_names",
    "intent_film_in_category",
    "intent_payment_sum_by_staff",
    "intent_rental_count_by_store",
    "intent_store_manager",
    "intent_store_staff_by_work",
    "isolated_runner",
    "run_isolated_scenario",
    "seed_negative_memory",
    "seed_rejected",
    "seed_template",
    "seeded_runner",
    "snapshot_store",
]

# --- engine live helpers ---


ENGINE_MODULE_FRAGMENTS = (
    "test_databricks",
    "test_mysql",
    "test_sqlserver",
    "test_oracle",
    "test_snowflake",
    "test_bigquery",
    "test_redshift",
    "test_mariadb",
    "test_duckdb",
    "test_sqlite",
    "test_mysql_dialect",
    "test_sqlserver_dialect",
    "test_oracle_dialect",
    "test_snowflake_dialect",
    "test_bigquery_dialect",
    "test_redshift_dialect",
    "test_duckdb_dialect",
    "test_sqlite_dialect",
)

_SESSION_ENGINE_CACHE: dict[tuple[str, str], AetherEngine] = {}

_RENTAL_SHOP_VIEWS_SQL = PROFILE_VIEWS_SQL

_RENTAL_SHOP_VIEW_NAMES = ("active_customer_v", "store_revenue_v", "film_catalog_v")


def _apply_rental_shop_views(instance: AetherEngine) -> None:
    """Create rental_shop views on the live engine connection when the DDL file is present."""
    if not _RENTAL_SHOP_VIEWS_SQL.is_file():
        return
    dialect = instance._dialect
    for name in _RENTAL_SHOP_VIEW_NAMES:
        dialect.execute(f'DROP VIEW IF EXISTS "{name}"')
    for stmt in _RENTAL_SHOP_VIEWS_SQL.read_text(encoding="utf-8").split(";"):
        stripped = stmt.strip()
        if stripped:
            dialect.execute(stripped)


def _runtime_schema_name() -> str:
    """Return the active engine schema/database name for information_schema queries."""
    runtime = EngineConfig.RUNTIME
    schema = getattr(runtime, "SCHEMA", None)
    if schema:
        return str(schema)
    database = getattr(runtime, "DATABASE", None)
    if database:
        return str(database)
    return "main"


def _rental_shop_views_from_catalog(instance: AetherEngine) -> dict[str, TableMetadata]:
    """Load rental_shop view columns from the engine catalog."""
    dialect_name = str(getattr(instance._dialect, "name", "") or EngineConfig.TYPE or "").lower()
    if dialect_name == "sqlite":
        grouped: dict[str, dict[str, ColumnMetadata]] = {}
        for view_name in _RENTAL_SHOP_VIEW_NAMES:
            rows = instance._dialect.execute(f'PRAGMA table_info("{view_name}")')
            columns: dict[str, ColumnMetadata] = {}
            for row in rows or ():
                if len(row) < 2:
                    continue
                col_name = str(row[1])
                col_type = str(row[2]) if len(row) > 2 and row[2] is not None else "TEXT"
                columns[col_name] = ColumnMetadata(name=col_name, data_type=col_type)
            if columns:
                grouped[view_name] = columns
        return {
            tname: TableMetadata(
                name=tname,
                columns=columns,
                primary_key=[],
                foreign_keys=[],
                kind="view",
            )
            for tname, columns in grouped.items()
        }

    schema_name = "main" if dialect_name == "duckdb" else _runtime_schema_name()
    names_sql = ", ".join(f"'{name}'" for name in _RENTAL_SHOP_VIEW_NAMES)
    sql = (
        "SELECT table_name, column_name, data_type "
        "FROM information_schema.columns "
        f"WHERE table_schema = '{schema_name}' "
        f"AND table_name IN ({names_sql}) "
        "ORDER BY table_name, ordinal_position"
    )
    rows = instance._dialect.execute(sql)
    grouped: dict[str, dict[str, ColumnMetadata]] = {}
    for table_name, column_name, data_type in rows:
        tname = str(table_name)
        grouped.setdefault(tname, {})
        grouped[tname][str(column_name)] = ColumnMetadata(
            name=str(column_name),
            data_type=str(data_type),
        )
    return {
        tname: TableMetadata(
            name=tname,
            columns=columns,
            primary_key=[],
            foreign_keys=[],
            kind="view",
        )
        for tname, columns in grouped.items()
    }


def _merge_rental_shop_views_into_graph(instance: AetherEngine) -> None:
    """Ensure bundled rental_shop views exist in the database and schema graph."""
    _apply_rental_shop_views(instance)
    for name, table in _rental_shop_views_from_catalog(instance).items():
        instance._schema_graph.tables[name] = table


def reflect_rental_shop_schema_for_live_test(
    instance: AetherEngine,
    *,
    include: SchemaInclude,
) -> SchemaGraph:
    """Reflect rental_shop tables and/or views for local-engine live view tests."""
    view_tables = _rental_shop_views_from_catalog(instance)
    base = instance._schema_graph
    if include == "views":
        tables = dict(view_tables)
        if not tables:
            tables = {name: table for name, table in base.tables.items() if table.kind == "view"}
    elif include == "both":
        tables = {name: table for name, table in base.tables.items() if table.kind != "view"}
        tables.update(view_tables)
    else:
        tables = {name: table for name, table in base.tables.items() if table.kind != "view"}
    return replace(base, tables=tables, include=include)


def _redirect_to_livetest_dir(instance: AetherEngine) -> str:
    """Swap artifact paths from ``artifacts_...`` to ``livetest_...`` for the engine instance."""
    original = instance._artifacts_dir
    parent = os.path.dirname(original)
    folder = os.path.basename(original)
    live_folder = folder.replace("artifacts_", "livetest_", 1)
    if live_folder == folder:
        live_folder = f"livetest_{folder}"
    live_dir = os.path.join(parent, live_folder)

    if os.path.isdir(original):
        if os.path.isdir(live_dir):
            shutil.rmtree(live_dir, ignore_errors=True)
        shutil.copytree(original, live_dir, dirs_exist_ok=True)
    else:
        os.makedirs(live_dir, exist_ok=True)

    schema_dst = os.path.join(live_dir, "schema_graph.json.gz")
    if not os.path.exists(schema_dst):
        schema_src_gz = os.path.join(original, "schema_graph.json.gz")
        schema_src_json = os.path.join(original, "schema_graph.json")
        if os.path.exists(schema_src_gz):
            shutil.copy2(schema_src_gz, schema_dst)
        elif os.path.exists(schema_src_json):
            with open(schema_src_json, encoding="utf-8") as sf:
                schema_payload = json.load(sf)
            write_gzip_json_atomic(schema_dst, schema_payload, sort_keys=True)

    template_store_dir = os.path.join(live_dir, "intent_templates")
    if os.path.isdir(template_store_dir):
        shutil.rmtree(template_store_dir, ignore_errors=True)

    instance._artifacts_dir = live_dir
    EngineConfig.SCHEMA_JSON_PATH = schema_dst
    EngineConfig.TEMPLATE_STORE_DIR = template_store_dir
    QSimConfig.SKELETONS_JSON_PATH = os.path.join(live_dir, "qsim_skeletons.json.gz")

    return live_dir


def build_engine_t2s(engine_name: str, schema: str) -> AetherEngine:
    """Build a ``AetherEngine`` instance configured for *engine_name* over the rental_shop schema."""
    from .conftest import (
        _domain_notes_path,
        _env_file,
        _relax_rental_shop_selectability,
    )

    cache_key = (engine_name, schema)
    cached = _SESSION_ENGINE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    sql_file = os.environ.get("SQL_FILE", str(PROFILE_SQL_DEFAULT))
    notes = _domain_notes_path()

    cfg_path = write_live_env_file_to_temp_config_toml(_env_file(), {"AETHERDIALECT_ENGINE": engine_name})
    try:
        with llm_usage_build_scope():
            instance = AetherEngine(
                EngineContext(
                    notes_file=str(notes) if notes else None,
                    sql_file=sql_file,
                ),
                artifacts_dir=tempfile.mkdtemp(prefix=f"live_{engine_name}_artifacts_"),
                config_file=cfg_path,
            )

            _redirect_to_livetest_dir(instance)

            fresh_store = TemplateOps.load_template_store(
                instance._schema_graph.effective_structural_hash, instance._schema_graph
            )
            instance._store = fresh_store
            instance._templates = TemplateOps.store_to_templates(fresh_store)
            instance._rejected = {}

            _relax_rental_shop_selectability(instance._schema_graph, schema)
            if engine_name in ("duckdb", "sqlite"):
                _merge_rental_shop_views_into_graph(instance)
                apply_synthetic_rental_partition_metadata(instance._schema_graph)

            _SESSION_ENGINE_CACHE[cache_key] = instance
            return instance
    finally:
        Path(cfg_path).unlink(missing_ok=True)


def build_runner(instance: AetherEngine) -> LiveTestRunner:
    """Wire a ``LiveTestRunner`` to an engine ``AetherEngine`` instance and instrument it."""
    from .conftest import _instrument_runner

    r = LiveTestRunner(
        schema=instance._schema_graph,
        store=instance._store,
        templates=instance._templates,
        rejected=instance._rejected,
        schema_terms=instance._schema_terms,
        csv_dir=instance._artifacts_dir,
        dialect=instance._dialect,
    )
    _instrument_runner(r)
    return r


def engine_schema(name: str, default: str) -> str:
    """Resolve the target schema/dataset env var for an engine, falling back to *default*."""
    return os.environ.get(name, default)


def skip_unless_configured(engine_name: str) -> Any:
    """Return a pytest skip marker reason when *engine_name* is not configured in the env file."""
    from .conftest import _env_file, _is_example_live_env

    env_path = _env_file()
    if _is_example_live_env(env_path) or not Path(env_path).is_file():
        return f"{engine_name} not configured in live env file"

    flat_present = False
    try:
        cfg_path = write_live_env_file_to_temp_config_toml(env_path, {"AETHERDIALECT_ENGINE": engine_name})
        text = Path(cfg_path).read_text(encoding="utf-8")
        Path(cfg_path).unlink(missing_ok=True)
        flat_present = f"[{engine_name}]" in text
    except Exception:
        flat_present = False
    return None if flat_present else f"{engine_name} not configured in live env file"
