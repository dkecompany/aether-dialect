"""Federation regression coverage for numeric agreement, artifacts, payloads, and sandbox seeds."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from aetherdialect._constants import FEDERATION_ARTIFACT_FORMAT_VERSION
from aetherdialect._contracts_base import FederationConfigError, MulGroup, NormalizedExpr
from aetherdialect._contracts_core import RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import (
    ColumnMetadata,
    FederationMappings,
    LogicalIntent,
    SchemaGraph,
    TableMetadata,
)
from aetherdialect._dialect import DialectRegistry
from aetherdialect._federation_compose import compose_composite_graph
from aetherdialect._federation_execute import (
    execute_federation_coordinator,
    load_federation_composite_graph,
    mappings_replay_matches,
    persist_federation_tree,
)
from aetherdialect._federation_manifest import (
    federation_artifact_paths,
    parse_federation_declaration,
)
from aetherdialect._federation_plan import (
    federation_plan_is_degenerate,
    plan_federated_intent,
)
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._pipeline_execute import prepare_federated_sql_plan
from aetherdialect._pipeline_generate import generate_and_validate_sql
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._sql_gen import build_deterministic_sql
from aetherdialect._templates_ops import TemplateOps
from tests.conftest import duckdb_engine_identity
from tests.federation_helpers import (
    build_two_member_federation,
    enriched_manifest,
    federation_member_graph,
    stamp_sandbox_payment_union_profiling,
)
from tests.test_federation_single_source import (
    _composed_manifest,
    _member_graphs,
    _runtime_manifest,
)

_DATA_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "data"


def _column(name: str, data_type: str = "integer") -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type=data_type, sensitivity="none")


def _table(
    name: str,
    *,
    source_id: str,
    columns: dict[str, ColumnMetadata],
    primary_key: list[str],
) -> TableMetadata:
    return TableMetadata(
        name=name,
        columns=columns,
        primary_key=primary_key,
        foreign_keys=[],
        source_id=source_id,
    )


def _member_graph(source_id: str, tables: dict[str, TableMetadata]) -> SchemaGraph:
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
    )


def _sandbox_federation_member_graphs() -> dict[str, SchemaGraph]:
    """Minimal member graphs covering sandbox declaration joins and replicas."""
    rental_cols = {
        "rental_id": _column("rental_id"),
        "inventory_id": _column("inventory_id"),
        "last_update": _column("last_update", "timestamp"),
    }
    inventory_cols = {
        "inventory_id": _column("inventory_id"),
        "last_update": _column("last_update", "timestamp"),
    }
    payment_cols = {
        "payment_id": _column("payment_id"),
        "rental_id": _column("rental_id"),
        "amount": _column("amount", "double"),
        "payment_date": _column("payment_date", "timestamp"),
    }
    customer_cols = {
        "customer_id": _column("customer_id"),
        "first_name": _column("first_name", "text"),
        "last_name": _column("last_name", "text"),
        "address_id": _column("address_id"),
        "create_date": _column("create_date", "timestamp"),
        "email": _column("email", "text"),
        "loyalty_tier": _column("loyalty_tier", "text"),
        "store_id": _column("store_id"),
    }
    crm_customer_cols = {
        "customer_id": _column("customer_id"),
        "first_name": _column("first_name", "text"),
        "last_name": _column("last_name", "text"),
        "address_id": _column("address_id"),
        "create_date": _column("create_date", "timestamp"),
        "email_addr": _column("email_addr", "text"),
        "loyalty_tier": _column("loyalty_tier", "text"),
        "store_id": _column("store_id"),
    }
    staff_storefront_cols = {
        "staff_id": _column("staff_id"),
        "first_name": _column("first_name", "text"),
        "last_name": _column("last_name", "text"),
        "store_id": _column("store_id"),
    }
    staff_crm_cols = {key: staff_storefront_cols[key] for key in staff_storefront_cols}
    item_cols = {"item_id": _column("item_id")}
    purchase_line_cols = {"item_id": _column("item_id")}
    delivery_cols = {"rental_id": _column("rental_id")}
    receipts_cols = {
        "rcpt_id": _column("rcpt_id"),
        "rent_id": _column("rent_id"),
        "amt": _column("amt", "double"),
        "dt": _column("dt", "timestamp"),
    }
    promotion_redemption_cols = {"rental_id": _column("rental_id")}
    city_cols = {
        "city_id": _column("city_id"),
        "city": _column("city", "text"),
        "country_id": _column("country_id"),
        "last_update": _column("last_update", "timestamp"),
    }
    country_cols = {
        "country_id": _column("country_id"),
        "country": _column("country", "text"),
        "last_update": _column("last_update", "timestamp"),
    }
    return {
        "storefront": _member_graph(
            "storefront",
            {
                "rental": _table("rental", source_id="storefront", columns=rental_cols, primary_key=["rental_id"]),
                "payment": _table("payment", source_id="storefront", columns=payment_cols, primary_key=["payment_id"]),
                "customer": _table(
                    "customer",
                    source_id="storefront",
                    columns=customer_cols,
                    primary_key=["customer_id"],
                ),
                "staff": _table(
                    "staff", source_id="storefront", columns=staff_storefront_cols, primary_key=["staff_id"]
                ),
                "city": _table("city", source_id="storefront", columns=city_cols, primary_key=["city_id"]),
                "country": _table("country", source_id="storefront", columns=country_cols, primary_key=["country_id"]),
            },
        ),
        "catalog": _member_graph(
            "catalog",
            {
                "inventory": _table(
                    "inventory",
                    source_id="catalog",
                    columns=inventory_cols,
                    primary_key=["inventory_id"],
                ),
                "item": _table("item", source_id="catalog", columns=item_cols, primary_key=["item_id"]),
                "payment": _table("payment", source_id="catalog", columns=payment_cols, primary_key=["payment_id"]),
                "city": _table("city", source_id="catalog", columns=city_cols, primary_key=["city_id"]),
                "country": _table("country", source_id="catalog", columns=country_cols, primary_key=["country_id"]),
            },
        ),
        "logistics": _member_graph(
            "logistics",
            {
                "purchase_line": _table(
                    "purchase_line",
                    source_id="logistics",
                    columns=purchase_line_cols,
                    primary_key=["item_id"],
                ),
                "delivery": _table("delivery", source_id="logistics", columns=delivery_cols, primary_key=["rental_id"]),
                "receipts": _table(
                    "receipts",
                    source_id="logistics",
                    columns=receipts_cols,
                    primary_key=["rcpt_id"],
                ),
            },
        ),
        "crm": _member_graph(
            "crm",
            {
                "promotion_redemption": _table(
                    "promotion_redemption",
                    source_id="crm",
                    columns=promotion_redemption_cols,
                    primary_key=["rental_id"],
                ),
                "customer": _table("customer", source_id="crm", columns=crm_customer_cols, primary_key=["customer_id"]),
                "staff": _table("staff", source_id="crm", columns=staff_crm_cols, primary_key=["staff_id"]),
            },
        ),
    }


def _assert_payload_has_no_source_leaks(payload: str) -> None:
    """Model-facing schema payloads must not expose member source identifiers."""
    lowered = payload.lower()
    assert '"source_id"' not in lowered
    assert '"member_source_ids"' not in lowered
    assert '"column_member_sources"' not in lowered


def _amount_graph(table: str, *, source_id: str) -> SchemaGraph:
    return federation_member_graph(
        table,
        source_id=source_id,
        columns={
            "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
            "amount": ColumnMetadata(name="amount", data_type="double", sensitivity="none"),
        },
    )


@pytest.mark.fast
def test_coordinator_inner_join_row_count_matches_manual_merge(two_member_federation) -> None:
    """Cross-source coordinator row counts must match a manual key merge, not a cartesian product."""
    fed = two_member_federation
    left = pd.DataFrame({"id": [1, 2, 3]})
    right = pd.DataFrame({"id": [2, 3, 4]})
    intent = RuntimeIntent(
        tables=[fed.left_table, fed.right_table],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, fed.composite, fed.manifest)
    frames = {fed.left_source: left, fed.right_source: right}
    result = execute_federation_coordinator(frames, plan, row_cap=100)
    manual = left.merge(right, on="id")
    assert len(result) == len(manual) == 2
    assert sorted(int(v) for v in result["id"].tolist()) == [2, 3]


@pytest.mark.fast
def test_degenerate_scalar_sum_agrees_between_direct_sql_and_coordinator() -> None:
    """A single-source federated scalar must match standalone SQL executed on the member frame."""
    manifest = enriched_manifest(
        {
            "a": _amount_graph("payment", source_id="a"),
            "b": _amount_graph("payment_b", source_id="b"),
        },
        {
            "federation_id": "fed_scalar",
            "cross_source_joins": [],
        },
        member_graphs={
            "a": _amount_graph("payment", source_id="a"),
            "b": _amount_graph("payment_b", source_id="b"),
        },
    )
    members = {
        "a": _amount_graph("payment", source_id="a"),
        "b": _amount_graph("payment_b", source_id="b"),
    }
    composite = compose_composite_graph(members, manifest)
    intent = RuntimeIntent(
        tables=["payment"],
        grain="scalar",
        select_cols=[
            SelectCol(
                expr=NormalizedExpr(
                    agg_func="sum",
                    add_groups=[MulGroup(multiply=["payment.amount"])],
                ),
            ),
        ],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest, member_graphs=members)
    assert federation_plan_is_degenerate(plan)
    member_frame = pd.DataFrame({"id": [1, 2, 3], "amount": [10.0, 20.5, 4.5]})
    expected = float(member_frame["amount"].sum())
    dialect = DialectRegistry.get("duckdb")
    direct_sql = build_deterministic_sql(intent, schema=members["a"], dialect=dialect)
    federated_sql = build_deterministic_sql(plan.steps[0].sub_intent, schema=members["a"], dialect=dialect)
    duckdb = pytest.importorskip("duckdb")
    conn = duckdb.connect()
    conn.register("payment", member_frame)
    direct_value = float(conn.execute(direct_sql).fetchone()[0])
    federated_value = float(conn.execute(federated_sql).fetchone()[0])
    assert direct_value == pytest.approx(expected)
    assert federated_value == pytest.approx(expected)
    assert federated_value == pytest.approx(direct_value)


@pytest.mark.fast
def test_degenerate_federated_prepare_matches_direct_member_sql() -> None:
    """Degenerate federated SQL preparation must be byte-identical to the standalone member path."""
    manifest = _composed_manifest()
    member_graphs = _member_graphs()
    composite = compose_composite_graph(member_graphs, manifest)
    intent = RuntimeIntent(
        tables=["film"],
        grain="many",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("film.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    assert federation_plan_is_degenerate(plan)
    default = DialectRegistry.get("duckdb")
    runtimes = MainExecutionOps._build_federation_source_runtimes(
        _runtime_manifest(), None, default, default_identity=duckdb_engine_identity()
    )
    store = TemplateOps.empty_template_store(composite.schema_graph_id)
    with patch(
        "aetherdialect._pipeline_generate.run_sql_validation_cascade",
        return_value=(True, "", None, []),
    ):

        class _Owner:
            _federation_member_graphs = member_graphs
            _federation_dialects = {sid: runtime.dialect for sid, runtime in runtimes.items()}

        owner = _Owner()
        single_source = MainExecutionOps.federation_single_source_sql_context(
            owner,
            intent,
            composite,
            manifest,
            None,
            default,
        )
        assert single_source is not None
        source_dialect, member_schema = single_source
        direct = generate_and_validate_sql(
            "list films",
            intent,
            member_schema,
            {},
            {},
            source_dialect,
            store,
        )
        fed = prepare_federated_sql_plan(
            "list films",
            plan,
            composite,
            dialect=default,
            dialects_by_source={sid: runtime.dialect for sid, runtime in runtimes.items()},
            join_candidates={},
            cmap={},
            store=store,
            source_runtimes=runtimes,
            manifest=manifest,
            member_graphs=member_graphs,
        )
    assert direct.success and fed.success
    assert direct.sql == fed.display_sql
    assert fed.steps[0].sql == direct.sql


@pytest.mark.fast
def test_federation_artifact_version_mismatch_rejects_stale_tree() -> None:
    """Persisted federation trees with an outdated artifact_format_version must fail closed."""
    fed = build_two_member_federation()
    mappings = FederationMappings(version="0.2.3")
    with tempfile.TemporaryDirectory() as tmp:
        persist_federation_tree(
            tmp,
            manifest=fed.manifest,
            mappings=mappings,
            composite=fed.composite,
            member_graphs=fed.member_graphs,
        )
        manifest_path = federation_artifact_paths(tmp)["artifact_manifest"]
        with open(manifest_path, encoding="utf-8") as handle:
            stored = json.load(handle)
        stored["artifact_format_version"] = "0.0.0"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(stored, handle)
        with pytest.raises(FederationConfigError, match=r"artifact_format_version") as exc_info:
            mappings_replay_matches(tmp, fed.member_graphs, fed.manifest, mappings)
        msg = str(exc_info.value)
        assert str(FEDERATION_ARTIFACT_FORMAT_VERSION) in msg
        with pytest.raises(FederationConfigError, match=r"artifact_format_version"):
            load_federation_composite_graph(tmp)


@pytest.mark.fast
def test_composite_schema_literal_payload_omits_source_id() -> None:
    """Composite schema literals sent to the model must not leak member source identifiers."""
    fed = build_two_member_federation()
    for payload in (
        fed.composite.schema_literal_json,
        fed.composite.structural_schema_literal_json([fed.left_table, fed.right_table]),
        fed.composite.schema_payload_compose([fed.left_table, fed.right_table]),
        fed.composite.schema_literal_json_for_tables(frozenset({fed.left_table})),
    ):
        _assert_payload_has_no_source_leaks(payload)
    assert fed.composite.tables[fed.left_table].source_id == fed.left_source


@pytest.mark.fast
def test_federation_mapping_confirm_scenario_targets_payment_union() -> None:
    """Sandbox mapping-confirm scenario must reference the payment union mapping."""
    scenarios = json.loads((_DATA_ROOT / "sandbox_scenarios.json").read_text(encoding="utf-8"))
    scenario = next(row for row in scenarios["scenarios"] if row["id"] == "federation_mapping_confirm")
    assert scenario["mechanism"] == "federation_mapping_confirm"
    assert scenario["recipe"] == "federation"
    question = str(scenario["question"])
    assert "payment" in question.lower()

    _, mappings = parse_federation_declaration(
        json.loads((_DATA_ROOT / "federation_declaration.json").read_text(encoding="utf-8")),
    )
    payment_union = next(row for row in mappings.logical_tables if row.logical == "payment")
    assert payment_union.semantics == "union"
    assert {member.source for member in payment_union.members} == {"storefront", "catalog", "logistics"}

    expectations = json.loads((_DATA_ROOT / "sandbox_expectations.json").read_text(encoding="utf-8"))
    payment_slot = next(row for row in expectations["slots"] if row.get("question") == question)
    assert payment_slot["expect"]["grain"] == "scalar"
    assert payment_slot["expect"]["scalar_value"] == pytest.approx(6947.73, abs=0.02)
    assert payment_slot["expect"]["terminal_status"] == "ok"


@pytest.mark.fast
def test_federation_sandbox_schema_files_present() -> None:
    """Bundled federation schemas under scripts/data must exist."""
    declaration_path = _DATA_ROOT / "federation_declaration.json"
    if not declaration_path.is_file():
        pytest.skip("federation sandbox schemas are not present under scripts/data")
    manifest, mappings = parse_federation_declaration(json.loads(declaration_path.read_text(encoding="utf-8")))
    assert manifest.federation_id == "sandbox_rental_shop"
    assert manifest.cross_source_joins
    assert mappings.logical_tables
    partition_path = _DATA_ROOT / "federation_partition.json"
    assert partition_path.is_file()
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    assert "rental" in partition["storefront"]
    assert "inventory" in partition["catalog"]


@pytest.mark.fast
def test_federation_sandbox_offline_smoke_when_schemas_exist() -> None:
    """When partition schemas exist, federation declaration composes a multi-source graph."""
    declaration_path = _DATA_ROOT / "federation_declaration.json"
    if not declaration_path.is_file():
        pytest.skip("federation sandbox schemas are not present under scripts/data")

    manifest, mappings = parse_federation_declaration(json.loads(declaration_path.read_text(encoding="utf-8")))
    members = _sandbox_federation_member_graphs()
    stamp_sandbox_payment_union_profiling(members)
    composite = compose_composite_graph(members, manifest, mappings)
    assert composite.tables["rental"].source_id == "storefront"
    assert composite.tables["inventory"].source_id == "catalog"
    assert set(composite.tables["customer"].member_source_ids) == {"storefront", "crm"}
    _assert_payload_has_no_source_leaks(composite.schema_literal_json)

    runtime_manifest = enriched_manifest(members, manifest, member_graphs=members, mappings=mappings)
    runtimes = MainExecutionOps._build_federation_source_runtimes(
        runtime_manifest, None, DialectRegistry.get("duckdb"), default_identity=duckdb_engine_identity()
    )
    assert set(runtimes) == {"storefront", "catalog", "logistics", "crm"}


@pytest.mark.fast
def test_federation_compose_prompt_avoids_union_vocabulary() -> None:
    """Federated compose prompts must not leak union/replica mapping vocabulary."""
    from aetherdialect._constants_runtime import FEDERATION_COMPOSE_SUPPORTED_CAPABILITIES
    from aetherdialect._contracts_schema import SchemaGraph
    from aetherdialect._intent_loop import _build_intent_compose_prompt
    from aetherdialect._schema_graph import recompute_join_paths_multi

    storefront = TableMetadata(
        name="rental",
        columns={"rental_id": ColumnMetadata(name="rental_id", data_type="integer", sensitivity="none")},
        primary_key=["rental_id"],
        foreign_keys=[],
        source_id="storefront",
    )
    catalog = TableMetadata(
        name="inventory",
        columns={"inventory_id": ColumnMetadata(name="inventory_id", data_type="integer", sensitivity="none")},
        primary_key=["inventory_id"],
        foreign_keys=[],
        source_id="catalog",
    )
    composite = SchemaGraph(
        tables={"rental": storefront, "inventory": catalog},
        join_paths_multi=recompute_join_paths_multi({"rental": storefront, "inventory": catalog}),
    )
    user_json = json.loads(
        _build_intent_compose_prompt(
            logical=LogicalIntent(tables=("rental",), select="count rentals"),
            structural_payload=json.dumps({"tables": []}),
            schema_graph=composite,
        )
    )
    caps = user_json["supported_capabilities"]
    assert caps == list(FEDERATION_COMPOSE_SUPPORTED_CAPABILITIES)
    joined = "\n".join(caps).lower()
    assert "do not emit sql union" in joined
    for phrase in ("logical union", "replica semantics", "logical table mappings"):
        assert phrase not in joined
    assert not any(cap.startswith("Not expressible: UNION") for cap in caps)


@pytest.mark.fast
def test_federation_batch_join_choice_attribution_phase() -> None:
    """Batch member join-choice LLM calls record join_choice usage attribution."""
    from aetherdialect._contracts_core import FederatedPlan, RuntimeIntent, SourceStep
    from aetherdialect._pipeline_execute import _federation_batch_member_join_presets
    from aetherdialect._utils import (
        drain_llm_usage_records,
        llm_usage_build_scope,
        llm_usage_session_scope,
        record_llm_usage,
        reset_llm_usage_accumulator,
    )

    reset_llm_usage_accumulator()
    fed = build_two_member_federation()
    plan = FederatedPlan(
        steps=(
            SourceStep(
                source_id=fed.left_source,
                sub_intent=RuntimeIntent(
                    tables=[fed.left_table, fed.right_table],
                    grain="many",
                    select_cols=[],
                    group_by_cols=[],
                    order_by_cols=[],
                    where=None,
                ),
            ),
        ),
        combine=(),
    )

    def _fake_join(*_args: object, **_kwargs: object) -> dict[str, str]:
        record_llm_usage(
            task="join_choice",
            logical_model="test",
            api_model="test",
            provider="sandbox",
            input_tokens=1,
            cached_input_tokens=0,
            output_tokens=1,
            cache_write_tokens=None,
            attempt=1,
            elapsed_ms=1,
        )
        return {"jc0": "J01"}

    with llm_usage_session_scope():
        with llm_usage_build_scope():
            with (
                patch(
                    "aetherdialect._pipeline_execute.join_scope_pass1_plan",
                    return_value=(
                        {},
                        [{"scope": "main", "candidates": [{"candidate_id": "J01"}, {"candidate_id": "J02"}]}],
                        {},
                        {},
                    ),
                ),
                patch("aetherdialect._pipeline_execute.get_join_choice_from_llm", side_effect=_fake_join),
            ):
                _federation_batch_member_join_presets(
                    "count rows",
                    plan,
                    fed.composite,
                    dialect=DialectRegistry.get("duckdb"),
                    dialects_by_source=None,
                    manifest=fed.manifest,
                    member_graphs=fed.member_graphs,
                    source_runtimes=None,
                )
            records = drain_llm_usage_records()
    assert records
    join_records = [row for row in records if row.task == "join_choice"]
    assert join_records
    assert join_records[0].phase == "join_choice"
    reset_llm_usage_accumulator()


@pytest.mark.fast
def test_sandbox_replica_mappings_compose() -> None:
    """Sandbox customer and staff replica mappings compose across storefront and crm."""
    manifest, mappings = parse_federation_declaration(
        json.loads((_DATA_ROOT / "federation_declaration.json").read_text(encoding="utf-8")),
    )
    members = _sandbox_federation_member_graphs()
    stamp_sandbox_payment_union_profiling(members)
    composite = compose_composite_graph(members, manifest, mappings)
    assert "customer" in composite.tables
    assert set(composite.tables["customer"].member_source_ids) == {"storefront", "crm"}
    staff = composite.tables["staff"]
    assert set(staff.member_source_ids) == {"storefront", "crm"}
    assert staff.source_id == "storefront"
