"""Regression checks that model-facing prompt text stays schema- neutral."""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Iterable
from typing import Any

import pytest

from aetherdialect._constants_runtime import (
    INTERPRET_FIELDS,
    PROMPT_NEUTRALITY_AUDIT_CONSTANTS,
    SANDBOX_MEMBER_SPACE_TABLES,
    UPLOAD_PROMPT_NEUTRALITY_AUDIT_CONSTANTS,
)
from aetherdialect._contracts_core import InterpretPlan
from aetherdialect._contracts_schema import ColumnMetadata, LogicalIntent, SchemaGraph, TableMetadata
from aetherdialect._federation_manifest import federation_prompt_fields_for_schema
from aetherdialect._intent_loop import (
    _build_intent_compose_prompt,
    build_intent_ground_prompt,
    build_intent_interpret_prompt,
    build_intent_semantic_repair_prompt,
)
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._sql_gen import _serialize_join_candidate_row, build_join_choice_prompt
from tests.federation_helpers import build_two_member_federation

_CONSTANTS = importlib.import_module("aetherdialect._constants")
_CONSTANTS_RUNTIME = importlib.import_module("aetherdialect._constants_runtime")

_SINGLE_DATABASE_PHRASES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bthis database\b", re.IGNORECASE),
    re.compile(r"\bsingle database\b", re.IGNORECASE),
    re.compile(r"\bone database\b", re.IGNORECASE),
    re.compile(r"\bone sql statement\b", re.IGNORECASE),
    re.compile(r"\bone join graph\b", re.IGNORECASE),
)

_MULTI_SOURCE_PHRASES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bmultiple databases?\b", re.IGNORECASE),
    re.compile(r"\bseveral databases?\b", re.IGNORECASE),
    re.compile(r"\bper-database\b", re.IGNORECASE),
    re.compile(r"\bper-source\b", re.IGNORECASE),
    re.compile(r"\bmember engines?\b", re.IGNORECASE),
    re.compile(r"\bacross members\b", re.IGNORECASE),
    re.compile(r"\bcross-source\b", re.IGNORECASE),
    re.compile(r"\bfederated plan\b", re.IGNORECASE),
    re.compile(r"\bfederation manifest\b", re.IGNORECASE),
    re.compile(r"\bfederation mappings?\b", re.IGNORECASE),
    re.compile(r"\bdifferent sources\b", re.IGNORECASE),
    re.compile(r"\bdecomposed across members\b", re.IGNORECASE),
    re.compile(r"\bmember statements?\b", re.IGNORECASE),
    re.compile(r"\blogical union\b", re.IGNORECASE),
    re.compile(r"\breplica semantics\b", re.IGNORECASE),
    re.compile(r"\blogical table mappings?\b", re.IGNORECASE),
)

_PIPELINE_JARGON_PHRASES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\baetherspace\b", re.IGNORECASE),
    re.compile(r"\bfederation\b", re.IGNORECASE),
    re.compile(r"\bmember\b", re.IGNORECASE),
    re.compile(r"\bsource_label\b", re.IGNORECASE),
    re.compile(r"\bwithheld\b", re.IGNORECASE),
    re.compile(r"\bhidden_columns\b", re.IGNORECASE),
    re.compile(r"\bout-of-scope\b", re.IGNORECASE),
    re.compile(r"\bactive space\b", re.IGNORECASE),
)

_PIPELINE_JARGON_EXEMPT_CONSTANTS: frozenset[str] = frozenset(
    {
        # SCHEMA_CLASSIFY may name restricted/hidden sensitivity enum values.
        "SCHEMA_CLASSIFY_SYSTEM",
    }
)

_VERTICAL_TOKENS: frozenset[str] = frozenset(
    {
        "dvdrental",
        "rental_shop",
        "film_actor",
        "game_supported_language",
        "inventory_status_history",
        "promotion_redemption",
        "purchase_line",
        "purchase_order",
        "stock_transfer",
        "damage_report",
        "item_category",
        "item_feature",
    }
) | frozenset(
    token
    for tables in SANDBOX_MEMBER_SPACE_TABLES.values()
    for token in tables
    if token
    not in {
        "actor",
        "address",
        "author",
        "book",
        "category",
        "city",
        "country",
        "courier",
        "customer",
        "delivery",
        "film",
        "game",
        "inventory",
        "item",
        "language",
        "payment",
        "promotion",
        "publisher",
        "rental",
        "reservation",
        "staff",
        "store",
        "supplier",
        "warehouse",
    }
)


def _iter_text_fragments(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_text_fragments(item)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _iter_text_fragments(item)


def _audit_constant_text(name: str) -> str:
    if hasattr(_CONSTANTS, name):
        value = getattr(_CONSTANTS, name)
    else:
        value = getattr(_CONSTANTS_RUNTIME, name)
    return "\n".join(_iter_text_fragments(value))


def _matching_patterns(text: str, patterns: tuple[re.Pattern[str], ...]) -> list[str]:
    return [pattern.pattern for pattern in patterns if pattern.search(text)]


def _graph_with_original_names(*, source_id: str = "main") -> SchemaGraph:
    table = TableMetadata(
        name="customer",
        original_name="Customer Account",
        columns={
            "customer_id": ColumnMetadata(
                name="customer_id",
                original_name="Customer ID",
                data_type="integer",
                sensitivity="none",
            ),
        },
        primary_key=["customer_id"],
        foreign_keys=[],
        source_id=source_id,
    )
    return SchemaGraph(
        tables={"customer": table},
        join_paths_multi=recompute_join_paths_multi({"customer": table}),
        effective_structural_hash="original_name_probe",
    )


@pytest.mark.fast
def test_model_facing_constants_avoid_single_database_claims() -> None:
    violations: list[str] = []
    for name in sorted(PROMPT_NEUTRALITY_AUDIT_CONSTANTS):
        matches = _matching_patterns(_audit_constant_text(name), _SINGLE_DATABASE_PHRASES)
        if matches:
            violations.append(f"{name}: {matches}")
    assert not violations


@pytest.mark.fast
def test_model_facing_constants_avoid_multi_source_disclosure() -> None:
    violations: list[str] = []
    for name in sorted(PROMPT_NEUTRALITY_AUDIT_CONSTANTS):
        matches = _matching_patterns(_audit_constant_text(name), _MULTI_SOURCE_PHRASES)
        if matches:
            violations.append(f"{name}: {matches}")
    assert not violations


@pytest.mark.fast
def test_model_facing_constants_avoid_vertical_table_names() -> None:
    violations: list[str] = []
    for name in sorted(PROMPT_NEUTRALITY_AUDIT_CONSTANTS):
        lowered = _audit_constant_text(name).lower()
        hits = sorted(token for token in _VERTICAL_TOKENS if re.search(rf"\b{re.escape(token)}\b", lowered))
        if hits:
            violations.append(f"{name}: {hits}")
    assert not violations


@pytest.mark.fast
def test_model_facing_constants_avoid_pipeline_jargon() -> None:
    violations: list[str] = []
    for name in sorted(PROMPT_NEUTRALITY_AUDIT_CONSTANTS):
        if name in _PIPELINE_JARGON_EXEMPT_CONSTANTS:
            continue
        matches = _matching_patterns(_audit_constant_text(name), _PIPELINE_JARGON_PHRASES)
        if matches:
            violations.append(f"{name}: {matches}")
    assert not violations


@pytest.mark.fast
def test_prompt_neutrality_audit_constants_cover_every_system_prompt() -> None:
    system_prompts = {
        name
        for name, val in sorted(vars(_CONSTANTS_RUNTIME).items())
        if name.endswith("_SYSTEM") and isinstance(val, str)
    }
    audited = PROMPT_NEUTRALITY_AUDIT_CONSTANTS | UPLOAD_PROMPT_NEUTRALITY_AUDIT_CONSTANTS
    missing = sorted(system_prompts - audited)
    assert not missing, f"prompt neutrality audit missing *_SYSTEM prompts: {missing}"
    non_system_extras = sorted(name for name in PROMPT_NEUTRALITY_AUDIT_CONSTANTS if not name.endswith("_SYSTEM"))
    assert non_system_extras, "PROMPT_NEUTRALITY_AUDIT_CONSTANTS should retain non-_SYSTEM rule constants"


@pytest.mark.fast
def test_interpret_and_ground_prompts_carry_no_federation_fields() -> None:
    graph = _graph_with_original_names()
    interpret_payload = graph.schema_payload_json(INTERPRET_FIELDS)
    ground_payload = graph.schema_payload_ground()
    interpret = json.loads(
        build_intent_interpret_prompt("how many customers?", interpret_payload, "", ()),
    )
    ground = json.loads(
        build_intent_ground_prompt(
            "how many customers?",
            InterpretPlan(
                approach="count customers",
                tables=("customer",),
                grounding=(),
                schema_invalid=False,
            ),
            ground_payload,
            "",
            "[]",
            (),
            (),
        ),
    )
    for payload in (interpret, ground):
        assert federation_prompt_fields_for_schema(graph) == {}
        assert not any(key.startswith("federation_") for key in payload)


@pytest.mark.fast
def test_schema_payloads_omit_original_name() -> None:
    graph = _graph_with_original_names()
    payloads = (
        graph.schema_payload_json(INTERPRET_FIELDS),
        graph.schema_payload_ground(),
        graph.schema_payload_compose(["customer"]),
        graph.schema_literal_json,
    )
    for payload in payloads:
        assert '"original_name"' not in payload.lower()


@pytest.mark.fast
def test_composite_schema_and_compose_payloads_omit_original_name() -> None:
    fed = build_two_member_federation()
    payloads = (
        fed.composite.schema_payload_compose([fed.left_table, fed.right_table]),
        fed.composite.schema_literal_json,
        _build_intent_compose_prompt(
            LogicalIntent(tables=(fed.left_table,), select="count rows"),
            fed.composite.schema_payload_compose([fed.left_table]),
            schema_graph=fed.composite,
        ),
    )
    for payload in payloads:
        assert '"original_name"' not in payload.lower()
        assert '"source_id"' not in payload.lower()


@pytest.mark.fast
def test_join_choice_and_semantic_repair_payloads_omit_original_name() -> None:
    graph = _graph_with_original_names()
    join_row = _serialize_join_candidate_row(
        {
            "candidate_id": "J01",
            "join_path_signature": ["customer.customer_id->orders.customer_id"],
            "edge_kinds": ["catalog_fk"],
            "candidate_tier": "base",
        },
    )
    assert "original_name" not in json.dumps(join_row).lower()
    _, join_user = build_join_choice_prompt(
        "how many customers?",
        "SELECT 1",
        [
            {
                "scope": "main",
                "tables": ["customer"],
                "candidates": [join_row],
            },
        ],
    )
    assert '"original_name"' not in join_user.lower()
    repair = build_intent_semantic_repair_prompt(
        "how many customers?",
        '{"tables":["customer"],"grain":"scalar","select_cols":[]}',
        [],
        [],
        graph.schema_literal_json,
    )
    assert '"original_name"' not in repair.lower()


@pytest.mark.fast
def test_compose_prompt_on_composite_avoids_federation_wording() -> None:
    fed = build_two_member_federation()
    payload = json.loads(
        _build_intent_compose_prompt(
            LogicalIntent(tables=(fed.left_table,), select="count rows"),
            fed.composite.schema_payload_compose([fed.left_table]),
            schema_graph=fed.composite,
        ),
    )
    joined = json.dumps(payload).lower()
    assert "do not emit sql union" in joined
    assert "federation" not in joined
    assert "logical union" not in joined
    assert "logical table mappings" not in joined
