"""Join-choice prompts for federation must not disclose topology or member execution context."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from aetherdialect._contracts_core import FederatedPlan, RuntimeIntent, SourceStep
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._dialect import DialectRegistry
from aetherdialect._federation import parse_federation_manifest, plan_federated_intent, resolve_federated_combine
from aetherdialect._pipeline import _federation_batch_member_join_presets
from aetherdialect._sql_gen import build_join_choice_prompt
from tests.federation_helpers import build_two_member_federation


def _two_join_manifest() -> tuple[object, SchemaGraph]:
    manifest = parse_federation_manifest(
        {
            "federation_id": "fed_join_neutral",
            "sources": [
                {"source_id": "a", "engine": "duckdb", "role": "owner"},
                {"source_id": "b", "engine": "duckdb", "role": "owner"},
            ],
            "table_namespace": {"t_a": "a", "t_b": "b", "t_a_alt": "a", "t_b_alt": "b"},
            "cross_source_joins": [
                {
                    "left": "t_a.alt_id",
                    "right": "t_b.alt_id",
                    "kind": "inner",
                    "logical_key": "alt_id",
                },
                {
                    "left": "t_a.id",
                    "right": "t_b.id",
                    "kind": "inner",
                    "logical_key": "id",
                },
            ],
        },
        include_derived_roster=True,
    )
    composite = SchemaGraph(
        tables={
            name: TableMetadata(
                name=name,
                columns={
                    "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                    "alt_id": ColumnMetadata(name="alt_id", data_type="integer", sensitivity="none"),
                },
                primary_key=["id"],
                foreign_keys=[],
                source_id="",
            )
            for name in ("t_a", "t_b", "t_a_alt", "t_b_alt")
        },
        join_paths_multi={},
    )
    return manifest, composite


@pytest.mark.fast
def test_cross_source_join_choice_scopes_and_candidates_are_neutral() -> None:
    manifest, composite = _two_join_manifest()
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    captured: dict[str, object] = {}

    def _fake_llm(
        _q: str,
        deterministic_sql: str,
        *,
        llm_scopes: list[dict[str, object]],
        **_kwargs: object,
    ) -> dict[str, str]:
        captured["sql"] = deterministic_sql
        captured["scopes"] = llm_scopes
        return {str(scope["scope"]): str(scope["candidates"][0]["candidate_id"]) for scope in llm_scopes}

    with patch("aetherdialect._federation.get_join_choice_from_llm", side_effect=_fake_llm):
        resolve_federated_combine("count rows", plan, manifest, composite)

    scopes = captured["scopes"]
    assert isinstance(scopes, list) and scopes
    for scope in scopes:
        scope_key = str(scope["scope"])
        assert not scope_key.startswith("federation:")
        assert "|" not in scope_key
        for cand in scope["candidates"]:
            assert isinstance(cand, dict)
            assert set(cand.keys()) <= {"candidate_id", "join_path_signature"}
            assert "logical_key" not in cand
            assert "left" not in cand
            assert "right" not in cand
            assert "kind" not in cand
    sql = str(captured["sql"])
    assert sql.strip().upper() == "SELECT 1"


@pytest.mark.fast
def test_cross_source_join_choice_prompt_user_payload_is_neutral() -> None:
    manifest, composite = _two_join_manifest()
    intent = RuntimeIntent(
        tables=["t_a", "t_b"],
        grain="many",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    plan = plan_federated_intent(intent, composite, manifest)
    captured: dict[str, object] = {}

    def _fake_llm(
        q: str,
        deterministic_sql: str,
        *,
        llm_scopes: list[dict[str, object]],
        **_kwargs: object,
    ) -> dict[str, str]:
        captured["scopes"] = llm_scopes
        _, user = build_join_choice_prompt(q, deterministic_sql, llm_scopes)
        captured["user"] = user
        return {str(scope["scope"]): str(scope["candidates"][0]["candidate_id"]) for scope in llm_scopes}

    with patch("aetherdialect._federation.get_join_choice_from_llm", side_effect=_fake_llm):
        resolve_federated_combine("count rows", plan, manifest, composite)

    payload = json.loads(str(captured["user"]))
    joined = json.dumps(payload)
    assert "federation:" not in joined
    assert "logical_key" not in joined
    for scope in payload["scopes"]:
        for cand in scope["candidates"]:
            assert set(cand.keys()) <= {"candidate_id", "join_path_signature"}


@pytest.mark.fast
def test_batch_member_join_choice_scopes_are_opaque() -> None:
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
    captured: dict[str, object] = {}

    def _fake_llm(
        _q: str,
        _sql: str,
        *,
        llm_scopes: list[dict[str, object]],
        **_kwargs: object,
    ) -> dict[str, str]:
        captured["scopes"] = llm_scopes
        return {str(scope["scope"]): "J01" for scope in llm_scopes}

    with (
        patch(
            "aetherdialect._pipeline.join_scope_pass1_plan",
            return_value=(
                {},
                [{"scope": "main", "candidates": [{"candidate_id": "J01"}, {"candidate_id": "J02"}]}],
                {},
                {},
            ),
        ),
        patch("aetherdialect._pipeline.get_join_choice_from_llm", side_effect=_fake_llm),
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

    scopes = captured["scopes"]
    assert isinstance(scopes, list) and scopes
    for scope in scopes:
        scope_key = str(scope["scope"])
        assert not scope_key.startswith("fed:")
        assert "M0" not in scope_key
        assert "M1" not in scope_key
