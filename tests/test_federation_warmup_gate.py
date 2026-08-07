"""Tests for federation seed-warmup, history, query-log, and QSim gates."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._contracts_base import (
    ConfigError,
    FederationCoordinatorConfig,
    FederationManifest,
    FederationPlanTemplate,
    FederationSourceBinding,
)
from aetherdialect._contracts_core import FederatedPreparedStep, FederatedPrepareOutcome, RuntimeIntent
from aetherdialect._contracts_schema import SchemaGraph
from aetherdialect._federation import (
    FederationConfigError,
    assert_query_log_warmup_allowed,
    qsim_intent_eligible_on_federation,
)
from aetherdialect._pipeline import execute_federated_warmup_intent, persist_federated_warmup_learning
from aetherdialect.aetherdialect import AetherFederation


def _manifest() -> FederationManifest:
    return FederationManifest(
        federation_id="fed_gate",
        sources=(
            FederationSourceBinding(source_id="a", engine="duckdb", connection="", context="master", role="owner"),
            FederationSourceBinding(source_id="b", engine="duckdb", connection="", context="master", role="owner"),
        ),
        table_namespace={"t_a": "a", "t_b": "b"},
        cross_source_joins=(),
        coordinator=FederationCoordinatorConfig(
            row_cap=1000,
            default_source_row_cap=1000,
            default_source_timeout_ms=1000,
            semijoin_key_cap=1000,
        ),
    )


def _federation_stub() -> AetherFederation:
    fed = AetherFederation.__new__(AetherFederation)
    fed._members = {}
    fed._federation_manifest = _manifest()
    fed._artifacts_dir = "."
    fed._closed = False
    fed._sandbox_closed = False
    return fed


def _intent(tables: list[str], *, sql_param: str = "") -> RuntimeIntent:
    return RuntimeIntent(
        tables=tables,
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        sql_param=sql_param,
    )


@pytest.mark.fast
def test_query_log_warmup_refused_on_federated_engine() -> None:
    engine = SimpleNamespace(_is_aether_federation=True)
    with pytest.raises(FederationConfigError, match="run them on each source engine individually"):
        assert_query_log_warmup_allowed(engine)


@pytest.mark.fast
def test_query_log_warmup_allowed_without_manifest() -> None:
    engine = SimpleNamespace(_is_aether_federation=False)
    assert_query_log_warmup_allowed(engine)


@pytest.mark.fast
def test_query_log_warmup_allowed_with_mock_owner() -> None:
    engine = MagicMock()
    engine._is_aether_federation = False
    assert_query_log_warmup_allowed(engine)


@pytest.mark.fast
def test_aether_federation_seed_warmup_routes_not_refused() -> None:
    fed = _federation_stub()
    with pytest.raises(ConfigError, match="warmup is not supported on AetherFederation"):
        fed.run_seed_warmup("seed.txt")


@pytest.mark.fast
def test_aether_federation_history_warmup_refused_with_member_guidance() -> None:
    fed = _federation_stub()
    with pytest.raises(ConfigError, match="warmup is not supported on AetherFederation"):
        fed.run_seed_warmup_from_history("history.sql")


@pytest.mark.fast
def test_aether_federation_query_log_warmup_refused_with_member_guidance() -> None:
    fed = _federation_stub()
    with pytest.raises(ConfigError, match="warmup is not supported on AetherFederation"):
        fed.run_seed_warmup_from_query_log()


@pytest.mark.fast
def test_qsim_eligibility_probe_uses_row_level_grain() -> None:
    grains: list[str] = []

    def _capture_sources(intent: object, *_args: object, **_kwargs: object) -> set[str]:
        grains.append(str(getattr(intent, "grain", "")))
        return {"a"}

    with patch("aetherdialect._federation.source_ids_for_intent", side_effect=_capture_sources):
        assert qsim_intent_eligible_on_federation(
            ["t_a"],
            SchemaGraph(tables={}, join_paths_multi={}),
            _manifest(),
        )
    assert grains == ["row_level"]


@pytest.mark.fast
def test_execute_federated_warmup_routes_per_source_dialects() -> None:
    plan = MagicMock()
    plan.ineligible_reason = None
    plan.steps = (SimpleNamespace(source_id="a"), SimpleNamespace(source_id="b"))
    prepared = FederatedPrepareOutcome(
        success=True,
        plan=plan,
        display_sql="-- a\nSELECT 1\n-- b\nSELECT 2",
        steps=(
            FederatedPreparedStep(
                source_id="a",
                sub_intent=_intent(["t_a"]),
                sql="SELECT a_col FROM t_a",
            ),
            FederatedPreparedStep(
                source_id="b",
                sub_intent=_intent(["t_b"]),
                sql="SELECT b_col FROM t_b",
            ),
        ),
        per_source_sql=(("a", "SELECT a_col FROM t_a"), ("b", "SELECT b_col FROM t_b")),
        combine_hash="combine",
        step_fingerprints=(("a", "fp_a"), ("b", "fp_b")),
        composite_schema_graph_id="comp",
    )
    dialect_a = MagicMock(name="dialect_a")
    dialect_b = MagicMock(name="dialect_b")
    captured: dict[str, object] = {}

    def _prepare(*_args: object, **kwargs: object) -> FederatedPrepareOutcome:
        captured["dialects_by_source"] = kwargs.get("dialects_by_source")
        captured["persist_template_learning"] = kwargs.get("persist_template_learning")
        return prepared

    with (
        patch("aetherdialect._pipeline.plan_federated_intent", return_value=plan),
        patch("aetherdialect._pipeline.generate_join_candidates", return_value=({}, {}, {})),
        patch("aetherdialect._pipeline.prepare_federated_sql_plan", side_effect=_prepare),
        patch(
            "aetherdialect._pipeline.execute_federated_prepare",
            return_value=SimpleNamespace(rows=((1,),), bundle=None),
        ),
    ):
        ok, err, rows, sql, learning = execute_federated_warmup_intent(
            "q",
            _intent(["t_a", "t_b"]),
            SchemaGraph(tables={}, join_paths_multi={}),
            MagicMock(),
            federation_manifest=_manifest(),
            stores_by_source={"a": {"templates": {}}, "b": {"templates": {}}},
            dialects_by_source={"a": dialect_a, "b": dialect_b},
            persist_template_learning=True,
        )
    assert ok is True
    assert err is None
    assert rows == [(1,)]
    assert learning is prepared
    assert captured["dialects_by_source"] == {"a": dialect_a, "b": dialect_b}
    assert captured["persist_template_learning"] is False
    assert sql == prepared.display_sql
    assert prepared.per_source_sql == (("a", "SELECT a_col FROM t_a"), ("b", "SELECT b_col FROM t_b"))


@pytest.mark.fast
def test_federated_warmup_learning_lands_in_member_stores_and_plan() -> None:
    plan = MagicMock()
    plan.residual = None
    prepared = FederatedPrepareOutcome(
        success=True,
        plan=plan,
        display_sql="display",
        steps=(
            FederatedPreparedStep(
                source_id="a",
                sub_intent=_intent(["t_a"], sql_param="SELECT 1"),
                sql="SELECT 1",
            ),
            FederatedPreparedStep(
                source_id="b",
                sub_intent=_intent(["t_b"], sql_param="SELECT 2"),
                sql="SELECT 2",
            ),
        ),
        combine_hash="ch",
        step_fingerprints=(("a", "fa"), ("b", "fb")),
        composite_schema_graph_id="comp_id",
    )
    store_a: dict[str, object] = {"templates": {}, "next_id": 1}
    store_b: dict[str, object] = {"templates": {}, "next_id": 1}
    saved_plan: list[FederationPlanTemplate] = []
    tmpl_a = MagicMock(id="ta")
    tmpl_b = MagicMock(id="tb")

    def _insert(store: object, *_args: object, **kwargs: object) -> MagicMock:
        assert kwargs.get("federation_plan_only") is True
        return tmpl_a if store is store_a else tmpl_b

    with (
        patch("aetherdialect._templates.TemplateOps.insert_template", side_effect=_insert),
        patch("aetherdialect._templates.TemplateOps.save_template_store") as save_store,
        patch("aetherdialect._pipeline.stamp_federation_member_template"),
        patch("aetherdialect._pipeline.member_schema_slice", return_value=SchemaGraph(tables={}, join_paths_multi={})),
        patch(
            "aetherdialect._pipeline.save_federation_plan_template",
            side_effect=lambda *_a, **_k: saved_plan.append(_a[1]),
        ),
        patch("aetherdialect._pipeline.federation_plan_residual_hash", return_value="rh"),
        patch("aetherdialect._pipeline.federation_plan_topology_identity", return_value=("mh", "mth")),
        patch("aetherdialect._pipeline.intent_key", return_value="plan_key"),
    ):
        created = persist_federated_warmup_learning(
            "warmup question",
            _intent(["t_a", "t_b"]),
            prepared,
            SchemaGraph(tables={}, join_paths_multi={}),
            stores_by_source={"a": store_a, "b": store_b},
            dialects_by_source={"a": MagicMock(), "b": MagicMock()},
            member_graphs={
                "a": SchemaGraph(tables={}, join_paths_multi={}),
                "b": SchemaGraph(tables={}, join_paths_multi={}),
            },
            federation_dir="fed_dir",
            federation_manifest=_manifest(),
            question_phrases=["warmup question"],
        )
    assert [t.id for t in created] == ["ta", "tb"]
    assert save_store.call_count == 2
    assert len(saved_plan) == 1
    plan_tmpl = saved_plan[0]
    assert plan_tmpl.plan_id == "plan_key"
    assert plan_tmpl.member_template_ids == (("a", "ta"), ("b", "tb"))


@pytest.mark.fast
def test_federated_warmup_missing_member_store_raises() -> None:
    prepared = FederatedPrepareOutcome(
        success=True,
        plan=MagicMock(residual=None),
        display_sql="display",
        steps=(
            FederatedPreparedStep(
                source_id="a",
                sub_intent=_intent(["t_a"]),
                sql="SELECT 1",
            ),
            FederatedPreparedStep(
                source_id="b",
                sub_intent=_intent(["t_b"]),
                sql="SELECT 2",
            ),
        ),
    )
    with pytest.raises(FederationConfigError, match="federation member store missing for source_id 'b'"):
        persist_federated_warmup_learning(
            "q",
            _intent(["t_a", "t_b"]),
            prepared,
            SchemaGraph(tables={}, join_paths_multi={}),
            stores_by_source={"a": {"templates": {}, "next_id": 1}},
            federation_dir="fed_dir",
            federation_manifest=_manifest(),
        )


@pytest.mark.fast
def test_federated_warmup_failed_turn_persists_nothing() -> None:
    plan = MagicMock()
    plan.ineligible_reason = None
    with (
        patch("aetherdialect._pipeline.plan_federated_intent", return_value=plan),
        patch("aetherdialect._pipeline.generate_join_candidates", return_value=({}, {}, {})),
        patch(
            "aetherdialect._pipeline.prepare_federated_sql_plan",
            return_value=FederatedPrepareOutcome(
                success=False,
                plan=plan,
                display_sql="",
                sql_validation_error="prepare failed",
            ),
        ),
        patch("aetherdialect._pipeline.persist_federated_warmup_learning") as persist_learning,
    ):
        ok, err, rows, _sql, learning = execute_federated_warmup_intent(
            "q",
            _intent(["t_a", "t_b"]),
            SchemaGraph(tables={}, join_paths_multi={}),
            MagicMock(),
            federation_manifest=_manifest(),
            stores_by_source={"a": {}, "b": {}},
            persist_template_learning=True,
        )
    assert ok is False
    assert err == "prepare failed"
    assert rows is None
    assert learning is None
    persist_learning.assert_not_called()


@pytest.mark.fast
def test_federated_warmup_partial_failure_persists_nothing() -> None:
    from aetherdialect._contracts_base import FederationPartialFailureError

    plan = MagicMock()
    plan.ineligible_reason = None
    prepared = FederatedPrepareOutcome(success=True, plan=plan, display_sql="sql", steps=())
    with (
        patch("aetherdialect._pipeline.plan_federated_intent", return_value=plan),
        patch("aetherdialect._pipeline.generate_join_candidates", return_value=({}, {}, {})),
        patch("aetherdialect._pipeline.prepare_federated_sql_plan", return_value=prepared),
        patch(
            "aetherdialect._pipeline.execute_federated_prepare",
            side_effect=FederationPartialFailureError("member failed", source_id="a", phase="member", succeeded=()),
        ),
        patch("aetherdialect._pipeline.persist_federated_warmup_learning") as persist_learning,
    ):
        ok, err, rows, _sql, learning = execute_federated_warmup_intent(
            "q",
            _intent(["t_a", "t_b"]),
            SchemaGraph(tables={}, join_paths_multi={}),
            MagicMock(),
            federation_manifest=_manifest(),
            stores_by_source={"a": {}, "b": {}},
            persist_template_learning=True,
        )
    assert ok is False
    assert "member failed" in (err or "")
    assert rows is None
    assert learning is None
    persist_learning.assert_not_called()
