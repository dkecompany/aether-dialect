"""Hermetic leak-proof checks: order independence and refusal uniformity."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._constants import (
    DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED,
    REFUSAL_TIMING_FLOOR_MS,
    SESSION_KIND_RESULT,
)
from aetherdialect._constants_runtime import (
    PERMISSION_DENIED_USER_MESSAGE,
)
from aetherdialect._contracts_base import (
    ConfigError,
    DomainKnowledgeEntry,
    KnowledgeScope,
    NormalizedExpr,
    SpaceContext,
)
from aetherdialect._contracts_core import RuntimeIntent, SelectCol, SessionStep
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, TableMetadata
from aetherdialect._intent_loop import full_intent_parse
from aetherdialect._main_session import PipelineSession
from aetherdialect._main_spaces import MainSpaceOps
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._schema_profile import NotesExtractionLedger, NotesExtractionResult
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils import domain_context_payload, stash_intent_parse_refusal
from tests.support.egress_assert import (
    assert_no_forbidden_identifiers,
    forbidden_identifiers_for_scope,
)
from tests.test_template_rbac_scope import _template_on_tables


def _session_owner() -> MagicMock:
    owner = MagicMock()
    owner._schema_graph = MagicMock()
    owner._schema_graph.effective_structural_hash = "test_hash"
    owner._store = {}
    owner._templates = {}
    owner._rejected = {}
    owner._schema_terms = set()
    owner._runtime_config = MagicMock(llm_execution=MagicMock())
    owner._audit_emit = MagicMock()
    return owner


def _runtime_intent() -> RuntimeIntent:
    return RuntimeIntent(
        tables=["secret"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("secret.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )


def _terminal_refusal_step(*, outcome: str) -> SessionStep:
    session = PipelineSession(_session_owner())
    session._turn_question = "blocked question"
    session._last_turn_outcome = {
        "outcome": outcome,
        "error": None,
        "sql": None,
        "rows": None,
        "columns": None,
        "rejection_bucket": None,
        "intent": _runtime_intent(),
        "matched_template": None,
        "template_history_index": None,
        "federated_bundle": None,
        "federated_plan": None,
        "generation_path": None,
        "federation_source_id": None,
        "federation_phase": None,
        "federation_succeeded": (),
        "failure_kind": None,
        "retryable": None,
        "refusal_diagnostic_code": DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED,
    }
    with (
        patch.object(session, "_emit_turn_llm_usage", return_value=()),
        patch("aetherdialect._main_session.turn_elapsed_ms", return_value=7),
    ):
        return session._completed_step()


def _step_payload(step: SessionStep) -> str:
    error_payload = None
    if step.error is not None:
        error_payload = {
            "code": step.error.code.value,
            "detail_code": step.error.detail_code,
            "source_id": step.error.source_id,
            "phase": step.error.phase,
            "limit_key": step.error.limit_key,
        }
    return json.dumps(
        {
            "done": step.done,
            "kind": step.kind,
            "answer": step.answer,
            "error": error_payload,
            "diagnostics": [
                {
                    "code": d.code,
                    "level": str(d.level),
                    "message": d.message,
                    "details": list(d.details),
                }
                for d in step.diagnostics
            ],
        },
        sort_keys=True,
    )


@pytest.mark.fast
@pytest.mark.parametrize(
    "outcome",
    [
        "permission_denied",
        "parse_failed",
        "not_available_in_context",
    ],
)
def test_refusal_uniformity_same_detail_code_for_permission_collapse(outcome: str) -> None:
    step = _terminal_refusal_step(outcome=outcome)
    assert step.error is not None
    assert step.error.detail_code == DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED
    assert step.answer is None


@pytest.mark.fast
def test_refusal_terminal_steps_set_elapsed_ms() -> None:
    step = _terminal_refusal_step(outcome="permission_denied")
    assert step.elapsed_ms is not None
    assert step.elapsed_ms >= 0


@pytest.mark.fast
def test_refusal_timing_floor_applies_across_refusal_paths() -> None:
    with patch("aetherdialect._main_session.turn_elapsed_ms", return_value=3):
        fast_path = _terminal_refusal_step(outcome="permission_denied")
    with patch("aetherdialect._main_session.turn_elapsed_ms", return_value=11):
        parse_path = _terminal_refusal_step(outcome="parse_failed")
    assert fast_path.elapsed_ms >= REFUSAL_TIMING_FLOOR_MS
    assert parse_path.elapsed_ms >= REFUSAL_TIMING_FLOOR_MS


def _scope_pair_schema() -> SchemaGraph:
    tables = {
        name: TableMetadata(
            name=name,
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
        )
        for name in ("orders", "payroll", "staff")
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id="leak_proof_scope_pairs",
        effective_structural_hash="leak_proof_scope_pairs",
    )


def _scoped_result_step(table: str) -> SessionStep:
    return SessionStep(
        done=True,
        prompt=None,
        kind=SESSION_KIND_RESULT,
        sql=f"SELECT id FROM {table}",
        answer=f"Returned rows from {table}.",
        diagnostics=(),
    )


_SCOPE_PAIRS: tuple[tuple[frozenset[str], frozenset[str]], ...] = (
    (frozenset({"orders", "payroll", "staff"}), frozenset({"orders"})),
    (frozenset({"orders", "payroll"}), frozenset({"orders"})),
    (frozenset({"orders", "payroll", "staff"}), frozenset({"orders", "payroll"})),
)


@pytest.mark.fast
@pytest.mark.parametrize(("wider_allow", "narrower_allow"), _SCOPE_PAIRS)
def test_scope_difference_sweep_narrower_egress_excludes_wider_only_tables(
    wider_allow: frozenset[str],
    narrower_allow: frozenset[str],
) -> None:
    """For A ⊃ B, forbidden identifiers for B never appear in B egress; A may include wider tables."""
    graph = _scope_pair_schema()
    wider_scope = KnowledgeScope.from_engine_context(graph, wider_allow)
    narrower_scope = KnowledgeScope.from_engine_context(graph, narrower_allow)
    forbidden_narrow = forbidden_identifiers_for_scope(graph, narrower_scope)
    forbidden_wide = forbidden_identifiers_for_scope(graph, wider_scope)
    wider_only = sorted(wider_allow - narrower_allow)
    assert wider_only, "expected a strict superset scope pair"
    for table in wider_only:
        assert table in forbidden_narrow
        assert table not in forbidden_wide

    narrow_allowed = sorted(narrower_allow)[0]
    assert_no_forbidden_identifiers(_scoped_result_step(narrow_allowed), forbidden_narrow)

    wide_only_table = wider_only[0]
    assert_no_forbidden_identifiers(_scoped_result_step(wide_only_table), forbidden_wide)

    with pytest.raises(AssertionError, match="forbidden identifier"):
        assert_no_forbidden_identifiers(_scoped_result_step(wide_only_table), forbidden_narrow)


@pytest.mark.fast
def test_owner_first_then_consumer_refusal_payloads_match_consumer_first() -> None:
    stash_intent_parse_refusal(DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED, PERMISSION_DENIED_USER_MESSAGE)
    consumer_first = _terminal_refusal_step(outcome="permission_denied")
    owner = _session_owner()
    owner._templates["T0001"] = MagicMock()
    owner._store = {"templates": {"T0001": {}}}
    stash_intent_parse_refusal(DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED, PERMISSION_DENIED_USER_MESSAGE)
    owner_first = _terminal_refusal_step(outcome="permission_denied")
    assert _step_payload(consumer_first) == _step_payload(owner_first)
    assert consumer_first.error is not None
    assert consumer_first.error.detail_code == DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED


def _orders_only_intent() -> RuntimeIntent:
    return RuntimeIntent(
        tables=["orders"],
        grain="scalar",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )


def _owner_with_partitioned_spaces(artifacts_dir: Path, schema: SchemaGraph) -> MagicMock:
    graph_id = schema.schema_graph_id
    for space in ("space_a", "space_b"):
        store = TemplateOps.empty_template_store_for_space(graph_id, artifacts_dir=str(artifacts_dir), space_name=space)
        TemplateOps.save_template_store(store)

    owner = MagicMock()
    owner._artifacts_dir = str(artifacts_dir)
    owner._schema_graph = schema
    owner._store_by_space = {}
    owner._templates_by_space = {}
    owner._rejected = {}
    owner._schema_terms = set()

    def _session(**kwargs):
        space = str(kwargs.get("space", "master")).strip().lower()
        return PipelineSession(
            owner,
            mode=kwargs.get("mode", "writer"),
            space_name=space,
        )

    owner.session = _session
    return owner


def _template_list_export(
    templates: dict[str, object],
    *,
    visible_tables: frozenset[str],
) -> str:
    dialect = MagicMock()
    dialect.sqlglot_dialect = "duckdb"
    summaries = TemplateOps.list_stored_template_summaries(
        templates,
        space="master",
        dialect=dialect,
        visible_tables=visible_tables,
    )
    return json.dumps(
        [{"id": summary.id, "approval_state": summary.approval_state} for summary in summaries],
        sort_keys=True,
    )


def _domain_context_export(entries: tuple[DomainKnowledgeEntry, ...]) -> str:
    payload = domain_context_payload(entries)
    return json.dumps(payload or [], sort_keys=True)


@pytest.mark.fast
def test_cross_space_partition_read_guard_session_cannot_list_other_space_templates(tmp_path: Path) -> None:
    """Space A sessions must not enumerate templates persisted only in space B's partition."""
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    schema = _scope_pair_schema()
    owner = _owner_with_partitioned_spaces(artifacts_dir, schema)

    sess_b = owner.session(mode="writer", space="space_b")
    _, store_b, templates_b, _, _ = sess_b._resources()
    TemplateOps.insert_template(
        store_b,
        templates_b,
        schema,
        "staff payroll join in space b",
        _orders_only_intent(),
        "SELECT id FROM orders",
        dialect=MagicMock(),
        record_accept=True,
    )
    TemplateOps.save_template_store(store_b)

    sess_a = owner.session(mode="writer", space="space_a")
    _, store_a, templates_a, _, _ = sess_a._resources()
    dialect = MagicMock()
    dialect.sqlglot_dialect = "duckdb"
    summaries_a = TemplateOps.list_stored_template_summaries(
        templates_a,
        space="space_a",
        dialect=dialect,
    )
    assert summaries_a == ()
    assert (
        TemplateOps.resolve_template_for_question("staff payroll join in space b", templates_a, template_store=store_a)
        is None
    )

    wrong_space_store = TemplateOps.load_template_store(
        schema.schema_graph_id,
        schema,
        artifacts_dir=str(artifacts_dir),
        space_name="space_a",
    )
    wrong_space_templates = dict(TemplateOps.store_to_templates(wrong_space_store))
    assert (
        TemplateOps.resolve_template_for_question(
            "staff payroll join in space b",
            wrong_space_templates,
            template_store=wrong_space_store,
        )
        is None
    )


@pytest.mark.fast
def test_post_define_rescoping_scrubs_out_of_scope_description_text() -> None:
    """Define-path snapshot enrichment scrubs LLM prose that names out- of-scope entities."""
    graph = _scope_pair_schema()
    snapshot = MainSpaceOps.subset_graph_for_space(graph, SpaceContext(tables=frozenset({"orders"})))
    space_ctx = SpaceContext(tables=frozenset({"orders"}))
    bad_classifications = {
        "orders": ("fact", "Linked to payroll compensation records.", {}),
    }
    with (
        patch("aetherdialect._main_spaces.EngineConfig.llm_credentials_configured", return_value=True),
        patch(
            "aetherdialect._main_spaces.extract_knowledge_from_notes",
            return_value=NotesExtractionResult((), (), NotesExtractionLedger(())),
        ),
        patch(
            "aetherdialect._main_spaces.llm_enrich_schema_from_structural_knowledge",
            return_value=bad_classifications,
        ),
    ):
        out = MainSpaceOps.enrich_space_snapshot_with_notes(
            snapshot,
            graph,
            space_ctx,
            notes="space-only notes.\n",
        )
    desc = str((out.get("table_descriptions") or {}).get("orders") or "")
    assert "payroll" not in desc.lower()
    assert "staff" not in desc.lower()


@pytest.mark.fast
def test_post_define_notes_naming_out_of_scope_raise() -> None:
    """User-authored space notes that name out-of-scope identifiers hard-fail."""
    graph = _scope_pair_schema()
    snapshot = MainSpaceOps.subset_graph_for_space(graph, SpaceContext(tables=frozenset({"orders"})))
    space_ctx = SpaceContext(tables=frozenset({"orders"}))
    with (
        patch("aetherdialect._main_spaces.EngineConfig.llm_credentials_configured", return_value=False),
        pytest.raises(ConfigError, match="out-of-scope identifier"),
    ):
        MainSpaceOps.enrich_space_snapshot_with_notes(
            snapshot,
            graph,
            space_ctx,
            notes="Orders join the staff table for payroll.\n",
        )


@pytest.mark.fast
def test_post_migration_rescoping_scrubs_out_of_scope_description_text(tmp_path: Path) -> None:
    """Re-enrichment after migration scrubs out-of-scope description prose on write."""
    graph = _scope_pair_schema()
    engine_dir = tmp_path / "engine"
    engine_dir.mkdir()
    uid = "orders_only"
    snapshot = MainSpaceOps.subset_graph_for_space(graph, SpaceContext(tables=frozenset({"orders"})))
    snapshot["uid"] = uid
    snapshot["name"] = uid
    snapshot["notes"] = "space notes after migration.\n"
    MainSpaceOps.save_aetherspace_snapshot(str(engine_dir), uid, snapshot)
    bad_classifications = {
        "orders": ("fact", "Compensation history references the staff table.", {}),
    }
    with (
        patch("aetherdialect._main_spaces.EngineConfig.llm_credentials_configured", return_value=True),
        patch(
            "aetherdialect._main_spaces.extract_knowledge_from_notes",
            return_value=NotesExtractionResult((), (), NotesExtractionLedger(())),
        ),
        patch(
            "aetherdialect._main_spaces.llm_enrich_schema_from_structural_knowledge",
            return_value=bad_classifications,
        ),
    ):
        MainSpaceOps.reenrich_aetherspace_snapshots_with_notes(str(engine_dir), graph)
    loaded = MainSpaceOps.load_aetherspace_snapshot(str(engine_dir), uid)
    assert loaded is not None
    desc = str((loaded.get("table_descriptions") or {}).get("orders") or "")
    assert "staff" not in desc.lower()


@pytest.mark.fast
def test_owner_first_then_consumer_template_list_export_matches_consumer_first() -> None:
    """Wider owner cache must not change consumer-visible template enumeration."""
    visible = frozenset({"orders"})
    in_scope = _template_on_tables("orders", template_id="T0001")
    out_of_scope = _template_on_tables("orders", "staff", template_id="T0002")
    consumer_templates = {"T0001": in_scope}
    consumer_export = _template_list_export(consumer_templates, visible_tables=visible)

    owner_templates = {"T0001": in_scope, "T0002": out_of_scope}
    owner_export = _template_list_export(owner_templates, visible_tables=visible)
    assert consumer_export == owner_export


@pytest.mark.fast
def test_owner_first_then_consumer_domain_context_export_matches_consumer_first() -> None:
    """Scoped domain-knowledge export must match whether owner-only entries were present upstream."""
    graph = _scope_pair_schema()
    visible = frozenset({"orders"})
    visible_scope = KnowledgeScope.from_visible_tables(graph, visible)
    forbidden = forbidden_identifiers_for_scope(graph, visible_scope)
    consumer_entries = (
        DomainKnowledgeEntry(
            key="orders_metric", text="Count of order rows.", kind="metric", referenced_entities=frozenset()
        ),
    )
    owner_entries = consumer_entries + (
        DomainKnowledgeEntry(
            key="staff_secret",
            text="Staff table shift roster.",
            kind="glossary",
            referenced_entities=frozenset({"staff"}),
        ),
    )

    def _scoped_export(entries: tuple[DomainKnowledgeEntry, ...]) -> str:
        scoped = MainSpaceOps.secure_domain_knowledge_for_visibility(
            entries,
            security_schema=graph,
            visible_table_names=visible,
            all_schema_table_names=set(graph.tables.keys()),
        )
        return _domain_context_export(scoped)

    consumer_export = _scoped_export(consumer_entries)
    owner_export = _scoped_export(owner_entries)
    assert consumer_export == owner_export
    assert_no_forbidden_identifiers(json.loads(consumer_export), forbidden)


@pytest.mark.fast
def test_provider_boundary_interpret_prompt_excludes_forbidden_identifiers() -> None:
    """Captured interpret prompts at the LLM provider must not name out- of-scope schema entities."""
    graph = _scope_pair_schema()
    visible = frozenset({"orders"})
    scope = KnowledgeScope.from_visible_tables(graph, visible)
    forbidden = forbidden_identifiers_for_scope(graph, scope)
    captured: list[tuple[str, str]] = []

    def _capture_chat(system: str, user: str, *, task: str = "default", **_kwargs: object) -> str:
        captured.append((system, user))
        return '{"approach":"","tables":[],"schema_invalid":true}'

    with patch("aetherdialect._intent_loop.LLMProvider.chat", side_effect=_capture_chat):
        full_intent_parse(
            "how many orders?",
            graph,
            store=None,
            max_retries=0,
            visible_objects=visible,
        )

    assert captured, "expected at least one provider-boundary prompt capture"
    for index, (system, user) in enumerate(captured):
        assert_no_forbidden_identifiers(system, forbidden, path=f"prompt[{index}].system")
        assert_no_forbidden_identifiers(user, forbidden, path=f"prompt[{index}].user")
