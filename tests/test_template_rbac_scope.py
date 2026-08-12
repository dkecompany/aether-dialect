"""Template enumeration and parameter binding RBAC scope."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aetherdialect._contracts_base import NormalizedExpr, PredicateGroup, WhereParam
from aetherdialect._contracts_core import (
    ConcreteIntent,
    FeedbackKind,
    QuestionFeedbackEntry,
    RejectionBucket,
    RuntimeIntent,
    SelectCol,
    Template,
    ValueHistory,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, SQLShape, TableMetadata, TemplateStats
from aetherdialect._intent_loop import collect_structural_match_templates, find_trusted_template_match
from aetherdialect._pipeline_generate import extract_fuzzy_reuse_params, match_question_level_template_reuse
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils import normalize_question
from aetherdialect._utils_intent import intent_key
from tests.template_fixtures import _minimal_template


def _template_on_tables(*table_names: str, template_id: str = "T0001") -> Template:
    tables = list(table_names)
    primary = tables[0]
    intent_sig = ConcreteIntent(
        intent_id="i1",
        tables=tables,
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column(f"{primary}.category"),
                    op="=",
                    value_type="string",
                    param_key="p1",
                )
            ]
        ),
    )
    return Template(
        id=template_id,
        intent_signature=intent_sig,
        intent_key="k1",
        tables_used=tables,
        sql_param=f"SELECT 1 FROM {primary} WHERE category = :p1",
        sql_fp=f"fp-{template_id}",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False, num_where=1),
        colmap_sig="cm",
        value_history=ValueHistory(
            param_values=[{"p1": "x"}],
            questions=["count of item in category x"],
            natural_language=["nl"],
            accept_counts=[1],
        ),
        stats=TemplateStats(accept=1, reject=0),
        footprint_tables=tuple(tables),
    )


@pytest.mark.fast
def test_list_templates_hides_out_of_scope_footprint() -> None:
    in_scope = _template_on_tables("orders", template_id="T0001")
    out_of_scope = _template_on_tables("orders", "staff", template_id="T0002")
    dialect = MagicMock()
    dialect.sqlglot_dialect = "duckdb"
    summaries = TemplateOps.list_stored_template_summaries(
        {"T0001": in_scope, "T0002": out_of_scope},
        space="master",
        dialect=dialect,
        visible_tables=frozenset({"orders"}),
    )
    assert [s.id for s in summaries] == ["T0001"]


@pytest.mark.fast
def test_build_parameter_bindings_redacts_denied_column_value() -> None:
    tmpl = _minimal_template()
    tmpl.intent_signature = ConcreteIntent(
        intent_id="i1",
        tables=["orders"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("orders.secret_col"),
                    op="=",
                    value_type="string",
                    param_key="p1",
                )
            ]
        ),
    )
    tmpl.tables_used = ["orders"]
    tmpl.footprint_tables = ("orders",)
    tmpl.value_history = ValueHistory(
        param_values=[{"p1": "top-secret"}],
        questions=["show secret"],
        natural_language=["nl"],
        accept_counts=[1],
    )
    schema = SchemaGraph(
        tables={
            "orders": TableMetadata(
                name="orders",
                columns={
                    "secret_col": ColumnMetadata(name="secret_col", data_type="text", sensitivity="none"),
                    "category": ColumnMetadata(name="category", data_type="text", sensitivity="none"),
                },
                primary_key=["id"],
                foreign_keys=[],
            )
        },
        join_paths_multi={},
        deny_columns={"orders": {"secret_col"}},
    )
    bindings = TemplateOps.build_parameter_bindings(
        tmpl,
        history_index=0,
        schema=schema,
    )
    assert len(bindings) == 1
    assert bindings[0].handle == "p1"
    assert bindings[0].current_value is None


def _orders_schema() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "orders": TableMetadata(
                name="orders",
                columns={
                    "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                    "category": ColumnMetadata(name="category", data_type="text", sensitivity="none"),
                },
                primary_key=["id"],
                foreign_keys=[],
            ),
            "staff": TableMetadata(
                name="staff",
                columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
                primary_key=["id"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
    )


@pytest.mark.fast
def test_match_question_level_template_reuse_hides_out_of_scope_footprint() -> None:
    question = normalize_question("count of item in category x")
    in_scope = _template_on_tables("orders", template_id="T0001")
    out_of_scope = _template_on_tables("orders", "staff", template_id="T0002")
    in_scope.value_history = ValueHistory(
        param_values=[{"p1": "x"}],
        questions=[question],
        natural_language=["nl"],
        accept_counts=[1],
    )
    out_of_scope.value_history = ValueHistory(
        param_values=[{"p1": "x"}],
        questions=[question],
        natural_language=["nl"],
        accept_counts=[1],
    )
    templates = {"T0001": in_scope, "T0002": out_of_scope}
    visible = frozenset({"orders"})
    hit = match_question_level_template_reuse(
        question,
        templates,
        schema=_orders_schema(),
        visible_tables=visible,
    )
    assert hit.best_template is not None
    assert hit.best_template.id == "T0001"

    denied = match_question_level_template_reuse(
        question,
        {"T0002": out_of_scope},
        schema=_orders_schema(),
        visible_tables=visible,
    )
    assert denied.best_template is None


@pytest.mark.fast
def test_find_trusted_template_match_hides_out_of_scope_footprint() -> None:
    question = normalize_question("count of item in category x")
    in_scope = _template_on_tables("orders", template_id="T0001")
    out_of_scope = _template_on_tables("orders", "staff", template_id="T0002")
    for tmpl in (in_scope, out_of_scope):
        tmpl.value_history = ValueHistory(
            param_values=[{"p1": "x"}],
            questions=[question],
            natural_language=["nl"],
            accept_counts=[1],
        )
    visible = frozenset({"orders"})
    hit = find_trusted_template_match(question, [in_scope, out_of_scope], visible_tables=visible)
    assert hit is not None
    assert hit.template.id == "T0001"
    assert find_trusted_template_match(question, [out_of_scope], visible_tables=visible) is None


@pytest.mark.fast
def test_collect_structural_match_templates_hides_out_of_scope_footprint() -> None:
    intent = RuntimeIntent(
        tables=["orders"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.category"))],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("orders.category"),
                    op="=",
                    value_type="string",
                    param_key="p1",
                )
            ]
        ),
    )
    in_scope = _template_on_tables("orders", template_id="T0001")
    out_of_scope = _template_on_tables("orders", template_id="T0002")
    out_of_scope.footprint_tables = ("orders", "staff")
    for tmpl in (in_scope, out_of_scope):
        tmpl.intent_signature.select_cols = [SelectCol(expr=NormalizedExpr.from_column("orders.category"))]
        tmpl.intent_key = intent_key(intent)
    visible = frozenset({"orders"})
    matched = collect_structural_match_templates(
        intent,
        {"T0001": in_scope, "T0002": out_of_scope},
        visible_tables=visible,
    )
    assert [t.id for t in matched] == ["T0001"]


@pytest.mark.fast
def test_extract_fuzzy_reuse_params_redacts_out_of_scope_literals() -> None:
    tmpl = _minimal_template()
    tmpl.intent_signature = ConcreteIntent(
        intent_id="i1",
        tables=["orders"],
        grain="row_level",
        select_cols=[],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("orders.secret_col"),
                    op="=",
                    value_type="string",
                    param_key="p1",
                )
            ]
        ),
    )
    tmpl.tables_used = ["orders"]
    tmpl.footprint_tables = ("orders",)
    tmpl.value_history = ValueHistory(
        param_values=[{"p1": "top-secret"}],
        questions=["show secret"],
        natural_language=["nl"],
        accept_counts=[1],
    )
    schema = SchemaGraph(
        tables={
            "orders": TableMetadata(
                name="orders",
                columns={
                    "secret_col": ColumnMetadata(name="secret_col", data_type="text", sensitivity="none"),
                    "category": ColumnMetadata(name="category", data_type="text", sensitivity="none"),
                },
                primary_key=["id"],
                foreign_keys=[],
            )
        },
        join_paths_multi={},
        deny_columns={"orders": {"secret_col"}},
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "aetherdialect._pipeline_generate.LLMProvider.chat",
            lambda *_a, **_k: '{"param_values": {"p1": "x"}}',
        )
        mp.setattr("aetherdialect._pipeline_generate.EngineConfig.llm_credentials_configured", lambda: True)
        mp.setattr(
            "aetherdialect._pipeline_generate.prompt_cache_schema_scope",
            lambda *_a, **_k: __import__("contextlib").nullcontext(),
        )
        user_payload = {}

        def _capture_chat(_system: str, user: str, **_kwargs: object) -> str:
            user_payload["user"] = user
            return '{"param_values": {"p1": "x"}}'

        mp.setattr("aetherdialect._pipeline_generate.LLMProvider.chat", _capture_chat)
        extract_fuzzy_reuse_params(
            "show secret",
            tmpl,
            history_index=0,
            literal_structural_only=False,
            schema=schema,
        )
    assert "top-secret" not in user_payload.get("user", "")


@pytest.mark.fast
def test_lookup_join_avoid_skips_out_of_scope_feedback() -> None:
    question = normalize_question("orders joined to staff")
    intent = RuntimeIntent(
        tables=["orders", "staff"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.id"))],
        group_by_cols=[],
        order_by_cols=[],
        chosen_join_candidate_id="J01",
    )
    _, intent_payload = TemplateOps._compute_intent_structural_signature(intent)
    row = QuestionFeedbackEntry(
        summary="wrong join path",
        buckets=(RejectionBucket.WRONG_TABLES_OR_JOINS,),
        kind=FeedbackKind.INTENT_REJECTED,
        effective_structural_hash="h",
        intent_structural_hash="ih",
        intent_payload=intent_payload,
        created_at="t0",
        updated_at="t0",
    ).to_dict()
    store: dict[str, object] = {"question_feedback": {question: [row]}}
    schema = _orders_schema()
    assert TemplateOps.lookup_join_avoid_candidate_ids_for_question(
        store,
        question,
        visible_tables=frozenset({"orders", "staff"}),
        schema=schema,
    ) == frozenset({"J01"})
    assert (
        TemplateOps.lookup_join_avoid_candidate_ids_for_question(
            store,
            question,
            visible_tables=frozenset({"orders"}),
            schema=schema,
        )
        == frozenset()
    )
