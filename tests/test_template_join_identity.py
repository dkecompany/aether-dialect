"""Template storage routes by join fingerprint and direct reuse pins stored paths."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherdialect._config import EngineConfig
from aetherdialect._contracts_core import GenerationPath, RuntimeIntent, SelectCol
from aetherdialect._contracts_schema import ColumnMetadata, FKEdge, SchemaGraph, TableMetadata
from aetherdialect._intent_process import NormalizedExpr
from aetherdialect._pipeline import execute_reuse_with_params
from aetherdialect._templates import TemplateOps, TemplateRefs


def _film_star_schema() -> SchemaGraph:
    category_fk = FKEdge(src_table="category", src_cols=["category_id"], dst_table="film", dst_cols=["category_id"])
    language_fk = FKEdge(src_table="language", src_cols=["language_id"], dst_table="film", dst_cols=["language_id"])
    tables = {
        "film": TableMetadata(
            name="film",
            columns={
                "film_id": ColumnMetadata(name="film_id", data_type="integer", sensitivity="none"),
                "category_id": ColumnMetadata(name="category_id", data_type="integer", sensitivity="none"),
                "language_id": ColumnMetadata(name="language_id", data_type="integer", sensitivity="none"),
            },
            primary_key=["film_id"],
            foreign_keys=[],
        ),
        "category": TableMetadata(
            name="category",
            columns={
                "category_id": ColumnMetadata(name="category_id", data_type="integer", sensitivity="none"),
            },
            primary_key=["category_id"],
            foreign_keys=[category_fk],
        ),
        "language": TableMetadata(
            name="language",
            columns={
                "language_id": ColumnMetadata(name="language_id", data_type="integer", sensitivity="none"),
            },
            primary_key=["language_id"],
            foreign_keys=[language_fk],
        ),
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi={},
        effective_structural_hash="h",
    )


def _parent_child_schema() -> SchemaGraph:
    fk = FKEdge(src_table="child", src_cols=["parent_id"], dst_table="parent", dst_cols=["id"])
    edge = {
        "src_table": "child",
        "src_cols": ["parent_id"],
        "dst_table": "parent",
        "dst_cols": ["id"],
    }
    path = [edge]
    tables = {
        "parent": TableMetadata(
            name="parent",
            columns={"id": ColumnMetadata(name="id", data_type="integer", sensitivity="none")},
            primary_key=["id"],
            foreign_keys=[],
        ),
        "child": TableMetadata(
            name="child",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                "parent_id": ColumnMetadata(name="parent_id", data_type="integer", sensitivity="none"),
            },
            primary_key=["id"],
            foreign_keys=[fk],
        ),
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi={"child": {"parent": [path]}},
        effective_structural_hash="h",
    )


def _join_intent(
    *,
    signature: list[str],
    candidate_id: str = "J01",
    tables: list[str] | None = None,
    select_col: str = "child.id",
) -> RuntimeIntent:
    resolved_tables = tables or ["child", "parent"]
    return RuntimeIntent(
        tables=resolved_tables,
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column(select_col))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        chosen_join_candidate_id=candidate_id,
        chosen_join_path_signature=list(signature),
    )


@pytest.mark.fast
def test_differing_join_paths_create_separate_templates() -> None:
    schema = _parent_child_schema()
    store: dict = {"templates": {}, "question_feedback": {}, "next_id": 1}
    templates: dict = {}
    direct_sig = ["child.parent_id->parent.id"]
    bridge_sig = ["child.bridge_id->bridge.id", "bridge.parent_id->parent.id"]

    first = TemplateOps.insert_template(
        store,
        templates,
        schema,
        "q1",
        _join_intent(signature=direct_sig),
        "SELECT child.id FROM child JOIN parent ON child.parent_id = parent.id",
    )
    second = TemplateOps.insert_template(
        store,
        templates,
        schema,
        "q2",
        _join_intent(signature=bridge_sig, candidate_id="J02"),
        (
            "SELECT child.id FROM child JOIN bridge ON child.bridge_id = bridge.id "
            "JOIN parent ON bridge.parent_id = parent.id"
        ),
    )

    assert first.id != second.id
    assert len(templates) == 2
    assert TemplateRefs.join_fingerprint_from_runtime_intent(
        _join_intent(signature=direct_sig)
    ) != TemplateRefs.join_fingerprint_from_runtime_intent(_join_intent(signature=bridge_sig))
    assert first.chosen_join_path_signature == direct_sig
    assert second.chosen_join_path_signature == bridge_sig
    assert first.sql_fp != second.sql_fp


@pytest.mark.fast
def test_matching_join_paths_merge_into_one_template() -> None:
    schema = _film_star_schema()
    store: dict = {"templates": {}, "question_feedback": {}, "next_id": 1}
    templates: dict = {}
    sig_a = ["category.category_id->film.category_id", "language.language_id->film.language_id"]
    sig_b = list(reversed(sig_a))
    sql = (
        "SELECT film.film_id FROM film "
        "JOIN category ON category.category_id = film.category_id "
        "JOIN language ON language.language_id = film.language_id"
    )
    tables = ["category", "film", "language"]

    TemplateOps.clear_sandbox_paraphrase_source()
    with patch.object(EngineConfig, "llm_credentials_configured", return_value=False):
        first = TemplateOps.insert_template(
            store,
            templates,
            schema,
            "q1",
            _join_intent(signature=sig_a, tables=tables, select_col="film.film_id"),
            sql,
        )
        second = TemplateOps.insert_template(
            store,
            templates,
            schema,
            "q2",
            _join_intent(signature=sig_b, tables=tables, select_col="film.film_id"),
            sql,
        )

        assert TemplateRefs.join_fingerprint_from_runtime_intent(
            _join_intent(signature=sig_a, tables=tables, select_col="film.film_id")
        ) == TemplateRefs.join_fingerprint_from_runtime_intent(
            _join_intent(signature=sig_b, tables=tables, select_col="film.film_id")
        )
        assert first.id == second.id
        assert len(templates) == 1
        assert second.stats.accept == 2
        assert set(second.value_history.questions) == {"q1", "q2"}
    TemplateOps.clear_sandbox_paraphrase_source()


@patch("aetherdialect._pipeline.LLMProvider.chat", return_value='{"aliases":{}}')
@patch("aetherdialect._templates.TemplateOps.save_template_store")
@patch("aetherdialect._templates.TemplateOps.templates_to_store", side_effect=lambda s, t: s)
@patch("aetherdialect._templates.TemplateOps.delete_rejected_templates_matching_question")
@patch("aetherdialect._pipeline.save_result_csv_for_store")
@patch("aetherdialect._pipeline.print_query_result")
@patch("aetherdialect._templates.TemplateOps.promote_trust")
@patch("aetherdialect._pipeline.validate_sql", return_value=(True, None, None, []))
@patch("aetherdialect._pipeline._resolve_joins_fresh")
@patch("aetherdialect._pipeline.generate_join_candidates")
@pytest.mark.fast
def test_direct_question_reuse_skips_join_enumeration(
    mock_generate_join_candidates,
    mock_resolve_joins_fresh,
    _mock_val,
    _mock_promote,
    _mock_print,
    _mock_csv,
    _mock_del,
    _mock_tts,
    _mock_save,
    _mock_llm,
) -> None:
    from aetherdialect._contracts_core import ConcreteIntent, SQLShape, Template, TemplateStats, ValueHistory

    pinned_sig = ["child.parent_id->parent.id"]
    concrete = ConcreteIntent(
        intent_id="id",
        tables=["child", "parent"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("child.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
        chosen_join_path_signature=pinned_sig,
        chosen_join_candidate_id="J01",
    )
    tmpl = Template(
        id="T1",
        effective_structural_hash="h",
        intent_signature=concrete,
        intent_key="ik",
        tables_used=["child", "parent"],
        sql_param="SELECT child.id FROM child JOIN parent ON child.parent_id = parent.id",
        sql_fp="fp",
        shape=SQLShape(num_joins=1, has_group_by=False, has_agg=False),
        colmap_sig="c",
        value_history=ValueHistory(
            param_values=[{}],
            questions=["norm_q"],
            natural_language=["nl"],
        ),
        stats=TemplateStats(accept=1, reject=0),
        trust_level=2,
    )
    dialect = MagicMock()
    dialect.finalize_render.return_value = "EXEC"
    dialect.explain_validation_sql = lambda sql, _pv: sql
    dialect.execute.return_value = [("row",)]
    store: dict = {"templates": {"T1": tmpl}}

    outcome = execute_reuse_with_params(
        "norm_q",
        tmpl,
        {},
        dialect,
        store,
        {"T1": tmpl},
        {},
        _parent_child_schema(),
        reuse_path=GenerationPath.EXACT_QUESTION_REUSE,
        prompt=False,
    )

    assert outcome is not None and outcome.success is True
    mock_generate_join_candidates.assert_not_called()
    mock_resolve_joins_fresh.assert_not_called()
    assert outcome.matched_template is not None
    assert outcome.matched_template.chosen_join_path_signature == pinned_sig
