"""Every SQL execution entry point must validate filter/HAVING bind parameters."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aetherdialect._contracts_base import (
    NormalizedExpr,
    PredicateGroup,
    WhereParam,
)
from aetherdialect._contracts_core import (
    ConcreteIntent,
    FederatedPlan,
    GenerationPath,
    ResidualSpec,
    RuntimeIntent,
    SelectCol,
    SourceStep,
    Template,
    ValueHistory,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, SQLShape, TableMetadata, TemplateStats
from aetherdialect._federation_execute import execute_federation_coordinator
from aetherdialect._main_execution import MainExecutionOps
from aetherdialect._pipeline_execute import (
    execute_reuse_with_params,
    execute_stored_template_by_ref,
)
from aetherdialect._schema_graph import recompute_join_paths_multi
from aetherdialect._seed_warmup import SeedWarmupCacheSession
from aetherdialect._validation_sql import (
    assert_execution_parameters_validated,
    assert_residual_execution_parameters_validated,
    execute_guarded_sql,
)

execute_warmup_sql_rows = SeedWarmupCacheSession.execute_warmup_sql_rows

EXECUTION_PATHS = (
    "interactive_ask",
    "template_replay_by_question",
    "template_replay_by_identifier",
    "warmup",
    "federated_member",
    "coordinator",
)


def _schema() -> SchemaGraph:
    tables = {
        "items": TableMetadata(
            name="items",
            columns={
                "id": ColumnMetadata(name="id", data_type="integer", sensitivity="none"),
                "status": ColumnMetadata(name="status", data_type="string", sensitivity="none"),
            },
            primary_key=["id"],
            foreign_keys=[],
            source_id="a",
        ),
    }
    return SchemaGraph(
        tables=tables,
        join_paths_multi=recompute_join_paths_multi(tables),
        schema_graph_id="sg_items",
        effective_structural_hash="eff_items",
    )


def _intent_with_bound_filter() -> RuntimeIntent:
    where = PredicateGroup.from_list(
        [
            WhereParam(
                left_expr=NormalizedExpr.from_column("items.status"),
                op="=",
                param_key="p1",
                value_type="string",
            )
        ]
    )
    return RuntimeIntent(
        tables=["items"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("items.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=where,
        param_values={"p1": "active"},
        sql_param="SELECT id FROM items WHERE status = :p1",
    )


def _template() -> Template:
    intent_sig = ConcreteIntent(
        intent_id="t1",
        tables=["items"],
        grain="row_level",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("items.id"))],
        group_by_cols=[],
        order_by_cols=[],
        where=PredicateGroup.from_list(
            [
                WhereParam(
                    left_expr=NormalizedExpr.from_column("items.status"),
                    op="=",
                    param_key="p1",
                    value_type="string",
                )
            ]
        ),
    )
    return Template(
        id="T0001",
        effective_structural_hash="eff_items",
        intent_signature=intent_sig,
        intent_key="ik_items",
        tables_used=["items"],
        sql_param="SELECT id FROM items WHERE status = :p1",
        sql_fp="fp_items",
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig="cm1",
        value_history=ValueHistory(
            param_values=[{"p1": "active"}],
            questions=["list active items"],
            natural_language=["list active items"],
        ),
        stats=TemplateStats(accept=1, reject=0),
        trust_level=1,
    )


def _dialect() -> MagicMock:
    dialect = MagicMock()
    dialect.finalize_render.side_effect = lambda sql, _params, **_kwargs: sql
    dialect.explain_validation_sql = lambda sql, _pv: sql
    dialect.execute.return_value = [(1,)]
    dialect.can_explain.return_value = False
    dialect.sqlglot_dialect = "duckdb"
    dialect.ast_validate_full.return_value = []
    return dialect


def _run_path(path_name: str) -> None:
    schema = _schema()
    intent = _intent_with_bound_filter()
    dialect = _dialect()
    if path_name == "interactive_ask":
        MainExecutionOps._run_pipeline_sql_rows(intent=intent, schema=schema, dialect=dialect, tmpl_sd=None)
        return
    if path_name == "template_replay_by_question":
        tmpl = _template()
        store: dict = {"templates": {tmpl.id: tmpl}}
        with (
            patch("aetherdialect._validation_sql.validate_sql", return_value=(True, None, None, [])),
            patch("aetherdialect._templates_ops.TemplateOps.save_template_store"),
            patch("aetherdialect._templates_ops.TemplateOps.templates_to_store", side_effect=lambda s, t: s),
            patch("aetherdialect._templates_ops.TemplateOps.delete_rejected_templates_matching_question"),
            patch("aetherdialect._pipeline_execute.save_result_csv_for_store"),
            patch("aetherdialect._pipeline_execute.print_query_result"),
            patch("aetherdialect._templates_ops.TemplateOps.promote_trust"),
            patch("aetherdialect._llm_provider.LLMProvider.chat", return_value='{"aliases":{}}'),
        ):
            execute_reuse_with_params(
                "list active items",
                tmpl,
                {"p1": "active"},
                dialect,
                store,
                {tmpl.id: tmpl},
                {},
                schema,
                reuse_path=GenerationPath.EXACT_QUESTION_REUSE,
                prompt=False,
            )
        return
    if path_name == "template_replay_by_identifier":
        tmpl = _template()
        store = {"templates": {tmpl.id: tmpl}}
        with (
            patch("aetherdialect._validation_sql.validate_sql", return_value=(True, None, None, [])),
            patch("aetherdialect._templates_ops.TemplateOps.save_template_store"),
            patch("aetherdialect._templates_ops.TemplateOps.templates_to_store", side_effect=lambda s, t: s),
            patch("aetherdialect._templates_ops.TemplateOps.delete_rejected_templates_matching_question"),
            patch("aetherdialect._pipeline_execute.save_result_csv_for_store"),
            patch("aetherdialect._pipeline_execute.print_query_result"),
            patch("aetherdialect._templates_ops.TemplateOps.promote_trust"),
            patch("aetherdialect._llm_provider.LLMProvider.chat", return_value='{"aliases":{}}'),
        ):
            execute_stored_template_by_ref(
                tmpl.id,
                {"p1": "active"},
                question="list active items",
                dialect=dialect,
                store=store,
                templates={tmpl.id: tmpl},
                rejected={},
                schema=schema,
            )
        return
    if path_name == "warmup":
        SeedWarmupCacheSession.execute_warmup_sql_rows(
            intent, schema, dialect, intent.sql_param, dict(intent.param_values)
        )
        return
    if path_name == "federated_member":
        with patch("aetherdialect._validation_sql.validate_sql", return_value=(True, None, None, [])):
            execute_guarded_sql(
                dialect,
                intent.sql_param,
                dict(intent.param_values),
                schema=schema,
                intent=intent,
            )
        return
    if path_name == "coordinator":
        residual = ResidualSpec(
            select_cols=(SelectCol(expr=NormalizedExpr.from_column("items.id")),),
            where=PredicateGroup.from_list(
                [
                    WhereParam(
                        left_expr=NormalizedExpr.from_column("items.status"),
                        op="=",
                        param_key="p1",
                        value_type="string",
                    )
                ]
            ),
        )
        plan = FederatedPlan(
            steps=(SourceStep(source_id="a", sub_intent=intent, projected_keys=("items.id",)),),
            residual=residual,
        )
        frames = {"a": pd.DataFrame({"items.id": [1], "items.status": ["active"]})}
        with (
            patch("aetherdialect._federation_execute._import_coordinator_duckdb") as import_mock,
            patch("aetherdialect._federation_execute._validate_coordinator_glue_sql"),
        ):
            conn = MagicMock()
            conn.execute.return_value = MagicMock(fetchdf=lambda: pd.DataFrame({"items.id": [1]}))
            duckdb_mod = MagicMock()
            duckdb_mod.connect.return_value = conn
            import_mock.return_value = duckdb_mod
            execute_federation_coordinator(
                frames,
                plan,
                schema=schema,
                param_values={"p1": "active"},
            )
        return
    raise AssertionError(f"unknown execution path {path_name!r}")


@pytest.mark.fast
@pytest.mark.parametrize("path_name", EXECUTION_PATHS)
def test_every_execution_path_validates_parameters(path_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    intent_calls: list[int] = []
    residual_calls: list[int] = []

    def _track_intent(intent: RuntimeIntent, schema: SchemaGraph) -> None:
        intent_calls.append(id(intent))
        assert_execution_parameters_validated(intent, schema)

    def _track_residual(residual: ResidualSpec, param_values: dict[str, object], schema: SchemaGraph) -> None:
        residual_calls.append(id(residual))
        assert_residual_execution_parameters_validated(residual, param_values, schema)

    monkeypatch.setattr(
        "aetherdialect._validation_sql.assert_execution_parameters_validated",
        _track_intent,
    )
    for target in (
        "aetherdialect._validation_sql.assert_execution_parameters_validated",
        "aetherdialect._seed_warmup.assert_execution_parameters_validated",
    ):
        monkeypatch.setattr(target, _track_intent)
    for target in (
        "aetherdialect._validation_sql.assert_residual_execution_parameters_validated",
        "aetherdialect._federation_execute.assert_residual_execution_parameters_validated",
    ):
        monkeypatch.setattr(target, _track_residual)

    _run_path(path_name)

    if path_name == "coordinator":
        assert residual_calls, f"{path_name} did not validate coordinator residual parameters"
    else:
        assert intent_calls, f"{path_name} did not validate execution parameters"


@pytest.mark.fast
def test_missing_filter_bind_value_is_refused_before_execute() -> None:
    schema = _schema()
    intent = replace(_intent_with_bound_filter(), param_values={})
    dialect = _dialect()
    with pytest.raises(Exception, match="no comparison value"):
        MainExecutionOps._run_pipeline_sql_rows(intent=intent, schema=schema, dialect=dialect, tmpl_sd=None)
    dialect.execute.assert_not_called()
