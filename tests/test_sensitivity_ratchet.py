"""Owner-level sensitivity ratchet: scrub and invalidation on classification change."""

from __future__ import annotations

from pathlib import Path

import pytest

from aetherdialect._contracts_base import DomainKnowledgeEntry, SensitivityClassification
from aetherdialect._contracts_core import RuntimeIntent, SelectCol, Template, ValueHistory
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, SQLShape, TableMetadata, TemplateStats
from aetherdialect._dialect import Dialect
from aetherdialect._schema_profile import filter_schema_anchored_domain_knowledge, on_sensitivity_classification_change
from aetherdialect._templates import TemplateRefs, TemplateStoreLifecycleOps
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils import canonicalize_sql, normalize_question, normalize_sql
from aetherdialect._utils_intent import intent_key


def _schema(*, sensitivity: SensitivityClassification = SensitivityClassification.NONE) -> SchemaGraph:
    amount = ColumnMetadata(name="amount", data_type="numeric")
    SensitivityClassification.apply_to(amount, sensitivity)
    return SchemaGraph(
        tables={
            "orders": TableMetadata(
                name="orders",
                columns={
                    "id": ColumnMetadata(name="id", data_type="integer"),
                    "amount": amount,
                },
                primary_key=["id"],
                foreign_keys=[],
            ),
        },
        join_paths_multi={},
        effective_structural_hash="eff_ratchet",
        schema_graph_id="graph_ratchet",
    )


@pytest.mark.fast
def test_scrub_removes_restricted_column_mentions() -> None:
    schema = _schema(sensitivity=SensitivityClassification.RESTRICTED)
    candidates = (
        DomainKnowledgeEntry(key="qualified", text="orders.amount is confidential.", kind="caveat"),
        DomainKnowledgeEntry(key="bare", text="Filter on amount when reporting.", kind="glossary"),
        DomainKnowledgeEntry(key="safe", text="Fiscal year starts in July.", kind="policy"),
    )
    kept = filter_schema_anchored_domain_knowledge(candidates, schema)
    assert [e.key for e in kept] == ["safe"]


@pytest.mark.fast
def test_change_event_invalidates_dk_and_template(tmp_path: Path) -> None:
    schema = _schema()
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    dk_entries = (
        DomainKnowledgeEntry(key="amt", text="orders.amount is the charged total.", kind="glossary"),
        DomainKnowledgeEntry(key="safe", text="Fiscal year starts in July.", kind="policy"),
    )
    from aetherdialect._utils_artifacts import save_domain_knowledge_artifact

    save_domain_knowledge_artifact(artifacts_dir, dk_entries, notes_sha256="notes")

    from aetherdialect._contracts_base import NormalizedExpr

    intent = RuntimeIntent(
        tables=["orders"],
        grain="scalar",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column("orders.amount"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )
    intent_sig = intent.to_concrete("")
    sql = "SELECT amount FROM orders"
    sql_canon = canonicalize_sql(sql)
    sql_param, _ = Dialect.parameter_abstract(
        normalize_sql(sql_canon), sqlglot_dialect=Dialect.active_sqlglot_dialect()
    )
    sql_fp = Dialect.compute_sql_fp(sql_param, sqlglot_dialect=Dialect.active_sqlglot_dialect())
    tmpl = Template(
        id="T1",
        schema_graph_id=schema.schema_graph_id,
        effective_structural_hash=schema.effective_structural_hash,
        intent_signature=intent_sig,
        intent_key=intent_key(intent),
        tables_used=["orders"],
        sql_param=sql_param,
        sql_fp=sql_fp,
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig="",
        value_history=ValueHistory(
            param_values=[{}],
            questions=[normalize_question("total amount")],
            natural_language=[""],
            accept_counts=[1],
        ),
        stats=TemplateStats(),
    )
    stamped = TemplateRefs.stamp_template_footprint(tmpl)
    store = TemplateOps.empty_template_store_for_space(schema.schema_graph_id, artifacts_dir=str(artifacts_dir))
    store.set_template_raw_dict(stamped.id, stamped.to_dict())
    TemplateOps.save_template_store(store)

    col = schema.tables["orders"].columns["amount"]
    SensitivityClassification.apply_to(col, SensitivityClassification.RESTRICTED)

    report = on_sensitivity_classification_change(
        schema,
        frozenset({"orders.amount"}),
        artifacts_dir=str(artifacts_dir),
        domain_knowledge=dk_entries,
    )

    assert report.domain_knowledge_dropped == 1
    assert report.domain_knowledge_entries is not None
    assert [e.key for e in report.domain_knowledge_entries] == ["safe"]
    assert report.templates_dropped == 1

    reloaded = TemplateOps.load_template_store(
        schema.schema_graph_id,
        schema,
        artifacts_dir=str(artifacts_dir),
    )
    assert list(reloaded.iter_templates_by_partition()) == []

    survives, reasons = TemplateRefs.footprint_survives(stamped, schema)
    assert not survives
    assert any(r.startswith("sensitive_column:") for r in reasons)

    precheck = TemplateStoreLifecycleOps._space_template_merge_precheck(stamped, schema)
    assert precheck == "entity_absent"
