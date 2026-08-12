"""Unit tests for space→master learning merge collision policy."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aetherdialect._contracts_base import ApprovalState, SpaceContext
from aetherdialect._contracts_core import (
    FeedbackKind,
    QuestionFeedbackEntry,
    RuntimeIntent,
    SelectCol,
    Template,
    ValueHistory,
)
from aetherdialect._contracts_schema import ColumnMetadata, SchemaGraph, SQLShape, TableMetadata, TemplateStats
from aetherdialect._dialect import Dialect
from aetherdialect._templates import SpaceLearningMergeCounts, TemplateRefs
from aetherdialect._templates_ops import TemplateOps
from aetherdialect._utils import canonicalize_sql, normalize_question, normalize_sql
from aetherdialect._utils_intent import intent_key


def _column(name: str, *, data_type: str = "integer") -> ColumnMetadata:
    return ColumnMetadata(name=name, data_type=data_type, sensitivity="none")


def _table(name: str, *, columns: dict[str, ColumnMetadata] | None = None) -> TableMetadata:
    cols = columns or {"id": _column("id")}
    return TableMetadata(name=name, columns=cols, primary_key=["id"], foreign_keys=[])


def _sample_schema() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "film": _table(
                "film",
                columns={
                    "film_id": _column("film_id"),
                    "title": _column("title", data_type="text"),
                },
            ),
        },
        join_paths_multi={},
        effective_structural_hash="eff_merge_test",
        schema_graph_id="graph_merge_test",
    )


def _scalar_intent(table: str, col: str) -> RuntimeIntent:
    from aetherdialect._contracts_base import NormalizedExpr

    return RuntimeIntent(
        tables=[table],
        grain="scalar",
        select_cols=[SelectCol(expr=NormalizedExpr.from_column(f"{table}.{col}"))],
        group_by_cols=[],
        order_by_cols=[],
        where=None,
    )


def _build_template(
    tid: str,
    schema: SchemaGraph,
    question: str,
    intent: RuntimeIntent,
    sql: str,
    *,
    join_sig: list[str] | None = None,
    trust_level: int = 1,
    stats: TemplateStats | None = None,
    federation_plan_only: bool = False,
    approval_state: ApprovalState = ApprovalState.APPROVED,
    effective_structural_hash: str | None = None,
) -> Template:
    intent_sig = intent.to_concrete("")
    if join_sig is not None:
        intent_sig = replace(intent_sig, chosen_join_path_signature=join_sig)
    sql_canon = canonicalize_sql(sql)
    sql_param, _ = Dialect.parameter_abstract(
        normalize_sql(sql_canon), sqlglot_dialect=Dialect.active_sqlglot_dialect()
    )
    sql_fp = Dialect.compute_sql_fp(sql_param, sqlglot_dialect=Dialect.active_sqlglot_dialect())
    q_norm = normalize_question(question)
    tmpl = Template(
        id=tid,
        schema_graph_id=schema.schema_graph_id,
        effective_structural_hash=effective_structural_hash or schema.effective_structural_hash,
        intent_signature=intent_sig,
        intent_key=intent_key(intent),
        tables_used=list(intent.tables or []),
        sql_param=sql_param,
        sql_fp=sql_fp,
        shape=SQLShape(num_joins=0, has_group_by=False, has_agg=False),
        colmap_sig="",
        value_history=ValueHistory(
            param_values=[{}],
            questions=[q_norm],
            natural_language=[""],
            accept_counts=[1],
        ),
        stats=stats or TemplateStats(accept=1, reject=0),
        trust_level=trust_level,
        federation_plan_only=federation_plan_only,
        approval_state=approval_state,
    )
    ft, fc = TemplateRefs.footprint_from_refs(TemplateRefs.template_schema_refs(tmpl))
    tmpl.footprint_tables = ft
    tmpl.footprint_columns = fc
    return tmpl


def _ensure_space_store(artifacts_dir: Path, schema: SchemaGraph, space: str) -> None:
    store = TemplateOps.load_template_store(
        schema.schema_graph_id,
        schema,
        artifacts_dir=str(artifacts_dir),
        space_name=space,
    )
    TemplateOps.save_template_store(store)


def _save_templates(
    artifacts_dir: Path,
    schema: SchemaGraph,
    space: str,
    templates: dict[str, Template],
) -> None:
    _ensure_space_store(artifacts_dir, schema, space)
    store = TemplateOps.load_template_store(
        schema.schema_graph_id,
        schema,
        artifacts_dir=str(artifacts_dir),
        space_name=space,
    )
    TemplateOps.templates_to_store(store, templates)
    TemplateOps.save_template_store(store)


def _load_templates(artifacts_dir: Path, schema: SchemaGraph, space: str) -> dict[str, Template]:
    store = TemplateOps.load_template_store(
        schema.schema_graph_id,
        schema,
        artifacts_dir=str(artifacts_dir),
        space_name=space,
    )
    return dict(TemplateOps.store_to_templates(store))


def _save_feedback(
    artifacts_dir: Path,
    schema: SchemaGraph,
    space: str,
    q_norm: str,
    entry: QuestionFeedbackEntry,
) -> None:
    _ensure_space_store(artifacts_dir, schema, space)
    store = TemplateOps.load_template_store(
        schema.schema_graph_id,
        schema,
        artifacts_dir=str(artifacts_dir),
        space_name=space,
    )
    TemplateOps.record_question_feedback(store, q_norm, entry)
    TemplateOps.save_template_store(store)


def _feedback_entry(schema: SchemaGraph, *, kind: FeedbackKind = FeedbackKind.INTENT_REJECTED) -> QuestionFeedbackEntry:
    ts = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return QuestionFeedbackEntry(
        summary="user rejected this question",
        buckets=(),
        kind=kind,
        effective_structural_hash=schema.effective_structural_hash,
        intent_structural_hash="ish_test_001",
        intent_payload="{}",
        created_at=ts,
        updated_at=ts,
    )


def _run_merge(artifacts_dir: Path, schema: SchemaGraph, space: str = "films_only") -> SpaceLearningMergeCounts:
    _ensure_space_store(artifacts_dir, schema, space)
    result = TemplateOps.merge_space_learning_into_master(
        str(artifacts_dir),
        space,
        schema.schema_graph_id,
        schema,
    )
    return result.counts


def _run_merge_sequence(
    artifacts_dir: Path,
    schema: SchemaGraph,
    spaces: list[str],
) -> SpaceLearningMergeCounts:
    total = SpaceLearningMergeCounts()
    for space in spaces:
        counts = _run_merge(artifacts_dir, schema, space=space)
        for field in total.__dataclass_fields__:
            setattr(total, field, getattr(total, field) + getattr(counts, field))
    return total


@pytest.mark.fast
def test_merge_case1_same_q_intent_join_merges(tmp_path: Path) -> None:
    schema = _sample_schema()
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    q = "how many films"
    master = _build_template(
        "T0001", schema, q, _scalar_intent("film", "film_id"), "SELECT film_id FROM film", trust_level=1
    )
    space = _build_template(
        "T0002",
        schema,
        q,
        _scalar_intent("film", "film_id"),
        "SELECT film_id FROM film",
        trust_level=3,
        stats=TemplateStats(accept=4, reject=0),
    )
    _save_templates(artifacts_dir, schema, "master", {"T0001": master})
    _save_templates(artifacts_dir, schema, "films_only", {"T0002": space})

    counts = _run_merge(artifacts_dir, schema)

    master_templates = _load_templates(artifacts_dir, schema, "master")
    assert len(master_templates) == 1
    merged = master_templates["T0001"]
    assert merged.trust_level == 3
    assert merged.stats.accept == 5
    assert counts.merged_same_identity == 1


@pytest.mark.fast
def test_merge_case2_same_q_diff_join_keeps_master(tmp_path: Path) -> None:
    schema = _sample_schema()
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    q = "how many films"
    join_a = ["film.actor_id->actor.actor_id"]
    join_b = ["film.category_id->category.category_id"]
    master = _build_template(
        "T0001",
        schema,
        q,
        _scalar_intent("film", "film_id"),
        "SELECT film_id FROM film",
        join_sig=join_a,
    )
    space = _build_template(
        "T0002",
        schema,
        q,
        _scalar_intent("film", "film_id"),
        "SELECT film_id FROM film",
        join_sig=join_b,
    )
    _save_templates(artifacts_dir, schema, "master", {"T0001": master})
    _save_templates(artifacts_dir, schema, "films_only", {"T0002": space})

    counts = _run_merge(artifacts_dir, schema)

    master_templates = _load_templates(artifacts_dir, schema, "master")
    assert len(master_templates) == 1
    kept = master_templates["T0001"]
    assert kept.intent_signature.chosen_join_path_signature == join_a
    assert counts.discarded_same_q_diff_join == 1


@pytest.mark.fast
def test_merge_case3_same_q_diff_intent_keeps_master(tmp_path: Path) -> None:
    schema = _sample_schema()
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    q = "how many films"
    master = _build_template("T0001", schema, q, _scalar_intent("film", "film_id"), "SELECT film_id FROM film")
    space = _build_template("T0002", schema, q, _scalar_intent("film", "title"), "SELECT title FROM film")
    _save_templates(artifacts_dir, schema, "master", {"T0001": master})
    _save_templates(artifacts_dir, schema, "films_only", {"T0002": space})

    counts = _run_merge(artifacts_dir, schema)

    master_templates = _load_templates(artifacts_dir, schema, "master")
    assert len(master_templates) == 1
    assert "film_id" in master_templates["T0001"].sql_param
    assert counts.discarded_same_q_diff_intent == 1


@pytest.mark.fast
def test_merge_case4_diff_q_same_intent_join_folds_paraphrase(tmp_path: Path) -> None:
    schema = _sample_schema()
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    intent = _scalar_intent("film", "film_id")
    master = _build_template("T0001", schema, "how many films", intent, "SELECT film_id FROM film")
    space = _build_template("T0002", schema, "count all films", intent, "SELECT film_id FROM film")
    _save_templates(artifacts_dir, schema, "master", {"T0001": master})
    _save_templates(artifacts_dir, schema, "films_only", {"T0002": space})

    counts = _run_merge(artifacts_dir, schema)

    master_templates = _load_templates(artifacts_dir, schema, "master")
    assert len(master_templates) == 1
    questions = set(master_templates["T0001"].value_history.questions)
    assert normalize_question("how many films") in questions
    assert normalize_question("count all films") in questions
    assert counts.folded_paraphrase == 1


@pytest.mark.fast
def test_merge_case12_master_rejection_discards_space_template(tmp_path: Path) -> None:
    schema = _sample_schema()
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    q = normalize_question("how many films")
    space = _build_template("T0002", schema, q, _scalar_intent("film", "film_id"), "SELECT film_id FROM film")
    _save_templates(artifacts_dir, schema, "master", {})
    _save_templates(artifacts_dir, schema, "films_only", {"T0002": space})
    _save_feedback(artifacts_dir, schema, "master", q, _feedback_entry(schema))

    counts = _run_merge(artifacts_dir, schema)

    assert _load_templates(artifacts_dir, schema, "master") == {}
    assert counts.discarded_template_master_rejection == 1


@pytest.mark.fast
def test_merge_case13_master_template_discards_space_rejection(tmp_path: Path) -> None:
    schema = _sample_schema()
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    q = normalize_question("how many films")
    master = _build_template("T0001", schema, q, _scalar_intent("film", "film_id"), "SELECT film_id FROM film")
    _save_templates(artifacts_dir, schema, "master", {"T0001": master})
    _ensure_space_store(artifacts_dir, schema, "films_only")
    _save_feedback(artifacts_dir, schema, "films_only", q, _feedback_entry(schema))

    counts = _run_merge(artifacts_dir, schema)

    master_templates = _load_templates(artifacts_dir, schema, "master")
    assert len(master_templates) == 1
    store = TemplateOps.load_template_store(
        schema.schema_graph_id,
        schema,
        artifacts_dir=str(artifacts_dir),
        space_name="master",
    )
    assert not TemplateOps.has_any_rejection_history_for_question(store, q, schema.schema_graph_id)
    assert counts.feedback_discarded_master_wins == 1


@pytest.mark.fast
def test_merge_case5_diff_q_same_intent_diff_join_carries_new_id(tmp_path: Path) -> None:
    schema = _sample_schema()
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    intent = _scalar_intent("film", "film_id")
    join_a = ["film.actor_id->actor.actor_id"]
    join_b = ["film.category_id->category.category_id"]
    master = _build_template(
        "T0001",
        schema,
        "how many films",
        intent,
        "SELECT film_id FROM film",
        join_sig=join_a,
    )
    space = _build_template(
        "T0002",
        schema,
        "count films by category",
        intent,
        "SELECT film_id FROM film",
        join_sig=join_b,
    )
    _save_templates(artifacts_dir, schema, "master", {"T0001": master})
    _save_templates(artifacts_dir, schema, "films_only", {"T0002": space})

    counts = _run_merge(artifacts_dir, schema)

    master_templates = _load_templates(artifacts_dir, schema, "master")
    assert len(master_templates) == 2
    assert counts.carried_new_id == 1


@pytest.mark.fast
def test_merge_case6_diff_q_diff_intent_carries_new_id(tmp_path: Path) -> None:
    schema = _sample_schema()
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    master = _build_template(
        "T0001", schema, "how many films", _scalar_intent("film", "film_id"), "SELECT film_id FROM film"
    )
    space = _build_template(
        "T0002", schema, "list film titles", _scalar_intent("film", "title"), "SELECT title FROM film"
    )
    _save_templates(artifacts_dir, schema, "master", {"T0001": master})
    _save_templates(artifacts_dir, schema, "films_only", {"T0002": space})

    counts = _run_merge(artifacts_dir, schema)

    master_templates = _load_templates(artifacts_dir, schema, "master")
    assert len(master_templates) == 2
    assert counts.carried_new_id == 1


@pytest.mark.fast
def test_merge_case7_template_id_collision_reassigns(tmp_path: Path) -> None:
    schema = _sample_schema()
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    master = _build_template(
        "T0001", schema, "how many films", _scalar_intent("film", "film_id"), "SELECT film_id FROM film"
    )
    space = _build_template(
        "T0001", schema, "list film titles", _scalar_intent("film", "title"), "SELECT title FROM film"
    )
    _save_templates(artifacts_dir, schema, "master", {"T0001": master})
    _save_templates(artifacts_dir, schema, "films_only", {"T0001": space})

    counts = _run_merge(artifacts_dir, schema)

    master_templates = _load_templates(artifacts_dir, schema, "master")
    assert len(master_templates) == 2
    assert "T0001" in master_templates
    assert counts.id_reassigned == 1
    assert counts.carried_new_id == 1


@pytest.mark.fast
def test_merge_case8_federation_plan_only_dropped(tmp_path: Path) -> None:
    schema = _sample_schema()
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    space = _build_template(
        "T0002",
        schema,
        "how many films",
        _scalar_intent("film", "film_id"),
        "SELECT film_id FROM film",
        federation_plan_only=True,
    )
    _save_templates(artifacts_dir, schema, "master", {})
    _save_templates(artifacts_dir, schema, "films_only", {"T0002": space})

    counts = _run_merge(artifacts_dir, schema)

    assert _load_templates(artifacts_dir, schema, "master") == {}
    assert counts.dropped_federation_plan_only == 1


@pytest.mark.fast
def test_merge_case9_pending_approval_dropped(tmp_path: Path) -> None:
    schema = _sample_schema()
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    space = _build_template(
        "T0002",
        schema,
        "how many films",
        _scalar_intent("film", "film_id"),
        "SELECT film_id FROM film",
        approval_state=ApprovalState.PENDING,
    )
    _save_templates(artifacts_dir, schema, "master", {})
    _save_templates(artifacts_dir, schema, "films_only", {"T0002": space})

    counts = _run_merge(artifacts_dir, schema)

    assert _load_templates(artifacts_dir, schema, "master") == {}
    assert counts.dropped_pending_approval == 1


@pytest.mark.fast
def test_merge_case10_structural_hash_mismatch_dropped(tmp_path: Path) -> None:
    schema = _sample_schema()
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    space = _build_template(
        "T0002",
        schema,
        "how many films",
        _scalar_intent("film", "film_id"),
        "SELECT film_id FROM film",
        effective_structural_hash="stale_hash",
    )
    _save_templates(artifacts_dir, schema, "master", {})
    _save_templates(artifacts_dir, schema, "films_only", {"T0002": space})

    counts = _run_merge(artifacts_dir, schema)

    assert _load_templates(artifacts_dir, schema, "master") == {}
    assert counts.dropped_structural_hash_mismatch == 1


@pytest.mark.fast
def test_merge_case11_entity_absent_dropped(tmp_path: Path) -> None:
    schema = _sample_schema()
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    space = _build_template(
        "T0002",
        schema,
        "how many actors",
        _scalar_intent("actor", "actor_id"),
        "SELECT actor_id FROM actor",
    )
    _save_templates(artifacts_dir, schema, "master", {})
    _save_templates(artifacts_dir, schema, "films_only", {"T0002": space})

    counts = _run_merge(artifacts_dir, schema)

    assert _load_templates(artifacts_dir, schema, "master") == {}
    assert counts.dropped_entity_absent == 1


@pytest.mark.fast
def test_merge_case14_duplicate_feedback_kind_keeps_master(tmp_path: Path) -> None:
    schema = _sample_schema()
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    q = normalize_question("how many films")
    master = _build_template("T0001", schema, q, _scalar_intent("film", "film_id"), "SELECT film_id FROM film")
    _save_templates(artifacts_dir, schema, "master", {"T0001": master})
    _save_feedback(artifacts_dir, schema, "master", q, _feedback_entry(schema))
    _save_feedback(artifacts_dir, schema, "films_only", q, _feedback_entry(schema))

    counts = _run_merge(artifacts_dir, schema)

    store = TemplateOps.load_template_store(
        schema.schema_graph_id,
        schema,
        artifacts_dir=str(artifacts_dir),
        space_name="master",
    )
    assert len(list(store._get_feedback_rows(q))) == 1
    assert counts.feedback_discarded_master_wins == 1


@pytest.mark.fast
def test_merge_case15_space_only_feedback_carried(tmp_path: Path) -> None:
    schema = _sample_schema()
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    q = normalize_question("how many films")
    _save_templates(artifacts_dir, schema, "master", {})
    _ensure_space_store(artifacts_dir, schema, "films_only")
    _save_feedback(artifacts_dir, schema, "films_only", q, _feedback_entry(schema))

    counts = _run_merge(artifacts_dir, schema)

    store = TemplateOps.load_template_store(
        schema.schema_graph_id,
        schema,
        artifacts_dir=str(artifacts_dir),
        space_name="master",
    )
    assert len(list(store._get_feedback_rows(q))) == 1
    assert counts.feedback_carried == 1


@pytest.mark.fast
def test_merge_case16_master_join_rejection_drops_conflicting_signature(tmp_path: Path) -> None:
    schema = _sample_schema()
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    q = normalize_question("how many films")
    join_a = ["film.actor_id->actor.actor_id"]
    master = _build_template(
        "T0001",
        schema,
        q,
        _scalar_intent("film", "film_id"),
        "SELECT film_id FROM film",
        join_sig=join_a,
    )
    _save_templates(artifacts_dir, schema, "master", {"T0001": master})
    _ensure_space_store(artifacts_dir, schema, "films_only")
    ts = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    space_feedback = QuestionFeedbackEntry(
        summary="bad join path",
        buckets=(),
        kind=FeedbackKind.VALIDATION_FAILURE,
        effective_structural_hash=schema.effective_structural_hash,
        intent_structural_hash="ish_test_001",
        intent_payload="{}",
        created_at=ts,
        updated_at=ts,
        rejected_join_path_signature=tuple(join_a),
    )
    _save_feedback(artifacts_dir, schema, "films_only", q, space_feedback)

    counts = _run_merge(artifacts_dir, schema)

    store = TemplateOps.load_template_store(
        schema.schema_graph_id,
        schema,
        artifacts_dir=str(artifacts_dir),
        space_name="master",
    )
    rows = list(store._get_feedback_rows(q))
    assert len(rows) == 1
    carried = QuestionFeedbackEntry.from_dict(rows[0])
    assert carried.rejected_join_path_signature == ()
    assert counts.feedback_join_rejection_dropped == 1
    assert counts.feedback_carried == 1


@pytest.mark.fast
def test_merge_case19_paraphrase_mapping_conflict_dropped(tmp_path: Path) -> None:
    schema = _sample_schema()
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    intent = _scalar_intent("film", "film_id")
    join_a = ["film.actor_id->actor.actor_id"]
    master_a = _build_template("T0001", schema, "how many films", intent, "SELECT film_id FROM film", join_sig=join_a)
    master_b = _build_template(
        "T0002",
        schema,
        "count rentals",
        _scalar_intent("film", "title"),
        "SELECT title FROM film",
    )
    conflict_q = normalize_question("count rentals")
    space = _build_template(
        "T0003",
        schema,
        "film count alias",
        intent,
        "SELECT film_id FROM film",
        join_sig=join_a,
    )
    space.value_history.questions = [normalize_question("film count alias"), conflict_q]
    space.value_history.param_values = [{}, {}]
    space.value_history.natural_language = ["", ""]
    space.value_history.accept_counts = [1, 1]
    _save_templates(artifacts_dir, schema, "master", {"T0001": master_a, "T0002": master_b})
    _save_templates(artifacts_dir, schema, "films_only", {"T0003": space})

    counts = _run_merge(artifacts_dir, schema)

    master_templates = _load_templates(artifacts_dir, schema, "master")
    merged = master_templates["T0001"]
    assert conflict_q not in set(merged.value_history.questions)
    assert counts.paraphrase_mapping_dropped >= 1


@pytest.mark.fast
def test_merge_case21_multi_space_uid_order_is_deterministic(tmp_path: Path) -> None:
    """Colliding ids across spaces: earlier merge keeps T0001; later space is reassigned."""
    schema = _sample_schema()
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    intent_a = _scalar_intent("film", "film_id")
    intent_b = _scalar_intent("film", "title")
    space_a = _build_template("T0001", schema, "how many films in space a", intent_a, "SELECT film_id FROM film")
    space_b = _build_template("T0001", schema, "how many titles in space b", intent_b, "SELECT title FROM film")
    for space, templates in (
        ("master", {}),
        ("space_aaa", {"T0001": space_a}),
        ("space_zzz", {"T0001": space_b}),
    ):
        store = TemplateOps.load_template_store(
            schema.schema_graph_id,
            schema,
            artifacts_dir=str(artifacts_dir),
            space_name=space,
        )
        TemplateOps.templates_to_store(store, templates)
        TemplateOps.save_template_store(store)

    for space in ("space_aaa", "space_zzz"):
        TemplateOps.merge_space_learning_into_master(
            str(artifacts_dir),
            space,
            schema.schema_graph_id,
            schema,
        )
    master = _load_templates(artifacts_dir, schema, "master")

    assert len(master) == 2
    assert "how many films in space a" in set(master["T0001"].value_history.questions)
    reassigned = next(tid for tid in master if tid != "T0001")
    assert "how many titles in space b" in set(master[reassigned].value_history.questions)


@pytest.mark.fast
def test_merge_case17_value_history_duplicate_row_dedupes(tmp_path: Path) -> None:
    from aetherdialect._templates import TemplateStoreLifecycleOps

    schema = _sample_schema()
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    q = normalize_question("how many films")
    intent = _scalar_intent("film", "film_id")
    master = _build_template("T0001", schema, q, intent, "SELECT film_id FROM film")
    master.value_history.param_values = [{"region": "us"}]
    space = _build_template(
        "T0002", schema, q, intent, "SELECT film_id FROM film", stats=TemplateStats(accept=3, reject=0)
    )
    space.value_history.questions = [q]
    space.value_history.param_values = [{"region": "us"}]
    space.value_history.natural_language = [""]
    space.value_history.accept_counts = [3]
    TemplateStoreLifecycleOps._merge_template_value_histories_master_first(master, space)
    assert len(master.value_history.questions) == 1
    assert master.value_history.accept_counts[0] == 4


@pytest.mark.fast
def test_merge_case18_value_history_union_cap_keeps_master_first(tmp_path: Path, monkeypatch) -> None:
    from aetherdialect._config import EngineLimits
    from aetherdialect._templates import TemplateStoreLifecycleOps

    monkeypatch.setattr(
        TemplateStoreLifecycleOps,
        "_resolve_engine_limits",
        lambda: EngineLimits(template_value_history_depth=3),
    )
    q = normalize_question("how many films")
    intent = _scalar_intent("film", "film_id")
    master = _build_template("T0001", _sample_schema(), q, intent, "SELECT film_id FROM film")
    master.value_history.questions = [q, normalize_question("count films in catalog")]
    master.value_history.param_values = [{}, {}]
    master.value_history.natural_language = ["", ""]
    master.value_history.accept_counts = [1, 1]
    space = _build_template("T0002", _sample_schema(), q, intent, "SELECT film_id FROM film")
    space.value_history.questions = [
        q,
        normalize_question("film total"),
        normalize_question("number of films"),
        normalize_question("all films count"),
    ]
    space.value_history.param_values = [{}, {}, {}, {}]
    space.value_history.natural_language = ["", "", "", ""]
    space.value_history.accept_counts = [1, 1, 1, 1]
    TemplateStoreLifecycleOps._merge_template_value_histories_master_first(master, space)
    assert len(master.value_history.questions) == 3
    assert q in set(master.value_history.questions)
    assert normalize_question("count films in catalog") in set(master.value_history.questions)


@pytest.mark.fast
def test_merge_case20_union_family_index_recomputed_after_merge(tmp_path: Path) -> None:
    from aetherdialect._constants import TEMPLATE_UNION_FAMILY_INDEX_KEY
    from aetherdialect._utils_intent import body_similarity_key_for_concrete, join_path_key_concrete

    schema = _sample_schema()
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    intent = _scalar_intent("film", "film_id")
    master = _build_template("T0001", schema, "how many films", intent, "SELECT film_id FROM film")
    space = _build_template("T0002", schema, "count all films", intent, "SELECT film_id FROM film")
    _save_templates(artifacts_dir, schema, "master", {"T0001": master})
    _save_templates(artifacts_dir, schema, "films_only", {"T0002": space})
    master_store = TemplateOps.load_template_store(
        schema.schema_graph_id,
        schema,
        artifacts_dir=str(artifacts_dir),
        space_name="master",
    )
    master_store._indexes[TEMPLATE_UNION_FAMILY_INDEX_KEY] = {"stale": ["T9999"]}
    TemplateOps.save_template_store(master_store)

    _run_merge(artifacts_dir, schema)

    store = TemplateOps.load_template_store(
        schema.schema_graph_id,
        schema,
        artifacts_dir=str(artifacts_dir),
        space_name="master",
    )
    family_index = store._indexes[TEMPLATE_UNION_FAMILY_INDEX_KEY]
    body_key = body_similarity_key_for_concrete(master.intent_signature)
    join_key = join_path_key_concrete(master.intent_signature)
    merged = _load_templates(artifacts_dir, schema, "master")["T0001"]
    assert merged.id in family_index.get(body_key, [])
    assert merged.id in family_index.get(f"{body_key}|{join_key}", [])
    assert "T9999" not in {tid for ids in family_index.values() for tid in ids}


@pytest.mark.fast
def test_merge_case22_delete_writes_master_before_purge(tmp_path: Path, monkeypatch) -> None:
    from aetherdialect._main_spaces import MainSpaceOps
    from aetherdialect._templates import TemplateStoreLifecycleOps

    schema = _sample_schema()
    artifacts_dir = tmp_path / "engine"
    artifacts_dir.mkdir()
    snap = MainSpaceOps.subset_graph_for_space(schema, SpaceContext(tables=frozenset({"film"})))
    MainSpaceOps.save_aetherspace_snapshot(str(artifacts_dir), "films_only", snap)
    space = _build_template(
        "T0002", schema, "how many films", _scalar_intent("film", "film_id"), "SELECT film_id FROM film"
    )
    _save_templates(artifacts_dir, schema, "master", {})
    _save_templates(artifacts_dir, schema, "films_only", {"T0002": space})
    calls: list[str] = []
    original_save = TemplateStoreLifecycleOps.save_template_store
    original_purge = TemplateOps.purge_space_learning_partition

    def _tracking_save(store) -> None:
        calls.append("save")
        return original_save(store)

    def _tracking_purge(artifacts: str, space_name: str) -> bool:
        calls.append("purge")
        return original_purge(artifacts, space_name)

    monkeypatch.setattr(TemplateStoreLifecycleOps, "save_template_store", _tracking_save)
    monkeypatch.setattr(TemplateOps, "purge_space_learning_partition", _tracking_purge)
    result = MainSpaceOps.delete_aetherspace(
        str(artifacts_dir),
        "films_only",
        persist_learning=True,
        schema_graph=schema,
    )
    assert result.deleted is True
    assert calls.index("save") < calls.index("purge")
    assert _load_templates(artifacts_dir, schema, "master")
    space_dir = artifacts_dir / "intent_templates" / "spaces" / "films_only"
    assert not space_dir.exists() or not any(space_dir.iterdir())
