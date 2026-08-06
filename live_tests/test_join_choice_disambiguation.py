"""Seeded live tests for the ``store``/``staff`` FK-choice ambiguity in rental_shop. Between ``store`` and ``staff`` the catalog exposes two foreign keys: * ``staff.store_id`` -> ``store.store_id`` (assignment: which store a staff member works at). * ``store.manager_staff_id`` -> ``staff.staff_id`` (manager: which staff member manages a store). Each test owns an ``isolated_runner`` so the join-choice assertion is never contaminated by prior runs, and uses ``runner.run`` directly so it can inspect ``StepResult.intent.chosen_join_path_signature`` before committing to the next step."""

from __future__ import annotations

from typing import Any

from aetherdialect._contracts_core import Expected, GenerationPath, LiveTestRunner, RuntimeIntent, Scenario
from aetherdialect._templates import TemplateOps
from aetherdialect._utils import normalize_question

from ._seed_helpers import (
    assert_new_template_forked,
    assert_template_unchanged,
    intent_store_manager,
    intent_store_staff_by_work,
    isolated_runner,
    seed_template,
    snapshot_store,
)

_WORK_EDGE = "staff.store_id->store.store_id"
_MANAGER_EDGE = "store.manager_staff_id->staff.staff_id"
_WORK_QUESTION = "list every staff first and last name together with the last update time of the store where they work"
_WORK_SEED_Q_NORM = normalize_question(_WORK_QUESTION)
_MANAGER_QUESTION = "show each store id together with the first and last name of the staff member that manages it"
_MANAGER_SEED_Q_NORM = normalize_question(_MANAGER_QUESTION)


def _join_signature_contains(intent: RuntimeIntent | None, edge: str) -> bool:
    """Return whether *intent*'s chosen join signature includes *edge*."""
    if intent is None:
        return False
    return edge in list(intent.chosen_join_path_signature or [])


def _run(runner: LiveTestRunner, question: str, scenario_id: str) -> Any:
    """Execute *question* against *runner* with permissive expectations; return ``StepResult``."""
    scenario = Scenario(
        id=scenario_id,
        question=question,
        expected=Expected(tables=["store", "staff"], min_rows=1),
        category="join_choice",
    )
    return runner.run(scenario)


def _assert_edge(result: Any, edge: str, label: str) -> None:
    """Assert *result*'s intent picked *edge*; include a diagnostic header on failure."""
    assert result is not None and result.intent is not None, f"[{label}] pipeline returned no intent"
    sig = list(result.intent.chosen_join_path_signature or [])
    assert edge in sig, f"[{label}] expected join edge {edge!r} to be chosen; got {sig!r}"


def _templates_with_edge(templates: dict[str, Any], edge: str) -> list[Any]:
    """Return stored templates whose ``chosen_join_path_signature`` contains *edge*."""
    return [t for t in templates.values() if edge in list(t.chosen_join_path_signature or [])]


def test_fresh_work_edge(schema, schema_terms, t2s) -> None:
    """Empty store + work-assignment NL picks the ``staff.store_id`` edge and stores a template."""
    with isolated_runner(schema, schema_terms, t2s, label="jc_work") as runner:
        result = _run(
            runner,
            _WORK_QUESTION,
            "JC-FRESH-WORK",
        )
        _assert_edge(result, _WORK_EDGE, "JC-FRESH-WORK")
        _work_template_msg = (
            f"[JC-FRESH-WORK] expected an accepted template on the work edge; got {sorted(runner.templates)!r}"
        )
        assert _templates_with_edge(runner.templates, _WORK_EDGE), _work_template_msg


def test_fresh_manager_edge(schema, schema_terms, t2s) -> None:
    """Empty store + manager NL picks the ``store.manager_staff_id`` edge and stores a template."""
    with isolated_runner(schema, schema_terms, t2s, label="jc_manager") as runner:
        result = _run(
            runner,
            _MANAGER_QUESTION,
            "JC-FRESH-MANAGER",
        )
        _assert_edge(result, _MANAGER_EDGE, "JC-FRESH-MANAGER")
        _manager_template_msg = (
            f"[JC-FRESH-MANAGER] expected an accepted template on the manager edge; got {sorted(runner.templates)!r}"
        )
        assert _templates_with_edge(runner.templates, _MANAGER_EDGE), _manager_template_msg


def test_two_edges_coexist_after_fresh_runs(schema, schema_terms, t2s) -> None:
    """Two disjoint NL questions produce two templates with distinct join signatures. Starts from an empty store, runs the work-assignment question first and the manager question second, then asserts both templates are persisted and each carries its own join signature."""
    with isolated_runner(schema, schema_terms, t2s, label="jc_coexist") as runner:
        work_result = _run(
            runner,
            _WORK_QUESTION,
            "JC-COEXIST-WORK",
        )
        _assert_edge(work_result, _WORK_EDGE, "JC-COEXIST-WORK")

        manager_result = _run(
            runner,
            _MANAGER_QUESTION,
            "JC-COEXIST-MANAGER",
        )
        _assert_edge(manager_result, _MANAGER_EDGE, "JC-COEXIST-MANAGER")

        work_templates = _templates_with_edge(runner.templates, _WORK_EDGE)
        manager_templates = _templates_with_edge(runner.templates, _MANAGER_EDGE)
        assert work_templates, "[JC-COEXIST] missing work-edge template"
        assert manager_templates, "[JC-COEXIST] missing manager-edge template"
        _distinct_templates_msg = "[JC-COEXIST] work and manager templates must be distinct rows"
        assert {t.id for t in work_templates} & {t.id for t in manager_templates} == set(), _distinct_templates_msg


def test_reuse_picks_matching_history(schema, schema_terms, t2s) -> None:
    """Both edges are seeded up front; each variant NL question reuses the matching template. The work question must route through a reuse path (``1`` / ``2.1`` / ``2.2`` / ``3``) on the work template; the manager question must reuse the manager template."""
    reuse_paths = (
        GenerationPath.EXACT_QUESTION_REUSE.code,
        GenerationPath.FUZZY_REUSE_LITERAL_STRUCTURAL.code,
        GenerationPath.FUZZY_REUSE_FULL_PARAMS.code,
        GenerationPath.INTENT_DIRECT_MATCH.code,
        GenerationPath.UNION_TEMPLATE_WIDEN.code,
        GenerationPath.UNION_TEMPLATE_AND_RUNTIME_WIDEN.code,
        GenerationPath.RUNTIME_SUBSET_TEMPLATE_WIDE.code,
    )

    with isolated_runner(schema, schema_terms, t2s, label="jc_reuse") as runner:
        work_intent = intent_store_staff_by_work()
        work_intent.chosen_join_candidate_id = "J01"
        work_intent.chosen_join_path_signature = [_WORK_EDGE]
        seed_template(
            runner,
            q_norm=_WORK_SEED_Q_NORM,
            intent=work_intent,
            sql="SELECT staff.first_name, staff.last_name, store.store_id, store.last_update "
            "FROM staff JOIN store ON staff.store_id = store.store_id",
            trust_level=2,
        )

        manager_intent = intent_store_manager()
        manager_intent.chosen_join_candidate_id = "J02"
        manager_intent.chosen_join_path_signature = [_MANAGER_EDGE]
        seed_template(
            runner,
            q_norm=_MANAGER_SEED_Q_NORM,
            intent=manager_intent,
            sql="SELECT store.store_id, staff.first_name, staff.last_name "
            "FROM store JOIN staff ON store.manager_staff_id = staff.staff_id",
            trust_level=2,
        )
        runner.templates = TemplateOps.store_to_templates(runner.store)

        before = snapshot_store(runner)

        work_result = _run(
            runner,
            _WORK_QUESTION,
            "JC-REUSE-WORK",
        )
        _assert_edge(work_result, _WORK_EDGE, "JC-REUSE-WORK")
        _work_reuse_msg = f"[JC-REUSE-WORK] expected a reuse path; got {work_result.generation_path!r}"
        assert work_result.generation_path in reuse_paths, _work_reuse_msg

        manager_result = _run(
            runner,
            _MANAGER_QUESTION,
            "JC-REUSE-MANAGER",
        )
        _assert_edge(manager_result, _MANAGER_EDGE, "JC-REUSE-MANAGER")
        _manager_reuse_msg = f"[JC-REUSE-MANAGER] expected a reuse path; got {manager_result.generation_path!r}"
        assert manager_result.generation_path in reuse_paths, _manager_reuse_msg

        after = snapshot_store(runner)
        assert after["template_ids"] == before["template_ids"], (
            f"[JC-REUSE] reuse must not fork new templates; new ids: "
            f"{sorted(set(after['template_ids']) - set(before['template_ids']))!r}"
        )


def test_reuse_then_fork(schema, schema_terms, t2s) -> None:
    """With only the work edge seeded, asking a manager question forks a new template. The forked template must carry the manager join signature; the original work-edge template must remain untouched."""
    with isolated_runner(schema, schema_terms, t2s, label="jc_fork") as runner:
        work_intent = intent_store_staff_by_work()
        work_intent.chosen_join_candidate_id = "J01"
        work_intent.chosen_join_path_signature = [_WORK_EDGE]
        seeded = seed_template(
            runner,
            q_norm=_WORK_SEED_Q_NORM,
            intent=work_intent,
            sql="SELECT staff.first_name, staff.last_name, store.store_id, store.last_update "
            "FROM staff JOIN store ON staff.store_id = store.store_id",
            trust_level=2,
        )
        runner.templates = TemplateOps.store_to_templates(runner.store)
        before = snapshot_store(runner)

        manager_result = _run(
            runner,
            _MANAGER_QUESTION,
            "JC-FORK-MANAGER",
        )
        _assert_edge(manager_result, _MANAGER_EDGE, "JC-FORK-MANAGER")

        after = snapshot_store(runner)
        new_ids = assert_new_template_forked(before, after)
        assert_template_unchanged(before, after, seeded.id)
        forked = [runner.templates[tid] for tid in new_ids]
        _fork_manager_msg = (
            f"[JC-FORK] expected at least one forked template on manager edge; got "
            f"{[list(t.chosen_join_path_signature or []) for t in forked]!r}"
        )
        assert any(_MANAGER_EDGE in list(t.chosen_join_path_signature or []) for t in forked), _fork_manager_msg
        assert seeded.id in runner.templates, "[JC-FORK] seeded work template must still exist"
