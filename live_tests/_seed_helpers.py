"""
Shared infrastructure for seeded live tests.

Exposes a function-scoped ``isolated_runner`` context manager that redirects ``EngineConfig.TEMPLATE_STORE_DIR`` to a unique subdirectory so every test starts with a guaranteed-empty template store. Adds intent builders for the canonical ``dvdrental`` shapes reused across multiple seeded live tests, plus thin wrappers for seeding templates and question-level feedback (rejections and structural validation rows). ``capture_parse_prompt`` records prior question feedback passed to ``_build_intent_parse_prompt`` so tests can assert which memory reaches the LLM.
"""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from aetherdialect._config import EngineConfig
from aetherdialect._contracts_base import TemplateStats
from aetherdialect._contracts_core import (
    FeedbackKind,
    FilterParam,
    NormalizedExpr,
    RuntimeIntent,
    SelectCol,
    Template,
    ValueHistory,
)
from aetherdialect._live_testing import LiveTestRunner
from aetherdialect._templates import (
    empty_template_store,
    insert_template,
    record_question_feedback,
    store_to_templates,
    summarize_failure_for_memory,
)
from aetherdialect._utils import intent_key

from .conftest import _instrument_runner


def _question_feedback_entry_count(store: dict[str, Any]) -> int:
    qf = store.get("question_feedback")
    if not isinstance(qf, dict):
        return 0
    n = 0
    for rows in qf.values():
        if isinstance(rows, list):
            n += len(rows)
    return n


def empty_store(effective_structural_hash: str):
    """Return a fresh empty partitioned template store (alias for :func:`empty_template_store`)."""

    return empty_template_store(effective_structural_hash)


@contextmanager
def isolated_runner(schema: Any, schema_terms: set[str], t2s: Any, *, label: str) -> Iterator[LiveTestRunner]:
    """
    Yield an instrumented ``LiveTestRunner`` with an isolated on-disk store.

    ``EngineConfig.TEMPLATE_STORE_DIR`` is redirected to a unique directory inside
    ``t2s._artifacts_dir`` and removed on teardown so residual state cannot leak.

    Args:

        schema: Profiled ``SchemaGraph`` shared across live tests.

        schema_terms: Schema term tokens for the runner.

        t2s: Session ``Text2SQL`` instance for the artifacts directory.

        label: Short identifier embedded in the isolated directory name.

    Yields:

        A ``LiveTestRunner`` bound to the fresh store.
    """

    original_dir = EngineConfig.TEMPLATE_STORE_DIR
    isolated_dir = os.path.join(
        str(t2s._artifacts_dir),
        f"seed_{label}_{uuid.uuid4().hex[:8]}_tmpl",
    )
    if os.path.isdir(isolated_dir):
        shutil.rmtree(isolated_dir, ignore_errors=True)
    os.makedirs(isolated_dir, exist_ok=True)
    EngineConfig.TEMPLATE_STORE_DIR = isolated_dir
    try:
        store = empty_template_store(schema.effective_structural_hash)
        runner = LiveTestRunner(
            schema=schema,
            store=store,
            templates=store_to_templates(store),
            rejected={},
            schema_terms=set(schema_terms),
            csv_dir=t2s._artifacts_dir,
        )
        _instrument_runner(runner)
        yield runner
    finally:
        EngineConfig.TEMPLATE_STORE_DIR = original_dir
        if os.path.isdir(isolated_dir):
            shutil.rmtree(isolated_dir, ignore_errors=True)


def intent_rental_count_by_store() -> RuntimeIntent:
    """Grouped rental count per ``inventory.store_id``."""

    return RuntimeIntent(
        tables=["rental", "inventory"],
        grain="grouped",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("inventory.store_id")),
            SelectCol(expr=NormalizedExpr.from_agg("count", "rental.rental_id")),
        ],
        group_by_cols=[NormalizedExpr.from_column("inventory.store_id")],
        order_by_cols=[],
        filters_param=[],
        having_param=[],
        natural_language="rentals per store",
    )


def intent_payment_sum_by_staff() -> RuntimeIntent:
    """Grouped sum of ``payment.amount`` per ``payment.staff_id``."""

    return RuntimeIntent(
        tables=["payment"],
        grain="grouped",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("payment.staff_id")),
            SelectCol(expr=NormalizedExpr.from_agg("sum", "payment.amount")),
        ],
        group_by_cols=[NormalizedExpr.from_column("payment.staff_id")],
        order_by_cols=[],
        filters_param=[],
        having_param=[],
        natural_language="total payments per staff",
    )


def intent_customer_first_names() -> RuntimeIntent:
    """Row-level select of ``customer.customer_id`` and ``customer.first_name``."""

    return RuntimeIntent(
        tables=["customer"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("customer.customer_id")),
            SelectCol(expr=NormalizedExpr.from_column("customer.first_name")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
        having_param=[],
        natural_language="list customer ids and first names",
    )


def intent_customer_full_names() -> RuntimeIntent:
    """Row-level select of customer id, first name, and last name."""

    return RuntimeIntent(
        tables=["customer"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("customer.customer_id")),
            SelectCol(expr=NormalizedExpr.from_column("customer.first_name")),
            SelectCol(expr=NormalizedExpr.from_column("customer.last_name")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
        having_param=[],
        natural_language="list customer ids first names and last names",
    )


def intent_customer_emails_only() -> RuntimeIntent:
    """Row-level select of ``customer.customer_id`` and ``customer.email``."""

    return RuntimeIntent(
        tables=["customer"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("customer.customer_id")),
            SelectCol(expr=NormalizedExpr.from_column("customer.email")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
        having_param=[],
        natural_language="list customer ids and emails",
    )


def intent_store_staff_by_work() -> RuntimeIntent:
    """Row-level ``store`` + ``staff`` join on the ``staff.store_id`` FK edge."""

    return RuntimeIntent(
        tables=["store", "staff"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("store.store_id")),
            SelectCol(expr=NormalizedExpr.from_column("store.last_update")),
            SelectCol(expr=NormalizedExpr.from_column("staff.first_name")),
            SelectCol(expr=NormalizedExpr.from_column("staff.last_name")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
        having_param=[],
        natural_language=(
            "list every staff first and last name together with the last update time of the store where they work"
        ),
    )


def intent_store_manager() -> RuntimeIntent:
    """Row-level ``store`` + ``staff`` join on the ``store.manager_staff_id`` FK edge."""

    return RuntimeIntent(
        tables=["store", "staff"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("store.store_id")),
            SelectCol(expr=NormalizedExpr.from_column("staff.first_name")),
            SelectCol(expr=NormalizedExpr.from_column("staff.last_name")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[],
        having_param=[],
        natural_language="who manages each store",
    )


def intent_film_in_category(category_name: str) -> RuntimeIntent:
    """Row-level ``film`` title filtered by ``film_category``/``category`` join."""

    return RuntimeIntent(
        tables=["film", "film_category", "category"],
        grain="row_level",
        select_cols=[
            SelectCol(expr=NormalizedExpr.from_column("film.title")),
        ],
        group_by_cols=[],
        order_by_cols=[],
        filters_param=[
            FilterParam(
                left_expr=NormalizedExpr.from_column("category.name"),
                op="=",
                value_type="text",
                bool_op="AND",
                raw_value=category_name,
            ),
        ],
        having_param=[],
        natural_language=f"films in the {category_name} category",
    )


def seed_template(
    runner: LiveTestRunner,
    *,
    q_norm: str,
    intent: RuntimeIntent,
    sql: str,
    trust_level: int = 1,
    stats: TemplateStats | None = None,
    value_history: ValueHistory | None = None,
    structural_override: dict[str, Any] | None = None,
    template_source: str = "human",
) -> Template:
    """
    Insert one template into *runner*'s store.

    When *structural_override* is non-empty, each key replaces the matching
    entry in ``Template.structural_defaults`` after insertion so the stored
    value_history row differs from the structural default (the lever that
    routes fuzzy reuse through ``GenerationPath.FUZZY_REUSE_FULL_PARAMS``).

    Args:

        runner: Target ``LiveTestRunner`` whose store/templates are mutated.

        q_norm: Normalised seed question stored in ``value_history.questions``.

        intent: Seed intent whose ``param_values`` feed structural defaults.

        sql: Seed SQL (canonicalised + parameterised by ``insert_template``).

        trust_level: Stored ``Template.trust_level``; default is 1.

        stats: Optional ``TemplateStats``; default is ``accept=3, reject=0``.

        value_history: Optional pre-built history; default is a single row.

        structural_override: Optional ``s*`` overrides applied post-insert.

        template_source: Stored ``Template.source`` marker.

    Returns:

        The newly inserted or merged ``Template``.
    """

    if stats is None:
        stats = TemplateStats(accept=3, reject=0)
    template = insert_template(
        runner.store,
        runner.templates,
        runner.schema,
        q_norm,
        intent,
        sql,
        template_source=template_source,
        template_trust_level=trust_level,
        template_initial_stats=stats,
        template_value_history=value_history,
    )
    if structural_override:
        for key, value in structural_override.items():
            template.structural_defaults[key] = value
    return template


def seed_rejected(
    runner: LiveTestRunner,
    *,
    q_norm: str,
    intent: RuntimeIntent,
    sql: str,
    reason: str = "seeded rejection",
) -> SimpleNamespace:
    """Insert one ``INTENT_REJECTED`` question-feedback row into *runner*'s store."""

    ent = summarize_failure_for_memory(
        question=q_norm,
        intent=intent,
        kind=FeedbackKind.INTENT_REJECTED,
        schema_hash=runner.schema.effective_structural_hash,
        user_reason=reason,
        sql=sql,
    )
    record_question_feedback(runner.store, q_norm, ent)
    return SimpleNamespace(
        id=f"qf:{q_norm}",
        intent_key=intent_key(intent),
        value_history=SimpleNamespace(rejection_reasons=[reason]),
    )


def seed_negative_memory(
    runner: LiveTestRunner,
    *,
    intent: RuntimeIntent,
    sql: str,
    reason: str,
    repeats: int = 1,
    q_norm: str | None = None,
) -> dict[str, str]:
    """
    Seed validation-failure feedback rows for *intent* (penalty / hint paths).

    Each repeat appends one ``question_feedback`` row scoped to the runner schema so
    :func:`aetherdialect._templates.compute_question_feedback_penalty` observes the seed.

    Returns the computed keys (``ikey``, ``sql_fp``, ``colmap_sig``, ``q_norm``) for assertions.
    """

    from aetherdialect._core_utils import (
        canonicalize_sql,
        colmap_signature,
        normalize_sql,
    )
    from aetherdialect._dialect import (
        active_sqlglot_dialect,
        compute_sql_fp,
        parameter_abstract,
    )

    sql_canon = canonicalize_sql(sql)
    sql_norm = normalize_sql(sql_canon)
    sql_param, _ = parameter_abstract(sql_norm, sqlglot_dialect=active_sqlglot_dialect())
    ikey = intent_key(intent)
    sql_fp = compute_sql_fp(sql_param, sqlglot_dialect=active_sqlglot_dialect())
    cmap_sig = colmap_signature(intent.column_map)
    qn = q_norm or intent.natural_language or f"seed-negative-memory::{ikey}"
    eff = runner.schema.effective_structural_hash
    for _ in range(max(1, repeats)):
        ent = summarize_failure_for_memory(
            question=qn,
            intent=intent,
            kind=FeedbackKind.VALIDATION_FAILURE,
            schema_hash=eff,
            validator_errors=[reason],
            sql=sql,
        )
        record_question_feedback(runner.store, qn, ent)
    return {"ikey": ikey, "sql_fp": sql_fp, "colmap_sig": cmap_sig, "q_norm": qn}


@contextmanager
def capture_parse_prompt() -> Iterator[list[dict[str, Any]]]:
    """
    Record calls into intent-parse prompt construction and forward them through.

    The captured list includes entries from ``_build_intent_parse_prompt`` (with
    ``prior_question_feedback``) and a slim record for each ``full_intent_parse`` /
    ``_invoke_intent_parse_with_hints`` call (``store`` / ``in_turn_seed`` / ``budget``).
    """

    import aetherdialect._intent_process as intent_process

    calls: list[dict[str, Any]] = []
    original_parse = intent_process.full_intent_parse
    original_invoke = intent_process._invoke_intent_parse_with_hints
    original_build = intent_process._build_intent_parse_prompt

    def _recording_build(
        question: str,
        schema_literal_json: str,
        table_list: list[str],
        prior_question_feedback: list[dict[str, str]] | None = None,
    ) -> tuple[str, str]:
        calls.append(
            {
                "via": "_build_intent_parse_prompt",
                "question": question,
                "prior_question_feedback": (list(prior_question_feedback) if prior_question_feedback else None),
            }
        )
        return original_build(question, schema_literal_json, table_list, prior_question_feedback)

    def _recording_parse(
        question: str,
        schema_graph: Any,
        max_retries: int = 3,
        *,
        store: Any | None = None,
        in_turn_seed: list[dict[str, str]] | None = None,
        budget: Any | None = None,
    ) -> Any:
        calls.append(
            {
                "via": "full_intent_parse",
                "question": question,
                "store": store is not None,
                "in_turn_seed": (list(in_turn_seed) if in_turn_seed else None),
                "budget": budget is not None,
            }
        )
        return original_parse(
            question,
            schema_graph,
            max_retries=max_retries,
            store=store,
            in_turn_seed=in_turn_seed,
            budget=budget,
        )

    def _recording_invoke(
        question: str,
        schema_graph: Any,
        *,
        max_retries: int = 3,
        store: dict[str, Any] | None = None,
        in_turn_seed: list[dict[str, str]] | None = None,
        budget: Any | None = None,
    ) -> Any:
        calls.append(
            {
                "via": "_invoke_intent_parse_with_hints",
                "question": question,
                "store": store is not None,
                "in_turn_seed": (list(in_turn_seed) if in_turn_seed else None),
                "budget": budget is not None,
            }
        )
        return original_invoke(
            question,
            schema_graph,
            max_retries=max_retries,
            store=store,
            in_turn_seed=in_turn_seed,
            budget=budget,
        )

    with patch.object(intent_process, "_build_intent_parse_prompt", _recording_build):
        with patch.object(intent_process, "full_intent_parse", _recording_parse):
            with patch.object(intent_process, "_invoke_intent_parse_with_hints", _recording_invoke):
                yield calls


def deterministic_join_choice_patch() -> Any:
    """
    Patch ``aetherdialect._sql_gen.get_join_choice_from_llm`` to pick the first candidate.

    Use this inside seeded generation-path tests where the specific join edge is irrelevant and the assertion is about the generation path code. Matches the keyword-only join-choice API and returns a scope-to-id dict merged with any preset choices.
    """

    def _first_candidate(
        q_norm: str,
        deterministic_sql: str,
        *,
        llm_scopes: list[dict[str, Any]],
        preset_choices: dict[str, str] | None = None,
        accept_na_by_scope: dict[str, bool] | None = None,
        require_final: bool = False,
    ) -> dict[str, str]:
        out = dict(preset_choices or {})
        for block in llm_scopes:
            sk = str(block.get("scope") or "")
            cands = list(block.get("candidates") or [])
            if not cands:
                out[sk] = "J00"
                continue
            out[sk] = str(cands[0].get("candidate_id") or "J00")
        return out

    return patch("aetherdialect._sql_gen.get_join_choice_from_llm", side_effect=_first_candidate)


def forced_join_choice_patch(
    predicate: Callable[[list[dict[str, Any]]], str],
) -> Any:
    """
    Patch ``aetherdialect._sql_gen.get_join_choice_from_llm`` to choose by *predicate*.

    *predicate* receives the raw main-query candidate list (as produced by ``join_hints_multi``) and must return one of the ``id`` strings in that list. Raises ``RuntimeError`` when *predicate* picks an id that is not valid so tests fail fast on fixture mistakes.
    """

    def _forced(
        q_norm: str,
        deterministic_sql: str,
        *,
        llm_scopes: list[dict[str, Any]],
        preset_choices: dict[str, str] | None = None,
        accept_na_by_scope: dict[str, bool] | None = None,
        require_final: bool = False,
    ) -> dict[str, str]:
        out = dict(preset_choices or {})
        for block in llm_scopes:
            sk = str(block.get("scope") or "")
            candidates = list(block.get("candidates") or [])
            valid = {str(c.get("candidate_id")) for c in candidates if c.get("candidate_id")}
            if not valid:
                out[sk] = "J00"
                continue
            chosen = predicate(candidates)
            if chosen not in valid:
                raise RuntimeError(f"forced_join_choice_patch picked {chosen!r} which is not in {sorted(valid)!r}")
            out[sk] = chosen
        return out

    return patch("aetherdialect._sql_gen.get_join_choice_from_llm", side_effect=_forced)


def capture_join_candidates() -> Any:
    """
    Capture arguments passed to ``aetherdialect._sql_gen.get_join_choice_from_llm``.

    Returns a ``patch`` context whose ``side_effect`` records ``q_norm`` and ``llm_scopes`` on the shared ``calls`` list. The patched callable still forwards to the real implementation so join choice proceeds as-is.
    """

    import aetherdialect._sql_gen

    calls: list[dict[str, Any]] = []
    original = aetherdialect._sql_gen.get_join_choice_from_llm

    def _recording(
        q_norm: str,
        deterministic_sql: str,
        *,
        llm_scopes: list[dict[str, Any]],
        preset_choices: dict[str, str] | None = None,
        accept_na_by_scope: dict[str, bool] | None = None,
        require_final: bool = False,
    ) -> dict[str, str]:
        calls.append({"q_norm": q_norm, "llm_scopes": list(llm_scopes)})
        return original(
            q_norm,
            deterministic_sql,
            llm_scopes=llm_scopes,
            preset_choices=preset_choices,
            accept_na_by_scope=accept_na_by_scope,
            require_final=require_final,
        )

    ctx = patch("aetherdialect._sql_gen.get_join_choice_from_llm", side_effect=_recording)
    ctx._calls = calls
    return ctx


def rejected_for_intent(runner: LiveTestRunner, intent: RuntimeIntent) -> None:
    """Deprecated live-test helper: legacy rejected map is empty; use ``question_feedback`` instead."""

    _ = (runner, intent)
    return None


def _kit_baseline_templates(runner: LiveTestRunner) -> dict[str, str]:
    """
    Seed four trust=2 templates with realistic per-pair feedback counts.

    Returns a dict mapping a short alias to the inserted template id so callers can reference seeded rows in assertions.
    """

    from aetherdialect._contracts_core import FeedbackCounts

    out: dict[str, str] = {}
    for alias, q_norm, intent, sql in (
        (
            "first_names",
            "list customer first names",
            intent_customer_first_names(),
            "SELECT customer.customer_id, customer.first_name FROM customer",
        ),
        (
            "full_names",
            "list customer first and last names",
            intent_customer_full_names(),
            "SELECT customer.customer_id, customer.first_name, customer.last_name FROM customer",
        ),
        (
            "rentals_per_store",
            "rental count per store baseline",
            intent_rental_count_by_store(),
            "SELECT inventory.store_id, COUNT(rental.rental_id) FROM rental "
            "JOIN inventory ON rental.inventory_id = inventory.inventory_id "
            "GROUP BY inventory.store_id",
        ),
        (
            "payments_per_staff",
            "total payments per staff baseline",
            intent_payment_sum_by_staff(),
            "SELECT staff_id, SUM(amount) FROM payment GROUP BY staff_id",
        ),
    ):
        tmpl = seed_template(
            runner,
            q_norm=q_norm,
            intent=intent,
            sql=sql,
            trust_level=2,
            stats=TemplateStats(accept=4, reject=1),
        )
        tmpl.feedback_by_question[q_norm] = FeedbackCounts(accepts=4, rejects=1, last_path=1)
        out[alias] = tmpl.id
    return out


def _kit_cold_templates(runner: LiveTestRunner) -> dict[str, str]:
    """Seed two trust=1 templates with stats=(1, 0) for promotion-gate tests."""

    out: dict[str, str] = {}
    for alias, q_norm, intent, sql in (
        (
            "first_names_cold",
            "list customer first names",
            intent_customer_first_names(),
            "SELECT customer.customer_id, customer.first_name FROM customer",
        ),
        (
            "full_names_cold",
            "list customer first and last names",
            intent_customer_full_names(),
            "SELECT customer.customer_id, customer.first_name, customer.last_name FROM customer",
        ),
    ):
        tmpl = seed_template(
            runner,
            q_norm=q_norm,
            intent=intent,
            sql=sql,
            trust_level=1,
            stats=TemplateStats(accept=1, reject=0),
        )
        out[alias] = tmpl.id
    return out


def _kit_rejected_aggregations(runner: LiveTestRunner) -> dict[str, str]:
    """Seed two ``INTENT_REJECTED`` feedback rows representing wrong-aggregation feedback."""

    out: dict[str, str] = {}
    rt_a = seed_rejected(
        runner,
        q_norm="rentals per store wrong agg",
        intent=intent_rental_count_by_store(),
        sql="SELECT inventory.store_id, SUM(rental.rental_id) FROM rental "
        "JOIN inventory ON rental.inventory_id = inventory.inventory_id "
        "GROUP BY inventory.store_id",
        reason="seeded wrong aggregation: SUM where COUNT was expected",
    )
    out["rentals_wrong_agg"] = str(rt_a.id)
    rt_b = seed_rejected(
        runner,
        q_norm="payments per staff wrong agg",
        intent=intent_payment_sum_by_staff(),
        sql="SELECT staff_id, COUNT(amount) FROM payment GROUP BY staff_id",
        reason="seeded wrong aggregation: COUNT where SUM was expected",
    )
    out["payments_wrong_agg"] = str(rt_b.id)
    return out


def _kit_rejected_join_paths(runner: LiveTestRunner) -> dict[str, str]:
    """Seed two ``INTENT_REJECTED`` feedback rows representing wrong-join-edge feedback."""

    work_intent = intent_store_staff_by_work()
    work_intent.chosen_join_candidate_id = "J99"
    work_intent.chosen_join_path_signature = ["store.manager_staff_id->staff.staff_id"]
    rt_work = seed_rejected(
        runner,
        q_norm="staff working at each store wrong edge",
        intent=work_intent,
        sql="SELECT staff.first_name, staff.last_name, store.store_id, store.last_update "
        "FROM store JOIN staff ON store.manager_staff_id = staff.staff_id",
        reason="seeded wrong join edge: chose manager FK for work-assignment question",
    )
    manager_intent = intent_store_manager()
    manager_intent.chosen_join_candidate_id = "J98"
    manager_intent.chosen_join_path_signature = ["staff.store_id->store.store_id"]
    rt_manager = seed_rejected(
        runner,
        q_norm="store manager wrong edge",
        intent=manager_intent,
        sql="SELECT store.store_id, staff.first_name, staff.last_name "
        "FROM staff JOIN store ON staff.store_id = store.store_id",
        reason="seeded wrong join edge: chose work FK for manager question",
    )
    return {
        "work_wrong_edge": str(rt_work.id),
        "manager_wrong_edge": str(rt_manager.id),
    }


def _kit_intent_failures(runner: LiveTestRunner) -> dict[str, str]:
    """Seed distinct structural validation rows scoped to the runner's schema."""

    from aetherdialect._core_utils import normalize_question

    seed_negative_memory(
        runner,
        intent=intent_customer_first_names(),
        sql="SELECT customer.customer_id FROM customer",
        reason="intent parse failed: unknown column 'thing'",
        q_norm=normalize_question("show me a thing that does not parse"),
    )
    seed_negative_memory(
        runner,
        intent=intent_payment_sum_by_staff(),
        sql="SELECT staff_id, SUM(amount) FROM payment",
        reason="validation failed: missing group_by_cols for grouped grain",
        q_norm=normalize_question("grouped query missing group by column"),
    )
    seed_negative_memory(
        runner,
        intent=intent_rental_count_by_store(),
        sql=(
            "SELECT inventory.store_id, COUNT(rental.rental_id) FROM rental "
            "JOIN inventory ON rental.inventory_id = inventory.inventory_id "
            "WHERE film.film_id = 1 GROUP BY inventory.store_id"
        ),
        reason="validation failed: filter references table not in tables list",
        q_norm=normalize_question("filter on unrelated table"),
    )
    seed_negative_memory(
        runner,
        intent=intent_customer_full_names(),
        sql="SELECT customer.customer_id, customer.first_name FROM customer",
        reason="validation failed: select_cols missing customer.last_name",
        q_norm=normalize_question("list customer first and last names with hints"),
    )
    return {}


def _kit_negative_memory_full(runner: LiveTestRunner) -> dict[str, str]:
    """
    Seed all three negative-memory sources for one canonical template/intent.

    Inserts the accepted template, a matching intent-level rejection for the same shape,
    and one validation row keyed to the same q_norm so ``compute_question_feedback_penalty`` observes stacked feedback.
    """

    intent = intent_rental_count_by_store()
    sql = (
        "SELECT inventory.store_id, COUNT(rental.rental_id) FROM rental "
        "JOIN inventory ON rental.inventory_id = inventory.inventory_id "
        "GROUP BY inventory.store_id"
    )
    q_norm = "rentals per store negative memory full"
    tmpl = seed_template(
        runner,
        q_norm=q_norm,
        intent=intent,
        sql=sql,
        trust_level=2,
        stats=TemplateStats(accept=4, reject=1),
    )
    rt = seed_rejected(
        runner,
        q_norm=q_norm,
        intent=intent,
        sql=sql,
        reason="seeded rejection co-located with accepted template",
    )
    pay = intent_payment_sum_by_staff()
    seed_negative_memory(
        runner,
        intent=pay,
        sql="SELECT staff_id, COUNT(amount) FROM payment GROUP BY staff_id",
        reason="seeded failure-log row co-located with accepted template",
        q_norm=q_norm,
    )
    return {"template": tmpl.id, "rejected": str(rt.id), "q_norm": q_norm}


def _kit_multi_pair_template(runner: LiveTestRunner) -> dict[str, str]:
    """
    Seed one trust=1 template with three distinct ``feedback_by_question`` pairs.

    Pair A: accepts=2, rejects=0 (eligible for promotion). Pair B: accepts=0, rejects=1 (mid-rejection). Pair C: accepts=1, rejects=0 (single accept).
    """

    from aetherdialect._contracts_core import FeedbackCounts

    tmpl = seed_template(
        runner,
        q_norm="multi pair template seed",
        intent=intent_customer_first_names(),
        sql="SELECT customer.customer_id, customer.first_name FROM customer",
        trust_level=1,
        stats=TemplateStats(accept=3, reject=1),
    )
    tmpl.feedback_by_question["pair a list customer first names"] = FeedbackCounts(accepts=2, rejects=0, last_path=1)
    tmpl.feedback_by_question["pair b show customer first names"] = FeedbackCounts(accepts=0, rejects=1, last_path=2)
    tmpl.feedback_by_question["pair c give me customer first names"] = FeedbackCounts(accepts=1, rejects=0, last_path=3)
    return {"template": tmpl.id}


_KITS: dict[str, Callable[[LiveTestRunner], dict[str, str]]] = {
    "baseline_templates": _kit_baseline_templates,
    "cold_templates": _kit_cold_templates,
    "rejected_aggregations": _kit_rejected_aggregations,
    "rejected_join_paths": _kit_rejected_join_paths,
    "intent_failures": _kit_intent_failures,
    "negative_memory_full": _kit_negative_memory_full,
    "multi_pair_template": _kit_multi_pair_template,
}


@contextmanager
def seeded_runner(
    schema: Any,
    schema_terms: set[str],
    t2s: Any,
    *,
    label: str,
    kits: tuple[str, ...] = (),
) -> Iterator[LiveTestRunner]:
    """
    Yield an isolated ``LiveTestRunner`` pre-populated by the requested *kits*.

    Each kit name in ``kits`` is applied in order against the fresh runner. Unknown kit names raise ``KeyError`` immediately so fixture typos surface at test setup time. The mapping ``runner.seeded_ids`` is attached so tests can look up artefact ids by alias (e.g. ``runner.seeded_ids["baseline_templates"]["first_names"]``).
    """

    with isolated_runner(schema, schema_terms, t2s, label=label) as runner:
        seeded_ids: dict[str, dict[str, str]] = {}
        for kit_name in kits:
            if kit_name not in _KITS:
                raise KeyError(f"unknown kit name: {kit_name!r}; available: {sorted(_KITS)!r}")
            seeded_ids[kit_name] = _KITS[kit_name](runner)
        runner.seeded_ids = seeded_ids
        yield runner


def snapshot_store(runner: LiveTestRunner) -> dict[str, Any]:
    """
    Return a structural snapshot of *runner*'s store sections for before/after diffing.

    The snapshot captures id sets, per-template trust level, per-template stats, and per-template feedback_by_question dict so tests can pinpoint which row changed.
    """

    return {
        "template_ids": frozenset(runner.templates),
        "rejected_ids": frozenset(),
        "rejected_intent_ids": frozenset(),
        "question_feedback_keys": frozenset((runner.store.get("question_feedback") or {}).keys()),
        "question_feedback_total_entries": _question_feedback_entry_count(runner.store),
        "intent_failure_count": _question_feedback_entry_count(runner.store),
        "trust_by_id": {tid: t.trust_level for tid, t in runner.templates.items()},
        "stats_by_id": {tid: (t.stats.accept, t.stats.reject) for tid, t in runner.templates.items()},
        "feedback_by_question_by_id": {
            tid: {qn: (fc.accepts, fc.rejects, fc.last_path) for qn, fc in (t.feedback_by_question or {}).items()}
            for tid, t in runner.templates.items()
        },
    }


def assert_template_unchanged(before: dict[str, Any], after: dict[str, Any], template_id: str) -> None:
    """Assert *template_id*'s trust_level, stats, and feedback_by_question are unchanged."""

    assert template_id in before["trust_by_id"], (
        f"[assert_template_unchanged] template {template_id!r} missing from before snapshot"
    )
    assert template_id in after["trust_by_id"], (
        f"[assert_template_unchanged] template {template_id!r} was deleted between snapshots"
    )
    assert before["trust_by_id"][template_id] == after["trust_by_id"][template_id], (
        f"[assert_template_unchanged] trust changed for {template_id!r}: "
        f"{before['trust_by_id'][template_id]} -> {after['trust_by_id'][template_id]}"
    )
    assert before["stats_by_id"][template_id] == after["stats_by_id"][template_id], (
        f"[assert_template_unchanged] stats changed for {template_id!r}: "
        f"{before['stats_by_id'][template_id]} -> {after['stats_by_id'][template_id]}"
    )
    assert before["feedback_by_question_by_id"][template_id] == after["feedback_by_question_by_id"][template_id], (
        f"[assert_template_unchanged] feedback_by_question changed for {template_id!r}: "
        f"{before['feedback_by_question_by_id'][template_id]} -> "
        f"{after['feedback_by_question_by_id'][template_id]}"
    )


def assert_new_template_forked(before: dict[str, Any], after: dict[str, Any]) -> set[str]:
    """Assert at least one new template id appeared; return the set of new ids."""

    new_ids = set(after["template_ids"]) - set(before["template_ids"])
    assert new_ids, (
        f"[assert_new_template_forked] no new templates inserted; "
        f"before={sorted(before['template_ids'])!r} after={sorted(after['template_ids'])!r}"
    )
    return new_ids


def assert_new_rejected_template(before: dict[str, Any], after: dict[str, Any]) -> set[str]:
    """Assert at least one new ``question_feedback`` row appeared (compat name for rejection tests)."""

    b = int(before.get("question_feedback_total_entries", 0))
    a = int(after.get("question_feedback_total_entries", 0))
    assert a > b, f"[assert_new_rejected_template] no new question_feedback rows; before_count={b!r} after_count={a!r}"
    return set()


__all__ = [
    "assert_new_rejected_template",
    "assert_new_template_forked",
    "assert_template_unchanged",
    "capture_join_candidates",
    "capture_parse_prompt",
    "deterministic_join_choice_patch",
    "empty_store",
    "forced_join_choice_patch",
    "intent_customer_emails_only",
    "intent_customer_first_names",
    "intent_customer_full_names",
    "intent_film_in_category",
    "intent_payment_sum_by_staff",
    "intent_rental_count_by_store",
    "intent_store_manager",
    "intent_store_staff_by_work",
    "isolated_runner",
    "rejected_for_intent",
    "seed_negative_memory",
    "seed_rejected",
    "seed_template",
    "seeded_runner",
    "snapshot_store",
]
