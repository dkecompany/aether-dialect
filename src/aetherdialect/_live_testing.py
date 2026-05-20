"""
Live pipeline tests: scenarios, soft asserts, I/O patching, and runners against real LLM/DB.

Fixtures stay caller-specific.
"""

from __future__ import annotations

import os
import re
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Literal
from unittest.mock import patch

import aetherdialect._core_utils
import aetherdialect._dialect
import aetherdialect._expansion_ops
import aetherdialect._intent_expr
import aetherdialect._intent_process
import aetherdialect._intent_repair
import aetherdialect._intent_resolve
import aetherdialect._main_execution
import aetherdialect._pipeline
import aetherdialect._qsim
import aetherdialect._qsim_ops
import aetherdialect._schema
import aetherdialect._schema_profiling
import aetherdialect._seed_warmup
import aetherdialect._sql_gen
import aetherdialect._templates
import aetherdialect._utils
import aetherdialect._validation_agg
import aetherdialect._validation_execute
import aetherdialect._validation_schema
import aetherdialect._validation_semantic

from ._config import GenerationPath, PolicyConfig
from ._contracts_core import QuestionFormStorage, RuntimeIntent, SqlGenerationOutcome
from ._core_utils import debug, normalize_question, substitute_params
from ._dialect import (
    Dialect,
    active_sqlglot_dialect,
    resolve_dialect,
    sql_outer_has_join_or_comma_from,
)
from ._intent_process import (
    collect_structural_match_templates,
    match_template_for_union,
)
from ._pipeline import (
    best_accepted_template_similarity,
    build_result_dataframe,
    compute_final_metrics,
    confirm_intent_with_user,
    display_final_results_to_stdout,
    generate_and_validate_sql,
    generate_join_candidates,
    handle_direct_sql_reuse,
    handle_user_feedback,
    load_pipeline_resources,
    match_question_level_template_reuse,
    merge_structural_defaults_for_reuse,
    parse_intent_via_llm,
    prepare_union_match_join_phase,
    save_result_csv,
)
from ._templates import (
    TemplateStoreView,
    has_any_rejection_history_for_question,
    should_auto_accept_for_question,
)
from ._utils import flatten_param_values, normalize_question_via_llm, validate_question

FeedbackMode = Literal["live", "deferred_test"]


def deterministic_generate_validate_execute(
    *,
    q_norm: str,
    intent: RuntimeIntent,
    schema: Any,
    dialect: str | Dialect,
    store: dict[str, Any] | TemplateStoreView | None = None,
) -> tuple[SqlGenerationOutcome, list[tuple] | None]:
    """
    Build SQL from a fixed ``RuntimeIntent`` (join candidates + ``generate_and_validate_sql``), then execute.

    Skips NL intent parsing entirely. A join-choice LLM may still run when the graph yields ambiguous join candidates; callers that require zero LLM traffic should patch ``get_join_choice_from_llm`` in tests.

    Args:

        dialect: Engine name (for example ``"postgresql"``) or a concrete ``Dialect`` instance from the registry.
    """
    if store is None:
        store = {"next_id": 1, "templates": {}, "question_feedback": {}}
    dialect_obj = resolve_dialect(dialect)
    join_candidates, cmap, cte_hints = generate_join_candidates(intent, schema)
    gen_out = generate_and_validate_sql(
        q_norm,
        intent,
        schema,
        join_candidates,
        cmap,
        dialect_obj,
        store,
        cte_join_hints=cte_hints,
        matched_template=None,
        structural_match_templates=[],
    )
    if not gen_out.success or not gen_out.sql:
        return gen_out, None
    params = dict(flatten_param_values(intent))
    tmpl_sd = getattr(gen_out.matched_template, "structural_defaults", None) if gen_out.matched_template else None
    exec_sql = dialect_obj.finalize_render(
        intent.sql_param or "",
        params,
        schema=schema,
        intent=intent,
        execution_sql_override=None,
        structural_defaults=tmpl_sd,
    )
    rows = dialect_obj.execute(exec_sql)
    return gen_out, rows


@dataclass
class PendingFeedback:
    """
    Deferred feedback payload for post-assertion commit.

    Attributes:

        canned_reject_reason: Text supplied to ``input()`` when persisting a deferred ``n`` answer.
    """

    choice: str
    intent: RuntimeIntent
    sql: str
    schema: Any
    store: dict[str, Any]
    templates: dict[str, Any]
    rejected: dict[str, Any]
    q_norm: str
    generation_path: GenerationPath
    matched_template: Any
    matched_rejected_template: Any | None
    dialect: Any
    canned_reject_reason: str = ""
    structural_match_templates: tuple[Any, ...] = ()
    join_matches_template: bool | None = None


@dataclass
class Expected:
    """
    Optional checks for one run; `None` or defaults skip the corresponding assertion.

    When ``reuse_type`` is checked, values align with pipeline routing: ``direct_reuse`` (question match, ``GenerationPath`` 1–2), ``intent_direct_reuse`` (union, same columns, path 3), ``intent_reuse`` (union, columns changed, path 4).
    """

    tables: list[str] | None = None
    tables_one_of: list[list[str]] | None = None
    grain_in: tuple[str, ...] | None = None
    min_rows: int | None = None
    max_rows: int | None = None
    min_confidence: float | None = None
    reuse_type: str | tuple[str, ...] | None = None
    contains_join: bool | None = None
    contains_group_by: bool | None = None
    contains_cte: bool | None = None
    sql_contains: list[str] | None = None
    sql_contains_one_of: list[list[str]] | None = None
    sql_excludes: list[str] | None = None
    grain: str | tuple[str, ...] | None = None
    should_fail_validation: bool = False
    column_names_one_of: list[list[str]] | None = None
    row_value_check: Callable[[list[tuple]], bool] | None = None
    min_semantic_warnings: int | None = None
    status: str | None = None
    status_in: tuple[str, ...] | None = None
    generation_path: str | None = None
    generation_path_in: tuple[str, ...] | None = None
    max_llm_calls: int | None = None


@dataclass
class Scenario:
    """One NL question, expectations, canned prompts, and metadata for a live run."""

    id: str
    question: str
    expected: Expected
    category: str = ""
    auto_responses: list[str] | None = None
    feedback: str = "y"
    reject_reason: str = "incorrect results"
    sequence_id: str | None = None


@dataclass
class SequenceScenario:
    """Ordered scenarios sharing template state for stateful live tests."""

    id: str
    steps: list[Scenario]
    category: str = ""


@dataclass
class SoftFailure:
    """One recorded mismatch from a soft assertion."""

    field: str
    expected: Any
    actual: Any
    message: str


@dataclass
class StepResult:
    """Mutable capture of pipeline outputs, metrics, and logs for one scenario run."""

    scenario_id: str
    question: str
    status: str = "unknown"
    intent: RuntimeIntent | None = None
    sql: str | None = None
    rows: list[tuple] | None = None
    confidence: float | None = None
    reuse_type: str | None = None
    template_id: str | None = None
    validation_failed: bool = False
    feedback: str | None = None
    error: str | None = None
    duration_seconds: float = 0.0
    captured_logs: list[str] = field(default_factory=list)
    semantic_warnings: list[str] = field(default_factory=list)
    soft_warnings: list[str] = field(default_factory=list)
    llm_calls: int = 0
    reject_reason_actual: str | None = None
    classified_category: str | None = None
    classified_reason: str | None = None
    generation_path: str | None = None
    pending_feedback: PendingFeedback | None = None


class SoftAssert:
    """Collect soft assertion failures; call `report()` to raise one combined error."""

    def __init__(self) -> None:
        """
        Initialize an empty failure list.

        Returns:

            None.
        """
        self.failures: list[SoftFailure] = []

    def check(
        self,
        condition: bool,
        field_name: str,
        expected: Any,
        actual: Any,
        message: str = "",
    ) -> None:
        """
        Append a `SoftFailure` when `condition` is false.

        Args:

            condition: Assertion predicate.

            field_name: Field label for reporting.

            expected: Expected value for reporting.

            actual: Observed value for reporting.

            message: Optional explanation.

        Returns:

            None.
        """
        if not condition:
            msg = message or f"{field_name}: expected {expected!r}, got {actual!r}"
            self.failures.append(SoftFailure(field=field_name, expected=expected, actual=actual, message=msg))

    @property
    def passed(self) -> bool:
        """
        True if no failures were recorded.

        Returns:

            Whether the collector has no recorded failures.
        """
        return len(self.failures) == 0

    def report(self, header: str = "") -> None:
        """
        Raise `AssertionError` with all failures, or return if `passed`.

        Args:

            header: Optional first line of the error text.

        Returns:

            None.
        """
        if self.passed:
            return
        lines = [header] if header else []
        for f in self.failures:
            lines.append(f"  [{f.field}] {f.message}")
        raise AssertionError("\n".join(lines))


def _make_prompt_responders(
    responses: list[str],
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """
    Build FIFO auto-responders for ``ask_user_choice`` and ``interactive_yes_no`` sharing one queue.

    Args:

        responses: FIFO list of ``y`` / ``n`` strings.

    Returns:

        ``(ask_user_choice_replacement, interactive_yes_no_replacement)``.
    """

    queue = list(responses)

    def _ask_user_choice(prompt: str, options: list[str], silent_no: bool = False) -> str | None:
        if queue:
            return queue.pop(0)
        return "y"

    def _interactive_yes_no(
        stage: str,
        prompt: str,
        options: list[str],
        silent_no: bool = False,
        *,
        choice_port: Any = None,
    ) -> str | None:
        if choice_port is not None:
            if queue:
                return queue.pop(0)
            return "y"
        if queue:
            return queue.pop(0)
        return "y"

    return _ask_user_choice, _interactive_yes_no


def _make_input_responder(reject_reason: str = "incorrect results") -> Callable:
    """
    Build a replacement for `builtins.input` that supplies canned text.

    The first call returns `reject_reason`; later calls return `n` to stop further prompts.

    Args:

        reject_reason: Text returned on the first `input()` call.

    Returns:

        Callable replacing the built-in `input`.
    """
    call_count = {"n": 0}

    def _fake_input(prompt: str = "") -> str:
        """Return canned rejection text once, then `n`."""
        call_count["n"] += 1
        if call_count["n"] == 1:
            return reject_reason
        return "n"

    return _fake_input


@contextmanager
def _pipeline_capture(
    auto_responses: list[str],
    reject_reason: str = "incorrect results",
    csv_dir: str = "",
):
    """
    Patch interactive I/O for programmatic pipeline runs.

    Replaces ``ask_user_choice`` / ``interactive_yes_no`` on ``core_utils`` and ``interactive_yes_no`` on ``pipeline`` and ``main_execution`` with FIFO auto-responders (shared queue), and ``builtins.input`` with canned text so the pipeline does not block on stdin. When ``csv_dir`` is set, redirects ``save_result_csv`` so CSV output lands in that directory.

    Args:

        auto_responses: FIFO list of ``y`` / ``n`` strings for interactive prompts.

        reject_reason: Canned rejection reason for `input()` prompts.

        csv_dir: If non-empty, redirect `results.csv` writes into this directory.

    Yields:

        Dict with key ``logs`` listing captured log lines during the run.
    """
    capture: dict[str, Any] = {"logs": []}
    ask_uc, iyn = _make_prompt_responders(auto_responses)
    input_responder = _make_input_responder(reject_reason)

    original_log = aetherdialect._core_utils.log
    original_debug = aetherdialect._core_utils.debug

    def _capturing_log(msg: str) -> None:
        """Append a log line to capture and forward to the original logger."""
        capture["logs"].append(f"[LOG] {msg}")
        original_log(msg)

    def _capturing_debug(msg: str) -> None:
        """Append a debug line to capture and forward to the original `debug`."""
        capture["logs"].append(f"[DEBUG] {msg}")
        original_debug(msg)

    _debug_modules = [
        aetherdialect._core_utils,
        aetherdialect._pipeline,
        aetherdialect._sql_gen,
        aetherdialect._validation_agg,
        aetherdialect._validation_execute,
        aetherdialect._validation_schema,
        aetherdialect._validation_semantic,
        aetherdialect._intent_expr,
        aetherdialect._intent_process,
        aetherdialect._intent_repair,
        aetherdialect._intent_resolve,
        aetherdialect._dialect,
        aetherdialect._expansion_ops,
        aetherdialect._utils,
        aetherdialect._templates,
        aetherdialect._schema,
        aetherdialect._schema_profiling,
        aetherdialect._qsim,
        aetherdialect._qsim_ops,
        aetherdialect._main_execution,
        aetherdialect._seed_warmup,
    ]
    _log_modules = [
        aetherdialect._core_utils,
        aetherdialect._pipeline,
        aetherdialect._main_execution,
        aetherdialect._expansion_ops,
        aetherdialect._seed_warmup,
    ]

    extra_patches: list[Any] = []
    for mod in _debug_modules:
        if hasattr(mod, "debug"):
            extra_patches.append(patch.object(mod, "debug", _capturing_debug))
    for mod in _log_modules:
        if hasattr(mod, "log"):
            extra_patches.append(patch.object(mod, "log", _capturing_log))

    _pt_orig = aetherdialect._core_utils.pipeline_trace

    def _capturing_pipeline_trace(heading: str, body: str) -> None:
        capture["logs"].append(f"[PIPELINE_TRACE] {heading}\n{body}")
        _pt_orig(heading, body)

    _ptl_orig = aetherdialect._core_utils.pipeline_trace_lazy

    def _capturing_pipeline_trace_lazy(heading: str, body_factory: Callable[[], str]) -> None:
        body = body_factory()
        capture["logs"].append(f"[PIPELINE_TRACE] {heading}\n{body}")
        _ptl_orig(heading, lambda: body)

    _trace_patch_modules = (
        aetherdialect._core_utils,
        aetherdialect._pipeline,
        aetherdialect._intent_process,
        aetherdialect._sql_gen,
        aetherdialect._validation_execute,
        aetherdialect._intent_resolve,
        aetherdialect._dialect,
        aetherdialect._intent_repair,
    )
    for mod in _trace_patch_modules:
        if hasattr(mod, "pipeline_trace"):
            extra_patches.append(patch.object(mod, "pipeline_trace", _capturing_pipeline_trace))
        if hasattr(mod, "pipeline_trace_lazy"):
            extra_patches.append(patch.object(mod, "pipeline_trace_lazy", _capturing_pipeline_trace_lazy))

    if csv_dir:
        _original_save = aetherdialect._pipeline.save_result_csv

        def _redirected_save(df: Any) -> None:
            orig_cwd = os.getcwd()
            try:
                os.chdir(csv_dir)
                _original_save(df)
            finally:
                os.chdir(orig_cwd)

        extra_patches.append(patch.object(aetherdialect._pipeline, "save_result_csv", _redirected_save))
        extra_patches.append(patch(__name__ + ".save_result_csv", _redirected_save))

    with (
        patch.object(aetherdialect._core_utils, "ask_user_choice", ask_uc),
        patch.object(aetherdialect._core_utils, "interactive_yes_no", iyn),
        patch.object(aetherdialect._pipeline, "interactive_yes_no", iyn),
        patch.object(aetherdialect._main_execution, "interactive_yes_no", iyn),
        patch("builtins.input", input_responder),
    ):
        for p in extra_patches:
            p.start()
        try:
            yield capture
        finally:
            for p in extra_patches:
                p.stop()


def _extract_reuse_sql(tmpl: Any, q_norm: str) -> str:
    """
    Reconstruct the final SQL that `handle_direct_sql_reuse` would produce.

    Args:

        tmpl: Template with `value_history` and SQL fields.

        q_norm: Normalized question string.

    Returns:

        Substituted parameterized SQL when params match; otherwise ``tmpl.sql_param``.
    """
    vh = tmpl.value_history
    matched_params: dict[str, str] = {}
    for i, hq in enumerate(vh.questions):
        if hq and q_norm == hq:
            matched_params = dict(vh.param_values[i])
            break
    if matched_params:
        merge_structural_defaults_for_reuse(
            tmpl.sql_param,
            matched_params,
            getattr(tmpl, "structural_defaults", None),
        )
        return substitute_params(tmpl.sql_param, matched_params)
    return tmpl.sql_param


def _build_reuse_intent(tmpl: Any) -> RuntimeIntent:
    """
    Build a lightweight `RuntimeIntent` from a template's intent signature.

    Args:

        tmpl: Template carrying `intent_signature`.

    Returns:

        Populated `RuntimeIntent` for direct-reuse display and checks.
    """
    sig = tmpl.intent_signature
    return RuntimeIntent(
        tables=sig.tables or [],
        grain=sig.grain or "row_level",
        select_cols=sig.select_cols or [],
        group_by_cols=sig.group_by_cols or [],
        order_by_cols=sig.order_by_cols or [],
        filters_param=sig.filters_param or [],
        having_param=getattr(sig, "having_param", None) or [],
        column_map=getattr(sig, "column_map", None) or {},
        natural_language="",
        chosen_join_candidate_id=getattr(sig, "chosen_join_candidate_id", None) or "",
        chosen_join_path_signature=list(getattr(sig, "chosen_join_path_signature", None) or []),
    )


def _scenario_requires_intent_prompt(scenario: Scenario) -> bool:
    """
    Return True when the scenario expects an intent decline on the first canned response.

    Returns:

        True when ``expected.status`` is ``intent_rejected`` and the first auto-response is ``n``.
    """

    if scenario.expected.status != "intent_rejected":
        return False
    ar = scenario.auto_responses
    if not ar:
        return False
    return str(ar[0]).strip().lower() == "n"


def _run_pipeline_core(
    question: str,
    schema: Any,
    store: dict[str, Any],
    templates: dict,
    rejected: dict,
    schema_terms: set[str],
    feedback: str,
    captured_logs: list[str],
    reject_reason: str = "",
    feedback_mode: FeedbackMode = "live",
    force_intent_confirm: bool = False,
) -> StepResult:
    """
    Execute pipeline steps for one question and return captured state.

    Mirrors `interactive_run_once` control flow with programmatic arguments and a `StepResult` instead of printing.

    Args:

        question: Natural-language question string.

        schema: Loaded `SchemaGraph`.

        store: Mutable template store dict.

        templates: Accepted templates dict.

        rejected: Rejected templates dict.

        schema_terms: Set of schema term tokens.

        feedback: Predetermined `y` / `n` feedback value.

        captured_logs: Mutable list to append log lines into.

        reject_reason: When feedback is `n`, canned reason recorded on ``StepResult``.

        feedback_mode: ``live`` applies feedback immediately; ``deferred_test`` records ``PendingFeedback``.

        force_intent_confirm: When True, intent confirmation is not skipped by similarity or empty warnings.

    Returns:

        Populated `StepResult`.
    """
    result = StepResult(scenario_id="", question=question, captured_logs=captured_logs)

    dialect, schema, store, templates, rejected, schema_terms = load_pipeline_resources(
        schema, store, templates, rejected, schema_terms
    )

    raw_question = question

    tmpl_pre = match_question_level_template_reuse(raw_question, templates, template_store=store)
    result.reuse_type = tmpl_pre.reuse_type
    if tmpl_pre.reuse_type == "direct_reuse":
        result.template_id = tmpl_pre.best_template.id if tmpl_pre.best_template else None
        assert tmpl_pre.reuse_candidate_normalized is not None
        reuse_pre = handle_direct_sql_reuse(
            tmpl_pre.reuse_candidate_normalized,
            tmpl_pre.best_template,
            dialect,
            store,
            templates,
            rejected,
            schema,
            existing_nl=None,
            reuse_history_index=tmpl_pre.reuse_history_index,
            form_storage=QuestionFormStorage(corrected=raw_question.strip()),
        )
        if reuse_pre is not None:
            result.generation_path = reuse_pre.generation_path.code if reuse_pre.generation_path is not None else None
        if reuse_pre is not None and reuse_pre.success:
            result.status = "ok"
            result.sql = _extract_reuse_sql(tmpl_pre.best_template, tmpl_pre.reuse_candidate_normalized)
            result.intent = _build_reuse_intent(tmpl_pre.best_template)
            return result

    valid, query_type, corrected = validate_question(raw_question)
    if not valid:
        result.status = "restricted" if query_type == "restricted" else "invalid_question"
        return result

    corrected_text = corrected

    tmpl_typo = match_question_level_template_reuse(corrected_text, templates, template_store=store)
    result.reuse_type = tmpl_typo.reuse_type

    if tmpl_typo.reuse_type == "direct_reuse":
        result.reuse_type = "direct_reuse"
        result.template_id = tmpl_typo.best_template.id if tmpl_typo.best_template else None
        assert tmpl_typo.reuse_candidate_normalized is not None
        reuse_out = handle_direct_sql_reuse(
            tmpl_typo.reuse_candidate_normalized,
            tmpl_typo.best_template,
            dialect,
            store,
            templates,
            rejected,
            schema,
            existing_nl=None,
            reuse_history_index=tmpl_typo.reuse_history_index,
            form_storage=QuestionFormStorage(corrected=corrected_text),
        )
        if reuse_out is not None:
            result.generation_path = reuse_out.generation_path.code if reuse_out.generation_path is not None else None
        if reuse_out is not None and reuse_out.success:
            result.status = "ok"
            result.sql = _extract_reuse_sql(tmpl_typo.best_template, tmpl_typo.reuse_candidate_normalized)
            result.intent = _build_reuse_intent(tmpl_typo.best_template)
            return result

    neg_drop = False
    normalized_canonical = normalize_question_via_llm(corrected_text, raw_original=raw_question)
    if (
        normalized_canonical != corrected_text
        and has_any_rejection_history_for_question(store, corrected_text)
    ):
        neg_drop = True
        normalized_canonical = corrected_text

    tmpl_norm = None
    if normalized_canonical != corrected_text:
        tmpl_norm = match_question_level_template_reuse(normalized_canonical, templates, template_store=store)
        if tmpl_norm.reuse_type == "direct_reuse":
            result.reuse_type = "direct_reuse"
            result.template_id = tmpl_norm.best_template.id if tmpl_norm.best_template else None
            assert tmpl_norm.reuse_candidate_normalized is not None
            reuse_n = handle_direct_sql_reuse(
                tmpl_norm.reuse_candidate_normalized,
                tmpl_norm.best_template,
                dialect,
                store,
                templates,
                rejected,
                schema,
                existing_nl=None,
                reuse_history_index=tmpl_norm.reuse_history_index,
                form_storage=QuestionFormStorage(
                    corrected=corrected_text,
                    normalized_optional=normalized_canonical,
                    normalized_negative_memory_dropped=neg_drop,
                    accept_via_normalized_lookup_only=True,
                ),
            )
            if reuse_n is not None:
                result.generation_path = reuse_n.generation_path.code if reuse_n.generation_path is not None else None
            if reuse_n is not None and reuse_n.success:
                result.status = "ok"
                result.sql = _extract_reuse_sql(tmpl_norm.best_template, tmpl_norm.reuse_candidate_normalized)
                result.intent = _build_reuse_intent(tmpl_norm.best_template)
                return result

    norm_opt = normalized_canonical if normalized_canonical != corrected_text else None
    form_storage_turn = QuestionFormStorage(
        corrected=corrected_text,
        normalized_optional=norm_opt,
        normalized_negative_memory_dropped=neg_drop,
        accept_via_normalized_lookup_only=False,
    )

    q_norm = normalize_question(corrected_text)

    parsed_intent, semantic_warnings, llm_calls = parse_intent_via_llm(
        corrected_text,
        schema,
        templates,
        store,
    )
    result.llm_calls = llm_calls
    if parsed_intent is None:
        result.status = "intent_parse_failed"
        return result
    intent = parsed_intent
    debug(f"[live_testing] intent parsed: tables={intent.tables} grain={intent.grain} llm_calls={llm_calls}")
    if llm_calls > 2:
        debug(f"[live_testing] WARNING: intent parse required {llm_calls} LLM calls for: {q_norm}")

    result.intent = intent

    result.semantic_warnings = [w.get("message", "") if isinstance(w, dict) else str(w) for w in semantic_warnings]

    union_result = match_template_for_union(intent, templates)
    structural_match_templates = collect_structural_match_templates(intent, templates)
    matched_template = None
    union_select_cols = None
    cols_changed = False
    union_sql_path: GenerationPath | None = None
    has_union_match = union_result is not None
    if union_result is not None:
        matched_template, union_select_cols, cols_changed, union_sql_path = union_result
        debug(
            f"[live_testing] union match: template={matched_template.id} "
            f"cols_changed={cols_changed} path={union_sql_path.code}"
        )

    intent_sim = best_accepted_template_similarity(intent, templates)
    if not confirm_intent_with_user(
        intent,
        store,
        semantic_warnings,
        similarity_score=intent_sim,
        has_union_match=has_union_match,
        cols_changed=cols_changed,
        rejected=rejected,
        q_norm=q_norm,
        schema=schema,
        force_intent_confirm=force_intent_confirm,
    ):
        result.status = "intent_rejected"
        return result

    (
        matched_template,
        union_select_cols,
        cols_changed,
        union_sql_path,
        has_union_match,
        join_candidates,
        cmap,
        cte_join_hints,
    ) = prepare_union_match_join_phase(q_norm, intent, schema, dialect, templates, store=store)

    if has_union_match:
        result.reuse_type = "intent_reuse" if cols_changed else "intent_direct_reuse"
        debug(
            f"[live_testing] union match: template={matched_template.id if matched_template else None} cols_changed={cols_changed}"
        )

    matched_rejected_template = None

    gen_out = generate_and_validate_sql(
        q_norm,
        intent,
        schema,
        join_candidates,
        cmap,
        dialect,
        store,
        cte_join_hints=cte_join_hints,
        matched_template=matched_template,
        union_select_cols=union_select_cols,
        cols_changed=cols_changed,
        structural_match_templates=structural_match_templates,
        union_sql_path=union_sql_path,
    )
    result.sql = gen_out.sql
    result.generation_path = gen_out.generation_path.code
    if not gen_out.success:
        debug(f"[live_testing] SQL validation failed for: {q_norm}")
        result.status = "validation_failed"
        result.validation_failed = True
        return result

    sql = gen_out.sql

    tmpl_sd = getattr(gen_out.matched_template, "structural_defaults", None) if gen_out.matched_template else None

    exec_sql = dialect.finalize_render(
        intent.sql_param or "",
        dict(flatten_param_values(intent)),
        schema=schema,
        intent=intent,
        execution_sql_override=None,
        structural_defaults=tmpl_sd,
    )
    rows = dialect.execute(exec_sql)
    result.sql = sql
    result.rows = rows
    debug(f"[live_testing] SQL generated ({result.generation_path}): rows={len(rows) if rows else 0}")
    if result.generation_path == GenerationPath.FRESH.code:
        debug(
            f"[live_testing.path5_trace] q_norm={q_norm!r} sql_param={(intent.sql_param or '')!r} substituted={sql!r}"
        )

    conf = compute_final_metrics(
        sql,
        intent,
        schema,
        templates,
        join_candidates,
        store,
        q_norm=q_norm,
        explain_soft_diagnostics=getattr(gen_out, "explain_soft_diagnostics", 0),
    )
    result.confidence = conf

    display_final_results_to_stdout(
        q_norm,
        intent,
        sql,
        rows,
        structural_defaults=tmpl_sd,
        template_display_alias_map=(
            getattr(gen_out.matched_template, "display_alias_map", None)
            if gen_out.matched_template
            else None
        ),
    )

    need_sql_feedback = (
        has_any_rejection_history_for_question(store, corrected_text)
        or (
            gen_out.matched_template is not None
            and not should_auto_accept_for_question(gen_out.matched_template, q_norm)
        )
        or conf < PolicyConfig.FINAL_SQL_AUTO_ACCEPT_THRESHOLD
    )
    if need_sql_feedback:
        effective_feedback = feedback
    else:
        effective_feedback = "y"

    result.feedback = effective_feedback

    if effective_feedback == "y" and intent.grain != "scalar":
        df_out = build_result_dataframe(
            rows,
            intent,
            sql,
            structural_defaults=tmpl_sd,
            q_norm=q_norm,
            template_display_alias_map=(
                getattr(gen_out.matched_template, "display_alias_map", None)
                if gen_out.matched_template
                else None
            ),
        )
        if df_out is not None:
            save_result_csv(df_out)

    if feedback_mode == "live":
        reject_info = handle_user_feedback(
            effective_feedback,
            intent,
            sql,
            schema,
            store,
            templates,
            rejected,
            q_norm,
            gen_out.generation_path,
            gen_out.matched_template,
            matched_rejected_template,
            dialect=dialect,
            structural_match_templates=gen_out.structural_match_templates,
            join_matches_template=gen_out.join_matches_template,
            form_storage=form_storage_turn,
        )
        if effective_feedback == "n" and reject_info:
            result.reject_reason_actual = reject_reason or reject_info.get("reject_reason")
            result.classified_category = reject_info.get("category")
            result.classified_reason = reject_info.get("normalized_reason")
    else:
        result.pending_feedback = PendingFeedback(
            choice=effective_feedback,
            intent=intent,
            sql=sql,
            schema=schema,
            store=store,
            templates=templates,
            rejected=rejected,
            q_norm=q_norm,
            generation_path=gen_out.generation_path,
            matched_template=gen_out.matched_template,
            matched_rejected_template=matched_rejected_template,
            dialect=dialect,
            canned_reject_reason=reject_reason,
            structural_match_templates=gen_out.structural_match_templates,
            join_matches_template=gen_out.join_matches_template,
        )

    if effective_feedback == "n":
        if feedback_mode == "deferred_test":
            result.status = "ok"
        else:
            result.status = "intent_rejected"
    else:
        result.status = "ok"
    return result


class LiveTestRunner:
    """
    Orchestrate single-scenario and sequence-scenario execution against the live pipeline.

    Holds pre-loaded resources; `run` / `run_deferred` wrap the pipeline in a capture context and return `StepResult` values for assertions (multi-step flows use `run_sequence_and_assert`).
    """

    def __init__(
        self,
        schema: Any,
        store: dict[str, Any],
        templates: dict,
        rejected: dict,
        schema_terms: set[str],
        csv_dir: str = "",
    ) -> None:
        """
        Initialize runner state from loaded pipeline resources.

        Args:

            schema: Profiled `SchemaGraph`.

            store: Mutable template store dict.

            templates: Accepted templates dict.

            rejected: Rejected templates dict.

            schema_terms: Set of schema term tokens.

            csv_dir: Directory for `results.csv` output; empty uses the current working directory.

        Returns:

            None.
        """
        self.schema = schema
        self.store = store
        self.templates = templates
        self.rejected = rejected
        self.schema_terms = schema_terms
        self.csv_dir = csv_dir

    def run(self, scenario: Scenario, retries: int = 0) -> StepResult:
        """
        Execute a single scenario against the live pipeline.

        Args:

            scenario: The `Scenario` to execute.

            retries: Additional attempts on failure; `0` means a single try.

        Returns:

            `StepResult` from the last attempt.
        """
        auto = scenario.auto_responses if scenario.auto_responses is not None else ["y", "y", "y"]
        last_result: StepResult | None = None

        for _ in range(1 + retries):
            t0 = time.monotonic()
            try:
                with _pipeline_capture(list(auto), scenario.reject_reason, csv_dir=self.csv_dir) as cap:
                    step = _run_pipeline_core(
                        question=scenario.question,
                        schema=self.schema,
                        store=self.store,
                        templates=self.templates,
                        rejected=self.rejected,
                        schema_terms=self.schema_terms,
                        feedback=scenario.feedback,
                        captured_logs=cap["logs"],
                        reject_reason=scenario.reject_reason,
                        feedback_mode="live",
                        force_intent_confirm=_scenario_requires_intent_prompt(scenario),
                    )
            except Exception:
                step = StepResult(
                    scenario_id=scenario.id,
                    question=scenario.question,
                    status="error",
                    error=traceback.format_exc(),
                    captured_logs=cap.get("logs", []) if "cap" in dir() else [],
                )

            step.scenario_id = scenario.id
            step.duration_seconds = time.monotonic() - t0
            last_result = step

            if step.status == "ok":
                break

        return last_result

    def run_deferred(self, scenario: Scenario, retries: int = 0) -> StepResult:
        """Execute one scenario while deferring feedback persistence."""
        auto = scenario.auto_responses if scenario.auto_responses is not None else ["y", "y", "y"]
        last_result: StepResult | None = None
        for _ in range(1 + retries):
            t0 = time.monotonic()
            try:
                with _pipeline_capture(list(auto), scenario.reject_reason, csv_dir=self.csv_dir) as cap:
                    step = _run_pipeline_core(
                        question=scenario.question,
                        schema=self.schema,
                        store=self.store,
                        templates=self.templates,
                        rejected=self.rejected,
                        schema_terms=self.schema_terms,
                        feedback=scenario.feedback,
                        captured_logs=cap["logs"],
                        reject_reason=scenario.reject_reason,
                        feedback_mode="deferred_test",
                        force_intent_confirm=_scenario_requires_intent_prompt(scenario),
                    )
            except Exception:
                step = StepResult(
                    scenario_id=scenario.id,
                    question=scenario.question,
                    status="error",
                    error=traceback.format_exc(),
                    captured_logs=cap.get("logs", []) if "cap" in dir() else [],
                )
            step.scenario_id = scenario.id
            step.duration_seconds = time.monotonic() - t0
            last_result = step
            if step.status == "ok":
                break
        return last_result

    def clone(self) -> LiveTestRunner:
        """Return an isolated runner with deep-copied mutable state."""
        return LiveTestRunner(
            schema=self.schema,
            store=deepcopy(self.store),
            templates=deepcopy(self.templates),
            rejected=deepcopy(self.rejected),
            schema_terms=set(self.schema_terms),
            csv_dir=self.csv_dir,
        )

    def adopt_state_from(self, other: LiveTestRunner) -> None:
        """Replace mutable state with another runner's state."""
        self.store = other.store
        self.templates = other.templates
        self.rejected = other.rejected
        self.schema_terms = other.schema_terms


def commit_pending_feedback(
    result: StepResult,
) -> None:
    """
    Persist deferred accept/reject feedback and clear ``pending_feedback`` on *result*.

    When the pending choice is ``n``, ``builtins.input`` is patched so classification receives ``canned_reject_reason`` outside the original pipeline capture context.

    Args:

        result: Step output that may hold ``pending_feedback`` from a deferred run.

    Returns:

        None.
    """
    pending = result.pending_feedback
    if pending is None:
        return

    def _invoke_feedback() -> dict[str, str] | None:
        return handle_user_feedback(
            pending.choice,
            pending.intent,
            pending.sql,
            pending.schema,
            pending.store,
            pending.templates,
            pending.rejected,
            pending.q_norm,
            pending.generation_path,
            pending.matched_template,
            pending.matched_rejected_template,
            dialect=pending.dialect,
            structural_match_templates=pending.structural_match_templates,
            join_matches_template=pending.join_matches_template,
        )

    if pending.choice == "n":
        reason = pending.canned_reject_reason or "incorrect results"

        def _stdin_reject(_prompt: str = "") -> str:
            return reason

        with patch("builtins.input", _stdin_reject):
            reject_info = _invoke_feedback()
    else:
        reject_info = _invoke_feedback()
    if pending.choice == "n" and reject_info:
        result.reject_reason_actual = reject_info.get("reject_reason")
        result.classified_category = reject_info.get("category")
        result.classified_reason = reject_info.get("normalized_reason")
    result.pending_feedback = None


def _live_sql_has_join_clause(sql: str) -> bool:
    """
    Return whether *sql* uses an explicit JOIN or a comma-separated multi-relation FROM in the outer SELECT.

    Implemented via the dialect-agnostic AST helper :func:`aetherdialect._dialect.sql_outer_has_join_or_comma_from`.
    """

    return sql_outer_has_join_or_comma_from(sql, sqlglot_dialect=active_sqlglot_dialect())


def _step_result_indicates_join(result: StepResult) -> bool:
    """Return True when rendered SQL or resolved intent shows a multi-table join."""

    if result.sql and _live_sql_has_join_clause(result.sql):
        return True
    it = result.intent
    if it is None:
        return False
    if it.chosen_join_path_signature:
        return True
    for step in it.cte_steps or []:
        if step.chosen_join_path_signature:
            return True
    if len(it.tables or []) >= 2:
        return True
    for step in it.cte_steps or []:
        if len(step.tables or []) >= 2:
            return True
    return False


def _internal_failure_log_offenders(captured_logs: list[str]) -> list[str]:
    """Return log lines that must not appear on a run whose final status is ``ok``."""

    offenders: list[str] = []
    for line in captured_logs or []:
        if "enforce_select_only FAILED" in line:
            offenders.append(line.strip())
        elif "phase-G post-processing revalidation failed" in line:
            offenders.append(line.strip())
        elif re.search(r"semantic errors:\s*[1-9]\d*", line, re.I):
            offenders.append(line.strip())
    return offenders


def _normalize_sql_for_match(sql: str) -> str:
    """Uppercase SQL with bracket/backtick/quote characters stripped for substring checks."""

    s = (sql or "").upper()
    s = re.sub(r"[`\"\[\]]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _assert_scenario(result: StepResult, expected: Expected, soft: SoftAssert | None = None) -> SoftAssert:
    """
    Evaluate a `StepResult` against an `Expected` specification.

    When `soft` is `None`, a new `SoftAssert` is created. Applicable checks run and failures accumulate on the returned instance.

    Args:

        result: The `StepResult` from a pipeline run.

        expected: The `Expected` to assert against.

        soft: Optional existing `SoftAssert` to append to.

    Returns:

        The `SoftAssert` instance (same object when one was passed in).
    """
    if soft is None:
        soft = SoftAssert()

    eff_status = expected.status
    eff_status_in = expected.status_in
    if (
        eff_status is None
        and eff_status_in is None
        and (
            expected.min_rows is not None
            or expected.sql_contains is not None
            or expected.column_names_one_of is not None
        )
    ):
        eff_status = "ok"

    allows_error = eff_status == "error" or (eff_status_in is not None and "error" in eff_status_in)
    if result.status == "error" and not allows_error:
        err_preview = (result.error or "unknown error").strip()
        if len(err_preview) > 4000:
            err_preview = f"{err_preview[:4000]}\n... (truncated)"
        soft.check(
            False,
            "pipeline_error",
            "ok",
            result.status,
            message=f"uncaught exception:\n{err_preview}",
        )

    if eff_status is not None:
        soft.check(
            result.status == eff_status,
            "status",
            eff_status,
            result.status,
        )
    elif eff_status_in is not None:
        soft.check(
            result.status in eff_status_in,
            "status_in",
            eff_status_in,
            result.status,
        )

    if result.intent is not None:
        actual_tables = sorted(result.intent.tables or [])
        if expected.tables_one_of is not None:
            allowed = [sorted(t) for t in expected.tables_one_of]
            soft.check(
                actual_tables in allowed,
                "tables",
                expected.tables_one_of,
                actual_tables,
            )
        elif expected.tables is not None:
            expected_tables = sorted(expected.tables)
            soft.check(
                actual_tables == expected_tables,
                "tables",
                expected_tables,
                actual_tables,
            )

    if expected.grain is not None and result.intent is not None:
        if isinstance(expected.grain, tuple):
            soft.check(
                result.intent.grain in expected.grain,
                "grain",
                expected.grain,
                result.intent.grain,
            )
        else:
            soft.check(
                result.intent.grain == expected.grain,
                "grain",
                expected.grain,
                result.intent.grain,
            )
    elif expected.grain_in is not None and result.intent is not None:
        soft.check(
            result.intent.grain in expected.grain_in,
            "grain",
            expected.grain_in,
            result.intent.grain,
        )

    if expected.reuse_type is not None:
        if isinstance(expected.reuse_type, tuple):
            soft.check(
                result.reuse_type in expected.reuse_type,
                "reuse_type",
                expected.reuse_type,
                result.reuse_type,
            )
        else:
            soft.check(
                result.reuse_type == expected.reuse_type,
                "reuse_type",
                expected.reuse_type,
                result.reuse_type,
            )

    if expected.generation_path is not None:
        soft.check(
            result.generation_path == expected.generation_path,
            "generation_path",
            expected.generation_path,
            result.generation_path,
        )
    elif expected.generation_path_in is not None:
        soft.check(
            result.generation_path in expected.generation_path_in,
            "generation_path_in",
            expected.generation_path_in,
            result.generation_path,
        )

    sql_upper = (result.sql or "").upper()
    sql_norm = _normalize_sql_for_match(result.sql or "")

    if expected.contains_join is not None:
        has_join = _step_result_indicates_join(result)
        soft.check(
            has_join == expected.contains_join,
            "contains_join",
            expected.contains_join,
            has_join,
        )

    if expected.contains_group_by is not None:
        has_gb = "GROUP BY" in sql_upper
        soft.check(
            has_gb == expected.contains_group_by,
            "contains_group_by",
            expected.contains_group_by,
            has_gb,
        )

    if expected.contains_cte is not None:
        has_cte = sql_upper.lstrip().startswith("WITH ")
        soft.check(
            has_cte == expected.contains_cte,
            "contains_cte",
            expected.contains_cte,
            has_cte,
        )

    if expected.sql_contains is not None and result.sql is not None:
        for substr in expected.sql_contains:
            found = _normalize_sql_for_match(substr) in sql_norm
            soft.check(found, "sql_contains", substr, f"not found in: {result.sql[:120]}")

    if expected.sql_contains_one_of is not None and result.sql is not None:
        ok_any = any(
            all(_normalize_sql_for_match(substr) in sql_norm for substr in group)
            for group in expected.sql_contains_one_of
        )
        soft.check(
            ok_any,
            "sql_contains_one_of",
            expected.sql_contains_one_of,
            f"not found in: {result.sql[:120]}",
        )

    if expected.sql_excludes is not None and result.sql is not None:
        for substr in expected.sql_excludes:
            found = substr.upper() in sql_upper
            soft.check(
                not found,
                "sql_excludes",
                f"absent: {substr}",
                f"found in: {result.sql[:120]}",
            )

    if expected.min_rows is not None and result.rows is not None:
        soft.check(
            len(result.rows) >= expected.min_rows,
            "min_rows",
            expected.min_rows,
            len(result.rows),
        )

    if expected.max_rows is not None and result.rows is not None:
        soft.check(
            len(result.rows) <= expected.max_rows,
            "max_rows",
            expected.max_rows,
            len(result.rows),
        )

    if expected.min_confidence is not None and result.confidence is not None:
        soft.check(
            result.confidence >= expected.min_confidence,
            "min_confidence",
            expected.min_confidence,
            result.confidence,
        )

    if expected.column_names_one_of is not None and result.rows is not None and result.intent is not None:
        actual_cols = []
        for c in result.intent.select_cols or []:
            name = getattr(c, "alias", None) or c.expr.primary_term
            actual_cols.append(name.split(".")[-1] if name and "." in name else (name or ""))
        allowed = [sorted(cols) for cols in expected.column_names_one_of]
        soft.check(
            sorted(actual_cols) in allowed,
            "column_names",
            expected.column_names_one_of,
            actual_cols,
        )

    if expected.row_value_check is not None and result.rows is not None:
        check_ok = expected.row_value_check(result.rows)
        soft.check(check_ok, "row_value_check", "True", check_ok)

    if expected.min_semantic_warnings is not None:
        soft.check(
            len(result.semantic_warnings) >= expected.min_semantic_warnings,
            "min_semantic_warnings",
            expected.min_semantic_warnings,
            len(result.semantic_warnings),
        )

    if expected.should_fail_validation:
        soft.check(
            result.validation_failed,
            "should_fail_validation",
            True,
            result.validation_failed,
        )

    if expected.max_llm_calls is not None:
        soft.check(
            result.llm_calls <= expected.max_llm_calls,
            "max_llm_calls",
            f"<={expected.max_llm_calls}",
            result.llm_calls,
        )

    if result.status == "ok":
        bad_lines = _internal_failure_log_offenders(result.captured_logs)
        if bad_lines:
            preview = "\n".join(bad_lines[:20])
            soft.check(
                False,
                "internal_failure_logs",
                "no internal failure tokens when status is ok",
                bad_lines,
                message=f"logged failures on accepted attempt:\n{preview}",
            )

    return soft


def run_and_assert(
    runner: LiveTestRunner,
    scenario: Scenario,
    header: str,
    max_attempts: int = 2,
    retries: int = 1,
) -> None:
    """
    Run a scenario and assert expectations, retrying from scratch on failure.

    On the first attempt the pipeline runs and assertions are checked. When any assertion fails and `max_attempts` > 1, the pipeline is re- run from scratch and assertions are re-evaluated.

    Args:

        runner: `LiveTestRunner` configured for the target database.

        scenario: The `Scenario` to execute.

        header: Label used in the `AssertionError` message.

        max_attempts: Total attempts including the initial run.

        retries: Per-attempt pipeline retry count passed to `runner.run`.

    Returns:

        None.
    """
    last_soft: SoftAssert | None = None
    for _ in range(max_attempts):
        attempt_runner = runner.clone()
        result = attempt_runner.run_deferred(scenario, retries=retries)
        last_soft = _assert_scenario(result, scenario.expected)
        if last_soft.passed:
            commit_pending_feedback(result)
            runner.adopt_state_from(attempt_runner)
            return
    if last_soft is not None:
        last_soft.report(header=header)


def run_sequence_and_assert(
    runner: LiveTestRunner,
    seq: SequenceScenario,
    max_attempts: int = 2,
    retries: int = 1,
) -> None:
    """
    Run a sequence of scenarios and assert each step, retrying on failure.

    When any step's assertions fail and `max_attempts` > 1, the entire sequence is re-executed from scratch.

    Args:

        runner: `LiveTestRunner` configured for the target database.

        seq: The `SequenceScenario` whose steps run in order.

        max_attempts: Total attempts including the initial run.

        retries: Per-step pipeline retry count passed to `runner.run`.

    Returns:

        None.
    """
    last_soft: SoftAssert | None = None
    for _ in range(max_attempts):
        attempt_runner = runner.clone()
        last_soft = SoftAssert()
        for step_scenario in seq.steps:
            step_with_seq = replace(step_scenario, sequence_id=seq.id)
            result = attempt_runner.run_deferred(step_with_seq, retries=retries)
            _assert_scenario(result, step_scenario.expected, soft=last_soft)
            if not last_soft.passed:
                break
            commit_pending_feedback(result)
        if last_soft.passed:
            runner.adopt_state_from(attempt_runner)
            return
    if last_soft is not None:
        last_soft.report(header=f"[{seq.id}]")


def run_seeded_schema_semantic_repair(
    question: str,
    seeded_intent: RuntimeIntent,
    schema_graph: Any,
    *,
    max_retries: int | None = None,
) -> tuple[RuntimeIntent | None, list[str], int]:
    """
    Run schema and semantic repair from a pre-built intent (live and harness entrypoint).

    Forwards the seed intent to :func:`aetherdialect._intent_process._run_schema_semantic_repair_loop` with
    no template store and no in-turn summary rows.
    """

    mr = PolicyConfig.MAX_STAGE_B_REPAIRS if max_retries is None else max_retries
    table_list = sorted(schema_graph.tables.keys())
    schema_literal_json = schema_graph.schema_literal_json
    system, _ = aetherdialect._intent_process._build_intent_parse_prompt(
        question,
        schema_literal_json,
        table_list,
    )
    return aetherdialect._intent_process._run_schema_semantic_repair_loop(
        intent=seeded_intent,
        question=question,
        system=system,
        schema_graph=schema_graph,
        schema_literal_json=schema_literal_json,
        table_list=table_list,
        max_retries=mr,
        llm_calls=0,
    )[:3]
