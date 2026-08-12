"""Live pipeline tests: scenarios, soft asserts, I/O patching, and runners against real LLM/DB. Absolute ``import aetherdialect._*`` names exist only so ``unittest.mock.patch`` can target stable module paths; production code uses relative imports. Fixtures stay caller-specific."""

from __future__ import annotations

import re
import time
import traceback
from dataclasses import replace
from typing import Any
from unittest.mock import patch

import aetherdialect._dialect
import aetherdialect._expansion_ops
import aetherdialect._intent_bind
import aetherdialect._intent_expr
import aetherdialect._intent_loop
import aetherdialect._intent_normalize
import aetherdialect._main_execution
import aetherdialect._pipeline_execute
import aetherdialect._pipeline_generate
import aetherdialect._qsim
import aetherdialect._schema_finalize
import aetherdialect._schema_graph
import aetherdialect._schema_profile
import aetherdialect._schema_reflect
import aetherdialect._seed_warmup
import aetherdialect._sql_gen
import aetherdialect._templates
import aetherdialect._utils
import aetherdialect._utils_intent
import aetherdialect._validation_rules
import aetherdialect._validation_shape
import aetherdialect._validation_sql

from ._config import PolicyConfig
from ._contracts_base import EngineIdentity
from ._contracts_core import (
    Expected,
    FeedbackMode,
    GenerationPath,
    LiveTestRunner,
    PendingFeedback,
    QuestionFormStorage,
    QuestionRoute,
    RuntimeIntent,
    Scenario,
    SequenceScenario,
    SoftAssert,
    SqlGenerationOutcome,
)
from ._dialect import Dialect, DialectRegistry
from ._federation_manifest import schema_spans_multiple_sources
from ._intent_loop import collect_structural_match_templates, match_template_for_union
from ._main_execution import MainExecutionOps
from ._pipeline_execute import (
    build_result_dataframe,
    display_final_results_to_stdout,
    handle_direct_sql_reuse,
    results_csv_output_path,
    save_result_csv,
)
from ._pipeline_generate import (
    best_accepted_template_similarity,
    confirm_intent_with_user,
    emit_explain_soft_diagnostics,
    generate_and_validate_sql,
    generate_join_candidates,
    handle_user_feedback,
    load_pipeline_resources,
    match_question_level_template_reuse,
    merge_structural_defaults_for_reuse,
    parse_intent_via_llm,
    prepare_union_match_join_phase,
    stamp_sql_shape,
)
from ._templates import TemplateStoreView
from ._templates_ops import TemplateOps
from ._utils import (
    StepResult,
    debug,
    pipeline_capture,
    pop_engine_identity,
    push_engine_identity,
    substitute_params,
)
from ._utils_intent import flatten_param_values, normalize_question_via_llm, validate_question

"""Live pipeline test runners and assertion helpers."""


def _push_runner_engine_identity(dialect: Any) -> Any | None:
    """Bind the dialect's runtime config for pipeline SQL finalization, matching session ask."""
    runtime_cfg = getattr(dialect, "config", None)
    engine_type = str(getattr(dialect, "name", "") or "").strip()
    if runtime_cfg is None or isinstance(runtime_cfg, type) or not engine_type:
        return None
    return push_engine_identity(EngineIdentity(engine_type=engine_type, runtime_config=runtime_cfg))


def run_live_test(runner: LiveTestRunner, scenario: Scenario, retries: int = 0) -> StepResult | None:
    """Execute a single scenario against the live pipeline."""
    auto = scenario.auto_responses if scenario.auto_responses is not None else ["y", "y", "y"]
    last_result: StepResult | None = None

    for _ in range(1 + retries):
        t0 = time.monotonic()
        try:
            with pipeline_capture(list(auto), scenario.reject_reason, csv_dir=runner.csv_dir) as cap:
                identity_token = _push_runner_engine_identity(runner.dialect)
                try:
                    step = _run_pipeline_core(
                        question=scenario.question,
                        schema=runner.schema,
                        store=runner.store,
                        templates=runner.templates,
                        rejected=runner.rejected,
                        schema_terms=runner.schema_terms,
                        feedback=scenario.feedback,
                        captured_logs=cap["logs"],
                        reject_reason=scenario.reject_reason,
                        feedback_mode=FeedbackMode.LIVE,
                        force_intent_confirm=_scenario_requires_intent_prompt(scenario),
                        dialect=runner.dialect,
                        csv_dir=runner.csv_dir,
                    )
                finally:
                    if identity_token is not None:
                        pop_engine_identity(identity_token)
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


def run_live_test_deferred(runner: LiveTestRunner, scenario: Scenario, retries: int = 0) -> StepResult | None:
    """Execute one scenario while deferring feedback persistence."""
    auto = scenario.auto_responses if scenario.auto_responses is not None else ["y", "y", "y"]
    last_result: StepResult | None = None
    for _ in range(1 + retries):
        t0 = time.monotonic()
        try:
            with pipeline_capture(list(auto), scenario.reject_reason, csv_dir=runner.csv_dir) as cap:
                identity_token = _push_runner_engine_identity(runner.dialect)
                try:
                    step = _run_pipeline_core(
                        question=scenario.question,
                        schema=runner.schema,
                        store=runner.store,
                        templates=runner.templates,
                        rejected=runner.rejected,
                        schema_terms=runner.schema_terms,
                        feedback=scenario.feedback,
                        captured_logs=cap["logs"],
                        reject_reason=scenario.reject_reason,
                        feedback_mode=FeedbackMode.DEFERRED_TEST,
                        force_intent_confirm=_scenario_requires_intent_prompt(scenario),
                        dialect=runner.dialect,
                        csv_dir=runner.csv_dir,
                    )
                finally:
                    if identity_token is not None:
                        pop_engine_identity(identity_token)
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


def deterministic_generate_validate_execute(
    *,
    q_norm: str,
    intent: RuntimeIntent,
    schema: Any,
    dialect: str | Dialect,
    store: dict[str, Any] | TemplateStoreView | None = None,
) -> tuple[SqlGenerationOutcome, list[tuple[Any, ...]] | None]:
    """Build SQL from a fixed ``RuntimeIntent`` (join candidates + ``generate_and_validate_sql``), then execute. Skips NL intent parsing entirely. A join-choice LLM may still run when the graph yields ambiguous join candidates; callers that require zero LLM traffic should patch ``get_join_choice_from_llm`` in tests."""
    if store is None:
        store = {"next_id": 1, "templates": {}, "question_feedback": {}}
    dialect_obj = DialectRegistry.resolve_dialect(dialect)
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
    rows = dialect_obj.execute(exec_sql, params)
    row_list = [tuple(row) for row in rows] if rows else []
    if len(row_list) == 0:
        fixed_intent, fixed_rows = MainExecutionOps.try_zero_row_where_remediation(intent, schema, dialect_obj, tmpl_sd)
        if fixed_rows is not None:
            intent = fixed_intent
            row_list = [tuple(row) for row in fixed_rows]
    return gen_out, row_list


def _extract_reuse_sql(tmpl: Any, q_norm: str) -> str:
    """Reconstruct the final SQL that `handle_direct_sql_reuse` would. produce."""
    vh = tmpl.value_history
    matched_params: dict[str, str] = {}
    for i, hq in enumerate(vh.questions):
        if hq and q_norm == hq:
            matched_params = dict(vh.param_values[i])
            break
    if matched_params:
        merge_structural_defaults_for_reuse(tmpl.sql_param, matched_params, getattr(tmpl, "structural_defaults", None))
        return substitute_params(tmpl.sql_param, matched_params)
    sql_param = getattr(tmpl, "sql_param", "")
    return sql_param if isinstance(sql_param, str) else str(sql_param)


def _build_reuse_intent(tmpl: Any) -> RuntimeIntent:
    """Build a lightweight `RuntimeIntent` from a template's intent. signature."""
    sig = tmpl.intent_signature
    return RuntimeIntent(
        tables=sig.tables or [],
        grain=sig.grain or "row_level",
        select_cols=sig.select_cols or [],
        group_by_cols=sig.group_by_cols or [],
        order_by_cols=sig.order_by_cols or [],
        where=getattr(sig, "where", None),
        having=getattr(sig, "having", None),
        column_map=getattr(sig, "column_map", None) or {},
        natural_language="",
        chosen_join_candidate_id=getattr(sig, "chosen_join_candidate_id", None) or "",
        chosen_join_path_signature=list(getattr(sig, "chosen_join_path_signature", None) or []),
    )


def _scenario_requires_intent_prompt(scenario: Scenario) -> bool:
    """Return True when the scenario expects an intent decline on the. first canned response."""
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
    templates: dict[str, Any],
    rejected: dict[str, Any],
    schema_terms: set[str],
    feedback: str,
    captured_logs: list[str],
    reject_reason: str = "",
    feedback_mode: FeedbackMode = FeedbackMode.LIVE,
    force_intent_confirm: bool = False,
    dialect: Any | None = None,
    csv_dir: str = "",
) -> StepResult:
    """Execute pipeline steps for one question and return captured. state. Mirrors `interactive_run_once` control flow with programmatic arguments and a `StepResult` instead of printing."""
    result = StepResult(scenario_id="", question=question, captured_logs=captured_logs)

    if schema_spans_multiple_sources(schema):
        result.status = "error"
        result.error = "LiveTestRunner does not support federated composite schemas; use AetherEngine.session() or AetherFederation.session() instead."
        return result

    dialect, schema, store, templates, rejected, schema_terms = load_pipeline_resources(
        schema, store, templates, rejected, schema_terms, dialect=dialect
    )

    raw_question = question

    tmpl_pre = match_question_level_template_reuse(raw_question, templates, template_store=store)
    result.reuse_type = tmpl_pre.reuse_type
    if tmpl_pre.reuse_type == "direct_reuse":
        best_template = tmpl_pre.best_template
        result.template_id = best_template.id if best_template else None
        if best_template is None or tmpl_pre.reuse_candidate_normalized is None:
            debug("[live_testing] direct_reuse candidate missing template/candidate; continuing")
        else:
            reuse_pre = handle_direct_sql_reuse(
                tmpl_pre.reuse_candidate_normalized,
                best_template,
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
                result.generation_path = (
                    reuse_pre.generation_path.code if reuse_pre.generation_path is not None else None
                )
            if reuse_pre is not None and reuse_pre.success:
                result.status = "ok"
                result.sql = _extract_reuse_sql(best_template, tmpl_pre.reuse_candidate_normalized)
                result.intent = _build_reuse_intent(best_template)
                return result

    validation = validate_question(raw_question)
    if not validation.accepted:
        result.status = "restricted" if validation.route == QuestionRoute.RESTRICTED else "invalid_question"
        return result

    corrected_text = validation.corrected

    tmpl_typo = match_question_level_template_reuse(corrected_text, templates, template_store=store)
    result.reuse_type = tmpl_typo.reuse_type

    if tmpl_typo.reuse_type == "direct_reuse":
        result.reuse_type = "direct_reuse"
        best_template = tmpl_typo.best_template
        result.template_id = best_template.id if best_template else None
        if best_template is None or tmpl_typo.reuse_candidate_normalized is None:
            debug("[live_testing] typo direct_reuse candidate missing template/candidate; continuing")
        else:
            reuse_out = handle_direct_sql_reuse(
                tmpl_typo.reuse_candidate_normalized,
                best_template,
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
                result.generation_path = (
                    reuse_out.generation_path.code if reuse_out.generation_path is not None else None
                )
            if reuse_out is not None and reuse_out.success:
                result.status = "ok"
                result.sql = _extract_reuse_sql(best_template, tmpl_typo.reuse_candidate_normalized)
                result.intent = _build_reuse_intent(best_template)
                return result

    neg_drop = False
    normalized_canonical = corrected_text
    if validation.route == QuestionRoute.ANALYTICAL:
        normalized_canonical = normalize_question_via_llm(corrected_text, raw_original=raw_question)
        if normalized_canonical != corrected_text and TemplateOps.has_any_rejection_history_for_question(
            store, corrected_text
        ):
            neg_drop = True
            normalized_canonical = corrected_text

    tmpl_norm = None
    if normalized_canonical != corrected_text:
        tmpl_norm = match_question_level_template_reuse(normalized_canonical, templates, template_store=store)
        if tmpl_norm.reuse_type == "direct_reuse":
            result.reuse_type = "direct_reuse"
            best_template = tmpl_norm.best_template
            result.template_id = best_template.id if best_template else None
            if best_template is None or tmpl_norm.reuse_candidate_normalized is None:
                debug("[live_testing] normalized direct_reuse missing template/candidate; continuing")
            else:
                reuse_n = handle_direct_sql_reuse(
                    tmpl_norm.reuse_candidate_normalized,
                    best_template,
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
                    result.generation_path = (
                        reuse_n.generation_path.code if reuse_n.generation_path is not None else None
                    )
                if reuse_n is not None and reuse_n.success:
                    result.status = "ok"
                    result.sql = _extract_reuse_sql(best_template, tmpl_norm.reuse_candidate_normalized)
                    result.intent = _build_reuse_intent(best_template)
                    return result

    norm_opt = normalized_canonical if normalized_canonical != corrected_text else None
    form_storage_turn = QuestionFormStorage(
        corrected=corrected_text,
        normalized_optional=norm_opt,
        normalized_negative_memory_dropped=neg_drop,
        accept_via_normalized_lookup_only=False,
    )

    q_norm = normalized_canonical

    parsed_intent, semantic_warnings, llm_calls, _ = parse_intent_via_llm(corrected_text, schema, templates, store)
    result.llm_calls = llm_calls
    if parsed_intent is None:
        result.status = "intent_parse_failed"
        return result
    intent = parsed_intent
    debug(f"[live_testing] intent parsed: tables={intent.tables} grain={intent.grain} llm_calls={llm_calls}")
    if llm_calls > 2:
        debug(f"[live_testing] WARNING: intent parse required {llm_calls} LLM calls for: {q_norm}")

    result.intent = intent

    result.semantic_warnings = [str(w) for w in semantic_warnings]

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

    exec_params = dict(flatten_param_values(intent))
    exec_sql = dialect.finalize_render(
        intent.sql_param or "",
        exec_params,
        schema=schema,
        intent=intent,
        execution_sql_override=None,
        structural_defaults=tmpl_sd,
    )
    rows = dialect.execute(exec_sql, aetherdialect._utils.reconcile_execute_bind_params(exec_sql, exec_params))
    row_list = [tuple(row) for row in rows] if rows else []
    if len(row_list) == 0:
        fixed_intent, fixed_rows = MainExecutionOps.try_zero_row_where_remediation(intent, schema, dialect, tmpl_sd)
        if fixed_rows is not None:
            intent = fixed_intent
            result.intent = intent
            row_list = [tuple(row) for row in fixed_rows]
    result.sql = sql
    result.rows = row_list
    debug(f"[live_testing] SQL generated ({result.generation_path}): rows={len(row_list)}")
    if result.generation_path == GenerationPath.FRESH.code:
        debug(
            f"[live_testing.path5_trace] q_norm={q_norm!r} sql_param={(intent.sql_param or '')!r} substituted={sql!r}"
        )

    stamp_sql_shape(sql, intent)
    emit_explain_soft_diagnostics(getattr(gen_out, "explain_soft_findings", ()))

    display_final_results_to_stdout(
        q_norm,
        intent,
        sql,
        row_list,
        structural_defaults=tmpl_sd,
        template_display_alias_map=(
            getattr(gen_out.matched_template, "display_alias_map", None) if gen_out.matched_template else None
        ),
    )

    need_sql_feedback = TemplateOps.should_prompt_sql_feedback(store, corrected_text, gen_out.matched_template)
    if need_sql_feedback:
        effective_feedback = feedback
    else:
        effective_feedback = "y"

    result.feedback = effective_feedback

    if effective_feedback == "y" and intent.grain != "scalar":
        df_out = build_result_dataframe(
            row_list,
            intent,
            sql,
            structural_defaults=tmpl_sd,
            q_norm=q_norm,
            template_display_alias_map=(
                getattr(gen_out.matched_template, "display_alias_map", None) if gen_out.matched_template else None
            ),
        )
        if df_out is not None:
            save_result_csv(df_out, output_path=results_csv_output_path(store, csv_dir=csv_dir or None))

    if feedback_mode == FeedbackMode.LIVE:
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
        if feedback_mode == FeedbackMode.DEFERRED_TEST:
            result.status = "ok"
        else:
            result.status = "intent_rejected"
    else:
        result.status = "ok"
    return result


def commit_pending_feedback(result: StepResult) -> None:
    """Persist deferred accept/reject feedback and clear. ``pending_feedback`` on *result*. When the pending choice is ``n``, ``builtins.input`` is patched so classification receives ``canned_reject_reason`` outside the original pipeline capture context."""
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

        def _stdin_reject(prompt: str = "") -> str:
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
    """Return whether *sql* uses an explicit JOIN or a comma- separated multi-relation FROM in the outer SELECT. Implemented via the dialect-agnostic AST helper :func:`aetherdialect._dialect.sql_outer_has_join_or_comma_from`."""
    return Dialect.sql_outer_has_join_or_comma_from(sql, sqlglot_dialect=Dialect.active_sqlglot_dialect())


def _step_result_indicates_join(result: StepResult) -> bool:
    """Return True when rendered SQL or resolved intent shows a multi- table join."""
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


def _assertion_table_names(intent: RuntimeIntent, sql: str | None) -> list[str]:
    """Return base schema table names used for live-test table assertions when CTEs are present."""
    cte_steps = intent.cte_steps or []
    if not cte_steps:
        return sorted(intent.tables or [])
    cte_aliases = {(step.cte_name or "").strip() for step in cte_steps if (step.cte_name or "").strip()}
    cte_aliases.update(str(name).strip() for name in (intent.interpret_cte_names or []) if str(name).strip())
    base_from_ctes: set[str] = set()
    for step in cte_steps:
        base_from_ctes.update(t for t in (step.tables or []) if t and t not in cte_aliases)
    main_base = {t for t in (intent.tables or []) if t and t not in cte_aliases}
    names: set[str] = set(base_from_ctes) | set(main_base)
    if sql:
        sql_tables = Dialect.sql_tables_referenced(sql, sqlglot_dialect=Dialect.active_sqlglot_dialect())
        names.update(t for t in sql_tables if t and t not in cte_aliases)
    return sorted(names)


def _assert_scenario(result: StepResult, expected: Expected, soft: SoftAssert | None = None) -> SoftAssert:
    """Evaluate a `StepResult` against an `Expected` specification. When. `soft` is `None`, a new `SoftAssert` is created. Applicable checks run and failures accumulate on the returned instance."""
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
        soft.check(False, "pipeline_error", "ok", result.status, message=f"uncaught exception:\n{err_preview}")

    if eff_status is not None:
        soft.check(result.status == eff_status, "status", eff_status, result.status)
    elif eff_status_in is not None:
        soft.check(result.status in eff_status_in, "status_in", eff_status_in, result.status)

    if result.intent is not None:
        actual_tables = _assertion_table_names(result.intent, result.sql if result.intent.cte_steps else None)
        actual_tables_cf = sorted(t.casefold() for t in actual_tables)
        if expected.tables_one_of is not None:
            allowed = [sorted(t.casefold() for t in group) for group in expected.tables_one_of]
            soft.check(actual_tables_cf in allowed, "tables", expected.tables_one_of, actual_tables)
        elif expected.tables is not None:
            expected_tables = sorted(expected.tables)
            expected_tables_cf = sorted(t.casefold() for t in expected_tables)
            soft.check(actual_tables_cf == expected_tables_cf, "tables", expected_tables, actual_tables)

    if expected.grain is not None and result.intent is not None:
        if isinstance(expected.grain, tuple):
            soft.check(result.intent.grain in expected.grain, "grain", expected.grain, result.intent.grain)
        else:
            soft.check(result.intent.grain == expected.grain, "grain", expected.grain, result.intent.grain)
    elif expected.grain_in is not None and result.intent is not None:
        soft.check(result.intent.grain in expected.grain_in, "grain", expected.grain_in, result.intent.grain)

    if expected.reuse_type is not None:
        if isinstance(expected.reuse_type, tuple):
            soft.check(result.reuse_type in expected.reuse_type, "reuse_type", expected.reuse_type, result.reuse_type)
        else:
            soft.check(result.reuse_type == expected.reuse_type, "reuse_type", expected.reuse_type, result.reuse_type)

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
        soft.check(has_join == expected.contains_join, "contains_join", expected.contains_join, has_join)

    if expected.contains_group_by is not None:
        has_gb = "GROUP BY" in sql_upper
        soft.check(has_gb == expected.contains_group_by, "contains_group_by", expected.contains_group_by, has_gb)

    if expected.contains_cte is not None:
        has_cte = sql_upper.lstrip().startswith("WITH ")
        soft.check(has_cte == expected.contains_cte, "contains_cte", expected.contains_cte, has_cte)

    if expected.sql_contains is not None and result.sql is not None:
        for substr in expected.sql_contains:
            found = _normalize_sql_for_match(substr) in sql_norm
            soft.check(found, "sql_contains", substr, f"not found in: {result.sql[:120]}")

    if expected.sql_contains_one_of is not None and result.sql is not None:
        ok_any = any(
            all(_normalize_sql_for_match(substr) in sql_norm for substr in group)
            for group in expected.sql_contains_one_of
        )
        soft.check(ok_any, "sql_contains_one_of", expected.sql_contains_one_of, f"not found in: {result.sql[:120]}")

    if expected.sql_excludes is not None and result.sql is not None:
        for substr in expected.sql_excludes:
            found = substr.upper() in sql_upper
            soft.check(not found, "sql_excludes", f"absent: {substr}", f"found in: {result.sql[:120]}")

    if expected.min_rows is not None and result.rows is not None:
        soft.check(len(result.rows) >= expected.min_rows, "min_rows", expected.min_rows, len(result.rows))

    if expected.max_rows is not None and result.rows is not None:
        soft.check(len(result.rows) <= expected.max_rows, "max_rows", expected.max_rows, len(result.rows))

    if expected.column_names_one_of is not None and result.rows is not None and result.intent is not None:
        actual_cols = []
        for c in result.intent.select_cols or []:
            name = getattr(c, "alias", None) or c.expr.primary_term
            actual_cols.append(name.split(".")[-1] if name and "." in name else (name or ""))
        allowed = [sorted(cols) for cols in expected.column_names_one_of]
        soft.check(sorted(actual_cols) in allowed, "column_names", expected.column_names_one_of, actual_cols)

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
        soft.check(result.validation_failed, "should_fail_validation", True, result.validation_failed)

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
    runner: LiveTestRunner, scenario: Scenario, header: str, max_attempts: int = 2, retries: int = 1
) -> None:
    """Run a scenario and assert expectations, retrying from scratch on. failure. On the first attempt the pipeline runs and assertions are checked. When any assertion fails and `max_attempts` > 1, the pipeline is re- run from scratch and assertions are re- evaluated."""
    last_soft: SoftAssert | None = None
    for _ in range(max_attempts):
        attempt_runner = runner.clone()
        result = attempt_runner.run_deferred(scenario, retries=retries)
        if result is None:
            result = StepResult(
                scenario_id=scenario.id,
                question=scenario.question,
                status="error",
                error="runner returned no result",
            )
        last_soft = _assert_scenario(result, scenario.expected)
        if last_soft.passed:
            commit_pending_feedback(result)
            runner.adopt_state_from(attempt_runner)
            return
    if last_soft is not None:
        last_soft.report(header=header)


def run_sequence_and_assert(
    runner: LiveTestRunner, seq: SequenceScenario, max_attempts: int = 2, retries: int = 1
) -> None:
    """Run a sequence of scenarios and assert each step, retrying on. failure. When any step's assertions fail and `max_attempts` > 1, the entire sequence is re-executed from scratch."""
    last_soft: SoftAssert | None = None
    for _ in range(max_attempts):
        attempt_runner = runner.clone()
        last_soft = SoftAssert()
        for step_scenario in seq.steps:
            step_with_seq = replace(step_scenario, sequence_id=seq.id)
            result = attempt_runner.run_deferred(step_with_seq, retries=retries)
            if result is None:
                result = StepResult(
                    scenario_id=step_with_seq.id,
                    question=step_with_seq.question,
                    status="error",
                    error="runner returned no result",
                )
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
    question: str, seeded_intent: RuntimeIntent, schema_graph: Any, *, max_retries: int | None = None
) -> tuple[RuntimeIntent | None, list[str], int]:
    """Run schema and semantic repair from a pre-built intent (live and harness entrypoint). Forwards the seed intent to :func:`aetherdialect._intent_loop.run_schema_semantic_repair_loop` with no template store and no in-turn summary rows."""
    mr = PolicyConfig.MAX_ASK_COMPOSE_REPAIRS if max_retries is None else max_retries
    table_list = sorted(schema_graph.tables.keys())
    schema_literal_json = schema_graph.schema_literal_json
    system, _ = aetherdialect._intent_loop.build_intent_parse_prompt(question, schema_literal_json, table_list)
    return aetherdialect._intent_loop.run_schema_semantic_repair_loop(
        intent=seeded_intent,
        question=question,
        system=system,
        schema_graph=schema_graph,
        schema_literal_json=schema_literal_json,
        table_list=table_list,
        max_retries=mr,
        llm_calls=0,
    )[:3]


LiveTestRunner.run_impl = run_live_test
LiveTestRunner.run_deferred_impl = run_live_test_deferred
