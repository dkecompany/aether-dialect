"""Intent fingerprinting, fuzzy question match, CTE/filter normalisation, and NL question helpers."""

from __future__ import annotations

import copy
import itertools
import random
import re
from dataclasses import replace
from typing import Any, cast

from ._config import EngineConfig, PolicyConfig, SeedWarmupConfig
from ._constants import (
    AGG_PATTERN,
    DO_NOT_LEMMATIZE,
    IRREGULAR_PLURALS_MAP,
    NORMALIZATION_ALLOWED_INTRODUCED_TOKENS,
    NORMALIZATION_JACCARD_FLOOR,
    PROMPT_SCALAR_VALUE_TYPES,
    QUESTION_CANONICALIZE_SYSTEM,
    QUESTION_FROM_SQL_SYSTEM,
    QUESTION_NORMALIZE_VOCABULARY_GUIDANCE,
    QUESTION_NORMALIZE_VOCABULARY_HEADING,
    QUESTION_STARTS_AGG,
    QUESTION_STARTS_GROUP,
    QUESTION_STARTS_LIST,
    QUESTION_VALIDATION_SYSTEM,
    REALISM_DROP_REASON_CATEGORIES,
    SCHEMA_DESCRIPTION_PROMPT_MAX_CHARS,
    SCHEMA_ENRICHED_LINES_MAX_CHARS,
    SHAPE_FORM_DATE_REGEX,
    SHAPE_FORM_NUM_REGEX,
    SHAPE_FORM_STR_REGEX,
    STOPWORDS_GRAMMATICAL_PARTICLES,
    VALID_AGGREGATION_FUNCTIONS,
    VALID_EXPECTED_ROWS,
    VALID_GRAINS,
    VALID_HAVING_OPS,
    VALID_WHERE_OPS,
    WARMUP_FREEFORM_QUESTIONS_SYSTEM,
    WARMUP_PARAPHRASES_BY_STYLE_SYSTEM,
)
from ._contracts_base import (
    CteEmissionKind,
    HavingParam,
    LlmJsonExhausted,
    NormalizedExpr,
    OrderByCol,
    PredicateGroup,
    QuestionRoute,
    QuestionValidationResult,
    WhereParam,
)
from ._contracts_core import (
    ConcreteIntent,
    QuestionReuseMatch,
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
    Template,
)
from ._contracts_schema import (
    CteOutputColumnMeta,
    SchemaGraph,
    SQLShape,
)
from ._core_utils import (
    build_case_folded_index,
    colmap_signature,
    debug,
    normalize_array_contains_param_value,
    normalize_question,
    sha256,
    stable_json,
)
from ._dialect import Dialect
from ._intent_resolve import join_path_key_concrete, join_path_key_runtime, sort_having, sort_where_predicates
from ._llm_provider import LLMProvider


def validate_question(question: str) -> QuestionValidationResult:
    """LLM gate: typo fix, refuse restricted/invalid, route analytical vs metadata questions."""
    try:
        result = LLMProvider.json(QUESTION_VALIDATION_SYSTEM, question, task="default")
    except LlmJsonExhausted as exc:
        debug(f"[utils.validate_question] llm_json exhausted: {exc}")
        return QuestionValidationResult(accepted=False, route=QuestionRoute.INVALID, corrected=question)
    query_type = str(result.get("query_type", "unspecified") or "").strip().lower()
    if query_type == "allowed":
        query_type = QuestionRoute.ANALYTICAL.value
    valid = str(result.get("valid_database_question", "") or "").strip().lower() == "yes"
    corrected = str(result.get("corrected", question) or question)
    if query_type == QuestionRoute.RESTRICTED.value:
        return QuestionValidationResult(accepted=False, route=QuestionRoute.RESTRICTED, corrected=corrected)
    if not valid:
        return QuestionValidationResult(accepted=False, route=QuestionRoute.INVALID, corrected=corrected)
    if query_type == QuestionRoute.SCHEMA_CATALOG.value:
        return QuestionValidationResult(accepted=True, route=QuestionRoute.SCHEMA_CATALOG, corrected=corrected)
    if query_type == QuestionRoute.BUSINESS_KNOWLEDGE.value:
        return QuestionValidationResult(accepted=True, route=QuestionRoute.BUSINESS_KNOWLEDGE, corrected=corrected)
    return QuestionValidationResult(accepted=True, route=QuestionRoute.ANALYTICAL, corrected=corrected)


def _suffix_lemmatize_token(token: str) -> str:
    """Apply conservative English plural-to-singular heuristics when the token is not in ``DO_NOT_LEMMATIZE``."""
    if token in DO_NOT_LEMMATIZE:
        return token
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4:
        for suf in ("ches", "shes", "xes", "zes"):
            if token.endswith(suf):
                return token[: -len(suf)]
    if token.endswith("ss"):
        return token
    if len(token) > 3 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _token_jaccard(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Return the Jaccard similarity between two token multisets treated as sets."""
    sa = set(tokens_a)
    sb = set(tokens_b)
    if not sa and not sb:
        return 1.0
    u = sa | sb
    if not u:
        return 1.0
    return len(sa & sb) / len(u)


def _compute_shape_form(question: str) -> str:
    """Mask numbers, dates, and quoted strings into placeholder tokens for coarse matching."""
    s = question.lower()
    s = SHAPE_FORM_NUM_REGEX.sub("<num>", s)
    s = SHAPE_FORM_DATE_REGEX.sub("<date>", s)
    s = SHAPE_FORM_STR_REGEX.sub("<str>", s)
    return s


def build_shape_question_index(templates: list[Template]) -> dict[str, list[str]]:
    """Map shape-form strings to template identifiers that carry matching history questions."""
    buckets: dict[str, set[str]] = {}
    for tpl in templates:
        for hq in tpl.value_history.questions:
            if not hq or not str(hq).strip():
                continue
            sf = _compute_shape_form(str(hq))
            buckets.setdefault(sf, set()).add(tpl.id)
    return {sf: sorted(ids) for sf, ids in buckets.items()}


def _enforce_normalization_guard(corrected: str, normalized: str, *, raw_original: str) -> tuple[bool, str]:
    """Validate LLM-normalized text against *corrected* and reject. unsafe expansions."""
    nstrip = normalized.strip()
    if not nstrip or len(nstrip) < 2:
        return False, "empty_short"
    corr_words = corrected.split()
    norm_words = nstrip.split()
    if len(norm_words) > len(corr_words):
        return False, "word_count_grew"
    ct = _tokenize(corrected)
    nt = _tokenize(nstrip)
    digit_pat = re.compile(r"^\d")
    for tok in ct:
        if digit_pat.match(tok):
            if tok not in nt:
                return False, "digit_token_lost"
    raw_words = raw_original.split()
    cap_tokens = {w.lower() for w in raw_words if len(w) >= 2 and w[:1].isupper() and w[1:].islower()}
    for ctok in cap_tokens:
        if ctok not in nt:
            return False, "capital_token_lost"
    stop_lower = frozenset(t.lower() for t in STOPWORDS_GRAMMATICAL_PARTICLES)
    ct_f = [t for t in ct if t.lower() not in stop_lower]
    nt_f = [t for t in nt if t.lower() not in stop_lower]
    if _token_jaccard(ct_f, nt_f) < float(NORMALIZATION_JACCARD_FLOOR):
        return False, "jaccard_floor"
    corr_word_set = set(corr_words)
    introduced = [w for w in norm_words if w not in corr_word_set]
    for w in introduced:
        wl = w.lower()
        if wl in corr_word_set:
            continue
        if wl in stop_lower:
            continue
        if wl in NORMALIZATION_ALLOWED_INTRODUCED_TOKENS:
            continue
        return False, "introduced_token"
    return True, "ok"


def normalize_question_via_llm(corrected: str, *, raw_original: str | None = None) -> str:
    """Canonicalize *corrected* via a dedicated LLM call separate from. typo validation."""
    raw_use = raw_original if raw_original is not None else corrected
    vocab_block = QUESTION_NORMALIZE_VOCABULARY_HEADING + "\n" + QUESTION_NORMALIZE_VOCABULARY_GUIDANCE
    user_obj: dict[str, Any] = {
        "question": corrected,
        "normalization_preferences": vocab_block,
    }
    try:
        result = LLMProvider.json(QUESTION_CANONICALIZE_SYSTEM, stable_json(user_obj), task="default")
    except LlmJsonExhausted as exc:
        debug(f"[utils.normalize_question_via_llm] llm_json exhausted: {exc}")
        return corrected
    normalized = str(result.get("normalized", corrected) or corrected).strip()
    ok, reason = _enforce_normalization_guard(corrected, normalized, raw_original=raw_use)
    if not ok:
        debug(f"[utils.normalize_question_via_llm] normalized_rejected reason={reason}")
        return corrected
    return normalized


def sql_shape(sql: str, intent: RuntimeIntent, *, sqlglot_dialect: str) -> SQLShape:
    """Count joins, CTEs, filters, having, and structural flags from. *sql* and *intent* via AST."""
    num_where = len(PredicateGroup.where_leaves(intent.where) or [])
    num_having = len(PredicateGroup.having_leaves(intent.having) or [])
    for cte in intent.cte_steps or []:
        num_where += len(PredicateGroup.where_leaves(cte.where) or [])
        num_having += len(PredicateGroup.having_leaves(cte.having) or [])
    return SQLShape(
        num_joins=Dialect.sql_count_outer_joins(sql, sqlglot_dialect=sqlglot_dialect),
        has_group_by=Dialect.sql_has_group_by(sql, sqlglot_dialect=sqlglot_dialect),
        has_agg=Dialect.sql_has_aggregate(sql, sqlglot_dialect=sqlglot_dialect),
        num_cte=len(intent.cte_steps or []),
        num_where=num_where,
        num_having=num_having,
        has_distinct=Dialect.sql_has_distinct(sql, sqlglot_dialect=sqlglot_dialect),
    )


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance between *s1* and *s2*."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row: list[int] = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def _tokenize(q: str) -> list[str]:
    """Unicode-aware tokens from *q*, suffix-lemmatized except for ``DO_NOT_LEMMATIZE``, stripped of ``PolicyConfig.STOPWORDS``, then sorted for multiset comparison."""
    out: list[str] = []
    buf: list[str] = []
    for ch in q.lower():
        if ch.isalnum() or ch == "_":
            buf.append(ch)
        elif buf:
            raw = "".join(buf)
            buf.clear()
            irr = IRREGULAR_PLURALS_MAP.get(raw, raw)
            step = _suffix_lemmatize_token(irr)
            if step not in PolicyConfig.STOPWORDS:
                out.append(step)
    if buf:
        raw = "".join(buf)
        irr = IRREGULAR_PLURALS_MAP.get(raw, raw)
        step = _suffix_lemmatize_token(irr)
        if step not in PolicyConfig.STOPWORDS:
            out.append(step)
    return sorted(out)


def _question_token_fingerprint_from_normalized(norm: str) -> str:
    """Sorted multiset fingerprint of stopword-stripped question tokens (see :func:`_tokenize`)."""
    return "\0".join(_tokenize(norm))


def question_token_fingerprint_from_raw(raw: str) -> str:
    """Fingerprint for a raw question string after :func:`aetherdialect._core_utils.normalize_question`."""
    return _question_token_fingerprint_from_normalized(normalize_question(raw))


def _neighboring_question_token_fingerprint_norms(norm: str) -> frozenset[str]:
    """Fingerprints for inverted-index lookup: the exact multiset plus neighbors from one in-token substitution. Bounded by :data:`PolicyConfig.QUESTION_TOKEN_INDEX_NEIGHBOR_CAP` for determinism."""
    toks = _tokenize(norm)
    base = "\0".join(toks)
    out: set[str] = {base}
    cap = int(PolicyConfig.QUESTION_TOKEN_INDEX_NEIGHBOR_CAP)
    if not toks:
        return frozenset(out)
    for i, tok in enumerate(toks):
        if len(out) >= cap:
            break
        for j in range(len(tok)):
            if len(out) >= cap:
                break
            for sub in "abcdefghijklmnopqrstuvwxyz0123456789_":
                if sub == tok[j]:
                    continue
                variant = tok[:j] + sub + tok[j + 1 :]
                lst = list(toks)
                lst[i] = variant
                lst.sort()
                out.add("\0".join(lst))
                if len(out) >= cap:
                    break
    return frozenset(out)


def _fuzzy_question_tokens_match_pair(
    q1_norm: str, q2_norm: str, max_distance: int, debug_label: str
) -> tuple[bool, int]:
    """Return whether stopword-stripped token lists align with summed. per-token edit distance within *max_distance*."""
    base = "[utils.exact_question_match]"
    tag = f"{base} {debug_label}".strip() if debug_label else base
    tokens1 = _tokenize(q1_norm)
    tokens2 = _tokenize(q2_norm)
    if len(tokens1) == 0 or len(tokens2) == 0:
        debug(f"{tag} FAIL empty_tokens t1={len(tokens1)} t2={len(tokens2)}")
        return False, 0
    if len(tokens1) != len(tokens2):
        debug(f"{tag} FAIL token_count t1={len(tokens1)} t2={len(tokens2)}")
        return False, 0
    total_dist = 0
    worst_pair = ("", "")
    for t1, t2 in zip(tokens1, tokens2, strict=True):
        dist = _levenshtein_distance(t1, t2)
        total_dist += dist
        if dist > 0:
            worst_pair = (t1, t2)
    if total_dist > max_distance:
        debug(f"{tag} FAIL total_dist={total_dist} worst_pair='{worst_pair[0]}'→'{worst_pair[1]}'")
        return False, total_dist
    debug(f"{tag} MATCH tokens={len(tokens1)} total_dist={total_dist}")
    return True, total_dist


def match_question_against_template_history(
    candidate_raw: str,
    templates: list[Template],
    *,
    max_token_edit_distance: int | None = None,
    shape_question_index: dict[str, list[str]] | None = None,
    question_token_index: dict[str, list[Any]] | None = None,
) -> QuestionReuseMatch | None:
    """Find the best trusted template whose stored question fuzzy- matches the candidate. Normalizes the candidate once, optionally narrows templates via a shape-form index, then scores every trusted history row using token edit distance with lexicographic ties broken by per-row accepts, template accepts, and template id."""
    budget = max_token_edit_distance if max_token_edit_distance is not None else PolicyConfig.FUZZY_MATCH_MAX_DISTANCE
    candidate_normalized = normalize_question(candidate_raw)
    scan_templates = templates
    if shape_question_index:
        cand_sf = _compute_shape_form(candidate_raw)
        allowed_ids = shape_question_index.get(cand_sf)
        if allowed_ids:
            allow_set = frozenset(allowed_ids)
            narrowed = [t for t in templates if t.id in allow_set]
            if narrowed:
                scan_templates = narrowed
    pair_filter: set[tuple[str, int]] | None = None
    if question_token_index:
        cand_fps = _neighboring_question_token_fingerprint_norms(candidate_normalized)
        acc: set[tuple[str, int]] = set()
        for fp in cand_fps:
            rows = question_token_index.get(fp)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, (list, tuple)) and len(row) == 2:
                    acc.add((str(row[0]), int(row[1])))
        if acc:
            pair_filter = acc
            tid_allow = {p[0] for p in acc}
            narrowed_t = [t for t in scan_templates if t.id in tid_allow]
            if narrowed_t:
                scan_templates = narrowed_t
    best: QuestionReuseMatch | None = None
    best_rank: tuple[int, int, int, str] | None = None
    for tpl in scan_templates:
        if tpl.trust_level < 1:
            continue
        approval = getattr(tpl, "approval_state", None)
        if approval is not None and str(getattr(approval, "value", approval)).lower() == "pending":
            continue
        for idx, hist_q in enumerate(tpl.value_history.questions):
            if not hist_q:
                continue
            if pair_filter is not None and (tpl.id, idx) not in pair_filter:
                continue
            stored_normalized = normalize_question(hist_q)
            ok, total = _fuzzy_question_tokens_match_pair(candidate_normalized, stored_normalized, budget, tpl.id)
            if not ok:
                continue
            row_ac = int(tpl.value_history.accept_counts[idx])
            tpl_ac = int(tpl.stats.accept)
            rank = (total, -row_ac, -tpl_ac, tpl.id)
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best = QuestionReuseMatch(
                    template_id=tpl.id,
                    history_index=idx,
                    stored_normalized_text=stored_normalized,
                    candidate_normalized=candidate_normalized,
                    token_edit_sum=total,
                )
    if best is not None:
        debug(
            f"[utils.match_question_against_template_history] tpl={best.template_id} "
            f"idx={best.history_index} exact={best.is_exact_string_reuse} token_edit_sum={best.token_edit_sum}"
        )
    return best


def exact_question_match(
    q1: str, q2: str, max_distance: int = PolicyConfig.FUZZY_MATCH_MAX_DISTANCE, label: str = ""
) -> bool:
    """True if normalised token sequences match 1:1 with summed edit. distance ≤ *max_distance*."""
    q1_norm = normalize_question(q1)
    q2_norm = normalize_question(q2)
    ok, _ = _fuzzy_question_tokens_match_pair(q1_norm, q2_norm, max_distance, label)
    return ok


def is_exact_question_text_match(q1: str, q2: str) -> bool:
    """Return True when *q1* and *q2* share the same normalised question string."""
    return normalize_question(q1) == normalize_question(q2)


def _normalize_where_predicates(filters: list[Any]) -> list[WhereParam]:
    """Coerce dicts / ``WhereParam`` to ``WhereParam`` and. ``sort_where_predicates``."""
    if not filters:
        return []
    out = []
    for f in filters:
        if isinstance(f, WhereParam):
            left_expr = f.left_expr
            op = f.op.strip().lower() if f.op else "="
            vtype = f.value_type.strip().lower() if isinstance(f.value_type, str) else "unknown"
            right_expr = f.right_expr
        elif isinstance(f, dict):
            left_raw = f.get("left_expr")
            if isinstance(left_raw, dict):
                left_expr = NormalizedExpr.from_dict(left_raw)
            else:
                col = f.get("column", "")
                left_expr = NormalizedExpr.from_column(col.strip().lower() if isinstance(col, str) else "")
            op = f.get("op", "=").strip().lower()
            vtype = f.get("value_type", "unknown").strip().lower()
            right_raw = f.get("right_expr")
            right_expr = NormalizedExpr.from_dict(right_raw) if isinstance(right_raw, dict) and right_raw else None
        else:
            continue
        if not left_expr or not isinstance(op, str):
            continue
        fp = WhereParam(left_expr=left_expr, op=op, value_type=vtype, param_key="", right_expr=right_expr)
        out.append(fp)
    return sort_where_predicates(out)


def _normalize_having_conditions(conditions: list[Any]) -> list[HavingParam]:
    """Coerce dicts / ``HavingParam``; clamp ops to. ``VALID_HAVING_OPS``; ``sort_having``."""
    if not conditions:
        return []
    out = []
    for c in conditions:
        if isinstance(c, HavingParam):
            left_expr = c.left_expr
            op = c.op.strip().lower() if c.op else "="
            value_type = c.value_type.strip().lower() if c.value_type else "number"
            right_expr = c.right_expr
        elif isinstance(c, dict):
            left_raw = c.get("left_expr")
            if isinstance(left_raw, dict):
                left_expr = NormalizedExpr.from_dict(left_raw)
            else:
                agg = c.get("aggregation", "")
                left_expr = NormalizedExpr.from_column(str(agg)) if agg else NormalizedExpr()
            op = c.get("op", "=")
            value_type = c.get("value_type", "number")
            right_raw = c.get("right_expr")
            right_expr = NormalizedExpr.from_dict(right_raw) if isinstance(right_raw, dict) and right_raw else None
        else:
            continue
        if not left_expr or not left_expr.primary_term:
            continue
        op_norm = op.strip().lower() if isinstance(op, str) else "="
        if op_norm not in VALID_HAVING_OPS:
            op_norm = "="
        hp = HavingParam(
            left_expr=left_expr, op=op_norm, value_type=str(value_type), param_key="", right_expr=right_expr
        )
        out.append(hp)
    return sort_having(out)


def _normalize_cte_steps(steps: Any, available_ctes: dict[str, list[str]] | None = None) -> list[RuntimeCteStep]:
    """Parse CTE steps from dicts or models; infer ``column_map`` and. output metadata."""
    if not isinstance(steps, list):
        return []
    if available_ctes is None:
        available_ctes = {}
    out = []
    for s in steps:
        if isinstance(s, RuntimeCteStep):
            cte_name = s.cte_name
            tables = s.tables or []
            grain = s.grain or "row_level"
            select_cols = s.select_cols or []
            group_by_cols = s.group_by_cols or []
            output_columns = s.output_columns or []
            where_params = PredicateGroup.where_leaves(s.where) or []
            having_param = PredicateGroup.having_leaves(s.having) or []
            param_values = s.param_values or {}
            order_by_cols = s.order_by_cols or []
            limit = s.limit
            column_map = s.column_map or {}
            output_column_metadata = s.output_column_metadata or {}
            chosen_join_candidate_id = s.chosen_join_candidate_id or ""
            chosen_join_path_signature = s.chosen_join_path_signature or []
            emission = CteEmissionKind.coerce(getattr(s, "emission", CteEmissionKind.JOIN_TABLE))
        elif isinstance(s, dict):
            cte_name = s.get("cte_name", "")
            tables = s.get("tables", [])
            grain = s.get("grain", "row_level")
            sc_raw = s.get("select_cols", [])
            select_cols = [
                (
                    SelectCol.from_dict(sc)
                    if isinstance(sc, dict)
                    else (SelectCol(expr=NormalizedExpr.from_column(sc)) if isinstance(sc, str) else sc)
                )
                for sc in sc_raw
            ]
            group_by_cols = s.get("group_by_cols", [])
            group_by_cols = [
                (
                    NormalizedExpr.from_dict(g)
                    if isinstance(g, dict)
                    else (NormalizedExpr.from_column(g) if isinstance(g, str) else g)
                )
                for g in group_by_cols
            ]
            output_columns = s.get("output_columns", [])
            fp_raw = s.get("where", s.get("where_param", []))
            fp_raw = fp_raw if isinstance(fp_raw, list) else []
            where_params = [WhereParam.from_dict(f) if isinstance(f, dict) else f for f in fp_raw]
            hp_raw = s.get("having", s.get("having_param", []))
            hp_raw = hp_raw if isinstance(hp_raw, list) else []
            having_param = [HavingParam.from_dict(h) if isinstance(h, dict) else h for h in hp_raw]
            param_values = s.get("param_values", {})
            obc_raw = s.get("order_by_cols", [])
            order_by_cols = [
                (
                    OrderByCol.from_dict(o)
                    if isinstance(o, dict)
                    else (OrderByCol(expr=NormalizedExpr.from_column(o)) if isinstance(o, str) else o)
                )
                for o in obc_raw
            ]
            limit = s.get("limit")
            column_map = s.get("column_map", {})
            ocm_raw = s.get("output_column_metadata", {})
            output_column_metadata = {
                k: CteOutputColumnMeta.from_dict(v) if isinstance(v, dict) else v for k, v in ocm_raw.items()
            }
            chosen_join_candidate_id = s.get("chosen_join_candidate_id", "")
            chosen_join_path_signature = s.get("chosen_join_path_signature", [])
            emission = CteEmissionKind.coerce(s.get("emission", CteEmissionKind.JOIN_TABLE))
        else:
            continue
        if not cte_name:
            continue
        if grain not in VALID_GRAINS:
            grain = "row_level"
        normalized_fp = []
        for f in where_params:
            if isinstance(f, WhereParam):
                op = f.op.strip().lower() if f.op else "="
                if op not in VALID_WHERE_OPS:
                    op = "="
                vtype = f.value_type.strip().lower() if f.value_type else "unknown"
                fp = replace(f, op=op, value_type=vtype)
                normalized_fp.append(fp)
        normalized_hp = []
        for h in having_param:
            if isinstance(h, HavingParam):
                op = h.op.strip().lower() if h.op else "="
                if op not in VALID_HAVING_OPS:
                    op = "="
                vtype = h.value_type.strip().lower() if h.value_type else "number"
                hp = replace(h, op=op, value_type=vtype)
                normalized_hp.append(hp)
        cte_column_map = {}
        all_cols_raw = [g.primary_column for g in group_by_cols] + [f.left_expr.primary_column for f in normalized_fp]
        for sc in select_cols:
            if isinstance(sc, SelectCol):
                all_cols_raw.append(sc.expr.primary_column)
        for col in all_cols_raw:
            if "." in col:
                parts = col.split(".", 1)
                table_ref = parts[0].lower()
                col_name = parts[1].lower()
                if table_ref in {t.lower() for t in tables} or table_ref in {c.lower() for c in available_ctes.keys()}:
                    cte_column_map[col_name] = table_ref
        if column_map:
            cte_column_map.update(column_map)
        ocm = {}
        for out_col in output_columns:
            out_col_lower = out_col.lower()
            if out_col_lower in output_column_metadata:
                ocm[out_col_lower] = output_column_metadata[out_col_lower]
            else:
                is_agg = any(out_col_lower.startswith(f"{agg}_") for agg in VALID_AGGREGATION_FUNCTIONS)
                agg_func = ""
                if is_agg:
                    for agg in VALID_AGGREGATION_FUNCTIONS:
                        if out_col_lower.startswith(f"{agg}_"):
                            agg_func = agg
                            break
                ocm[out_col_lower] = CteOutputColumnMeta(
                    source="aggregation" if is_agg else "passthrough",
                    agg_func=agg_func,
                    filterable=True,
                    aggregatable=True,
                    data_type=("integer" if is_agg and agg_func == "count" else "unknown"),
                )
        normalized_select_cols = (
            select_cols
            if select_cols and isinstance(select_cols[0], SelectCol)
            else (
                [SelectCol(expr=NormalizedExpr.from_column(c) if isinstance(c, str) else c.expr) for c in select_cols]
                if select_cols
                else []
            )
        )
        normalized_order_by = (
            order_by_cols
            if order_by_cols and isinstance(order_by_cols[0], OrderByCol)
            else (
                [(OrderByCol(expr=NormalizedExpr.from_column(c)) if isinstance(c, str) else c) for c in order_by_cols]
                if order_by_cols
                else []
            )
        )
        cte = RuntimeCteStep(
            cte_name=str(cte_name),
            tables=sorted(set(str(t) for t in tables)),
            grain=grain,
            select_cols=normalized_select_cols,
            group_by_cols=group_by_cols,
            output_columns=list(str(c) for c in output_columns),
            where=PredicateGroup.from_list(sort_where_predicates(normalized_fp)),
            having=PredicateGroup.from_list(sort_having(normalized_hp)),
            param_values=param_values,
            order_by_cols=normalized_order_by,
            limit=limit,
            column_map=cte_column_map,
            output_column_metadata=ocm,
            chosen_join_candidate_id=chosen_join_candidate_id,
            chosen_join_path_signature=chosen_join_path_signature,
            emission=emission,
        )
        out.append(cte)
        available_ctes[cte_name] = output_columns
    return out


def _normalize_cte_steps_for_key(
    steps: list[RuntimeCteStep], *, include_join_skeleton: bool = True
) -> list[dict[str, Any]]:
    """Projection of CTE steps to signature strings for structural intent hashes. When *include_join_skeleton* is false (``body_similarity_key``), CTE join-emission metadata is omitted so join-path variants group together."""
    result = []
    for cte in steps:
        select_sigs: list[str] = []
        for sc in cast(list[Any], cte.select_cols or []):
            if isinstance(sc, SelectCol):
                select_sigs.append(sc.signature_key)
            elif isinstance(sc, str):
                select_sigs.append(sc.strip())
        order_sigs: list[str] = []
        for obc in cast(list[Any], cte.order_by_cols or []):
            if isinstance(obc, OrderByCol):
                order_sigs.append(obc.signature_key)
            elif isinstance(obc, str):
                order_sigs.append(obc.strip())
        cte_dict = {
            "cte_name": cte.cte_name,
            "tables": sorted(cte.tables or []),
            "select_cols": sorted(select_sigs),
            "group_by_cols": sorted([g.signature_key for g in (cte.group_by_cols or [])]),
            "output_columns": sorted(cte.output_columns or []),
            "where": [
                f.signature_key
                for f in sorted(
                    PredicateGroup.where_leaves(cte.where) or [],
                    key=lambda x: (
                        x.left_expr.signature_key,
                        x.op,
                        x.right_expr.signature_key if x.right_expr else "",
                        x.value_type,
                    ),
                )
            ],
            "having_param": [
                h.signature_key
                for h in sorted(
                    PredicateGroup.having_leaves(cte.having) or [],
                    key=lambda x: (
                        x.left_expr.signature_key,
                        x.op,
                        x.right_expr.signature_key if x.right_expr else "",
                        x.value_type,
                    ),
                )
            ],
            "order_by_cols": sorted(order_sigs),
            "window_registry": sorted(s.signature_key for s in (cte.window_registry or [])),
            "case_registry": sorted(s.signature_key for s in (cte.case_registry or [])),
        }
        if include_join_skeleton:
            cte_dict["emission"] = str(getattr(cte, "emission", "") or "")
        result.append(cte_dict)
    return result


def _contains_where_param_keys(intent: RuntimeIntent) -> set[str]:
    keys: set[str] = set()
    for cte in intent.cte_steps or []:
        for fp in PredicateGroup.where_leaves(cte.where) or []:
            if fp.op == "contains" and fp.param_key and fp.right_expr is None:
                keys.add(fp.param_key)
    for fp in PredicateGroup.where_leaves(intent.where) or []:
        if fp.op == "contains" and fp.param_key and fp.right_expr is None:
            keys.add(fp.param_key)
    return keys


def flatten_param_values(intent: RuntimeIntent) -> dict[str, Any]:
    """Merge CTE ``param_values`` then main; main overrides duplicate. keys. Applies:func:`core_utils.normalize_array_contains_param_value` to keys for WHERE rows with ``op == "contains"``. Dialect SQL for those predicates also normalizes stored array elements at execution time."""
    merged = {}
    for cte in intent.cte_steps or []:
        merged.update(cte.param_values or {})
    merged.update(intent.param_values or {})
    ckeys = _contains_where_param_keys(intent)
    if not ckeys:
        return merged
    out = dict(merged)
    for k in ckeys:
        if k in out:
            out[k] = normalize_array_contains_param_value(out[k])
    return out


def _structural_intent_hash(intent: RuntimeIntent, *, include_join_skeleton: bool) -> str:
    """Shared structural hash for ``intent_key`` and ``body_similarity_key``."""
    select_cols = intent.select_cols or []

    grain = intent.grain or "row_level"
    if grain not in VALID_GRAINS:
        debug(f"[utils.intent_key] invalid_grain: '{grain}' defaulting to 'row_level'")
        grain = "row_level"

    expected_rows = intent.expected_rows or "many"
    if expected_rows not in VALID_EXPECTED_ROWS:
        if grain == "scalar":
            expected_rows = "one"
        elif grain == "grouped":
            expected_rows = "few"
        else:
            expected_rows = "many"
        debug(f"[utils.intent_key] inferred_expected_rows: grain={grain} -> expected_rows={expected_rows}")

    filters_normalized = _normalize_where_predicates(PredicateGroup.where_leaves(intent.where) or [])
    having_conditions_normalized = _normalize_having_conditions(PredicateGroup.having_leaves(intent.having) or [])
    cte_steps_normalized = _normalize_cte_steps(intent.cte_steps or [])
    cte_steps_for_key = _normalize_cte_steps_for_key(cte_steps_normalized, include_join_skeleton=include_join_skeleton)

    select_cols_sorted = sorted([s.signature_key for s in select_cols])
    order_by_sorted = sorted([o.signature_key for o in (intent.order_by_cols or [])])

    normalized = {
        "tables": sorted(intent.tables or []),
        "select_cols": select_cols_sorted,
        "where": [f.to_dict() for f in filters_normalized],
        "group_by_cols": sorted([g.signature_key for g in (intent.group_by_cols or [])]),
        "order_by_cols": order_by_sorted,
        "having_conditions": [hc.to_dict() for hc in having_conditions_normalized],
        "cte_steps": cte_steps_for_key,
        "window_registry": sorted(s.signature_key for s in (getattr(intent, "window_registry", None) or [])),
        "case_registry": sorted(s.signature_key for s in (getattr(intent, "case_registry", None) or [])),
    }
    key = sha256(stable_json(normalized))
    debug(f"[utils.intent_key] computed: tables={normalized['tables']} key={key[:16]}...")
    return key


def intent_key(intent: RuntimeIntent) -> str:
    """SHA-256 of normalised structural intent: tables, selects, filters, group/order/having, CTEs. Omits ``grain``, ``limit``, and raw param values; uses normalised filter/having dicts and CTE key skeletons. Differs from :func:`aetherdialect._intent_process.intent_similarity`, which scores overlap via weighted clause similarity (including a separate CTE blend) rather than a single hash."""
    return _structural_intent_hash(intent, include_join_skeleton=True)


def body_similarity_key(intent: RuntimeIntent) -> str:
    """Structural body fingerprint excluding grain, limit, column_map, join path, and param values."""
    return _structural_intent_hash(intent, include_join_skeleton=False)


def body_similarity_key_for_concrete(concrete: ConcreteIntent) -> str:
    """``body_similarity_key`` for a stored ``ConcreteIntent``."""
    return body_similarity_key(concrete.to_runtime_skeleton())


def template_instance_key_from_parts(
    body_key: str,
    join_fp: str,
    sql_fp_val: str,
    *,
    colmap_sig: str = "",
    grain: str = "",
    limit: int | None = None,
    params_fp: str = "",
) -> str:
    """Stable key for an executable template row: body + join fingerprint + parameterized SQL fingerprint."""
    return sha256(
        stable_json(
            {
                "b": body_key,
                "j": join_fp,
                "s": sql_fp_val,
                "c": colmap_sig,
                "g": grain,
                "l": limit,
                "p": params_fp,
            }
        )
    )


def template_instance_key_for_concrete(
    concrete: ConcreteIntent,
    sql_fp_val: str,
    *,
    param_values: dict[str, Any] | None = None,
) -> str:
    """Warmup/template-store instance key from a stored concrete intent."""
    params = param_values if param_values is not None else dict(concrete.param_values or {})
    return template_instance_key_from_parts(
        body_similarity_key_for_concrete(concrete),
        join_path_key_concrete(concrete),
        sql_fp_val,
        colmap_sig=colmap_signature(concrete.column_map or {}),
        grain=concrete.grain or "row_level",
        limit=concrete.limit,
        params_fp=stable_json(params),
    )


def template_instance_key_for_runtime(runtime: RuntimeIntent, sql_fp_val: str) -> str:
    """Warmup instance key from a post-join runtime intent."""
    return template_instance_key_from_parts(
        body_similarity_key(runtime),
        join_path_key_runtime(runtime),
        sql_fp_val,
        colmap_sig=colmap_signature(runtime.column_map or {}),
        grain=runtime.grain or "row_level",
        limit=runtime.limit,
        params_fp=stable_json(flatten_param_values(runtime)),
    )


def extract_tables_from_sql(sql: str, known_tables: list[str], *, sqlglot_dialect: str) -> list[str]:
    """Subset of *known_tables* referenced as physical tables in *sql* via AST inspection. CTE definition names are excluded by :func:`aetherdialect._dialect.sql_tables_referenced`."""
    referenced = Dialect.sql_tables_referenced(sql, sqlglot_dialect=sqlglot_dialect)
    hits = [t for t in known_tables if t.lower() in referenced]
    return sorted(set(hits))


def _describe_operation(select_terms: list[str]) -> str:
    """First ``AGG_PATTERN`` aggregation name in *select_terms*, else. ``list``."""
    for sc in select_terms:
        m = AGG_PATTERN.match(sc)
        if m:
            return m.group(1).lower()
    return "list"


def pick_question_style(select_terms: list[str], has_grouping: bool) -> str:
    """Random NL question opener from structure (group / agg / list)."""
    agg_funcs = []
    for sc in select_terms:
        m = AGG_PATTERN.match(sc)
        if m:
            agg_funcs.append(m.group(1).lower())

    if has_grouping:
        return random.choice(QUESTION_STARTS_GROUP)
    elif agg_funcs:
        agg = agg_funcs[0]
        if agg == "count":
            return random.choice(["How many", "Count", "What is the number of"])
        elif agg == "sum":
            return random.choice(["What is the total", "Find the sum of", "Calculate the total"])
        elif agg == "avg":
            return random.choice(["What is the average", "Calculate the average", "Find the mean"])
        elif agg in ("min", "max"):
            return random.choice([f"What is the {agg}imum", f"Find the {agg}imum", f"Get the {agg}"])
        return random.choice(QUESTION_STARTS_AGG)
    else:
        return random.choice(QUESTION_STARTS_LIST)


def generate_question(
    tables: list[str],
    select_terms: list[str],
    filter_descriptions: list[dict[str, str]],
    group_by_terms: list[str],
    having_descriptions: list[dict[str, str]],
    schema: SchemaGraph,
) -> str | None:
    """LLM: intent JSON → one sentence; must start with chosen style prefix."""
    semantics = {}
    for table in tables:
        table_ir = schema.tables.get(table)
        if table_ir:
            semantics[table] = {
                "description": table_ir.description or f"{table} records",
                "columns": {col: (getattr(meta, "description", None) or col) for col, meta in table_ir.columns.items()},
            }

    roles = {}
    all_columns = set()
    for fd in filter_descriptions:
        all_columns.add(fd.get("column", ""))
    for col in group_by_terms:
        all_columns.add(col)
    for col in all_columns:
        parts = col.split(".")
        if len(parts) == 2 and parts[0] in schema.tables:
            col_meta = schema.tables[parts[0]].columns.get(parts[1])
            if col_meta and col_meta.role:
                roles[col] = col_meta.role

    operation = _describe_operation(select_terms)

    intent_structure = {
        "tables": tables,
        "operation": operation,
        "columns": select_terms,
        "filters": filter_descriptions if filter_descriptions else None,
        "grouping": group_by_terms if group_by_terms else None,
        "having": having_descriptions if having_descriptions else None,
    }

    selected_style = pick_question_style(select_terms, bool(group_by_terms))

    system = (
        "You are a natural language question generator for database queries. Convert structured query intent to conversational questions.\n\n"
        "Output Requirements:\n"
        '- Output ONLY valid JSON with fields "question" (string) and "is_realistic" (boolean)\n'
        "- Do NOT include markdown, explanations, or commentary\n"
        "- Identical inputs must produce identical outputs\n\n"
        "Generation Rules:\n"
        "- Question MUST reflect EXACTLY the intent structure (same tables, columns, filters, aggregation)\n"
        "- Do NOT add columns, tables, or filters not in the intent\n"
        "- Do NOT change aggregation type or omit filter values\n"
        "- Use natural, conversational language - avoid SQL jargon and raw table names\n"
        "- Use semantic context to refer to business concepts naturally\n"
        "- For multi-table queries, describe relationships naturally\n"
        "- Inject filter values naturally into the question\n"
        "- ONE sentence only\n"
        "- Vary phrasing naturally within the required start constraint\n"
    )

    user_prompt = {
        "task": "Generate a natural language question for this query intent",
        "intent": intent_structure,
        "semantic_context": semantics,
        "column_roles": roles,
        "phrasing_constraint": {
            "required_start": selected_style,
            "description": f"Question MUST start with exactly: '{selected_style}'",
            "strict_mode": not filter_descriptions and not group_by_terms,
            "phrasing_flexibility": "Vary word choice and sentence structure while keeping the required start",
        },
        "output_format": {
            "question": "Your natural language question here",
            "is_realistic": True,
        },
    }

    response = LLMProvider.json(system, stable_json(user_prompt), retries=1, task="synth_variety")
    question = response.get("question")
    ir = response.get("is_realistic", True)
    if isinstance(ir, str):
        ir = str(ir).lower() in ("true", "1", "yes")
    if not ir:
        debug("[utils.generate_question] is_realistic=false")
        return None
    if question and isinstance(question, str):
        question = question.strip()
        template_start = selected_style.split("{")[0].strip()
        if template_start and not question.startswith(template_start):
            debug(f"[utils.generate_question] phrasing_violation: expected_start={template_start}, got={question[:30]}")
            return None
        debug(f"[utils.generate_question] generated: {question[:50]}")
        return question
    debug("[utils.generate_question] missing_question_field")
    return None


def schema_table_descriptions_for_tables(schema: SchemaGraph, tables: list[str]) -> str:
    """Render table names and optional table descriptions for warmup NL prompts."""
    blocks: list[str] = []
    for table in tables:
        table_meta = schema.tables.get(table)
        if not table_meta:
            continue
        lines: list[str] = [f"TABLE {table}"]
        td = getattr(table_meta, "description", "") or ""
        if str(td).strip():
            lines.append(f"  table_description: {str(td).strip()}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _truncate_enriched_description(text: str) -> str:
    """Bound one enriched description line to the schema prompt char cap."""
    td = str(text).strip()
    if not td:
        return ""
    if len(td) <= SCHEMA_DESCRIPTION_PROMPT_MAX_CHARS:
        return td
    return td[: SCHEMA_DESCRIPTION_PROMPT_MAX_CHARS - 3] + "..."


def schema_context_enriched_lines_for_tables(schema: SchemaGraph, tables: list[str]) -> str:
    """Render table descriptions plus column roles and optional column descriptions for prompts."""
    blocks: list[str] = []
    budget = SCHEMA_ENRICHED_LINES_MAX_CHARS
    used = 0
    separator = "\n\n"
    for table in tables:
        table_meta = schema.tables.get(table)
        if not table_meta:
            continue
        lines: list[str] = [f"TABLE {table}"]
        td = _truncate_enriched_description(getattr(table_meta, "description", "") or "")
        if td:
            lines.append(f"  table_description: {td}")
        col_lines: list[str] = []
        for col_name, col_meta in table_meta.columns.items():
            piece = f"{col_name} ({col_meta.data_type or 'unknown'})"
            if col_meta.role:
                piece += f" [{col_meta.role}]"
            cd = _truncate_enriched_description(col_meta.description or "")
            if cd:
                piece += f" — {cd}"
            col_lines.append(piece)
        if col_lines:
            lines.append("  columns:")
            for cl in col_lines:
                lines.append(f"    {cl}")
        block = "\n".join(lines)
        extra = len(separator) if blocks else 0
        if used + extra + len(block) > budget:
            remaining = budget - used - extra
            if remaining <= 0:
                break
            if remaining > 3:
                block = block[: remaining - 3] + "..."
                blocks.append(block)
            break
        blocks.append(block)
        used += extra + len(block)
    return separator.join(blocks)


def _phrase_jaccard_tokens(text: str) -> frozenset[str]:
    """Token set for paraphrase diversity scoring."""
    words: list[str] = []
    buf: list[str] = []
    for ch in normalize_question(text):
        if ch.isalnum():
            buf.append(ch)
        elif buf:
            words.append("".join(buf))
            buf.clear()
    if buf:
        words.append("".join(buf))
    return frozenset(words)


def _phrase_jaccard_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity between two phrase token sets."""
    if not a and not b:
        return 1.0
    union_n = len(a | b)
    if union_n == 0:
        return 0.0
    return len(a & b) / union_n


def select_diverse_paraphrases(candidates: list[str], *, max_count: int, lambda_mmr: float | None = None) -> list[str]:
    """Pick up to *max_count* paraphrases with maximum marginal relevance over token Jaccard similarity."""
    uniq: list[str] = []
    seen: set[str] = set()
    for cand in candidates:
        nn = normalize_question(cand)
        text = str(cand).strip()
        if not nn or not text or nn in seen:
            continue
        seen.add(nn)
        uniq.append(text)
    if len(uniq) <= max_count:
        return uniq
    sigs = {i: _phrase_jaccard_tokens(t) for i, t in enumerate(uniq)}
    remaining = set(range(len(uniq)))
    first = 0
    selected: list[int] = [first]
    remaining.remove(first)
    lam = float(lambda_mmr if lambda_mmr is not None else SeedWarmupConfig.WARMUP_MMR_LAMBDA)
    while remaining and len(selected) < max_count:
        best_i = -1
        best_score = -1e9
        for i in remaining:
            max_sim = 0.0
            for j in selected:
                max_sim = max(max_sim, _phrase_jaccard_similarity(sigs[i], sigs[j]))
            novelty = 1.0 - max_sim
            score = lam * novelty - (1.0 - lam) * max_sim
            if score > best_score:
                best_score = score
                best_i = i
        if best_i < 0:
            best_i = min(remaining)
        selected.append(best_i)
        remaining.remove(best_i)
    return [uniq[i] for i in selected]


def generate_warmup_paraphrases_by_style(
    schema: SchemaGraph, tables: list[str], *, sql: str | None = None, seed_question: str | None = None
) -> dict[str, list[str]] | None:
    """LLM paraphrases grouped by every configured warmup style (up to policy max per style)."""
    if not EngineConfig.llm_credentials_configured():
        return None
    if not sql and not seed_question:
        return None
    styles = SeedWarmupConfig.WARMUP_QUESTION_STYLES
    gu = SeedWarmupConfig.WARMUP_QUESTION_STYLE_GUIDANCE
    per_max = SeedWarmupConfig.WARMUP_PARAPHRASES_PER_STYLE_MAX
    slots = [{"style": s, "guidance": gu.get(s, ""), "max_count": per_max} for s in styles]
    body: dict[str, Any] = {
        "schema": schema_table_descriptions_for_tables(schema, tables),
        "style_slots": slots,
        "output_format": {"paraphrases_by_style": {s: ["string"] for s in styles}},
    }
    if sql:
        body["sql"] = sql
    if seed_question:
        body["seed_question"] = seed_question
    try:
        response = LLMProvider.json(
            WARMUP_PARAPHRASES_BY_STYLE_SYSTEM, stable_json(body), retries=1, task="synth_variety"
        )
    except (TypeError, ValueError, LlmJsonExhausted):
        return None
    raw = response.get("paraphrases_by_style")
    if not isinstance(raw, dict):
        return None
    out: dict[str, list[str]] = {}
    for style in styles:
        vals = raw.get(style)
        phrases: list[str] = []
        if isinstance(vals, list):
            for item in vals:
                if isinstance(item, str) and item.strip():
                    phrases.append(item.strip())
                if len(phrases) >= per_max:
                    break
        out[style] = select_diverse_paraphrases(phrases, max_count=per_max)
    if not any(out.values()):
        return None
    return out


def flatten_warmup_paraphrases_by_style(by_style: dict[str, list[str]]) -> list[str]:
    """Flatten per-style paraphrase buckets into one deduplicated list preserving style order."""
    flat: list[str] = []
    seen: set[str] = set()
    per_max = SeedWarmupConfig.WARMUP_PARAPHRASES_PER_STYLE_MAX
    for style in SeedWarmupConfig.WARMUP_QUESTION_STYLES:
        for phrase in by_style.get(style) or []:
            nn = normalize_question(phrase)
            if not nn or nn in seen:
                continue
            seen.add(nn)
            flat.append(phrase)
            if len([p for p in flat if normalize_question(p)]) >= per_max * len(
                SeedWarmupConfig.WARMUP_QUESTION_STYLES
            ):
                break
    return flat


def generate_paraphrases_of_seed_question(
    seed_question: str, schema: SchemaGraph, tables: list[str], *, style_pair: tuple[str, str] | None = None
) -> list[str] | None:
    """LLM paraphrases of an existing seed question grouped by warmup styles."""
    del style_pair
    by_style = generate_warmup_paraphrases_by_style(schema, tables, seed_question=seed_question)
    if not by_style:
        return None
    return flatten_warmup_paraphrases_by_style(by_style)


def generate_warmup_questions_freeform(
    schema: SchemaGraph, tables: list[str], *, sql: str | None = None, seed_question: str | None = None
) -> list[str] | None:
    """Single-call NL question generation when styled paraphrase buckets are empty."""
    if not EngineConfig.llm_credentials_configured():
        return None
    if not sql and not seed_question:
        return None
    body: dict[str, Any] = {
        "schema": schema_table_descriptions_for_tables(schema, tables),
        "output_format": {"questions": ["string"]},
    }
    if sql:
        body["sql"] = sql
    if seed_question:
        body["seed_question"] = seed_question
    try:
        response = LLMProvider.json(
            WARMUP_FREEFORM_QUESTIONS_SYSTEM, stable_json(body), retries=0, task="synth_variety"
        )
    except LlmJsonExhausted:
        return None
    raw = response.get("questions")
    if not isinstance(raw, list):
        q0 = response.get("question")
        raw = [q0] if isinstance(q0, str) and q0.strip() else []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            continue
        nn = normalize_question(item)
        if not nn or nn in seen:
            continue
        seen.add(nn)
        out.append(item.strip())
        if len(out) >= 3:
            break
    return out or None


def generate_question_from_sql(
    sql: str, schema: SchemaGraph, tables: list[str], *, intent_source: str | None = None
) -> dict[str, Any] | None:
    """LLM: *sql* plus column context yields NL questions and a realism flag."""
    schema_context = schema_table_descriptions_for_tables(schema, tables)
    user_body: dict[str, Any] = {
        "sql": sql,
        "schema": schema_context,
        "output_format": {
            "questions": ["string"],
            "question": "natural language question or empty string",
            "is_realistic": True,
            "drop_reason": None,
            "drop_reason_category": "other",
        },
    }
    user_prompt = stable_json(user_body)

    response = LLMProvider.json(QUESTION_FROM_SQL_SYSTEM, user_prompt, retries=1, task="synth_variety")
    is_realistic = response.get("is_realistic", False)
    question = response.get("question", "")
    drop_reason = response.get("drop_reason")

    if not isinstance(is_realistic, bool):
        is_realistic = str(is_realistic).lower() in ("true", "1", "yes")

    if not is_realistic and intent_source != "sql_history":
        debug(f"[utils.generate_question_from_sql] dropped: reason={drop_reason}")
        raw_cat = response.get("drop_reason_category")
        cat = str(raw_cat).strip() if isinstance(raw_cat, str) else "other"
        if cat not in REALISM_DROP_REASON_CATEGORIES:
            cat = "other"
        return {
            "questions": [],
            "question": "",
            "is_realistic": False,
            "drop_reason": drop_reason or "unrealistic",
            "drop_reason_category": cat,
        }

    if not is_realistic and intent_source == "sql_history":
        debug("[utils.generate_question_from_sql] sql_history provenance; parsing LLM output despite realism=false")

    by_style = generate_warmup_paraphrases_by_style(schema, tables, sql=sql)
    phrases: list[str] = []
    if by_style:
        phrases = flatten_warmup_paraphrases_by_style(by_style)
    if not phrases:
        raw_list = response.get("questions")
        if isinstance(raw_list, list):
            for x in raw_list:
                if isinstance(x, str) and x.strip():
                    phrases.append(x.strip())
        if not phrases:
            q0 = response.get("question", "")
            if isinstance(q0, str) and q0.strip():
                phrases = [q0.strip()]
            elif isinstance(question, str) and question.strip():
                phrases = [question.strip()]
    out_phrases: list[str] = []
    seen_norm: set[str] = set()
    for p in phrases:
        nn = normalize_question(p)
        if not nn or nn in seen_norm:
            continue
        seen_norm.add(nn)
        out_phrases.append(p.strip())
    if not out_phrases:
        debug("[utils.generate_question_from_sql] empty questions after parse")
        return None

    first = out_phrases[0]
    debug(f"[utils.generate_question_from_sql] generated: {first[:60]}")
    return {
        "questions": out_phrases,
        "question": first,
        "is_realistic": True,
        "drop_reason": None,
        "drop_reason_category": None,
        "paraphrases_by_style": by_style or {},
    }


def _merge_intent_param_values(intent: RuntimeIntent) -> dict[str, Any]:
    """Merge CTE and main param value maps with main overriding duplicate keys."""
    merged: dict[str, Any] = {}
    for cte in intent.cte_steps or []:
        merged.update(cte.param_values or {})
    merged.update(intent.param_values or {})
    return merged


def _parse_qualified_column_ref(left_expr: Any) -> tuple[str, str] | None:
    """Return ``(table, column)`` when *left_expr* is a qualified column reference."""
    term = (getattr(left_expr, "column_ref", None) or left_expr.primary_term or "").strip()
    if "." not in term:
        return None
    table, column = term.split(".", 1)
    table = table.strip()
    column = column.strip()
    if not table or not column:
        return None
    return table, column


def _where_equality_literal(where_param: WhereParam, param_values: dict[str, Any]) -> str | None:
    """Return a string literal for an equality filter when one is present."""
    if where_param.op not in ("=", "=="):
        return None
    if where_param.right_expr is not None:
        return None
    value_type = (where_param.value_type or "string").strip().lower()
    if value_type not in ("string", "categorical", "free_text"):
        return None
    if where_param.param_key and where_param.param_key in param_values:
        raw = param_values[where_param.param_key]
    elif where_param.raw_value is not None:
        raw = where_param.raw_value
    else:
        return None
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def column_cached_distinct_values(schema: SchemaGraph, table: str, column: str) -> list[str]:
    """Return cached distinct sample values for a schema column when profiling stored them."""
    table_meta = schema.tables.get(table)
    if table_meta is None:
        return []
    column_meta = table_meta.columns.get(column)
    if column_meta is None:
        return []
    if column_meta.prompt_value_type() not in PROMPT_SCALAR_VALUE_TYPES:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for source in (column_meta.value_overlap_sample or [], column_meta.frequent_values or []):
        for value in source:
            text = str(value).strip()
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            out.append(text)
    return out


def _morph_plural_candidates(token: str) -> list[str]:
    """Return plural-form candidates for a single lowercase token."""
    candidates: list[str] = []
    if token.endswith("y") and len(token) > 1 and token[-2] not in "aeiou":
        candidates.append(token[:-1] + "ies")
    if token.endswith("um"):
        candidates.append(token[:-2] + "a")
    if token.endswith("us"):
        candidates.append(token[:-2] + "i")
    candidates.append(token + "s")
    candidates.append(token + "es")
    if token.endswith(("s", "x", "z")) or token.endswith("ch") or token.endswith("sh"):
        extra = token + "es"
        if extra not in candidates:
            candidates.append(extra)
    return [candidate for candidate in candidates if candidate != token]


def _morph_singular_candidates(token: str) -> list[str]:
    """Return singular-form candidates for a single lowercase token."""
    candidates: list[str] = []
    if token.endswith("ies") and len(token) > 3:
        candidates.append(token[:-3] + "y")
    if token.endswith("a") and len(token) > 1:
        candidates.append(token[:-1] + "um")
    if token.endswith("i") and len(token) > 1:
        candidates.append(token[:-1] + "us")
    if token.endswith("es") and len(token) > 2:
        candidates.append(token[:-2])
    if token.endswith("s") and len(token) > 1:
        candidates.append(token[:-1])
    return [candidate for candidate in candidates if candidate != token]


def morph_variants(token: str) -> list[str]:
    """Return morphological variants for a single token including the original spelling."""
    base = token.strip()
    if not base:
        return []
    lower = base.lower()
    seen: set[str] = {lower}
    ordered: list[str] = [base]
    for candidate in _morph_plural_candidates(lower) + _morph_singular_candidates(lower):
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return ordered


def _cache_canonical_index(cached: list[str]) -> dict[str, str]:
    """Map lowercase cached values to their canonical profiling spellings."""
    return build_case_folded_index(cached, kind="cached value")


def zero_row_where_remediation_candidates(literal: str, cached: list[str]) -> list[str]:
    """Build ordered filter literal candidates from cached distinct values for zero-row remediation."""
    text = literal.strip()
    if not text or not cached:
        return []
    canonical = _cache_canonical_index(cached)
    cache_keys = set(canonical)
    original_key = text.lower()
    seen_keys: set[str] = {original_key}
    candidates: list[str] = []

    def add_candidate(raw: str) -> None:
        key = raw.lower()
        if key in seen_keys or key not in cache_keys:
            return
        seen_keys.add(key)
        candidates.append(canonical[key])

    if " " in text:
        add_candidate(text.replace(" ", "_"))
    if "_" in text:
        add_candidate(text.replace("_", " "))

    tokens = re.split(r"[ _]+", text)
    if len(tokens) >= 2:
        variant_lists = [morph_variants(token) for token in tokens]
        for join_char in (" ", "_"):
            for combo in itertools.product(*variant_lists):
                add_candidate(join_char.join(combo))

    if " " not in text:
        for variant in morph_variants(text):
            add_candidate(variant)

    return candidates


def _where_param_matches(left: WhereParam, right: WhereParam) -> bool:
    """Return True when two filter params refer to the same equality slot."""
    if left.param_key and right.param_key and left.param_key == right.param_key:
        return True
    left_ref = _parse_qualified_column_ref(left.left_expr)
    right_ref = _parse_qualified_column_ref(right.left_expr)
    return left_ref is not None and left_ref == right_ref


def patch_where_literal_on_intent(
    intent: RuntimeIntent, where_param: WhereParam, canonical_value: str
) -> RuntimeIntent:
    """Return a deep copy of *intent* with one equality filter literal replaced."""
    patched = copy.deepcopy(intent)
    for candidate in PredicateGroup.where_leaves(patched.where) or []:
        if not _where_param_matches(candidate, where_param):
            continue
        candidate.raw_value = canonical_value
        if candidate.param_key:
            patched.param_values[candidate.param_key] = canonical_value
        break
    return patched


def enumerate_zero_row_equality_where(
    intent: RuntimeIntent, schema: SchemaGraph
) -> list[tuple[WhereParam, str, str, list[str]]]:
    """List equality filters whose literals are absent from cached distinct values."""
    param_values = _merge_intent_param_values(intent)
    targets: list[tuple[WhereParam, str, str, list[str]]] = []
    for where_param in PredicateGroup.where_leaves(intent.where) or []:
        literal = _where_equality_literal(where_param, param_values)
        if literal is None:
            continue
        ref = _parse_qualified_column_ref(where_param.left_expr)
        if ref is None:
            continue
        table, column = ref
        cached = column_cached_distinct_values(schema, table, column)
        if not cached:
            continue
        if literal.lower() in {value.lower() for value in cached}:
            continue
        targets.append((where_param, literal, column, cached))
    return targets


def zero_row_where_suggestions(intent: RuntimeIntent, schema: SchemaGraph) -> list[str]:
    """Suggest cached distinct-value corrections for equality filters after a zero-row execute."""
    suggestions: list[str] = []
    for _where_param, literal, column, cached in enumerate_zero_row_equality_where(intent, schema):
        best: str | None = None
        best_distance: int | None = None
        for value in cached:
            distance = _levenshtein_distance(literal.lower(), value.lower())
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best = value
        if best is None or best_distance is None:
            continue
        if best_distance > PolicyConfig.ZERO_ROW_WHERE_FUZZY_MAX_DISTANCE:
            continue
        suggestions.append(f"No rows for {column}={literal!r}. Did you mean {best!r}?")
    return suggestions
