"""Intent fingerprinting, fuzzy question match, CTE/filter normalisation, and NL question helpers."""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, replace
from typing import Any

from ._config import (
    AGG_PATTERN,
    DO_NOT_LEMMATIZE,
    IRREGULAR_PLURALS_MAP,
    NORMALIZATION_JACCARD_FLOOR,
    QUESTION_STARTS_AGG,
    QUESTION_STARTS_GROUP,
    QUESTION_STARTS_LIST,
    REALISM_DROP_REASON_CATEGORIES,
    SHAPE_FORM_DATE_REGEX,
    SHAPE_FORM_NUM_REGEX,
    SHAPE_FORM_STR_REGEX,
    VALID_AGGREGATION_FUNCTIONS,
    VALID_EXPECTED_ROWS,
    VALID_FILTER_OPS,
    VALID_GRAINS,
    VALID_HAVING_OPS,
    EngineConfig,
    PolicyConfig,
    SeedWarmupConfig,
    STOPWORDS_GRAMMATICAL_PARTICLES,
)
from ._contracts_base import (
    CteOutputColumnMeta,
    LlmJsonExhausted,
    RuntimeConfig,
    SchemaGraph,
    SQLShape,
    SurfaceTemplateSpec,
)
from ._contracts_core import (
    ConcreteIntent,
    FilterParam,
    HavingParam,
    NormalizedExpr,
    OrderByCol,
    RuntimeCteStep,
    RuntimeIntent,
    SelectCol,
    Template,
    concrete_intent_to_runtime_skeleton,
)
from ._core_utils import (
    debug,
    llm_json,
    normalize_array_contains_param_value,
    normalize_question,
    notify,
    safe_json_loads,
    sha256,
    stable_json,
)
from ._dialect import (
    sql_count_outer_joins,
    sql_has_aggregate,
    sql_has_distinct,
    sql_has_group_by,
    sql_tables_referenced,
)
from ._intent_resolve import sort_filters, sort_having

NORMALIZATION_ALLOWED_INTRODUCED_TOKENS: frozenset[str] = frozenset(
    {"list", "count", "sum", "average", "max", "min", "total", "of"},
)
REALISM_CATEGORY_LIST: str = ", ".join(sorted(REALISM_DROP_REASON_CATEGORIES))

QUESTION_NORMALIZE_VOCABULARY_HEADING: str = (
    "Vocabulary preferences (apply only when context fits; preserve logical negation and constraints):"
)
QUESTION_NORMALIZE_VOCABULARY_GUIDANCE: str = (
    "Verb phrasing: prefer concise analytic wording over vague conversational fillers whenever the question intent is unchanged. "
    "Aggregation: align stated aggregation language with the grain implied by the question. "
    "Temporal: when any time scope appears, state it explicitly enough for deterministic routing. "
    "Negation: preserve every explicit negator on the predicate it scopes; do not drop or soften negation."
)

_QUESTION_VALIDATION_SYSTEM = (
    "You decide if user input is a database query request or not.\n\n"
    "Treat the input as a VALID database query request whenever a reasonable relational database could store rows that answer it.\n"
    "That includes list, show, get, find, count, sum, average, min, max, filter, sort, group, compare, rank, top-N, trend, or per-entity questions, including bounded counts such as listing two named entities.\n"
    "When the utterance is ambiguous but still plausibly data-seeking, choose VALID.\n\n"
    "Mark as INVALID only when it is clearly one of the following:\n"
    '- Chitchat or meta conversation (e.g. "hello", "thanks", "who are you")\n'
    '- A request for SQL tutoring, query help, or how-to without asking for actual rows (e.g. "how do I write a join")\n'
    '- General world knowledge or opinion with no plausible tabular backing (e.g. "does the Earth orbit the Sun")\n\n'
    "The label restricted applies only when the user asks for a DML, DDL, or administrative database operation. "
    "DML covers any data mutation (delete, update, insert, merge, truncate, copy). "
    "DDL covers schema mutation (create, drop, alter, rename). "
    "Administrative covers privilege management, indexing, vacuuming, configuration, and any other non-analytical operation. "
    "Analytical questions never receive restricted, including questions that describe their solution using analytical primitives such as CTEs, subqueries, joins, aggregations, window functions, distinct, recursion, or set operations. "
    "Use of the literal words CTE, subquery, with, join, group, order, window, partition, recursive, or similar terms in the question never alone implies restricted.\n\n"
    "Respond with JSON containing exactly three fields:\n"
    '- "valid_database_question": "yes" or "no"\n'
    '- "query_type": "allowed" if read/SELECT operation, "restricted" if write or schema-modifying, "unspecified" if unclear.\n'
    '- "corrected": the input with spelling typos fixed only. Do NOT remove, reorder, or rephrase any words.\n\n'
    "Respond ONLY with valid JSON, no explanation."
)

_QUESTION_NORMALIZE_SYSTEM = (
    "You rewrite a typo-corrected database query into a canonical short query so that semantically identical "
    "questions hash to the same string.\n\n"
    "When the user message is JSON, field ``question`` carries the rewrite target; optional ``normalization_preferences`` "
    "is advisory context only.\n\n"
    "Apply these rules IN ORDER:\n"
    "0. Before any other rewrite, normalize quantifier and aggregation openers: map phrases such as "
    '"how many", "number of", and bare "count" asking for cardinality to the two-token prefix "count of"; '
    'map "total of", "totals for", and bare "sum" used as an aggregation opener to "sum of"; '
    'map "average of", "mean of", and bare "avg" used as an aggregation opener to "avg of"; '
    'map "maximum of", "largest", "highest", "max" used as an aggregation opener to "max of"; '
    'map "minimum of", "smallest", "lowest", "min" used as an aggregation opener to "min of". '
    "Preserve trailing nouns and filters after those prefixes.\n"
    '1. Replace any verb phrase whose only purpose is to ask for non-aggregated rows with the single token "list"; '
    "do not replace the aggregation prefixes introduced in rule 0, and do not replace aggregation verbs "
    "such as count, sum, average, max, min, or total when they already head a normalized aggregation phrase.\n"
    "2. Drop polite or filler clauses that do not carry analytical meaning.\n"
    "3. Replace plural common nouns with their singular base form. Do NOT singularize verbs.\n"
    "4. Preserve every number, date, quoted literal, comparison word, adjective, named entity, "
    "and any preposition immediately before a number/date/literal.\n"
    "5. Preserve original word order.\n"
    "6. If no rule applies, return the input unchanged.\n"
    "7. Never add a word that did not appear in the input (including any inflected form already present).\n\n"
    "Examples of rule 0 (JSON only illustrates the normalized field):\n"
    '{"question":"how many films are in the action category","normalized":"count of film in the action category"}\n'
    '{"question":"total payments last month by store","normalized":"sum of payment last month by store"}\n\n'
    "Respond with JSON containing exactly one field:\n"
    '- "normalized": the rewritten canonical short query.\n\n'
    "Respond ONLY with valid JSON, no explanation."
)


def validate_question(question: str) -> tuple[bool, str, str]:
    """
    LLM gate: data question vs chitchat; typo fix; allowed vs restricted.

    Args:

        question: Raw user text.

    Returns:

        ``(ok, kind, corrected)``: ``ok`` True only for allowed reads; ``kind`` is ``allowed``, ``restricted``, or ``invalid``.
    """
    try:
        result = llm_json(_QUESTION_VALIDATION_SYSTEM, question, task="default")
    except LlmJsonExhausted as exc:
        debug(f"[utils.validate_question] llm_json exhausted: {exc}")
        return False, "invalid", question
    query_type = str(result.get("query_type", "unspecified") or "").strip().lower()
    valid = str(result.get("valid_database_question", "") or "").strip().lower() == "yes"
    corrected = result.get("corrected", question) or question
    if query_type == "restricted":
        return False, "restricted", corrected
    if not valid:
        return False, "invalid", corrected
    return True, "allowed", corrected


def _suffix_lemmatize_token(token: str) -> str:
    """
    Apply conservative English plural-to-singular heuristics when the token is not in ``DO_NOT_LEMMATIZE``.
    """

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


def compute_shape_form(question: str) -> str:
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
            sf = compute_shape_form(str(hq))
            buckets.setdefault(sf, set()).add(tpl.id)
    return {sf: sorted(ids) for sf, ids in buckets.items()}


def _enforce_normalization_guard(corrected: str, normalized: str, *, raw_original: str) -> tuple[bool, str]:
    """
    Validate LLM-normalized text against *corrected* and reject unsafe expansions.

    Args:

        corrected: Typo-corrected user question.

        normalized: Candidate canonical question string.

        raw_original: Original casing slice used for capital-token preservation checks.

    Returns:

        ``(accept, reason_code)`` where *accept* is False when the normalized form must be discarded.
    """

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
    """
    Canonicalize *corrected* via a dedicated LLM call separate from typo validation.

    Args:

        corrected: Typo-corrected question text.

        raw_original: Original user question casing reference for guard checks; defaults to *corrected*.

    Returns:

        Canonical question string, or *corrected* when the model output fails validation.
    """

    raw_use = raw_original if raw_original is not None else corrected
    vocab_block = QUESTION_NORMALIZE_VOCABULARY_HEADING + "\n" + QUESTION_NORMALIZE_VOCABULARY_GUIDANCE
    user_obj: dict[str, Any] = {
        "question": corrected,
        "normalization_preferences": vocab_block,
    }
    try:
        result = llm_json(_QUESTION_NORMALIZE_SYSTEM, stable_json(user_obj), task="default")
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
    """
    Count joins, CTEs, filters, having, and structural flags from *sql* and *intent* via AST.

    Args:

        sql: Generated SQL.

        intent: Intent (main + CTE filter/having lists).

        sqlglot_dialect: sqlglot dialect token (``"postgres"`` or ``"spark"``).

    Returns:

        ``SQLShape`` with structural counts and booleans.
    """
    num_filters = len(intent.filters_param or [])
    num_having = len(intent.having_param or [])
    for cte in intent.cte_steps or []:
        num_filters += len(cte.filters_param or [])
        num_having += len(cte.having_param or [])
    return SQLShape(
        num_joins=sql_count_outer_joins(sql, sqlglot_dialect=sqlglot_dialect),
        has_group_by=sql_has_group_by(sql, sqlglot_dialect=sqlglot_dialect),
        has_agg=sql_has_aggregate(sql, sqlglot_dialect=sqlglot_dialect),
        num_cte=len(intent.cte_steps or []),
        num_filters=num_filters,
        num_having=num_having,
        has_distinct=sql_has_distinct(sql, sqlglot_dialect=sqlglot_dialect),
    )


def _levenshtein_distance(s1: str, s2: str) -> int:
    """
    Levenshtein edit distance between *s1* and *s2*.

    Args:

        s1: First string.

        s2: Second string.

    Returns:

        Minimum insert/delete/replace steps.
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
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
    """
    Lowercase ``[a-z0-9_]+`` tokens from *q*, suffix-lemmatized except for ``DO_NOT_LEMMATIZE``, stripped of ``PolicyConfig.STOPWORDS``, then sorted for multiset comparison.

    Args:

        q: Question text (typically already passed through ``normalize_question``).

    Returns:

        Sorted token list (repeated content words appear repeatedly).
    """

    out: list[str] = []
    for raw in re.findall(r"[a-z0-9_]+", q.lower()):
        if not raw:
            continue
        irr = IRREGULAR_PLURALS_MAP.get(raw, raw)
        step = _suffix_lemmatize_token(irr)
        if step in PolicyConfig.STOPWORDS:
            continue
        out.append(step)
    return sorted(out)


def question_token_fingerprint_from_normalized(norm: str) -> str:
    """Sorted multiset fingerprint of stopword-stripped question tokens (see :func:`_tokenize`)."""

    return "\0".join(_tokenize(norm))


def question_token_fingerprint_from_raw(raw: str) -> str:
    """Fingerprint for a raw question string after :func:`aetherdialect._core_utils.normalize_question`."""

    return question_token_fingerprint_from_normalized(normalize_question(raw))


def neighboring_question_token_fingerprint_norms(norm: str) -> frozenset[str]:
    """
    Fingerprints for inverted-index lookup: the exact multiset plus neighbors from one in-token substitution.

    Bounded by :data:`PolicyConfig.QUESTION_TOKEN_INDEX_NEIGHBOR_CAP` for determinism.
    """

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
    q1_norm: str,
    q2_norm: str,
    max_distance: int,
    debug_label: str,
) -> tuple[bool, int]:
    """
    Return whether stopword-stripped token lists align with summed per-token edit distance within *max_distance*.

    Args:

        q1_norm: First question already passed through ``normalize_question``.

        q2_norm: Second question already passed through ``normalize_question``.

        max_distance: Maximum summed Levenshtein distance across aligned token pairs.

        debug_label: Suffix for ``debug`` log lines (may be empty).

    Returns:

        ``(matched, total_edit_distance)`` where *total_edit_distance* is defined only when lengths match.
    """

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


@dataclass(frozen=True, slots=True)
class QuestionReuseMatch:
    """Trusted template history row selected by fuzzy token match against a candidate question."""

    template_id: str
    history_index: int
    stored_normalized_text: str
    candidate_normalized: str
    token_edit_sum: int

    @property
    def is_exact_string_reuse(self) -> bool:
        """True when normalized candidate text equals the stored normalized history string (generation path 1)."""

        return self.candidate_normalized == self.stored_normalized_text


def match_question_against_template_history(
    candidate_raw: str,
    templates: list[Template],
    *,
    max_token_edit_distance: int | None = None,
    shape_question_index: dict[str, list[str]] | None = None,
    question_token_index: dict[str, list[Any]] | None = None,
) -> QuestionReuseMatch | None:
    """
    Find the best trusted template whose stored question fuzzy-matches the candidate.

    Normalizes the candidate once, optionally narrows templates via a shape-form index, then scores every
    trusted history row using token edit distance with lexicographic ties broken by per-row accepts,
    template accepts, and template id.

    Args:

        candidate_raw: User or corrected question text.

        templates: Templates to scan in list order; entries with ``trust_level < 1`` are skipped.

        max_token_edit_distance: Per-call override for summed token edit budget; defaults to policy.

        shape_question_index: Optional coarse map from ``compute_shape_form`` to candidate template ids.

        question_token_index: Optional inverted index from :func:`question_token_fingerprint_from_raw` keys to
            ``[template_id, history_index]`` rows.

    Returns:

        ``QuestionReuseMatch`` for the best hit, or ``None``.
    """

    budget = max_token_edit_distance if max_token_edit_distance is not None else PolicyConfig.FUZZY_MATCH_MAX_DISTANCE
    candidate_normalized = normalize_question(candidate_raw)
    scan_templates = templates
    if shape_question_index:
        cand_sf = compute_shape_form(candidate_raw)
        allowed_ids = shape_question_index.get(cand_sf)
        if allowed_ids:
            allow_set = frozenset(allowed_ids)
            narrowed = [t for t in templates if t.id in allow_set]
            if narrowed:
                scan_templates = narrowed
    pair_filter: set[tuple[str, int]] | None = None
    if question_token_index:
        cand_fps = neighboring_question_token_fingerprint_norms(candidate_normalized)
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
        for idx, hist_q in enumerate(tpl.value_history.questions):
            if not hist_q:
                continue
            if pair_filter is not None and (tpl.id, idx) not in pair_filter:
                continue
            stored_normalized = normalize_question(hist_q)
            ok, total = _fuzzy_question_tokens_match_pair(
                candidate_normalized,
                stored_normalized,
                budget,
                tpl.id,
            )
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
    q1: str,
    q2: str,
    max_distance: int = PolicyConfig.FUZZY_MATCH_MAX_DISTANCE,
    label: str = "",
) -> bool:
    """
    True if normalised token sequences match 1:1 with summed edit distance ≤ *max_distance*.

    Args:

        q1: First question.

        q2: Second question.

        max_distance: Max sum of per-token Levenshtein distances.

        label: Optional tag for ``debug`` logs.

    Returns:

        True when token counts match and total distance is within budget.
    """

    q1_norm = normalize_question(q1)
    q2_norm = normalize_question(q2)
    ok, _ = _fuzzy_question_tokens_match_pair(q1_norm, q2_norm, max_distance, label)
    return ok


def is_exact_question_text_match(q1: str, q2: str) -> bool:
    """Return True when *q1* and *q2* share the same normalised question string."""

    return normalize_question(q1) == normalize_question(q2)


def _normalize_filters(filters: list[Any]) -> list[FilterParam]:
    """
    Coerce dicts / ``FilterParam`` to ``FilterParam`` and ``sort_filters``.

    Args:

        filters: LLM dicts or instances.

    Returns:

        Sorted list for stable keys.
    """
    if not filters:
        return []
    out = []
    for f in filters:
        if isinstance(f, FilterParam):
            left_expr = f.left_expr
            op = f.op.strip().lower() if f.op else "="
            vtype = f.value_type.strip().lower() if isinstance(f.value_type, str) else "unknown"
            right_expr = f.right_expr
            bool_op = f.bool_op
            filter_group = f.filter_group
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
            bool_op = f.get("bool_op", "AND")
            fg_raw = f.get("filter_group")
            filter_group = int(fg_raw) if fg_raw is not None else None
        else:
            continue
        if not left_expr or not isinstance(op, str):
            continue
        fp = FilterParam(
            left_expr=left_expr,
            op=op,
            value_type=vtype,
            param_key="",
            right_expr=right_expr,
            bool_op=bool_op,
            filter_group=filter_group,
        )
        out.append(fp)
    return sort_filters(out)


def _normalize_having_conditions(conditions: list[Any]) -> list[HavingParam]:
    """
    Coerce dicts / ``HavingParam``; clamp ops to ``VALID_HAVING_OPS``; ``sort_having``.

    Args:

        conditions: LLM dicts or instances.

    Returns:

        Sorted list for stable keys.
    """
    if not conditions:
        return []
    out = []
    for c in conditions:
        if isinstance(c, HavingParam):
            left_expr = c.left_expr
            op = c.op.strip().lower() if c.op else "="
            value_type = c.value_type.strip().lower() if c.value_type else "number"
            right_expr = c.right_expr
            bool_op = c.bool_op
            filter_group = c.filter_group
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
            bool_op = c.get("bool_op", "AND")
            fg_raw = c.get("filter_group")
            filter_group = int(fg_raw) if fg_raw is not None else None
        else:
            continue
        if not left_expr or not left_expr.primary_term:
            continue
        op_norm = op.strip().lower() if isinstance(op, str) else "="
        if op_norm not in VALID_HAVING_OPS:
            op_norm = "="
        hp = HavingParam(
            left_expr=left_expr,
            op=op_norm,
            value_type=str(value_type),
            param_key="",
            right_expr=right_expr,
            bool_op=bool_op,
            filter_group=filter_group,
        )
        out.append(hp)
    return sort_having(out)


def _normalize_cte_steps(steps: Any, available_ctes: dict[str, list[str]] | None = None) -> list[RuntimeCteStep]:
    """
    Parse CTE steps from dicts or models; infer ``column_map`` and output metadata.

    Args:

        steps: List of steps, or non-list → ``[]``.

        available_ctes: CTE name → output columns; mutated as steps are appended.

    Returns:

        ``RuntimeCteStep`` list in input order.
    """
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
            filters_param = s.filters_param or []
            having_param = s.having_param or []
            param_values = s.param_values or {}
            order_by_cols = s.order_by_cols or []
            limit = s.limit
            column_map = s.column_map or {}
            output_column_metadata = s.output_column_metadata or {}
            chosen_join_candidate_id = s.chosen_join_candidate_id or ""
            chosen_join_path_signature = s.chosen_join_path_signature or []
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
            fp_raw = s.get("filters_param", [])
            filters_param = [FilterParam.from_dict(f) if isinstance(f, dict) else f for f in fp_raw]
            hp_raw = s.get("having_param", [])
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
        else:
            continue
        if not cte_name:
            continue
        if grain not in VALID_GRAINS:
            grain = "row_level"
        normalized_fp = []
        for f in filters_param:
            if isinstance(f, FilterParam):
                op = f.op.strip().lower() if f.op else "="
                if op not in VALID_FILTER_OPS:
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
            elif isinstance(sc, str):
                all_cols_raw.append(sc)
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
            else ([SelectCol(expr=NormalizedExpr.from_column(c)) for c in select_cols] if select_cols else [])
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
            filters_param=sorted(
                normalized_fp,
                key=lambda x: (
                    x.filter_group if x.filter_group is not None else -1,
                    x.left_expr.signature_key,
                    x.op,
                    x.right_expr.signature_key if x.right_expr else "",
                    x.value_type,
                ),
            ),
            having_param=sorted(
                normalized_hp,
                key=lambda x: (
                    x.filter_group if x.filter_group is not None else -1,
                    x.left_expr.signature_key,
                    x.op,
                    x.right_expr.signature_key if x.right_expr else "",
                    x.value_type,
                ),
            ),
            param_values=param_values,
            order_by_cols=normalized_order_by,
            limit=limit,
            column_map=cte_column_map,
            output_column_metadata=ocm,
            chosen_join_candidate_id=chosen_join_candidate_id,
            chosen_join_path_signature=chosen_join_path_signature,
        )
        out.append(cte)
        available_ctes[cte_name] = output_columns
    return out


def _normalize_cte_steps_for_key(steps: list[RuntimeCteStep]) -> list[dict[str, Any]]:
    """
    Projection of CTE steps to signature strings for ``intent_key`` JSON.

    Args:

        steps: Normalised ``RuntimeCteStep`` instances.

    Returns:

        List of plain dicts safe for ``stable_json``.
    """
    result = []
    for cte in steps:
        select_sigs = []
        for sc in cte.select_cols or []:
            if isinstance(sc, SelectCol):
                select_sigs.append(sc.signature_key)
            elif isinstance(sc, str):
                select_sigs.append(sc)
        order_sigs = []
        for obc in cte.order_by_cols or []:
            if isinstance(obc, OrderByCol):
                order_sigs.append(obc.signature_key)
            elif isinstance(obc, str):
                order_sigs.append(obc)
        cte_dict = {
            "cte_name": cte.cte_name,
            "tables": sorted(cte.tables or []),
            "select_cols": sorted(select_sigs),
            "group_by_cols": sorted([g.signature_key for g in (cte.group_by_cols or [])]),
            "output_columns": sorted(cte.output_columns or []),
            "filters_param": [
                f"{f.signature_key}|{'AND' if f.filter_group is not None else f.bool_op}|{f.filter_group}"
                for f in sorted(
                    cte.filters_param or [],
                    key=lambda x: (
                        x.filter_group if x.filter_group is not None else -1,
                        x.left_expr.signature_key,
                        x.op,
                        x.right_expr.signature_key if x.right_expr else "",
                        x.value_type,
                    ),
                )
            ],
            "having_param": [
                f"{h.signature_key}|{'AND' if h.filter_group is not None else h.bool_op}|{h.filter_group}"
                for h in sorted(
                    cte.having_param or [],
                    key=lambda x: (
                        x.filter_group if x.filter_group is not None else -1,
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
        result.append(cte_dict)
    return result


def _contains_filter_param_keys(intent: RuntimeIntent) -> set[str]:
    keys: set[str] = set()
    for cte in intent.cte_steps or []:
        for fp in cte.filters_param or []:
            if fp.op == "contains" and fp.param_key and fp.right_expr is None:
                keys.add(fp.param_key)
    for fp in intent.filters_param or []:
        if fp.op == "contains" and fp.param_key and fp.right_expr is None:
            keys.add(fp.param_key)
    return keys


def flatten_param_values(intent: RuntimeIntent) -> dict[str, Any]:
    """
    Merge CTE ``param_values`` then main; main overrides duplicate keys.

    Applies:func:`core_utils.normalize_array_contains_param_value` to keys for ``filters_param`` rows with ``op == "contains"``. Dialect SQL for those filters also normalizes stored array elements at execution time.

    Args:

        intent: Runtime intent with optional CTE steps.

    Returns:

        Single map for execution/substitution.
    """
    merged = {}
    for cte in intent.cte_steps or []:
        merged.update(cte.param_values or {})
    merged.update(intent.param_values or {})
    ckeys = _contains_filter_param_keys(intent)
    if not ckeys:
        return merged
    out = dict(merged)
    for k in ckeys:
        if k in out:
            out[k] = normalize_array_contains_param_value(out[k])
    return out


def intent_key(intent: RuntimeIntent) -> str:
    """
    SHA-256 of normalised structural intent: tables, selects, filters, group/order/having, CTEs.

    Omits ``grain``, ``limit``, and raw param values; uses normalised filter/having dicts and CTE key skeletons. Differs from :func:`aetherdialect._intent_process.intent_similarity`, which scores overlap via weighted clause similarity (including a separate CTE blend) rather than a single hash.

    Args:

        intent: Intent to fingerprint.

    Returns:

        64-character hex digest.
    """
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

    filters_normalized = _normalize_filters(intent.filters_param or [])
    having_conditions_normalized = _normalize_having_conditions(intent.having_param or [])
    cte_steps_normalized = _normalize_cte_steps(intent.cte_steps or [])
    cte_steps_for_key = _normalize_cte_steps_for_key(cte_steps_normalized)

    select_cols_sorted = sorted([s.signature_key for s in select_cols])
    order_by_sorted = sorted([o.signature_key for o in (intent.order_by_cols or [])])

    normalized = {
        "tables": sorted(intent.tables or []),
        "select_cols": select_cols_sorted,
        "filters": [f.to_dict() for f in filters_normalized],
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


def body_similarity_key(intent: RuntimeIntent) -> str:
    """Structural body fingerprint excluding grain, limit, column_map, join path, and param values."""

    return intent_key(intent)


def body_similarity_key_for_concrete(concrete: ConcreteIntent) -> str:
    """``body_similarity_key`` for a stored ``ConcreteIntent``."""

    return body_similarity_key(concrete_intent_to_runtime_skeleton(concrete))


def template_instance_key_from_parts(body_key: str, join_fp: str, sql_fp_val: str) -> str:
    """Stable key for an executable template row: body + join fingerprint + parameterized SQL fingerprint."""

    return sha256(stable_json({"b": body_key, "j": join_fp, "s": sql_fp_val}))


def extract_tables_from_sql(sql: str, known_tables: list[str], *, sqlglot_dialect: str) -> list[str]:
    """
    Subset of *known_tables* referenced as physical tables in *sql* via AST inspection.

    CTE definition names are excluded by :func:`aetherdialect._dialect.sql_tables_referenced`.

    Args:

        sql: SQL text.

        known_tables: Candidate physical table names.

        sqlglot_dialect: sqlglot dialect token (``"postgres"`` or ``"spark"``).

    Returns:

        Sorted unique hits.
    """
    referenced = sql_tables_referenced(sql, sqlglot_dialect=sqlglot_dialect)
    hits = [t for t in known_tables if t.lower() in referenced]
    return sorted(set(hits))


def _describe_operation(select_terms: list[str]) -> str:
    """
    First ``AGG_PATTERN`` aggregation name in *select_terms*, else ``list``.

    Args:

        select_terms: SELECT list item strings.

    Returns:

        Lowercase agg name or ``"list"``.
    """
    for sc in select_terms:
        m = AGG_PATTERN.match(sc)
        if m:
            return m.group(1).lower()
    return "list"


def _pick_question_style(select_terms: list[str], has_grouping: bool) -> str:
    """
    Random NL question opener from structure (group / agg / list).

    Args:

        select_terms: SELECT strings for agg detection.

        has_grouping: Whether GROUP BY is present.

    Returns:

        Phrase the generated question must start with (e.g. ``How many``).
    """
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
    """
    LLM: intent JSON → one sentence; must start with chosen style prefix.

    Args:

        tables: Intent tables.

        select_terms: SELECT expression strings.

        filter_descriptions: ``column`` / ``condition`` entries.

        group_by_terms: GROUP BY strings.

        having_descriptions: HAVING ``column`` / ``condition`` entries.

        schema: For semantic context in the prompt.

    Returns:

        Question text, or ``None`` if LLM fails or prefix check fails.
    """
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

    selected_style = _pick_question_style(select_terms, bool(group_by_terms))

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

    response = llm_json(system, stable_json(user_prompt), retries=1, task="default")
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


_QUESTION_FROM_SQL_SYSTEM = (
    "You are given a SQL query and a schema description. "
    "Your job is to decide whether the query represents a realistic, "
    "meaningful business question and, if so, produce natural-language paraphrases "
    "that a human analyst would ask to obtain this query's result.\n\n"
    "Rules:\n"
    "- If the query is unrealistic, nonsensical, or produces meaningless "
    "results, set is_realistic to false and explain why in drop_reason.\n"
    "- If realistic, set questions to an array of up to three distinct, "
    "conversational paraphrases a non-technical user would ask. "
    "Do NOT use SQL jargon or raw column names — use natural business language.\n"
    "- Do not phrase the output as numbered steps, subqueries, JOIN recipes, or procedural SQL instructions; "
    "each entry must read as one coherent analyst question.\n"
    "- You may also set question (string) to the first paraphrase for compatibility; "
    "when questions is non-empty, question should match questions[0].\n"
    "- Output ONLY valid JSON with fields: "
    '"questions" (array of strings), "question" (string, optional legacy), '
    '"is_realistic" (boolean), "drop_reason" (string or null), and optionally '
    '"drop_reason_category" (string) when is_realistic is false. '
    f"If present, drop_reason_category must be one of: {REALISM_CATEGORY_LIST}.\n"
)

_QUESTION_FROM_SQL_SYSTEM_WARMUP_STYLED = (
    _QUESTION_FROM_SQL_SYSTEM + "\n"
    "- When style_slots is present in the input JSON, produce exactly three strings in "
    "questions[0], questions[1], and questions[2] aligned to each slot in order; "
    "question must equal questions[0]. Each question follows its slot style and guidance.\n"
)

_PARAPHRASE_SEED_QUESTION_SYSTEM = (
    "You rephrase one analyst question into alternative natural-language wordings. "
    "Preserve entities, filters, metrics, grouping, ordering, and limits implied by the seed text. "
    "Use schema descriptions only for terminology consistency. "
    "Do not answer the question. "
    "Do not output SQL, identifiers, numbered steps, or JOIN recipes.\n\n"
    "Output ONLY valid JSON with fields questions (array of strings), "
    "length equal to style_slots, each entry matching the corresponding slot style and guidance."
)


def schema_context_enriched_lines_for_tables(schema: SchemaGraph, tables: list[str]) -> str:
    """
    Render table descriptions plus column roles and optional column descriptions for prompts.

    Args:

        schema: Live schema graph.

        tables: Table names to include in document order.

    Returns:

        Multi-table text block without sample values.
    """

    blocks: list[str] = []
    for table in tables:
        table_meta = schema.tables.get(table)
        if not table_meta:
            continue
        lines: list[str] = [f"TABLE {table}"]
        td = getattr(table_meta, "description", "") or ""
        if str(td).strip():
            lines.append(f"  table_description: {str(td).strip()}")
        col_lines: list[str] = []
        for col_name, col_meta in table_meta.columns.items():
            piece = f"{col_name} ({col_meta.data_type or 'unknown'})"
            if col_meta.role:
                piece += f" [{col_meta.role}]"
            cd = col_meta.description or ""
            if str(cd).strip():
                piece += f" — {str(cd).strip()}"
            col_lines.append(piece)
        if col_lines:
            lines.append("  columns:")
            for cl in col_lines:
                lines.append(f"    {cl}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def select_three_warmup_styles(seed_index: int, intent_id: str) -> tuple[str, str, str]:
    """
    Deterministically pick three distinct warmup question styles for synthetic NL diversity.

    Args:

        seed_index: Expansion batch row index or zero when absent.

        intent_id: Stable intent identifier mixed into the digest.

    Returns:

        Three entries drawn from :attr:`SeedWarmupConfig.WARMUP_QUESTION_STYLES` without replacement.
    """

    styles = SeedWarmupConfig.WARMUP_QUESTION_STYLES
    digest = sha256(
        f"warmup_styles:{SeedWarmupConfig.WARMUP_SAMPLING_POLICY_VERSION}:{seed_index}:{intent_id}",
    )
    used: set[int] = set()
    chosen: list[str] = []
    for i in range(0, len(digest) - 1, 2):
        if len(chosen) >= 3:
            break
        slot = int(digest[i : i + 2], 16) % len(styles)
        if slot in used:
            continue
        used.add(slot)
        chosen.append(styles[slot])
    k = 0
    while len(chosen) < 3:
        while k in used:
            k = (k + 1) % len(styles)
        used.add(k)
        chosen.append(styles[k])
        k += 1
    return chosen[0], chosen[1], chosen[2]


_NLG_SURFACE_BANK: tuple[SurfaceTemplateSpec, ...] = (
    SurfaceTemplateSpec(
        "grain_overview",
        (
            "Summarize {tables} at {grain} grain.",
            "What picture emerges for {tables} at {grain} grain?",
            "Give an analyst-facing overview of {tables} with {grain} grain.",
        ),
    ),
    SurfaceTemplateSpec(
        "filters",
        (
            "Show rows from {tables} narrowed by {n_filters} filter conditions.",
            "Which records in {tables} satisfy the current filters?",
        ),
    ),
    SurfaceTemplateSpec(
        "shape",
        (
            "Break down {tables} for reporting.",
            "List the figures implied over {tables}.",
        ),
    ),
)


def generate_bulk_anchors(
    intent: RuntimeIntent,
    _schema: SchemaGraph,
    count: int,
) -> tuple[str, ...]:
    """
    Emit deterministic natural-language anchors from intent shape without LLM calls.

    Args:

        intent: Executable runtime intent after warmup substitution.

        _schema: Schema graph reserved for richer entity labeling in future realizations.

        count: Maximum anchors to return after deduplication.

    Returns:

        Distinct question-shaped strings bounded by *count*.
    """

    tables_label = ", ".join(intent.tables or []) or "the scoped tables"
    grain = intent.grain or "row_level"
    n_filters = len(intent.filters_param or [])
    flat_forms: list[str] = []
    for spec in _NLG_SURFACE_BANK:
        flat_forms.extend(spec.surface_forms)
    if not flat_forms or count <= 0:
        return ()

    base = sha256(
        stable_json(
            {
                "tables": intent.tables,
                "grain": intent.grain,
                "nf": n_filters,
                "ng": len(intent.group_by_cols or []),
                "cte": len(intent.cte_steps or []),
            },
        ),
    )
    out: list[str] = []
    seen: set[str] = set()
    for i in range(max(count * 8, count)):
        if len(out) >= count:
            break
        pos = (i * 2) % max(len(base) - 1, 1)
        mix = int(base[pos : pos + 2], 16)
        form_idx = (mix + i) % len(flat_forms)
        raw_form = flat_forms[form_idx]
        text = raw_form.format(tables=tables_label, grain=grain, n_filters=n_filters)
        nn = normalize_question(text)
        if nn and nn not in seen:
            seen.add(nn)
            out.append(text.strip())
    return tuple(out[:count])


def generate_paraphrases_of_seed_question(
    seed_question: str,
    schema: SchemaGraph,
    tables: list[str],
    *,
    style_pair: tuple[str, str],
) -> list[str] | None:
    """
    LLM paraphrases of an existing seed question (not SQL-derived), aligned to two style slots.

    Args:

        seed_question: Gold seed text to rephrase.

        schema: Table and column metadata for terminology grounding.

        tables: Tables referenced by the intent.

        style_pair: Two distinct style keys from :attr:`SeedWarmupConfig.WARMUP_QUESTION_STYLES`.

    Returns:

        Zero to two non-empty paraphrase strings, or ``None`` when the call fails to yield usable text.
    """

    gu = SeedWarmupConfig.WARMUP_QUESTION_STYLE_GUIDANCE
    s0, s1 = style_pair
    slots = [
        {"style": s0, "guidance": gu.get(s0, "")},
        {"style": s1, "guidance": gu.get(s1, "")},
    ]
    user_prompt = stable_json(
        {
            "seed_question": seed_question,
            "schema": schema_context_enriched_lines_for_tables(schema, tables),
            "style_slots": slots,
            "output_format": {"questions": ["", ""]},
        }
    )
    try:
        response = llm_json(
            _PARAPHRASE_SEED_QUESTION_SYSTEM,
            user_prompt,
            retries=1,
            task="default",
        )
    except (TypeError, ValueError, LlmJsonExhausted):
        return None
    raw_list = response.get("questions")
    out: list[str] = []
    if isinstance(raw_list, list):
        for x in raw_list:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
            if len(out) >= 2:
                break
    return out if out else None


def schema_context_lines_for_tables(schema: SchemaGraph, tables: list[str]) -> str:
    """
    Render table and column role lines for prompt context.

    Args:

        schema: Live schema graph.

        tables: Table names to include.

    Returns:

        One line per table listing columns with types and roles.
    """

    col_descriptions: list[str] = []
    for table in tables:
        table_meta = schema.tables.get(table)
        if not table_meta:
            continue
        cols = []
        for col_name, col_meta in table_meta.columns.items():
            desc = f"{col_name} ({col_meta.data_type or 'unknown'})"
            if col_meta.role:
                desc += f" [{col_meta.role}]"
            cols.append(desc)
        col_descriptions.append(f"TABLE {table}: {', '.join(cols)}")
    return "\n".join(col_descriptions)


def select_best_question_via_judge(
    sql: str,
    schema_context: str,
    candidates: list[str],
) -> int:
    """
    Pick the candidate NL question that best matches *sql* given *schema_context*.

    Uses a JSON-mode judge call with ``task=\"judge\"``. On parse failure or any invalid index, returns ``0``.

    Args:

        sql: Executable or literal SQL text shown to the judge.

        schema_context: Concatenated table and column descriptions.

        candidates: Non-empty paraphrase strings in display order.

    Returns:

        Index into *candidates*.
    """

    if not candidates:
        return 0
    if len(candidates) == 1:
        return 0
    system = (
        "You compare natural-language question candidates against a SQL query and schema notes. "
        "Choose exactly one candidate index that best captures what the SQL computes for an analyst. "
        "Output ONLY valid JSON matching the requested shape."
    )
    payload = stable_json(
        {
            "sql": sql,
            "schema_context": schema_context,
            "candidates": [{"index": i, "text": t} for i, t in enumerate(candidates)],
            "instructions": (
                "chosen_index must be the zero-based index of the single best candidate. "
                "Prefer faithful semantics over stylistic flair."
            ),
            "output_format": {"chosen_index": 0},
        }
    )
    try:
        response = llm_json(system, payload, retries=0, task="judge")
        raw = response.get("chosen_index", 0)
        if isinstance(raw, bool):
            idx = 0
        elif isinstance(raw, int):
            idx = raw
        elif isinstance(raw, float):
            idx = int(raw)
        elif isinstance(raw, str):
            try:
                idx = int(raw.strip())
            except ValueError:
                idx = 0
        else:
            idx = 0
        if 0 <= idx < len(candidates):
            return idx
    except (TypeError, ValueError, LlmJsonExhausted):
        pass
    return 0


def generate_question_from_sql(
    sql: str,
    schema: SchemaGraph,
    tables: list[str],
    *,
    warmup_style_triple: tuple[str, str, str] | None = None,
    intent_source: str | None = None,
) -> dict[str, Any] | None:
    """
    LLM: *sql* plus column context yields NL questions and a realism flag.

    Args:

        sql: Executable or literal SQL.

        schema: Table and column metadata for the prompt.

        tables: Tables to describe in the prompt.

        warmup_style_triple: When set, uses enriched descriptions and three ordered style slots for synthetic warmup.

        intent_source: When ``sql_history``, executed-SQL provenance bypasses the realism rejection gate.

    Returns:

        Dict with ``question``, ``is_realistic``, ``drop_reason``, optional ``drop_reason_category`` when unrealistic, or ``None`` on error.
    """

    if warmup_style_triple is None:
        schema_context = schema_context_lines_for_tables(schema, tables)
        system = _QUESTION_FROM_SQL_SYSTEM
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
    else:
        schema_context = schema_context_enriched_lines_for_tables(schema, tables)
        system = _QUESTION_FROM_SQL_SYSTEM_WARMUP_STYLED
        gu = SeedWarmupConfig.WARMUP_QUESTION_STYLE_GUIDANCE
        s0, s1, s2 = warmup_style_triple
        user_body = {
            "sql": sql,
            "schema": schema_context,
            "style_slots": [
                {"style": s0, "guidance": gu.get(s0, "")},
                {"style": s1, "guidance": gu.get(s1, "")},
                {"style": s2, "guidance": gu.get(s2, "")},
            ],
            "output_format": {
                "questions": ["string", "string", "string"],
                "question": "natural language question or empty string",
                "is_realistic": True,
                "drop_reason": None,
                "drop_reason_category": "other",
            },
        }

    user_prompt = stable_json(user_body)

    response = llm_json(
        system,
        user_prompt,
        retries=1,
        task="default",
    )
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

    nmax = SeedWarmupConfig.WARMUP_QUESTIONS_MAX
    raw_list = response.get("questions")
    phrases: list[str] = []
    if isinstance(raw_list, list):
        for x in raw_list:
            if isinstance(x, str) and x.strip():
                phrases.append(x.strip())
            if len(phrases) >= nmax + 8:
                break
    if len(phrases) > nmax:
        debug(f"[utils.generate_question_from_sql] truncating questions to {nmax}")
        phrases = phrases[:nmax]
    if not phrases:
        q0 = response.get("question", "")
        if isinstance(q0, str) and q0.strip():
            phrases = [q0.strip()]
    if not phrases and isinstance(question, str) and question.strip():
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
    }
