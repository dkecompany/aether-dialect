"""Session, prompt, corpus, and refusal prose constants (data-only)."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

from ._constants import (
    DIAGNOSTIC_CODE_REFUSAL_AGGREGATE_FAN_OUT,
    DIAGNOSTIC_CODE_REFUSAL_AMBIGUOUS_DATE_LITERAL,
    DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP,
    DIAGNOSTIC_CODE_REFUSAL_CLAUSE_WIDENED_ROWSET,
    DIAGNOSTIC_CODE_REFUSAL_CONVERSATIONAL_DENY,
    DIAGNOSTIC_CODE_REFUSAL_CTE_CAP,
    DIAGNOSTIC_CODE_REFUSAL_DECLINED_SCHEMA,
    DIAGNOSTIC_CODE_REFUSAL_HOP_CEILING,
    DIAGNOSTIC_CODE_REFUSAL_INSUFFICIENT_KNOWLEDGE,
    DIAGNOSTIC_CODE_REFUSAL_INVALID_QUESTION,
    DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_TIE_CAP,
    DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE,
    DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT,
    DIAGNOSTIC_CODE_REFUSAL_NULL_IN_NEGATED_LIST,
    DIAGNOSTIC_CODE_REFUSAL_OPAQUE_EXPR,
    DIAGNOSTIC_CODE_REFUSAL_OPERATION_NOT_SUPPORTED,
    DIAGNOSTIC_CODE_REFUSAL_PARSE_FAILURE,
    DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED,
    DIAGNOSTIC_CODE_REFUSAL_PROBE_CTE_PLACEMENT,
    DIAGNOSTIC_CODE_REFUSAL_SCOPE_VIOLATION,
    DIAGNOSTIC_CODE_REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN,
    DIAGNOSTIC_CODE_REFUSAL_UNION_COLUMN_MISSING,
    DIAGNOSTIC_CODE_REFUSAL_UNMAPPABLE_QUESTION,
    DIAGNOSTIC_CODE_REFUSAL_UNSUPPORTED_COLUMN_TYPE,
    INSTRUCTIONAL_BRIDGE_TABLE_PLACEHOLDER,
    INSTRUCTIONAL_DATE_COLUMN_PLACEHOLDER,
    INSTRUCTIONAL_INTEGER_COLUMN_PLACEHOLDER,
    INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER,
    INSTRUCTIONAL_LINK_TABLE_PLACEHOLDER,
    INSTRUCTIONAL_OTHER_COLUMN_PLACEHOLDER,
    INSTRUCTIONAL_OTHER_DATE_COLUMN_PLACEHOLDER,
    INSTRUCTIONAL_OTHER_QUALIFIED_COLUMN_PLACEHOLDER,
    INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER,
    INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER,
    INSTRUCTIONAL_TABLE_PLACEHOLDER,
    RENTAL_SHOP_BUNDLE_MEMBERS,
    REPHRASE_HINT_REFUSAL_CODES,
    UPLOAD_COLUMN_TRANSFORM_IDS,
)

SANDBOX_SCHEMA_LITERALS_FILENAME: str = "schema_literals.json"

SANDBOX_INTERPRET_DOMAIN_FILENAME: str = "schema_interpret_domain.json"

DATA_QUALITY_SEVERITY_BLOCKING: str = "blocking"

SESSION_PROMPT_YESNO: str = "Is this correct? (y/n): "

SESSION_PROMPT_REASON: str = "Please provide a reason: "

SESSION_USER_FEEDBACK_BODY: str = (
    "What was wrong?\n"
    "Tip: a single sentence is enough — for example 'wrong table', 'missing date filter', or 'should aggregate by month'."
)

SESSION_INTENT_FEEDBACK_BODY: str = (
    "What should change about this interpretation?\n"
    "Tip: a single sentence is enough — for example 'wrong table', 'missing date filter', or 'should aggregate by month'."
)

ASK_PHASE_A: str = "A:intake.reuse"

ASK_PHASE_B: str = "B:intent.interpret"

ASK_PHASE_C: str = "C:intent.ground"

ASK_PHASE_D: str = "D:intent.compose"

ASK_PHASE_E: str = "E:intent.repair_structural"

ASK_PHASE_F: str = "F:intent.validate_schema"

ASK_PHASE_G: str = "G:intent.validate_semantic"

ASK_PHASE_H: str = "H:intent.finalize"

ASK_PHASE_I: str = "I:plan.decompose"

ASK_PHASE_J: str = "J:sql.build_joins"

ASK_PHASE_K: str = "K:sql.validate_scope"

ASK_PHASE_L: str = "L:sql.execute"

ASK_PHASE_M: str = "M:plan.combine"

ASK_PHASE_N: str = "N:feedback"

SCHEMA_BUILD_PHASE_A: str = "A:cache_gate"

SCHEMA_BUILD_PHASE_B: str = "B:diff"

SCHEMA_BUILD_PHASE_C: str = "C:reflect"

SCHEMA_BUILD_PHASE_D: str = "D:scope_filter"

SCHEMA_BUILD_PHASE_E: str = "E:profile"

SCHEMA_BUILD_PHASE_F: str = "F:classify"

SCHEMA_BUILD_PHASE_G: str = "G:coerce_ops"

SCHEMA_BUILD_PHASE_H: str = "H:join_stats"

SCHEMA_BUILD_PHASE_I: str = "I:viability_hash"

SCHEMA_BUILD_PHASE_J: str = "J:persist"

SCHEMA_BUILD_PHASE_K: str = "K:overrides_health"

QSIM_PHASE_A: str = "A:bootstrap"

QSIM_PHASE_B: str = "B:skeletons"

QSIM_PHASE_C: str = "C:targets"

QSIM_PHASE_D: str = "D:select"

QSIM_PHASE_E: str = "E:fill"

QSIM_PHASE_F: str = "F:validate"

QSIM_PHASE_G: str = "G:normalize_dedup"

QSIM_PHASE_H: str = "H:instantiate"

QSIM_PHASE_I: str = "I:nl_synthesis"

QSIM_PHASE_J: str = "J:emit"

WARMUP_PHASE_A: str = "A:setup"

WARMUP_PHASE_B: str = "B:ingest"

WARMUP_PHASE_C: str = "C:expand"

WARMUP_PHASE_D: str = "D:dedupe_classify"

WARMUP_PHASE_E: str = "E:join_prep"

WARMUP_PHASE_F: str = "F:cache_open"

WARMUP_PHASE_G: str = "G:execute"

WARMUP_PHASE_H: str = "H:sample"

WARMUP_PHASE_I: str = "I:nl_generate"

WARMUP_PHASE_J: str = "J:template_learn"

WARMUP_PHASE_K: str = "K:persist"

FEDERATION_COMPOSITION_PHASE_A: str = "A:roster"

FEDERATION_COMPOSITION_PHASE_B: str = "B:merge"

FEDERATION_COMPOSITION_PHASE_C: str = "C:collapse"

FEDERATION_COMPOSITION_PHASE_D: str = "D:unify"

FEDERATION_COMPOSITION_PHASE_E: str = "E:edges"

FEDERATION_COMPOSITION_PHASE_F: str = "F:reconcile"

FEDERATION_COMPOSITION_PHASE_G: str = "G:identity"

FEDERATION_COMPOSITION_PHASE_H: str = "H:persist"

PERMISSION_DRIFT_CONTACT_ADMIN_MESSAGE: str = (
    "This query could not be authorized against the current database grants. Please contact your administrator."
)

INTERPRET_SUPPORTED_CAPABILITIES: tuple[str, ...] = (
    "Identify the semantic entities, measures, conditions, grouping, ordering, ranking, row cap, conditional labeling, and time reasoning needed to answer the question.",
    "Reformulate unsupported full-SQL constructs into supported analysis shapes in plain language without naming IR or SQL operators.",
    "Infer whether the question needs row-level output, grouped output, a scalar answer, staged intermediate computation, a windowed comparison, or a conditional bucketed result.",
    "Recognize existence, absence, set difference, one-row-per-partition, and outer-join preservation needs and describe them in plain language without SQL operators.",
    "Use only the domain schema descriptions and enum heads to ground domain concepts; capture any missing or ambiguous binding as internal planning uncertainty rather than refusing.",
    "Record grounding traceability for tables and for filter, having, or group_by constraints using only table names or "
    f"{INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} refs; never bare column or enum tokens; do not enumerate select output columns in grounding.",
)

GROUND_SUPPORTED_CAPABILITIES: tuple[str, ...] = (
    "Bind the interpret pathway to real schema identifiers and natural-language clause descriptions without emitting runtime IR.",
    "Populate select, filter, group_by, having, order_by, limit, window, and case prose fields; copy every literal into the matching prose field.",
    f"List semantic base tables in tables; omit {INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER} from tables when its columns appear in prose.",
    f"Keep a table in join scope only via qualified {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} tokens in select, filter, group_by, having, order_by, or registry prose — not from the tables list alone.",
    f"When membership or existence requires {INSTRUCTIONAL_LINK_TABLE_PLACEHOLDER} or {INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER}, "
    f"name {INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_OTHER_COLUMN_PLACEHOLDER} or "
    f"{INSTRUCTIONAL_BRIDGE_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_OTHER_COLUMN_PLACEHOLDER} in select prose, not only join-equality narration in filter prose.",
    "Use cte_steps when staged computation is needed; each step tables list may name base schema tables and prior cte_steps names this step reads from.",
    "Describe semi_join or anti_join probe CTE steps when existence, absence, or set difference is needed; mention distinct_on and preserve_tables in prose when the approach requires them.",
    "Never author join paths; the engine discovers foreign-key paths after Compose.",
    "When answer shape matters for later Compose, state in prose whether the result is one row, one row per group, or a row listing.",
    "When null ordering matters, state in order prose whether nulls sort first or last.",
    "When ranking, running totals, or prior/next-row offsets are required, say so in the window prose field without inventing registry ids.",
    "When membership vs pattern match or case-sensitive vs case-insensitive match is easy to confuse, prefer phrasing consistent with nl_phrase_mappings hard cases.",
)

COMPOSE_SUPPORTED_CAPABILITIES: tuple[str, ...] = (
    "Return columns or computed expressions; optionally return only distinct rows.",
    "Aggregate with count, sum, avg, min, max, string_agg, stddev, variance, or median; sum, avg, stddev, variance, and median require numeric columns; string_agg concatenates text with a separator param and optional within-aggregate ordering.",
    "Group by one or more columns (grouped grain), or compute a single aggregate over all rows (scalar grain); a query with no aggregation is row-level.",
    "Filter rows with comparison and membership operators listed in operator_reference and schema-gated capabilities (including like, in, between, null checks, and when supported ilike and array contains).",
    "Encode WHERE and HAVING boolean logic as nested PredicateGroup trees (op and/or with predicate leaves and child groups).",
    "Restrict aggregated results with a HAVING comparison using =, !=, <, <=, >, >=, in, not in, or between.",
    "Order by any column or expression ascending or descending with optional nulls first or nulls last, and cap rows with a limit.",
    "Compute arithmetic with +, -, *, / and wrap arithmetic in an aggregate.",
    "Apply scalar functions upper, lower, trim, ltrim, rtrim, length, abs, round, floor, ceil, date_trunc, date_part, extract (never epoch), coalesce, concat, year, month, day.",
    "Concatenate text with concat, never the || operator.",
    "Rank or number rows per group and compute running or offset values with window functions row_number, rank, dense_rank, ntile, percent_rank, cume_dist, sum, avg, lag, lead, first_value, last_value, nth_value, using partitioning, ordering, numeric_argument where required, and row or range frames.",
    "Produce conditional labels or buckets with CASE.",
    "Filter on a relative time window (last N days, weeks, months, quarters, or years) or on the difference between two date columns.",
    "Compare or shift a date by an integer number of days, and reference the current date or current timestamp.",
    "Break a computation into intermediate steps (WITH steps) for staged aggregation, self-comparison, per-entity ranking, or reuse; a later step reads an earlier step's named outputs.",
    "When the question asks which entities have a matching row elsewhere, declare a CTE with emission semi_join projecting the join keys.",
    "When the question asks which entities lack a matching row or differ from another set, declare a CTE with emission anti_join projecting the compared tuple.",
    "Return one row per partition with distinct_on and order_by_cols for within-partition ranking.",
    "Preserve anchor tables through joins with preserve_tables even when their columns are not selected.",
    "Every bare wNN in select_cols requires a same-scope window_registry entry with that registry_id; every bare cNN requires a same-scope case_registry entry.",
    "Omit emission for ordinary CTEs, or emit join_table / scalar_subquery as a hint the engine reclassifies from CTE shape.",
    "Emit only operators and expression forms present in operator_reference and schema-gated capabilities; nl_phrase_mappings only disambiguate among those forms.",
    "PredicateGroup nesting depth must not exceed 3; CTE steps must stay within configured MAX_CTE_STEPS.",
    "in and not in take a list of literal values in raw_value; contains takes a single element value for array columns when listed in operator_reference.",
    "Not expressible: UNION, INTERSECT, recursive CTEs, and LATERAL joins.",
    "Join paths are discovered by the engine from foreign keys; never author joins or name junction tables except when the many-to-many set itself is requested; list only the base tables whose columns are used.",
)

FEDERATION_COMPOSE_SUPPORTED_CAPABILITIES: tuple[str, ...] = (
    *COMPOSE_SUPPORTED_CAPABILITIES[:-2],
    "Do not emit SQL UNION; each listed table is already the complete relation for its entity.",
    "Not expressible: INTERSECT, recursive CTEs, and LATERAL joins.",
    COMPOSE_SUPPORTED_CAPABILITIES[-1],
)

IR_SUPPORTED_CAPABILITIES: tuple[str, ...] = COMPOSE_SUPPORTED_CAPABILITIES

SANDBOX_MIN_FIXTURE_COUNT: int = 100

SANDBOX_MIN_INTENT_FIXTURE_COUNT: int = 50

REALISM_DROP_REASON_CATEGORIES: frozenset[str] = frozenset(
    {
        "nonsensical_sql",
        "tautology",
        "overly_narrow_filter",
        "sensitive_smell",
        "unmeasurable_metric",
        "other",
    }
)

USER_INVALID_INPUT_LINE: str = "\nInvalid input."

JOIN_PRIOR_FEEDBACK_HEADING: str = "Previously rejected joins for this question (avoid these table sets / FK paths):"

JOIN_PRIOR_FEEDBACK_PATH_LABEL: str = "FK path:"

INTERPRET_SINGLE_OUTPUT_WINDOW_NO_CTE_RULE: str = "Do not introduce a CTE for a question answerable by a single SELECT with ordinary joins, filters, or a window function on the primary table. Use CTEs only for reuse across multiple SELECT bodies, staged aggregation, self-references, or per-entity ranking."

DATA_QUALITY_ISSUE_RAGGED_ROW: str = "ragged_row"

DATA_QUALITY_ISSUE_INVALID_MERGE_RANGE: str = "invalid_merge_range"

DATA_QUALITY_DETAIL_CANDIDATE_HEADER_ROW: str = "candidate_header_row"

DATA_QUALITY_DETAIL_CANDIDATE_SHEET: str = "candidate_sheet"

DATA_QUALITY_DETAIL_CANDIDATE_TABLE_RANGE: str = "candidate_table_range"

DATA_QUALITY_SQL_RESERVED_WORDS: frozenset[str] = frozenset(
    {
        "all",
        "and",
        "any",
        "as",
        "between",
        "case",
        "column",
        "create",
        "date",
        "delete",
        "desc",
        "distinct",
        "drop",
        "else",
        "end",
        "except",
        "exists",
        "false",
        "from",
        "full",
        "group",
        "having",
        "in",
        "inner",
        "insert",
        "into",
        "is",
        "join",
        "left",
        "like",
        "limit",
        "not",
        "null",
        "on",
        "or",
        "order",
        "outer",
        "right",
        "select",
        "set",
        "table",
        "then",
        "true",
        "union",
        "update",
        "using",
        "values",
        "when",
        "where",
        "with",
    }
)

QUESTION_STARTS_AGG: tuple[str, ...] = (
    "How many",
    "What is the total",
    "What is the average",
    "What is the minimum",
    "What is the maximum",
    "Find the sum of",
    "Calculate the",
    "Show the count of",
    "Get the total",
)

COLUMN_DEFINITION_STOP_WORDS: frozenset[str] = frozenset(
    {
        "NOT",
        "DEFAULT",
        "NULL",
        "PRIMARY",
        "UNIQUE",
        "REFERENCES",
        "CHECK",
        "CONSTRAINT",
        "GENERATED",
        "AUTO_INCREMENT",
        "AUTOINCREMENT",
        "COMMENT",
        "AS",
        "ROW",
        "FORMAT",
        "SIGNED",
        "UNSIGNED",
        "ZEROFILL",
        "ENCODE",
        "COLLATE",
    }
)

RENTAL_SHOP_VIEW_NAMES: tuple[str, ...] = ("active_customer_v", "store_revenue_v", "film_catalog_v")

SCHEMA_INSTRUCTION_LIKE_LINE_PATTERNS: tuple[str, ...] = (
    r"(?i)^\s*ignore\s+(all\s+)?(prior|previous)\b",
    r"(?i)^\s*disregard\s+(all\s+)?(prior|previous)\b",
    r"(?i)^\s*system\s*:",
    r"(?i)^\s*assistant\s*:",
    r"(?i)^\s*you\s+are\b",
    r"(?i)^\s*always\s+filter\b",
    r"(?i)^\s*never\s+reveal\b",
    r"(?i)^\s*do\s+not\s+follow\b",
)

BOOLEAN_TRUTH_PATTERN_MAP: Mapping[frozenset[str], str] = MappingProxyType(
    {
        frozenset({"0", "1"}): "1",
        frozenset({"true", "false"}): "true",
        frozenset({"yes", "no"}): "yes",
        frozenset({"y", "n"}): "y",
        frozenset({"t", "f"}): "t",
        frozenset({"on", "off"}): "on",
        frozenset({"active", "inactive"}): "active",
        frozenset({"enabled", "disabled"}): "enabled",
        frozenset({"pass", "fail"}): "pass",
        frozenset({"passed", "failed"}): "passed",
        frozenset({"pass", "failed"}): "pass",
        frozenset({"passed", "fail"}): "passed",
        frozenset({"open", "closed"}): "open",
        frozenset({"opened", "closed"}): "opened",
        frozenset({"open", "close"}): "open",
        frozenset({"opened", "close"}): "opened",
        frozenset({"success", "failure"}): "success",
        frozenset({"succeeded", "failed"}): "succeeded",
        frozenset({"success", "failed"}): "success",
        frozenset({"present", "absent"}): "present",
    }
)

META_INSUFFICIENT_KNOWLEDGE_RESPONSE_KIND: str = "insufficient_knowledge"

META_KNOWLEDGE_ANSWER_SCHEMA: dict[str, Any] = {
    "oneOf": [
        {
            "type": "object",
            "required": ["response_kind", "message"],
            "additionalProperties": False,
            "properties": {
                "response_kind": {"type": "string", "const": "domain_knowledge"},
                "message": {"type": "string", "minLength": 1},
            },
        },
        {
            "type": "object",
            "required": ["response_kind"],
            "additionalProperties": False,
            "properties": {
                "response_kind": {"type": "string", "const": META_INSUFFICIENT_KNOWLEDGE_RESPONSE_KIND},
            },
        },
    ]
}

META_SCHEMA_AND_KNOWLEDGE_ANSWER_SCHEMA: dict[str, Any] = {
    "oneOf": [
        {
            "type": "object",
            "required": ["response_kind", "message"],
            "additionalProperties": False,
            "properties": {
                "response_kind": {"type": "string", "const": "schema_and_knowledge"},
                "message": {"type": "string", "minLength": 1},
                "notes": {"type": "string"},
            },
        },
        {
            "type": "object",
            "required": ["response_kind"],
            "additionalProperties": False,
            "properties": {
                "response_kind": {"type": "string", "const": META_INSUFFICIENT_KNOWLEDGE_RESPONSE_KIND},
            },
        },
    ]
}

KNOWLEDGE_NOTES_EXTRACT_SYSTEM: str = (
    "You extract knowledge records from operator notes in a single pass.\n"
    "Input JSON has domain_notes and schema_names (complete whitelist of relation and relation.field names).\n"
    'Return ONLY JSON {"records": [...], "coverage": [...]}. No markdown fences. '
    "Top-level keys must be exactly records and coverage.\n"
    "Each record has exactly key, kind, text, referenced_entities; payload only when required by kind.\n"
    '- "key": non-empty concept slug when referenced_entities is empty; omit or null when referenced_entities is non-empty\n'
    '- "kind": structural category (relation|field|join|grain|cardinality|lifecycle|declared_value_set|sentinel_semantics|unit_of_measure|relation_shape|term_binding|period_convention|concept_absence) when anchored, or glossary|policy|metric|synonym|caveat when unanchored\n'
    '- "text": non-empty fact preserving operator wording\n'
    '- "referenced_entities": names from schema_names this fact is a property of; empty list when unanchored\n'
    "Anchoring rule: non-empty referenced_entities means the fact is anchored (structural); empty means unanchored (domain). "
    "Decide anchoring per fact using schema_names — do not emit separate domain vs structural passes.\n"
    "referenced_entities must be drawn only from schema_names when non-empty.\n"
    "Coverage: partition domain_notes into spans (exact substrings). Each coverage element has span, disposition (fact or no_fact), and record_index (required for fact, omitted for no_fact). Every part of domain_notes must appear in exactly one coverage span.\n"
    "Sparsity is expected. Do not invent facts. Preserve operator wording.\n"
    "Reason internally, output only the JSON object."
)

KNOWLEDGE_NOTES_EXTRACT_REPAIR_SYSTEM: str = (
    "You repair a failed single-pass knowledge extraction from operator notes.\n"
    "Input JSON has domain_notes, schema_names, validation_error, and previous_raw.\n"
    'Return ONLY JSON {"records": [...], "coverage": [...]} with the same shape as the extract pass. '
    "Fix validation_error. Keep operator facts; do not invent new ones. "
    "Coverage must partition domain_notes exactly.\n"
    "Reason internally, output only the JSON object."
)

DOMAIN_KNOWLEDGE_SPACE_MERGE_SYSTEM: str = (
    "You merge two domain-knowledge lists.\n"
    "Input JSON has two entry arrays (each element has key, kind, text, referenced_entities).\n"
    "The second array weighs more when facts are contradictory. Keep both when complementary — even if they share the same key. Do not drop a unique fact. Do not invent facts or keys.\n"
    "key is a concept label, not a unique merge identity.\n"
    "Do not promote schema inventory into domain knowledge.\n"
    'Return JSON {"entries": [{"key","kind","text","referenced_entities"}, ...]} only. '
    "Top-level keys must be exactly entries. Each entry must have exactly key, kind, text, and referenced_entities.\n"
    "Respond ONLY with valid JSON, no explanation."
)

DOMAIN_KNOWLEDGE_FEDERATION_MERGE_SYSTEM: str = (
    "You merge domain-knowledge lists from equal peer sources.\n"
    "Input JSON has multiple peer entry arrays. Each peer bundle includes an opaque identifier and an entries array; never echo any opaque identifier in output text.\n"
    "Treat all peers equally. Keep complementary facts even when keys overlap. "
    "On true contradiction, synthesize one entry or keep disambiguated keys. Never prefer one peer over another.\n"
    "Do not promote schema inventory into domain knowledge.\n"
    'Return JSON {"entries": [{"key","kind","text","referenced_entities"}, ...]} only. '
    "Top-level keys must be exactly entries. Each entry must have exactly key, kind, text, and referenced_entities.\n"
    "Respond ONLY with valid JSON, no explanation."
)

STRUCTURAL_KNOWLEDGE_SPACE_MERGE_SYSTEM: str = (
    "You merge two structural knowledge fact lists.\n"
    "Input JSON has two fact arrays (each element has kind, text, referenced_entities, payload?).\n"
    "Facts from the second array weigh more when contradictory. Keep both when complementary — even with the same kind. "
    "Do not drop a unique fact. Do not invent facts.\n"
    "referenced_entities names the relations and relation.field identifiers each fact is about.\n"
    'Return JSON {"facts": [{"kind","text","referenced_entities",[payload]}, ...]} only. '
    "Top-level keys must be exactly facts. Each fact must have kind, text, and referenced_entities.\n"
    "Respond ONLY with valid JSON, no explanation."
)

STRUCTURAL_KNOWLEDGE_FEDERATION_MERGE_SYSTEM: str = (
    "You merge structural knowledge facts from equal peer sources.\n"
    "Input JSON has multiple peer fact arrays. Each peer bundle includes an opaque identifier and a facts array; never echo any opaque identifier in output text.\n"
    "Equal peers: keep complementary facts; on contradiction synthesize or disambiguate. Never prefer one peer over another.\n"
    'Return JSON {"facts": [{"kind","text","referenced_entities",[payload]}, ...]} only. '
    "Top-level keys must be exactly facts. Each fact must have kind, text, and referenced_entities.\n"
    "Respond ONLY with valid JSON, no explanation."
)

DOMAIN_KNOWLEDGE_REFINER_SYSTEM: str = (
    "You refine domain-knowledge entries for a text-to-SQL system. "
    "Domain knowledge depends on the domain of the data, not on schema structure. "
    'Input is JSON with \'entries\' (objects with key, kind, text). Return ONLY JSON of the form {"entries": [{"key": "...", "kind": "...", "text": "..."}]} with the same keys as the input (or fewer when merging duplicate keys). '
    "Do not invent keys that were not in the input. Keep kind in {glossary, policy, metric, synonym, caveat}. "
    "Tighten prose; preserve human keywords; do not invent facts. "
    "Keys are concept slugs, not relation or relation.field identifiers. "
    "Do not turn entries into relation/field catalog blurbs — those belong in schema descriptions."
)

META_EMPTY_DOMAIN_KNOWLEDGE_MESSAGE: str = "No domain knowledge entries are configured for this engine or space."

STRUCTURE_EDITABLE_ENUMS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "table_role": ("dimension", "fact", "bridge", "unknown"),
        "column_role": (
            "identifier",
            "categorical",
            "numeric_categorical",
            "numeric_measure",
            "temporal",
            "boolean",
            "free_text",
            "audit",
        ),
        "column_sensitivity": (
            "none",
            "restricted",
            "hidden",
        ),
        "foreign_key_kind": ("structural", "semantic"),
    }
)

FEDERATION_METHOD_SEMANTICS: dict[str, str] = {
    "add_engine": "composite",
    "aetherspace": "composite",
    "apply_federation": "composite",
    "apply_knowledge": "composite",
    "apply_migration_map": "composite",
    "apply_structure": "composite",
    "asession": "composite",
    "clear_all_learning": "both",
    "clear_simulation_caches": "both",
    "clear_template_store": "both",
    "close": "composite",
    "delete_aetherspace": "composite",
    "execute_template": "composite",
    "export_context": "composite",
    "export_federation": "composite",
    "export_knowledge": "composite",
    "export_structure": "composite",
    "fetch_template": "composite",
    "get_qsim_summary": "composite",
    "get_questions_only": "composite",
    "get_seed_warmup_summary": "composite",
    "list_aetherspaces": "composite",
    "list_contexts": "composite",
    "list_templates": "composite",
    "mapping_suggestions": "composite",
    "prepared_federated_outcome": "composite",
    "refresh": "composite",
    "remove_engine": "composite",
    "run_interactive": "composite",
    "run_qsim": "composite",
    "run_seed_warmup": "unsupported",
    "run_seed_warmup_from_history": "unsupported",
    "run_seed_warmup_from_query_log": "unsupported",
    "session": "composite",
}

FEDERATION_COMPOSITE_RECONCILIATION_NOTE: str = (
    "When two descriptions refer to the same domain concept, choose one canonical wording and role."
)

FEDERATION_WARMUP_UNSUPPORTED_MESSAGE: str = "warmup is not supported on AetherFederation"

CONFIG_ERROR_SCHEMA_CONTEXT_COLUMN_SPEC: str = "deny_columns and allow_columns entries must be qualified as 'table.column' or '*.column'; bare column names are not permitted; got {spec!r}"

REPHRASE_HINT_MESSAGES: dict[str, str] = {
    "sql_validation_failed": (
        "Please rephrase or retry.\n\n"
        "Tips: simplify filters, be explicit about columns, or split a complex question into smaller ones.\n"
    ),
    "user_rejected_intent": (
        "Please rephrase your question.\n"
        "Tips: be more specific about which columns, filters, grouping, or time range you want."
    ),
    "user_rejected_result": (
        "Please retry or rephrase your question.\n\n"
        "Tips: be more specific about columns, filters, grouping, or time range.\n"
    ),
    "federation_partial_failure": (
        "A federated query could not complete because one member failed after others succeeded. "
        "No partial answer was returned.\n\n"
        "Tips: retry the question, or ask on each member engine individually.\n"
    ),
    "federation_cap_exceeded": (
        "A federated query could not complete because a coordinator or member resource limit was exceeded.\n\n"
        "Tips: narrow the question, add filters, or reduce the result scope.\n"
    ),
    "federation_member_execution_failed": (
        "A federated query could not complete because one member failed.\n\n"
        "Tips: retry the question, or ask on each member engine individually.\n"
    ),
    "federation_member_probe_failed": (
        "Federation initialization could not connect to a member database.\n\n"
        "Tips: verify member connectivity and credentials, then retry federation setup.\n"
    ),
    "federation_turn_cancelled": (
        "The federated query was cancelled before it could finish.\n\n"
        "Tips: retry the question if you still need an answer.\n"
    ),
}

INTENT_PLACEHOLDER_FORMAT_REPAIR_PARSE_ERROR: str = (
    "Instructional placeholder tokens appear in expression strings. Replace each with exact table.column names from schema_info. Do not leave angle-bracket markup, table_N or column_N instructional tokens, "
    f"or synthetic shape tokens from the prompt ({INSTRUCTIONAL_TABLE_PLACEHOLDER}, "
    f"{INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER}, {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER}, "
    f"{INSTRUCTIONAL_OTHER_QUALIFIED_COLUMN_PLACEHOLDER})."
)

NATURAL_LANGUAGE_REFUSAL_PARSE_ERROR: str = (
    "natural_language contains refusal or permission prose while select_cols remain populated"
)

META_SCHEMA_ANSWER_SCHEMA: dict[str, Any] = {
    "oneOf": [
        {
            "type": "object",
            "required": ["response_kind", "headline", "counts", "tables", "relationships", "notes"],
            "additionalProperties": False,
            "properties": {
                "response_kind": {"type": "string", "const": "schema_catalog"},
                "headline": {"type": "string", "minLength": 1},
                "counts": {
                    "type": "object",
                    "required": ["tables", "columns", "members", "columns_in_table", "tables_in_member"],
                    "additionalProperties": False,
                    "properties": {
                        "tables": {"type": ["integer", "null"]},
                        "columns": {"type": ["integer", "null"]},
                        "members": {"type": ["integer", "null"]},
                        "columns_in_table": {
                            "oneOf": [
                                {"type": "null"},
                                {
                                    "type": "object",
                                    "required": ["table", "columns"],
                                    "additionalProperties": False,
                                    "properties": {
                                        "table": {"type": "string", "minLength": 1},
                                        "columns": {"type": "integer"},
                                    },
                                },
                            ]
                        },
                        "tables_in_member": {
                            "oneOf": [
                                {"type": "null"},
                                {
                                    "type": "object",
                                    "required": ["source_id", "tables"],
                                    "additionalProperties": False,
                                    "properties": {
                                        "source_id": {"type": "string", "minLength": 1},
                                        "tables": {"type": "integer"},
                                    },
                                },
                            ]
                        },
                    },
                },
                "tables": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "source_id", "description", "columns"],
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string", "minLength": 1},
                            "source_id": {"type": "string", "minLength": 1},
                            "description": {"type": "string"},
                            "columns": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["name", "data_type", "role", "description"],
                                    "additionalProperties": False,
                                    "properties": {
                                        "name": {"type": "string", "minLength": 1},
                                        "data_type": {"type": "string"},
                                        "role": {"type": "string"},
                                        "description": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                },
                "relationships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["left", "right", "kind"],
                        "additionalProperties": False,
                        "properties": {
                            "left": {"type": "string", "minLength": 1},
                            "right": {"type": "string", "minLength": 1},
                            "kind": {"type": "string", "enum": ["fk", "semantic"]},
                        },
                    },
                },
                "notes": {"type": "array", "items": {"type": "string"}},
            },
        },
        {
            "type": "object",
            "required": ["response_kind"],
            "additionalProperties": False,
            "properties": {
                "response_kind": {"type": "string", "const": META_INSUFFICIENT_KNOWLEDGE_RESPONSE_KIND},
            },
        },
    ]
}

SCHEMA_CLASSIFY_SYSTEM: str = (
    "Classify every table's role and every column listed in the input for each table.\n\n"
    "INPUT SCOPE:\n"
    "Each table object lists only the columns you must classify under columns. "
    "Return JSON with exactly those table keys. For each table, include exactly those column keys under columns — no additional columns and no omissions.\n\n"
    "TABLE ROLES:\n"
    "- dimension: reference/lookup table referenced by others, descriptive attributes\n"
    "- fact: event or measure table with FKs to dimensions, contains measures\n"
    "- bridge: junction table for many-to-many, mostly FKs, few own columns\n"
    "- unknown: cannot confidently classify\n"
    "Use FK topology: tables referenced by many others are dimension; tables with many outbound FKs are fact; tables with only 2+ FKs and minimal columns are bridge.\n\n"
    "COLUMN ROLE DECISION PRIORITY (evaluate in order, first match wins):\n"
    "1. is_primary_key or is_foreign_key → identifier\n"
    "2. value_type is date → temporal (unless name clearly marks audit-only system timestamp like last_update → audit)\n"
    "3. name suggests binary state (is_*, has_*, active) and distinct_count = 2 → boolean\n"
    "4. value_type is integer or number and name suggests quantity/amount/price/count/size/distance → numeric_measure\n"
    "4b. value_type is integer and name suggests duration/period/lead_time/lag/tenure/offset in time units → temporal\n"
    "5. value_type is integer or number and name suggests code/rating/level/rank/status/tier/year → numeric_categorical\n"
    "6. value_type is integer or number with no clear name signal → numeric_measure (default for numeric value types)\n"
    "7. value_type is string and very high distinct_ratio → free_text\n"
    "8. value_type is string → categorical (default for string value types)\n\n"
    "AMBIGUOUS TWO-VALUE COLUMNS:\n"
    "When a column has exactly two sampled categorical values (e.g. positive/negative-style pairs), do not assume boolean unless name, value_type, FK topology, and profile_hints clearly support a flag.\n\n"
    "PROFILE HINTS (supporting evidence only — never override name/value_type signals):\n"
    "Each column may include a profile_hints object with distinct_count, distinct_ratio, and null_ratio. Use these to confirm or disambiguate when name and value_type are ambiguous.\n"
    "Do NOT use profile_hints as the primary reason to choose a role.\n\n"
    "CROSS-TABLE CONSISTENCY:\n"
    "- Columns with the same name and value_type across tables MUST receive the same role.\n"
    "- Deduce roles from names, value_type, FK topology, and profile_hints using the priority above.\n\n"
    "COLUMN DESCRIPTIONS:\n"
    "For each column, provide a short semantic description (max 8 words) describing what the column represents in domain terms. Every column object MUST include a non-empty description.\n"
    "Role-based guidance for column descriptions:\n"
    "- identifier columns: describe what entity the ID refers to.\n"
    "- numeric_measure columns: state the unit or what is measured.\n"
    "- categorical columns: mention common category values or groupings.\n"
    "- temporal columns with value_type date: state what event the date/time marks.\n"
    "- temporal columns with value_type integer: state that the column holds a day-count or period length (not a calendar date).\n"
    "- audit columns: state that the column records when a row was last changed by the system.\n"
    "- boolean columns: describe the yes/no condition.\n"
    "- FK columns: MUST state what related data the target table provides when joined. Name the key descriptive columns on the target table (e.g. 'links to target_table for name, title, description').\n\n"
    "TABLE DESCRIPTIONS:\n"
    "For each table provide a one-line domain purpose that includes: (a) what entity or event the table represents, (b) which related tables it connects to via foreign keys, and (c) the notable descriptive or measure columns it provides that users commonly ask about. Every table MUST include a non-empty description.\n\n"
    "SENSITIVITY (per column, optional):\n"
    'Include "sensitivity" in each column object: always null in this pass.\n'
    'A later second-pass refine step may set "restricted" or "hidden" only when domain notes explicitly require it.\n\n'
    "Reason internally, output only JSON:\n"
    '{"table": {"table_role": "...", "description": "...", "columns": {"column": {"role": "...", "description": "...", "sensitivity": null}, ...}}, ...}'
)

SCHEMA_ENTITY_ENRICH_SYSTEM: str = (
    "You enrich one schema entity's description using structural facts routed to it.\n\n"
    "Input JSON has entity (relation or relation.field identifier), entity_type (table or column), "
    "base_classification (current role and description for this entity only), and structural_facts "
    "(facts whose reference set names this entity).\n"
    "These facts describe this entity; write its description incorporating them. "
    "Do not decide which facts apply — they were routed to this entity already.\n"
    "Preserve substantive meaning from base_classification. "
    "Do not name relations or fields that are not this entity, not in structural_facts' referenced_entities, "
    "and not columns of this table when entity_type is table.\n"
    "Override role or sensitivity only when a structural fact explicitly supports the change; otherwise keep base values.\n"
    'For entity_type table output: {"table_role":"...", "description":"..."}.\n'
    'For entity_type column output: {"role":"...", "description":"...", "sensitivity": null or a sensitivity level}.\n'
    "Reason internally, output only JSON."
)

SCHEMA_CONSISTENCY_REFINE_SYSTEM: str = (
    "You receive base_classification JSON describing every table and column. The user message contains only base_classification under that key.\n\n"
    "Preserve the base output unless you detect a genuine cross-table inconsistency — for example the same column name and value_type assigned different roles in different tables. When you fix such an inconsistency, align the conflicting entries to the role that best matches the shared name, value_type, and FK topology.\n\n"
    "Do not invent new descriptions: keep each table description and column description from the base unless a detected inconsistency forces a minimal coordinated rewrite.\n"
    "Do not change sensitivity values from the base.\n"
    "Do not change column roles when the base assignment is already internally consistent.\n"
    "Do not remove tables or columns from base_classification. Do not add new tables or columns.\n\n"
    "Emit JSON identical in shape to base_classification.\n"
    "Reason internally, output only JSON:\n"
    '{"table": {"table_role": "...", "description": "...", "columns": {"column": {"role": "...", "description": "...", "sensitivity": null}, ...}}, ...}'
)

UPLOAD_PROMPT_NEUTRALITY_AUDIT_CONSTANTS: frozenset[str] = frozenset(
    {
        "CSV_IDENTIFIER_NAMING_SYSTEM",
        "UPLOAD_COLUMN_TRANSFORMS_SYSTEM",
        "UPLOAD_INTERPRET_SYSTEM",
        "UPLOAD_SUMMARY_SYSTEM",
    }
)

CSV_SCHEMA_LITERAL_ORIGINAL_NAME_NOTE: str = "For file-sourced tables, name is the normalized SQL identifier and original_name is the label from the uploaded file."

INTENT_NON_SELECTABLE_PREDICATE_MESSAGE_BY_SENSITIVITY_VALUE: MappingProxyType[str, str] = MappingProxyType(
    {
        "restricted": "{location}: column {table}.{column} cannot appear in {surface} (restricted classification).",
        "hidden": "{location}: column {table}.{column} cannot appear in {surface} (hidden classification).",
    }
)

INTENT_NON_SELECTABLE_PREDICATE_MESSAGE_DEFAULT: str = (
    "{location}: column {table}.{column} cannot appear in {surface} (sensitivity policy)."
)

_INTENT_DATE_UNIT_AMOUNT_VALUE: dict[str, Any] = {
    "type": "object",
    "required": ["unit", "amount"],
    "properties": {
        "unit": {"enum": ["day", "week", "month", "quarter", "year"]},
        "amount": {"type": "integer", "minimum": 1},
    },
}

_INTENT_WHERE_ITEM_ALLOF: list[dict[str, Any]] = [
    {
        "if": {
            "allOf": [
                {"required": ["value_type"]},
                {"properties": {"value_type": {"const": "date_diff"}}},
            ]
        },
        "then": {"properties": {"value": _INTENT_DATE_UNIT_AMOUNT_VALUE}},
    },
    {
        "if": {
            "allOf": [
                {"required": ["value_type"]},
                {"properties": {"value_type": {"const": "date_window"}}},
            ]
        },
        "then": {"properties": {"value": _INTENT_DATE_UNIT_AMOUNT_VALUE}},
    },
]

INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["tables"],
    "additionalProperties": False,
    "properties": {
        "tables": {
            "oneOf": [
                {"type": "array", "items": {"type": "string"}},
                {"type": "string"},
            ]
        },
        "select_cols": {
            "type": "array",
            "items": {"oneOf": [{"type": "string"}, {"type": "object"}]},
        },
        "group_by_cols": {"type": "array", "items": {"type": "string"}},
        "order_by_cols": {
            "type": "array",
            "items": {
                "oneOf": [
                    {"type": "string"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "expr": {"oneOf": [{"type": "string"}, {"type": "object"}]},
                            "direction": {
                                "oneOf": [
                                    {"type": "string", "enum": ["asc", "desc", "ASC", "DESC"]},
                                    {"type": "null"},
                                ]
                            },
                            "nulls": {"type": "string", "enum": ["first", "last"]},
                        },
                    },
                ]
            },
        },
        "where": {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": ["and", "or"]},
                        "predicates": {"type": "array"},
                        "groups": {"type": "array"},
                    },
                },
                {"type": "null"},
            ]
        },
        "having": {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": ["and", "or"]},
                        "predicates": {"type": "array"},
                        "groups": {"type": "array"},
                    },
                },
                {"type": "null"},
            ]
        },
        "cte_steps": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["cte_name", "select_cols", "output_columns"],
                "additionalProperties": False,
                "properties": {
                    "cte_name": {"type": "string"},
                    "description": {"type": "string"},
                    "tables": {"type": "array", "items": {"type": "string"}},
                    "grain": {
                        "type": "string",
                        "enum": ["row_level", "grouped", "scalar"],
                    },
                    "select_cols": {
                        "type": "array",
                        "items": {
                            "oneOf": [
                                {"type": "string"},
                                {
                                    "type": "object",
                                    "required": ["expr"],
                                    "properties": {
                                        "expr": {"type": "string"},
                                        "alias": {"type": "string"},
                                    },
                                },
                            ]
                        },
                    },
                    "group_by_cols": {"type": "array"},
                    "order_by_cols": {"type": "array"},
                    "where": {
                        "oneOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "op": {"type": "string", "enum": ["and", "or"]},
                                    "predicates": {"type": "array"},
                                    "groups": {"type": "array"},
                                },
                            },
                            {"type": "null"},
                        ]
                    },
                    "having": {
                        "oneOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "op": {"type": "string", "enum": ["and", "or"]},
                                    "predicates": {"type": "array"},
                                    "groups": {"type": "array"},
                                },
                            },
                            {"type": "null"},
                        ]
                    },
                    "output_columns": {
                        "type": "array",
                        "items": {"type": "string", "pattern": "^[a-z_][a-z0-9_]*$"},
                    },
                    "window_registry": {"type": "array"},
                    "case_registry": {"type": "array"},
                    "distinct_on": {"type": "array", "items": {"type": "string"}},
                    "preserve_tables": {"type": "array", "items": {"type": "string"}},
                    "emission": {
                        "type": "string",
                        "enum": ["semi_join", "anti_join", "join_table", "scalar_subquery"],
                    },
                    "limit": {"oneOf": [{"type": "integer"}, {"type": "null"}]},
                    "param_values": {"type": "object"},
                },
            },
        },
        "window_registry": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "registry_id": {"type": "string"},
                    "label": {"type": "string"},
                    "window_spec": {
                        "type": "object",
                        "properties": {
                            "function": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                            "partition_by": {"type": "array"},
                            "order_by": {"type": "array"},
                            "argument": {},
                            "numeric_argument": {"oneOf": [{"type": "integer"}, {"type": "null"}]},
                            "frame_kind": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                            "frame_start": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                            "frame_end": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                            "frame_start_offset": {"oneOf": [{"type": "integer"}, {"type": "null"}]},
                            "frame_end_offset": {"oneOf": [{"type": "integer"}, {"type": "null"}]},
                        },
                    },
                },
            },
        },
        "case_registry": {"type": "array"},
        "limit": {"oneOf": [{"type": "integer"}, {"type": "null"}]},
        "natural_language": {"oneOf": [{"type": "string"}, {"type": "null"}]},
        "distinct_on": {"type": "array", "items": {"type": "string"}},
        "preserve_tables": {"type": "array", "items": {"type": "string"}},
        "grain": {
            "type": "string",
            "enum": ["row_level", "grouped", "scalar"],
        },
        "param_values": {"type": "object"},
        "expected_rows": {"type": "string"},
        "chosen_join_candidate_id": {"type": "string"},
        "chosen_join_path_signature": {"type": "array"},
        "column_map": {"type": "object"},
        "comparison_only_tables": {"type": "array"},
        "distinct_select_index": {"type": "integer"},
        "sql_param": {"type": "string"},
        "schema_invalid": {"type": "boolean"},
    },
}

LOGICAL_INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["tables", "select"],
    "additionalProperties": False,
    "properties": {
        "tables": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "select": {"type": "string", "minLength": 1},
        "where": {"type": "string"},
        "group_by": {"type": "string"},
        "having": {"type": "string"},
        "order_by": {"type": "string"},
        "limit": {"oneOf": [{"type": "string"}, {"type": "null"}]},
        "window": {"type": "string"},
        "case": {"type": "string"},
        "cte_steps": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "tables", "select"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "tables": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "select": {"type": "string", "minLength": 1},
                    "where": {"type": "string"},
                    "group_by": {"type": "string"},
                    "having": {"type": "string"},
                    "order_by": {"type": "string"},
                    "limit": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                    "window": {"type": "string"},
                    "case": {"type": "string"},
                },
            },
        },
    },
}

MOCK_FIXTURE_STUB_SCHEMA_LITERALS: dict[str, str] = {"owner": "{}", "consumer": "{}"}

INTENT_SYSTEM_SEMANTIC_JOIN_INTERMEDIATES: str = (
    "Semantically-required join intermediates. "
    "The join enumerator only considers the SHORTEST FK paths connecting the tables in `tables`, and any table listed in `tables` whose columns are not referenced elsewhere in the intent (in `select_cols`, filters, `group_by_cols`, `having`, `order_by_cols`, or registry expressions) is pruned before enumeration. "
    "Therefore, listing an extra table in `tables` alone is never enough to force the join through it. "
    "When the descriptions on the endpoint tables (or on tables related to them) indicate that two semantically distinct FK paths connect the same pair of tables, and one path is strictly LONGER than the other, the shorter path will always be chosen unless you force the longer one explicitly. "
    "To force the longer path, reference at least one column from each required intermediate table inside the intent. Adding such a column to `select_cols` is preferred — typically the intermediate table's primary key or its most descriptive column — because it both keeps the table from being pruned and surfaces context to the user that this specific semantic was intended. "
    f"Concretely, suppose the intent connects `{INSTRUCTIONAL_TABLE_PLACEHOLDER}` to `{INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER}` and the descriptions indicate two semantic "
    f"paths: `{INSTRUCTIONAL_TABLE_PLACEHOLDER} -> {INSTRUCTIONAL_LINK_TABLE_PLACEHOLDER} -> {INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER}` for one semantic and "
    f"`{INSTRUCTIONAL_TABLE_PLACEHOLDER} -> {INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER} -> {INSTRUCTIONAL_BRIDGE_TABLE_PLACEHOLDER} -> {INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER}` for another. "
    f"If the question requires the second semantic, add at least one column from `{INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER}` or one column from "
    f"`{INSTRUCTIONAL_BRIDGE_TABLE_PLACEHOLDER}` (preferably to `select_cols`, ideally a primary key or the most descriptive column from each). "
    "If the question requires the first semantic, no extra columns are needed because the shorter path is already what the resolver will pick. When descriptions are silent about distinct paths, or when only one FK path exists between the endpoints, no extra columns are needed. "
    "COUNT(*) does not count as a column reference for prune purposes; when more than one physical table is in scope, emit COUNT(table.primary_key_column) on the table being counted (and reference any other required tables via qualified columns) instead of COUNT(*)."
)

INTENT_SYSTEM_NATURAL_LANGUAGE_FIELD_SHAPE: str = (
    "`natural_language` field shape. Write one short conversational sentence a non-technical user would recognize as their question restated — what answer they will get, in everyday words. "
    "Never name physical tables or columns, never use qualified identifiers like table.column, never narrate join paths or FK arrows, never mention CTE names or SQL operators/clauses. "
    'The reader of this field is the end user being asked "I understood: …; is this what you wanted?"'
)

INTENT_CRITICAL_RULES: tuple[str, ...] = (
    "Every JSON object uses only keys listed in structural_json_keys for its structural type (the root object follows RuntimeIntent; nested rows follow their named type; SQL expressions are always a single string field such as expr or left_expr per structural_json_keys.sql_expression). "
    "Do not emit extra sibling keys at any level.",
    f"Qualify every column reference as {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} using names from the schema text and allowed_tables; "
    "never emit bare column names except the bare wNN and cNN registry tokens on select_cols that point at window_registry and case_registry entries. "
    "Qualify columns inside every window_registry.window_spec.partition_by, order_by, and argument, and inside every case_registry case_when branch (condition sides, result, else_result).",
    "Use only tables and columns from the provided schema text and allowed_tables; do not invent identifiers.",
    "Join path discovery, foreign-key traversal, and bridge or junction tables are handled only by the downstream engine after this JSON is parsed; never refuse or shrink the intent because tables look disconnected in the structural payload.",
    "Do not judge whether the question is answerable from schema connectivity; translate the interpret prose into IR using only the listed interpret tables and their columns.",
    "Grain must match structure: grouped requires group_by_cols and aggregation in select_cols; row_level means no GROUP BY and no aggregation in select_cols; scalar is a single aggregated result with no GROUP BY.",
    "Row-level predicates belong in where; predicates on aggregates belong in having. "
    "Never put join predicates in where or having.",
    (
        f"A where predicate may compare {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} to "
        f"{INSTRUCTIONAL_OTHER_QUALIFIED_COLUMN_PLACEHOLDER} when the question asks for a comparison "
        "between two values. This is a comparison, not a relationship: it states how two values rank against each other, never how the two tables connect. Join paths remain discovered downstream from foreign keys."
    ),
    "SUM and AVG apply only to numeric measure columns; use COUNT for non-measure columns.",
    "Columns whose schema type is unknown may appear in select_cols only; do not filter or aggregate them.",
    "Nested aggregation is forbidden; compute inner aggregates in a CTE step, then aggregate in the main query.",
    "Do not use EXTRACT(EPOCH FROM ...) for time differences; subtract date columns directly or use supported date functions.",
    f"CTE output_columns are snake_case alias tokens matching ^[a-z_][a-z0-9_]*$; never qualified {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER}, never function call text, never AS clauses; align positionally with select_cols. Reference CTE outputs only via cteN.<output_columns_token> in window_registry, where, having, order_by_cols, and select_cols.",
    'Relative date-window filters use value_type date_window with value {"unit", "amount"}; column-to-column date spans use value_type date_diff with value {"unit", "amount"}. Use singular unit names. '
    'Absolute calendar bounds in date_window use ISO 8601 start/end strings such as "2020-01-15".',
    "Integer columns with schema role temporal represent day-count durations; compare them to elapsed day expressions (date subtraction or keyword minus date), not as calendar dates.",
    "BETWEEN uses op between with value [lower, upper]. NULL checks use op is null or is not null without a value field.",
    (
        "Negated comparisons (!=, not in, not like) include rows where the filtered column value is unknown when that column is nullable; use is null or is not null when the question is only about missing values."
    ),
    (
        "Encode WHERE and HAVING boolean logic as PredicateGroup trees: op and/or with predicate leaves and nested groups (nesting depth capped at 3). "
        "For (predicate_A OR predicate_B) emit op or with two predicate leaves. "
        "For (predicate_A AND predicate_B) OR (predicate_C AND predicate_D) emit op or with two op and groups, each holding two predicate leaves."
    ),
    "window_registry defines registry_id and window_spec; select_cols reference entries with bare wNN tokens. "
    "Never put a window_spec key on a select_cols entry.",
    "case_registry defines registry_id and case_when with non-empty branches; select_cols reference entries with bare cNN tokens. "
    "When the question asks for conditional labels or buckets over columns, populate case_registry rather than dropping the derived column.",
    "Reject drafts that reference a bare wNN or cNN with no matching registry_id in the same scope's window_registry or case_registry.",
    "Do not emit param_key, param_values, limit_param_key, or other engine-owned fields listed in ir_assembly_rules.",
    "Ops in where and having must be taken from the allowlist for that clause kind in operator_reference and capabilities.",
    "SELECT DISTINCT prefixes the column expr with the bare token DISTINCT and a space "
    f"('DISTINCT {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER}'). Use COUNT(DISTINCT {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER}) for distinct counts; "
    f"for COUNT(DISTINCT CONCAT({INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER}, {INSTRUCTIONAL_OTHER_QUALIFIED_COLUMN_PLACEHOLDER})) emit Shape A: COUNT MulGroup with distinct=true whose single multiply child is a CONCAT MulGroup (scalar_func='concat'); "
    "do not wrap DISTINCT around arbitrary expressions except as COUNT(DISTINCT ...). "
    "Do not embed COUNT(*) inside arithmetic subexpressions—use COUNT(*) only as a top-level aggregate where appropriate.",
    f"Arithmetic combines expressions with +, -, *, /; aggregations may wrap arithmetic (SUM({INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} * {INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_OTHER_COLUMN_PLACEHOLDER})). "
    f"Subtract date columns directly ({INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_DATE_COLUMN_PLACEHOLDER} - "
    f"{INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_OTHER_DATE_COLUMN_PLACEHOLDER}) for day differences. "
    f"Add or subtract integer day counts from date columns ({INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_DATE_COLUMN_PLACEHOLDER} + "
    f"{INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_INTEGER_COLUMN_PLACEHOLDER} or "
    f"{INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_DATE_COLUMN_PLACEHOLDER} + 7) for due-date comparisons.",
    "String concatenation uses CONCAT(expr1, ' ', expr2, ...) in expr strings; do not use the SQL || operator (pipe-pipe).",
    f"Apply scalar functions such as ROUND after aggregates when needed (ROUND(SUM({INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER}), 2)).",
    "Use exact identifiers from the provided schema text; never leave synthetic shape tokens from this prompt "
    f"({INSTRUCTIONAL_TABLE_PLACEHOLDER}, {INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER}, column, "
    f"{INSTRUCTIONAL_DATE_COLUMN_PLACEHOLDER}, {INSTRUCTIONAL_OTHER_DATE_COLUMN_PLACEHOLDER}), generic instructional tokens (table_N, column_N), or angle-bracket markup in expressions.",
)

INTENT_PARSE_RULES_APPEND: tuple[str, ...] = (
    "output_format lists every required top-level key; use [] for empty arrays and null for unused scalars.",
    "natural_language is required: describe exactly what the emitted IR returns (tables, columns, filters, aggregates) using real entity names; never include refusal, unavailability, permission, or denial language.",
    (
        "Encode WHERE and HAVING boolean logic as PredicateGroup trees: op and/or with predicate leaves and nested groups. "
        "For (predicate_A OR predicate_B) emit a where or having PredicateGroup with op or and two predicate leaves. "
        "For (predicate_A AND predicate_B) OR (predicate_C AND predicate_D) emit op or with two op and groups, each holding two predicate leaves. "
        "Use where and having PredicateGroup trees for all intents. "
        "Use qualified column references from the schema for left_expr; bind literals via value placeholders as elsewhere in these rules. "
        "Each where or having predicate leaf must include op."
    ),
)

INTENT_FORMAT_REPAIR_JSON_RULES: tuple[str, ...] = (
    "Return JSON only with no prose, markdown fences, or trailing commas.",
    "Preserve intent content while correcting syntax; ensure all required fields are present.",
    "Use [] for empty array fields and null for absent optional scalars.",
)

INTENT_INTERPRET_SYSTEM: str = (
    "You are the Interpret stage. Your role is to lay out a thinking pathway for answering the data question, not to author a query structure, join plan, or runtime representation. "
    "Output ONLY valid JSON matching interpret_plan_schema in the user payload. "
    "approach: plain-language steps describing entities to return, row conditions, grouping or per-entity breakdown, ordering or ranking, row caps, aggregates, conditional labels, and time-window or duration reasoning. Do not use "
    f"{INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} syntax, SQL, IR tokens, join paths, or set operators. "
    "tables: semantic base tables whose concepts are needed; omit junction tables unless the many-to-many set itself is the answer. "
    "grounding: traceability only. Each ref must be either a table name listed in tables, or a qualified "
    f"{INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} from the payload — never a bare column name, bare enum token, or unqualified identifier. "
    "Record each table in tables, and any column or enum column specifically driving filter, having, or group_by reasoning named in approach as "
    f"{INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER}. Do not enumerate select output columns in grounding. Each entry is ref plus used_for. "
    "When the question cannot be answered from the tables and columns in this payload (for example it needs entities not present in the schema), set schema_invalid true, put a short note in missing, and leave approach as an empty string, tables as [], and grounding as []. Do not invent substitute tables. "
    "When schema binding is only partially incomplete but you can still outline a usable plan on in-scope tables, set schema_invalid true as a UI signal and still complete approach and tables from in-scope names only. "
    "Use only names from the payload. Express only computations in supported_capabilities; reformulate unsupported constructs in plain language."
)

INTENT_GROUND_SYSTEM: str = (
    "You are the Ground stage. Your role is to convert interpret_plan into logical intent JSON: referenced tables plus natural-language descriptions of what belongs in each clause field. "
    "Output ONLY valid JSON matching logical_intent_json_schema in the user payload. "
    "interpret_plan is the semantic source; schema_literal_json supplies identifier descriptions, roles, and types. "
    "Follow nl_conventions. Never emit SQL, IR operators, EXISTS, NOT EXISTS, UNION, INTERSECT, EXCEPT, join paths, or join types. The Compose stage never sees the question or schema payload; your JSON is the sole semantic contract for literals, table choice, and clause content. "
    "Populate select, filter, group_by, having, order_by, limit, window, and case as natural language using qualified "
    f"{INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} where needed. Copy every literal into the matching prose field. "
    "Scope preservation: after structural encoding, a table remains in join scope only when a qualified column from that table appears in clause prose at that scope; name columns in the clause that uses them, not in select when only needed for join reachability. "
    "Bridge tables: name bridge columns in filter prose when existence or membership is required. "
    "Per-entity breakdown maps to group_by, not row-level DISTINCT. "
    "cte_steps: each step has name, tables, and the same prose fields. tables may list base schema tables and prior cte_steps names this step reads from. Express only supported_capabilities."
)

INTENT_COMPOSE_SYSTEM: str = (
    "You are the Compose stage. Your role is to encode logical_intent natural language into runtime intermediate representation JSON. You do not re-read the question, re-plan semantics, add tables, or author joins. Output ONLY valid JSON matching output_format. "
    "logical_intent prose fields are the sole semantic source. Map each populated prose field to its IR slot using logical_to_ir_field_map in the user payload. Use nl_phrase_mappings and only identifiers from structural_schema_for_chosen_tables. "
    "Translate select, filter, group_by, having, order_by, limit, window, and case prose mechanically. Emit a "
    f"qualified column reference in IR for every {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} named in the matching clause prose at that scope. "
    "For cte_steps, set cte_name from name and tables from tables; tables may contain base schema tables and prior interpret step names matching earlier cte_steps names. "
    "Emit only operators in operator_reference, value types in value_type_reference, and constructs in supported_capabilities. NEVER emit param_key, param_values, or harvested-literal mappings; emit raw_value for Where and Having literals; leave select and aggregate aliases empty strings. "
    "natural_language must be one conversational sentence for the end user (no table.column names, join paths, FK arrows, CTE names, or SQL clause words)."
)

PROMPT_NEUTRALITY_AUDIT_CONSTANTS: frozenset[str] = frozenset(
    {
        "INTENT_INTERPRET_SYSTEM",
        "INTENT_GROUND_SYSTEM",
        "INTENT_COMPOSE_SYSTEM",
        "INTENT_CRITICAL_RULES",
        "INTENT_PARSE_RULES_APPEND",
        "INTENT_FORMAT_REPAIR_JSON_RULES",
        "COMPOSE_IR_ASSEMBLY_RULES",
        "COMPOSE_NL_TO_IR_GUIDANCE",
        "COMPOSE_NL_PHRASE_MAPPINGS",
        "REPAIR_INSTRUCTIONS",
        "LOGICAL_DECOMPOSITION_GUIDANCE",
        "FORMAT_STRUCTURAL_GUIDANCE",
        "COMPOSE_SUPPORTED_CAPABILITIES",
        "FEDERATION_COMPOSE_SUPPORTED_CAPABILITIES",
        "INTERPRET_SUPPORTED_CAPABILITIES",
        "GROUND_SUPPORTED_CAPABILITIES",
        "FEDERATION_COMPOSITE_RECONCILIATION_NOTE",
        "INTENT_SYSTEM_SEMANTIC_JOIN_INTERMEDIATES",
        "INTENT_SYSTEM_NATURAL_LANGUAGE_FIELD_SHAPE",
        "SEED_QUESTION_CLARIFY_SYSTEM",
        "INTERPRET_CARDINALITY_RELATIONSHIP_RULE",
        "INTERPRET_SHARED_PK_TABLE_SCOPE_RULE",
        "INTERPRET_SINGLE_OUTPUT_WINDOW_NO_CTE_RULE",
        "SCHEMA_CLASSIFY_SYSTEM",
        "SCHEMA_ENTITY_ENRICH_SYSTEM",
        "SCHEMA_CONSISTENCY_REFINE_SYSTEM",
        "KNOWLEDGE_NOTES_EXTRACT_SYSTEM",
        "KNOWLEDGE_NOTES_EXTRACT_REPAIR_SYSTEM",
        "DOMAIN_KNOWLEDGE_SPACE_MERGE_SYSTEM",
        "DOMAIN_KNOWLEDGE_FEDERATION_MERGE_SYSTEM",
        "STRUCTURAL_KNOWLEDGE_SPACE_MERGE_SYSTEM",
        "STRUCTURAL_KNOWLEDGE_FEDERATION_MERGE_SYSTEM",
        "QUESTION_VALIDATION_SYSTEM",
        "META_SCHEMA_CATALOG_SYSTEM",
        "META_DOMAIN_KNOWLEDGE_SYSTEM",
        "META_SCHEMA_AND_KNOWLEDGE_SYSTEM",
        "PARAM_QUESTION_REMAP_SYSTEM",
        "DESCRIPTION_REFINER_SYSTEM",
        "DOMAIN_KNOWLEDGE_REFINER_SYSTEM",
        "QSIM_FILL_SYSTEM",
        "QUESTION_CANONICALIZE_SYSTEM",
        "WARMUP_PARAPHRASES_BY_STYLE_SYSTEM",
        "WARMUP_FREEFORM_QUESTIONS_SYSTEM",
        "QUESTION_FROM_SQL_SYSTEM",
    }
)

REFUSAL_CATALOGUE: dict[str, dict[str, str]] = {
    DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED: {
        "user_text": ("Unable to locate the requested data. Please contact your administrator."),
        "reformulation_hint": ("Try rephrasing using only tables and columns available in your current schema view."),
    },
    DIAGNOSTIC_CODE_REFUSAL_SCOPE_VIOLATION: {
        "user_text": (
            "This question cannot be answered with the information currently available.\n\n"
            "Try rephrasing to ask about tables and columns you can see in the schema."
        ),
        "reformulation_hint": ("Try rephrasing to ask about tables and columns you can see in the schema."),
    },
    DIAGNOSTIC_CODE_REFUSAL_INVALID_QUESTION: {
        "user_text": (
            "I could not pin this question to specific tables or columns.\n\n"
            "Try naming the entity you care about, the metric you want, and any filter such as a date range, status, or region."
        ),
        "reformulation_hint": (
            "Try naming the entity you care about, the metric you want, and any filter such as a date range, status, or region."
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_CONVERSATIONAL_DENY: {
        "user_text": "I can only help with questions about your data.",
        "reformulation_hint": "",
    },
    DIAGNOSTIC_CODE_REFUSAL_UNMAPPABLE_QUESTION: {
        "user_text": (
            "I could not pin this question to specific tables or columns.\n\n"
            "Try naming the entity you care about, the metric you want, and any filter such as a date range, status, or region."
        ),
        "reformulation_hint": (
            "Try naming the entity you care about, the metric you want, and any filter such as a date range, status, or region."
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_OPERATION_NOT_SUPPORTED: {
        "user_text": "This type of operation is not supported. I can only answer questions that read from your data.",
        "reformulation_hint": "",
    },
    DIAGNOSTIC_CODE_REFUSAL_INSUFFICIENT_KNOWLEDGE: {
        "user_text": (
            "The available schema descriptions and domain knowledge do not contain enough information to answer this question."
        ),
        "reformulation_hint": (
            "Try asking about a specific table, column, or glossary term that appears in your schema or domain knowledge."
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_PARSE_FAILURE: {
        "user_text": (
            "I could not understand the structure of this question.\n\n"
            "Try naming specific tables or columns, keeping filters simple and references clear."
        ),
        "reformulation_hint": (
            "Please rephrase your question.\n\n"
            "Tips: mention specific tables or columns, keep filters simple, and avoid ambiguous references.\n"
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_DECLINED_SCHEMA: {
        "user_text": (
            "The proposed table and column mapping was declined.\n\n"
            "Try rephrasing using tables and columns that exist in this database, or ask about a related concept."
        ),
        "reformulation_hint": (
            "Please rephrase your question.\n"
            "Tips: use tables and columns that exist in this database, or ask about a related concept."
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_UNAVAILABLE: {
        "user_text": (
            "These tables could not be connected: {tables}. "
            "Declare a foreign-key or semantic relationship between them, or ask using fewer tables."
        ),
        "reformulation_hint": (
            "These tables cannot be joined with the relationships currently in this schema.\n\n"
            "Tips: declare a foreign-key or semantic link between the tables named in the error, or narrow the question to tables that already connect.\n"
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_JOIN_PATH_TIE_CAP: {
        "user_text": (
            "Too many equally short join paths between {source_table} and {target_table} ({path_count} paths; limit {ceiling}). Narrow the tables in your question or declare which relationship to use."
        ),
        "reformulation_hint": (
            "Too many equally short join paths were found between the named tables.\n\n"
            "Tips: narrow the question to fewer tables or declare which relationship should connect them.\n"
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_AGGREGATE_FAN_OUT: {
        "user_text": (
            "This aggregate would duplicate parent rows because of how the tables connect.\n\n"
            "Try grouping at the parent grain first, or use a join path that does not multiply rows."
        ),
        "reformulation_hint": (
            "Tips: group at the parent grain first, or choose a join path that does not multiply rows.\n"
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_HOP_CEILING: {
        "user_text": (
            "This comparison would require joining across too many tables.\n\n"
            "Try comparing values on tables that are closer together in the schema."
        ),
        "reformulation_hint": (
            "Tips: compare values on tables that are closer together, or simplify the comparison.\n"
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_CTE_CAP: {
        "user_text": (
            "This question needs too many intermediate query steps.\n\n"
            "Try splitting the question into smaller parts or simplifying the logic."
        ),
        "reformulation_hint": ("Tips: split the question into smaller parts or simplify the intermediate logic.\n"),
    },
    DIAGNOSTIC_CODE_REFUSAL_CAPABILITY_GAP: {
        "user_text": (
            "This question shape cannot be answered with the databases currently available.\n\n"
            "Try a simpler question or ask on each source individually."
        ),
        "reformulation_hint": (
            "Please rephrase your question.\n\nTips: try a simpler question or ask on each source individually.\n"
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_CLAUSE_WIDENED_ROWSET: {
        "user_text": (
            "A limit, sort, or distinct cannot be applied cleanly when a join multiplies rows.\n\n"
            "Group first or simplify joins before limiting or deduplicating results."
        ),
        "reformulation_hint": (
            "Please rephrase or retry.\n\n"
            "Tips: group first or simplify joins before limiting, sorting, or deduplicating results.\n"
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_PROBE_CTE_PLACEMENT: {
        "user_text": (
            "A filter step cannot be used in this join position.\n\n"
            "Restructure the question so filtering happens on the correct side of the join."
        ),
        "reformulation_hint": (
            "Please rephrase or retry.\n\n"
            "Tips: restructure the question so filtering happens on the correct side of the join.\n"
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_NULL_IN_NEGATED_LIST: {
        "user_text": (
            "This filter cannot be expressed: a NOT IN list cannot include null. "
            "Ask whether the column is null or not, or name only the non-null values to exclude."
        ),
        "reformulation_hint": ("Ask whether the column is null or not, or name only the non-null values to exclude.\n"),
    },
    DIAGNOSTIC_CODE_REFUSAL_OPAQUE_EXPR: {
        "user_text": (
            "This question uses an expression structure that cannot be compiled safely.\n\n"
            "Rephrase using explicit columns, filters, and aggregates supported by the schema."
        ),
        "reformulation_hint": (
            "Please rephrase your question.\n\n"
            "Tips: use explicit table.column references and supported filters or aggregates.\n"
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN: {
        "user_text": (
            "This filter cannot be expressed: the column stores dates without time-of-day, so hour, minute, or second windows cannot be answered. Ask for a day-level window instead."
        ),
        "reformulation_hint": "Ask for a day-level date window instead of hours, minutes, or seconds.\n",
    },
    DIAGNOSTIC_CODE_REFUSAL_AMBIGUOUS_DATE_LITERAL: {
        "user_text": (
            "This filter cannot be expressed: the date bound is ambiguous. "
            "Use ISO 8601 form such as 2020-01-15 or 2020-01-15T14:30:00."
        ),
        "reformulation_hint": ("Use an unambiguous ISO 8601 date such as 2020-01-15 or 2020-01-15T14:30:00.\n"),
    },
    DIAGNOSTIC_CODE_REFUSAL_UNION_COLUMN_MISSING: {
        "user_text": (
            "A column needed for this answer is missing from one or more databases in the group.\n\n"
            "Try asking over the sources that have the column, or declare a shared column mapping."
        ),
        "reformulation_hint": (
            "Try asking over the sources that have the column, or declare a shared column mapping.\n"
        ),
    },
    DIAGNOSTIC_CODE_REFUSAL_UNSUPPORTED_COLUMN_TYPE: {
        "user_text": (
            "This question cannot be answered: the {column} column has an unsupported data type and cannot be filtered or aggregated."
        ),
        "reformulation_hint": ("Try asking about a different column or a supported data type.\n"),
    },
    DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT: {
        "user_text": ("This question refers to information that is not available in this context."),
        "reformulation_hint": ("Try rephrasing to ask about information that is available in this context.\n"),
    },
}

REPHRASE_HINT_MESSAGES.update(
    {
        key: REFUSAL_CATALOGUE[code]["reformulation_hint"] or REFUSAL_CATALOGUE[code]["user_text"]
        for key, code in REPHRASE_HINT_REFUSAL_CODES.items()
    }
)

PERMISSION_DENIED_USER_MESSAGE: str = REFUSAL_CATALOGUE[DIAGNOSTIC_CODE_REFUSAL_PERMISSION_DENIED]["user_text"]

REFUSAL_NOT_AVAILABLE_IN_CONTEXT_MESSAGE: str = REFUSAL_CATALOGUE[DIAGNOSTIC_CODE_REFUSAL_NOT_AVAILABLE_IN_CONTEXT][
    "user_text"
]

REFUSAL_NULL_IN_NEGATED_LIST_MESSAGE: str = REFUSAL_CATALOGUE[DIAGNOSTIC_CODE_REFUSAL_NULL_IN_NEGATED_LIST]["user_text"]

REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN_MESSAGE: str = REFUSAL_CATALOGUE[
    DIAGNOSTIC_CODE_REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN
]["user_text"]

REFUSAL_AMBIGUOUS_DATE_LITERAL_MESSAGE: str = REFUSAL_CATALOGUE[DIAGNOSTIC_CODE_REFUSAL_AMBIGUOUS_DATE_LITERAL][
    "user_text"
]

REFUSAL_UNSUPPORTED_COLUMN_TYPE_MESSAGE: str = REFUSAL_CATALOGUE[DIAGNOSTIC_CODE_REFUSAL_UNSUPPORTED_COLUMN_TYPE][
    "user_text"
]

INELIGIBLE_ANSWERABLE_HINTS_BY_CODE: dict[str, str] = {
    "no_tables": "Try naming the entities and metrics you need from the schema.",
    "space_scope": "Try asking within the active space tables, or widen the space definition.",
    "projection_not_single_member": (
        "Try asking for columns that live on one member, or declare a logical column mapping."
    ),
    "union_column_missing": (
        "Try asking over the members that do have the column, or declare a logical column mapping spanning all members."
    ),
    "undeclared_join_path": ("Try naming entities that share a declared relationship in the federation manifest."),
    "cross_source_aggregate": "Try asking for the metric per entity or per category separately.",
    "cross_source_or_filter": "Try splitting the question into separate filters on each entity.",
    "cross_source_where_group_disjunction": "Try splitting the question into separate filters on each entity.",
    "cross_source_distinct": "Try asking for the list or count from one entity, then combine results manually.",
    "cross_source_order_by": "Try asking for the list or count from one entity, then combine results manually.",
    "cross_source_window": "Try ranking or windowing within one entity before combining, or ask without the window.",
    "cross_source_correlated_subquery": "Try expressing the lookup as a join or an intermediate step on one source.",
    "cross_source_scalar_subquery": "Try expressing the lookup as a join or an intermediate step on one source.",
    "cross_source_having": "Try asking for the metric per entity or per category separately.",
    "cross_source_distinct_on": "Try asking for one row per entity within each member before combining.",
    "cross_source_semijoin": "Try declaring a join key between the probe and driver tables in the federation manifest.",
    "cross_source_antijoin": "Try declaring a join key between the probe and driver tables in the federation manifest.",
    "cross_source_predicate_disjunction": "Try splitting the question into separate filters on each entity.",
    "semi_join_unsupported": "Try rephrasing using constructs every member supports, or ask on one member at a time.",
    "anti_join_unsupported": "Try rephrasing using constructs every member supports, or ask on one member at a time.",
    "distinct_on_unsupported": "Try rephrasing using constructs every member supports, or ask on one member at a time.",
    "preserve_tables_unsupported": "Try rephrasing using constructs every member supports, or ask on one member at a time.",
    "nested_predicate_groups": "Try simplifying boolean logic to AND-only filters on one member at a time.",
    "distinct_on_requires_order_by": "Try specifying how to rank rows within each partition (for example by date descending).",
    "unattributable_raw_sql": "Try rephrasing using explicit table.column references from the schema.",
    "member_capability": "Try rephrasing using constructs every member supports, or ask on one member at a time.",
    "unknown": "Try asking for each part separately, or rephrasing without cross-source aggregation.",
}

SCHEMA_FIELD_DESCRIPTION: str = "description"

SCHEMA_FIELD_ROLE: str = "role"

SCHEMA_FIELD_TYPE: str = "type"

SCHEMA_FIELD_TRUTH_VALUE: str = "truth_value"

SCHEMA_FIELD_KEYS: str = "keys"

SCHEMA_FIELD_ENUM: str = "enum"

SCHEMA_FIELD_SAMPLES: str = "samples"

SCHEMA_FIELD_DERIVED: str = "derived"

SCHEMA_FIELD_RAW_TYPE: str = "raw_type"

SCHEMA_INSTRUCTION_SCRUB_REPLACEMENT: str = "[scrubbed]"

SANDBOX_DEFAULT_SCHEMA_SQL: str = "rental_shop.sql"

INTERPRET_PROMPT_KEY_ORDER: tuple[str, ...] = (
    "task",
    "interpret_plan_schema",
    "supported_capabilities",
    "schema_domain",
    "domain_context",
    "question",
    "prior_question_feedback",
    "prior_user_corrections",
)

GROUND_PROMPT_KEY_ORDER: tuple[str, ...] = (
    "task",
    "logical_intent_json_schema",
    "nl_conventions",
    "logical_schema_rules",
    "supported_capabilities",
    "schema_literal_json",
    "interpret_plan",
    "domain_context",
    "question",
    "logical_decomposition_guidance",
    "prior_question_feedback",
    "prior_user_corrections",
    "prior_grounding_failures",
)

COMPOSE_PROMPT_KEY_ORDER: tuple[str, ...] = (
    "task",
    "field_specifications",
    "output_format",
    "structural_json_keys",
    "critical_rules",
    "parse_rules_append",
    "format_structural_guidance",
    "supported_capabilities",
    "nl_phrase_mappings",
    "nl_to_ir_guidance",
    "ir_assembly_rules",
    "operator_reference",
    "value_type_reference",
    "structural_pattern_rules",
    "compose_priority_rules",
    "logical_to_ir_field_map",
    "cte_tables_encoding",
    "structural_schema_for_chosen_tables",
    "logical_intent",
)

SEMANTIC_REPAIR_PROMPT_KEY_ORDER: tuple[str, ...] = (
    "task",
    "critical_rules",
    "field_specifications",
    "structural_json_keys",
    "output_format",
    "schema_info",
    "domain_context",
    "errors_to_fix",
    "suggestions",
    "current_intent",
    "question",
    "prior_question_feedback",
)

FORMAT_REPAIR_PROMPT_KEY_ORDER: tuple[str, ...] = (
    "task",
    "field_specifications",
    "structural_json_keys",
    "instructions",
    "output_format",
    "question",
    "invalid_response",
    "parse_error",
)

PARSE_PROMPT_KEY_ORDER: tuple[str, ...] = (
    "task",
    "field_specifications",
    "structural_json_keys",
    "expression_format",
    "rules",
    "logical_decomposition_guidance",
    "format_structural_guidance",
    "output_format",
    "operator_reference",
    "value_type_reference",
    "naming_conventions",
    "schema_summary",
    "domain_context",
    "allowed_tables",
    "question",
    "prior_question_feedback",
)

FUZZY_REUSE_PARAM_PROMPT_KEY_ORDER: tuple[str, ...] = (
    "task",
    "extraction_rules",
    "output_format",
    "matched_question",
    "matched_values",
    "param_keys",
    "param_slots",
    "question",
)

DISPLAY_ALIAS_PROMPT_KEY_ORDER: tuple[str, ...] = (
    "task",
    "columns",
    "question",
)

JOIN_CHOICE_PROMPT_KEY_ORDER: tuple[str, ...] = (
    "task",
    "output_format",
    "scopes",
    "deterministic_sql",
    "question",
    "prior_join_feedback",
)

META_SCHEMA_CATALOG_PROMPT_KEY_ORDER: tuple[str, ...] = (
    "schema",
    "question",
)

META_SCHEMA_CATALOG_REPAIR_PROMPT_KEY_ORDER: tuple[str, ...] = (
    "schema",
    "question",
    "previous_answer",
    "error",
)

REALISM_CATEGORY_LIST: str = ", ".join(sorted(REALISM_DROP_REASON_CATEGORIES))

QUESTION_FROM_SQL_SYSTEM: str = (
    "You are given a SQL query and a schema description. "
    "Your job is to decide whether the query represents a realistic, meaningful domain question and, if so, produce natural-language paraphrases that a human analyst would ask to obtain this query's result.\n\n"
    "Rules:\n"
    "- If the query is unrealistic, nonsensical, or produces meaningless results, set is_realistic to false and explain why in drop_reason.\n"
    "- If realistic, set questions to an array of up to three distinct, conversational paraphrases a non-technical user would ask. "
    "Do NOT use SQL jargon or raw column names — use natural domain language.\n"
    "- Do not phrase the output as numbered steps, subqueries, JOIN recipes, or procedural SQL instructions; each entry must read as one coherent analyst question.\n"
    "- You may also set question (string) to the first paraphrase; when questions is non-empty, question should match questions[0].\n"
    "- Output ONLY valid JSON with fields: "
    '"questions" (array of strings), "question" (string, optional), "is_realistic" (boolean), "drop_reason" (string or null), and optionally '
    f'"drop_reason_category" (string) when is_realistic is false. '
    f"If present, drop_reason_category must be one of: {REALISM_CATEGORY_LIST}.\n"
)

MIGRATION_HEADER_BY_TIER: Mapping[str, str] = MappingProxyType(
    {
        "additive": "Schema expanded with new tables or columns. Existing learning is kept.",
        "soft_refresh": "Refreshing cached metadata. Existing learning is kept.",
        "remap": "Schema renames detected. Mapping existing learning to the new names.",
        "destructive": "Learning reset: cache rebuilt from scratch (schema changed in ways that cannot be remapped).",
    }
)

SAVED_LINE: str = "Saved."

FEEDBACK_NOTED_LINE: str = "Feedback noted. Try rephrasing your question for a better match."

QUERY_RESULTS_HEADER: str = "Query Results"

USER_ERROR_PREFIX: str = "Error: "

USER_WARN_PREFIX: str = "! "

USER_TERMINATED_LINE: str = "\nUser terminated."

REMEDIATION_RESTRICTED_QUESTION: str = (
    "The user asked for a write or administrative operation. Text-to-SQL only supports read queries."
)

USER_REJECTED_RESULT_BUCKET_TIPS: dict[str, str] = {
    "MISSING_FILTER": "Tips: name the filter or dimension you care about (time range, status, category).",
    "WRONG_GROUPING": "Tips: say whether you want totals per entity, per period, or overall.",
    "WRONG_AGGREGATION": "Tips: specify sum, average, count, or another metric clearly.",
    "WRONG_TIME_RANGE": "Tips: give an explicit date range or relative window.",
    "WRONG_TABLES_OR_JOINS": "Tips: name the tables or relationships that should connect your answer.",
    "WRONG_SORT_OR_LIMIT": "Tips: say how results should be ordered or how many rows you need.",
    "OTHER": "Tips: be more specific about columns, filters, grouping, or time range.",
    "MALFORMED_MEMBER_ANSWER": (
        "Tips: the member result did not match the requested projection; retry or narrow the question."
    ),
    "JOIN_FAN_OUT": (
        "Tips: the cross-source join multiplied rows; check that join keys are unique on the declared side."
    ),
}

TABLE_SCOPE_REPAIR_REASON_TEXT: dict[str, str] = {
    "interpret_align": "aligned with interpret scope (join bridge)",
    "expression_reference": "referenced in query expressions",
    "unreferenced_table": "not referenced in query expressions",
    "join_bridge": "required by the chosen join path",
}

INTERPRET_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["approach", "tables"],
    "additionalProperties": False,
    "properties": {
        "approach": {"type": "string"},
        "tables": {"type": "array", "items": {"type": "string"}},
        "grounding": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["ref", "used_for"],
                "additionalProperties": False,
                "properties": {"ref": {"type": "string"}, "used_for": {"type": "string"}},
            },
        },
        "schema_invalid": {"type": "boolean"},
        "missing": {"type": "string"},
    },
}

PARAM_QUESTION_REMAP_SYSTEM: str = (
    "You rewrite stored natural-language questions for a new parameter binding.\n"
    "Input JSON has questions (list of strings), old_params and new_params (handle→value maps).\n"
    'Return JSON {"questions": [...]} with the SAME length and order as input questions, each rewritten so temporal/numeric wording matches new_params while preserving meaning and structure.\n'
    "Do not add or drop questions. Do not invent unrelated content.\n"
    "Respond ONLY with valid JSON, no explanation."
)

DESCRIPTION_REFINER_SYSTEM: str = (
    "You refine human-written database descriptions so a downstream text-to-SQL LLM can use them effectively. "
    "When previous_text is non-empty, mirror its prose style, length, and structural pattern (sentence shape, role mentions, qualifier ordering). Preserve every keyword and identifier the human wrote in text (column names, table names, units, values, conditions, references). Tighten phrasing, remove fluff, make role and domain meaning explicit, and keep wording in plain prose. Do not invent facts the human did not state. "
    "Output ONLY valid JSON."
)

QSIM_FILL_SYSTEM: str = (
    "You are a SQL intent generator."
    " Given structural skeleton constraints, generate a valid query intent filling in specific columns and filters from the available schema."
    " Rules:"
    " 1. STRICTLY follow skeleton constraints for tables, aggregation presence, filter count, groupby count, having, orderby."
    " 2. Use ONLY columns from the provided schema in the specified tables."
    " 3. For filters, choose columns with meaningful filter potential (categorical, boolean/flag, or temporal columns)."
    " 4. For aggregation, use COUNT/SUM/AVG/MIN/MAX wrapping table.column in select_cols; aggregate numeric columns only (not IDs or foreign keys); COUNT may use any column or *."
    " 5. For groupby, choose categorical or temporal columns that make semantic sense."
    " 6. Ensure all column references use table.column format from the specified tables."
    " 7. Return ONLY valid JSON matching the specified format."
    " 8. For expr_comparison (filter expr-vs-expr), both expressions must be from different tables with compatible types."
    " 9. DISTINCT is only valid for non-aggregated queries."
    " 10. For orderby, include ASC or DESC direction suffix."
    " 11. DO NOT return columns or tables not in the provided schema."
    " 12. For having, expression must be an aggregation matching a select_cols aggregation."
)

CSV_IDENTIFIER_NAMING_SYSTEM: str = "You propose concise snake_case SQL identifiers for tabular upload labels. Output ONLY valid JSON matching identifier_naming_schema in the user payload. Use lowercase letters, digits, and underscores only. Start with a letter. Keep names short but readable. Do not invent domain meaning beyond the supplied label text."

CSV_IDENTIFIER_NAMING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["identifier"],
    "additionalProperties": False,
    "properties": {"identifier": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"}},
}

UPLOAD_COLUMN_TRANSFORMS_SYSTEM: str = (
    "You propose column transforms for one tabular upload relation. Output ONLY valid JSON matching column_transforms_schema in the user payload. Use transform_id values from upload_transform_ids "
    f"in the user payload ({', '.join(UPLOAD_COLUMN_TRANSFORM_IDS)}). Each proposal must name the "
    "target column label when required and supply params fields defined in the schema. Mark requires_review true for shape-changing transforms. Proposals are verified deterministically on the full column before apply; invalid proposals are rejected without changing data."
)

UPLOAD_COLUMN_TRANSFORMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["column_transforms"],
    "additionalProperties": False,
    "properties": {
        "column_transforms": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["transform_id"],
                "additionalProperties": False,
                "properties": {
                    "transform_id": {"type": "string", "enum": list(UPLOAD_COLUMN_TRANSFORM_IDS)},
                    "column": {"type": "string"},
                    "requires_review": {"type": "boolean"},
                    "params": {"type": "object"},
                },
            },
        },
    },
}

UPLOAD_SUMMARY_SYSTEM: str = "You summarize structured upload inspection findings for an operator. Output ONLY valid JSON with one summary field. Use issue codes, locations, severities, and suggested_selections from the user payload. Do not invent row values or domain meaning beyond the supplied findings."

UPLOAD_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["summary"],
    "additionalProperties": False,
    "properties": {"summary": {"type": "string"}},
}

UPLOAD_INTERPRET_SYSTEM: str = "You propose layout interpretation for one ambiguous tabular upload. Output ONLY valid JSON matching upload_interpret_schema in the user payload. Suggest header_row, table_range, append_regions, or merge_regions when structural scoring is inconclusive. Use only fields present in the schema. Proposals are verified against the grid before use."

UPLOAD_INTERPRET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "header_row": {"type": "integer"},
        "table_range": {"type": "string"},
        "append_regions": {"type": "array", "items": {"type": "string"}},
        "merge_regions": {"type": "array", "items": {"type": "string"}},
    },
}

INTERPRET_PROSE_FIELDS: tuple[str, ...] = (
    "select",
    "where",
    "group_by",
    "having",
    "order_by",
    "window",
    "case",
)

LLM_JSON_ONLY_FOOTER: str = "Respond ONLY with valid JSON, no explanation."

META_DOMAIN_KNOWLEDGE_SYSTEM: str = (
    "You answer questions using only the JSON `domain_knowledge` list in the user message.\n"
    "Each entry has key, kind, and text. Answer only from that list; never invent terms, tables, columns, or numeric facts not stated there.\n"
    "If the list is empty, set `message` to exactly the configured empty-knowledge reply "
    f"({META_EMPTY_DOMAIN_KNOWLEDGE_MESSAGE!r}).\n"
    "If entries exist but none answer the question, return JSON with response_kind "
    f"{META_INSUFFICIENT_KNOWLEDGE_RESPONSE_KIND!r} and no other fields. Do not invent an answer.\n"
    "Return JSON matching META_KNOWLEDGE_ANSWER_SCHEMA exactly (response_kind domain_knowledge, message as natural-language prose).\n"
    f"{LLM_JSON_ONLY_FOOTER}"
)

META_SCHEMA_AND_KNOWLEDGE_SYSTEM: str = (
    "You answer questions that need BOTH schema structure and domain knowledge.\n"
    "Use only the JSON `schema` object and `domain_knowledge` list in the user message.\n"
    "Never invent tables, columns, relationships, metrics, or glossary terms absent from those payloads.\n"
    "Never name tables, columns, or relationships absent from those payloads. Do not invent row values.\n"
    "If the provided schema and domain knowledge cannot answer the question, return JSON with response_kind "
    f"{META_INSUFFICIENT_KNOWLEDGE_RESPONSE_KIND!r} and no other fields. Do not invent an answer.\n"
    "Return JSON with keys: response_kind (must be schema_and_knowledge), message (non-empty prose), and optional notes (string). Put all user-facing content in message.\n"
    f"{LLM_JSON_ONLY_FOOTER}"
)

META_SCHEMA_CATALOG_SYSTEM: str = (
    "You answer questions about the active database schema using only the JSON `schema` object in the user message.\n"
    "Never invent table names, column names, relationships, row values, or metrics absent from `schema`.\n"
    "Never name tables, columns, or relationships absent from `schema`.\n"
    "Copy inventory counts only from `schema.inventory` and `schema.members` into the `counts` fields; do not recount by scanning table lists or prose.\n"
    "Leave a `counts` field null when the question does not ask for that metric.\n"
    "Leave `tables` and `relationships` as empty arrays for pure count or inventory questions that need no detail list.\n"
    "If the schema dump cannot answer the question, return JSON with response_kind "
    f"{META_INSUFFICIENT_KNOWLEDGE_RESPONSE_KIND!r} and no other fields. Do not invent an answer.\n"
    "Return JSON matching META_SCHEMA_ANSWER_SCHEMA exactly (response_kind, headline, counts, tables, relationships, notes).\n"
    f"{LLM_JSON_ONLY_FOOTER}"
)

LLM_PRESERVE_ANALYTICAL_CONTENT: str = (
    "Preserve all entities, filters, metrics, grouping, ordering, and limits implied by the source. "
    "Do not add or remove requirements."
)

QUESTION_NORMALIZE_VOCABULARY_HEADING: str = (
    "Vocabulary preferences (apply only when context fits; preserve logical negation and constraints):"
)

QUESTION_NORMALIZE_VOCABULARY_GUIDANCE: str = (
    "Verb phrasing: prefer concise analytic wording over vague conversational fillers whenever the question intent is unchanged. "
    "Aggregation: align stated aggregation language with the grain implied by the question. "
    "Temporal: when any time scope appears, state it explicitly enough for deterministic routing. "
    "Negation: preserve every explicit negator on the predicate it scopes; do not drop or soften negation."
)

QUESTION_VALIDATION_SYSTEM: str = (
    "You decide whether user input is a Text-to-SQL analytical request, a metadata/domain question about the configured schema, a forbidden mutation/admin request, or invalid.\n\n"
    "The user message is JSON with fields question (the user text) and inventory (table_names and domain_knowledge_keys visible to this caller). "
    "Use inventory only to judge whether a metadata route can be grounded; never invent tables or glossary keys outside inventory.\n\n"
    "Default bias: when the user asks to retrieve, filter, count, aggregate, sort, group, rank, compare, or otherwise materialize rows or scalars from stored tabular data, choose ANALYTICAL. "
    "Do not mark such asks INVALID merely because wording is terse, uses past tense, names a calendar day/range, mentions null/empty fields, or omits explicit SQL vocabulary.\n\n"
    "ANALYTICAL (valid_database_question=yes, query_type=analytical) when the user wants an answer that maps to a SELECT-style SQL query over their stored tabular data: list/show/get/find/return rows, counts, sums, averages, filters, sorts, groups, ranks, trends, comparisons, or per-entity metrics. "
    "Date, status, and attribute filters are analytical. Questions that ask for the contents of a view or table-shaped result are analytical.\n"
    "Do NOT choose analytical for generic data advice, tutoring, world knowledge, product how-tos, or vague 'tell me about data' questions that do not ask for queryable rows or aggregates.\n\n"
    "METADATA routes (valid_database_question=yes) when the user asks about the configured schema's structure and/or its domain knowledge (glossary, policies, metrics, synonyms, caveats, relation/field meanings). "
    "When uncertain between INVALID and a metadata route, prefer the narrowest metadata route that inventory can ground.\n"
    "Choose exactly one metadata query_type:\n"
    "- schema_catalog: structure/inventory only (tables, columns, types, keys, relationships, counts) with no need for domain-knowledge definitions\n"
    "- domain_knowledge: glossary/policy/metric/synonym/caveat definitions only, without needing a schema inventory dump\n"
    "- schema_and_knowledge: both structure and domain knowledge are required (for example: which relations relate to a named domain concept that needs definitions plus schema)\n\n"
    "When ambiguous between analytical and metadata, prefer analytical if a concrete SQL result set is clearly intended; otherwise prefer the narrowest metadata route.\n\n"
    "Mark INVALID (valid_database_question=no) with query_type conversational or unmappable:\n"
    "- conversational: chitchat or meta conversation (e.g. hello, thanks, who are you) with no schema or data question\n"
    "- unmappable: SQL tutoring/how-to without asking for rows, general world knowledge, generic data advice not about this schema or glossary, or analytical wording that cannot be grounded even after inventory review\n\n"
    "RESTRICTED (valid_database_question=no, query_type=restricted) ONLY for DML, DDL, or administrative operations (delete/update/insert/merge/truncate/copy; create/drop/alter/rename; grants, vacuum, config). "
    "Analytical questions never receive restricted merely because they mention CTE, join, window, distinct, or similar analytical primitives.\n\n"
    "Respond with JSON containing exactly three fields:\n"
    '- "valid_database_question": "yes" or "no"\n'
    '- "query_type": "analytical", "schema_catalog", "domain_knowledge", "schema_and_knowledge", "restricted", "conversational", "unmappable", or "unspecified"\n'
    '- "corrected": the input with spelling typos fixed only. Do NOT remove, reorder, or rephrase any words.\n\n'
    f"{LLM_JSON_ONLY_FOOTER}"
)

QUESTION_CANONICALIZE_SYSTEM: str = (
    "You rewrite a typo-corrected database query into a canonical short query so that semantically identical questions hash to the same string.\n\n"
    "When the user message is JSON, field ``question`` carries the rewrite target; optional ``normalization_preferences`` is advisory context only.\n\n"
    "Apply these rules IN ORDER:\n"
    '0. Before any other rewrite, normalize quantifier and aggregation openers: map phrases such as "how many", "number of", and bare "count" asking for cardinality to the two-token prefix "count of"; map "total of", "totals for", and bare "sum" used as an aggregation opener to "sum of"; map "average of", "mean of", and bare "avg" used as an aggregation opener to "avg of"; map "maximum of", "largest", "highest", "max" used as an aggregation opener to "max of"; map "minimum of", "smallest", "lowest", "min" used as an aggregation opener to "min of". '
    "Preserve trailing nouns and filters after those prefixes.\n"
    '1. Replace any verb phrase whose only purpose is to ask for non-aggregated rows with the single token "list"; do not replace the aggregation prefixes introduced in rule 0, and do not replace aggregation verbs such as count, sum, average, max, min, or total when they already head a normalized aggregation phrase.\n'
    "2. Drop polite or filler clauses that do not carry analytical meaning.\n"
    "3. Replace plural common nouns with their singular base form. Do NOT singularize verbs.\n"
    "4. Preserve every number, date, quoted literal, comparison word, adjective, named entity, and any preposition immediately before a number/date/literal.\n"
    "5. Preserve original word order.\n"
    "6. If no rule applies, return the input unchanged.\n"
    "7. Never add a word that did not appear in the input (including any inflected form already present).\n\n"
    "Examples of rule 0 (JSON only illustrates the normalized field):\n"
    '{"question":"how many films are in the action category","normalized":"count of film in the action category"}\n'
    '{"question":"total payments last month by store","normalized":"sum of payment last month by store"}\n\n'
    "Respond with JSON containing exactly one field:\n"
    '- "normalized": the rewritten canonical short query.\n\n'
    f"{LLM_JSON_ONLY_FOOTER}"
)

WARMUP_PARAPHRASES_BY_STYLE_SYSTEM: str = (
    "Generate natural-language analyst questions grouped by stylistic palette slots. "
    f"{LLM_PRESERVE_ANALYTICAL_CONTENT} "
    "Use schema descriptions only for terminology consistency. "
    "Do not answer the question. Do not output SQL, identifiers, numbered steps, or JOIN recipes. "
    "For each style slot you may return zero paraphrases when that style does not fit; never force a poor match. "
    "Output ONLY valid JSON with field paraphrases_by_style mapping each style name to an array of strings."
)

WARMUP_FREEFORM_QUESTIONS_SYSTEM: str = (
    "You write natural-language analyst questions that faithfully describe what a SQL query computes. "
    "Use schema table descriptions only for domain terminology. "
    "Do not output SQL, identifiers, numbered steps, or JOIN recipes. "
    "Return 1–3 concise questions in a JSON object with field questions as an array of strings."
)

SEED_QUESTION_CLARIFY_SYSTEM: str = (
    "You rephrase database analyst questions for clarity only. Do not answer them. "
    f"{LLM_PRESERVE_ANALYTICAL_CONTENT} "
    "Do not use SQL or qualified identifiers unless the source already does. "
    'Output only valid JSON: {"lines":[{"index":<int>,"clarified":"<string>"}]} with exactly one object per input index, indices matching the batch, no extra keys, no markdown.'
)

SANDBOX_STATIC_FAITHFULNESS_SPECS: dict[str, dict[str, object]] = {
    "which games support english?": {
        "sql_contains": ("game_supported_language",),
        "contains_join": True,
    },
    "which city has the most customers?": {
        "required_tables": ("city", "customer"),
        "contains_join": True,
    },
    "how many customers are in each country?": {
        "required_tables": ("country", "customer"),
        "contains_join": True,
    },
    "film title and replacement cost minus rental rate as profit margin": {
        "sql_excludes": ("interval",),
    },
    "what is the best pizza topping?": {"status": "invalid_question"},
    "what's the weather today?": {"status": "invalid_question"},
}

SANDBOX_TOUR_EXPECT_NO_SQL: frozenset[str] = frozenset(
    {
        "What's the weather today?",
        "Show payroll deductions by employee SSN.",
        "How many rentals happened on 2025-01-01?",
    }
)

SANDBOX_VALIDATION_FAILURE_QUESTIONS: frozenset[str] = frozenset(
    {
        "Show payroll deductions by employee SSN.",
        "Show me all staff salaries.",
        "How many rentals happened on 2025-01-01?",
        "How many rentals were made in total?",
        "How many items are there?",
    }
)

SANDBOX_VALIDATION_FAILURE_EXPECT_NO_SQL: frozenset[str] = frozenset()

SANDBOX_MALFORMED_MOCK_FIXTURE_QUESTIONS: tuple[str, ...] = (
    "Show rentals with malformed compose output for repair exercise.",
)

SANDBOX_MALFORMED_MOCK_FIXTURE_SPECS: tuple[dict[str, object], ...] = (
    {
        "question": "Show rentals with malformed compose output for repair exercise.",
        "malformed_output": '{"tables":["rental"],"select":"count rentals","filter":}',
        "repair_output": '{"tables":["rental"],"select":"count rental rows","filter":"","group_by":"","having":"","order_by":"","limit":null,"window":"","case":""}',
        "repair_stage": "compose",
    },
)

REPAIR_INSTRUCTIONS: dict[str, str] = {
    "extract_epoch": (
        "Do not use EXTRACT(EPOCH FROM ...). Use date column subtraction "
        f"({INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_DATE_COLUMN_PLACEHOLDER} - "
        f"{INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_OTHER_DATE_COLUMN_PLACEHOLDER}) or other supported "
        "date functions for time differences."
    ),
    "grain_validity": "Use one of the allowed grain values: 'scalar', 'grouped', or 'row_level'.",
    "grain_consistency": "Ensure grain matches the query structure. 'grouped' requires group_by_cols and aggregation in select_cols. 'row_level' means no GROUP BY and no aggregation. 'scalar' means a single aggregated value with no GROUP BY.",
    "schema_validation": "One or more tables or columns do not exist in the schema. Check allowed_tables and use only exact column names from each table.",
    "unknown_table": "The table does not exist in the schema. Remove it from tables and rewrite any references to use only tables that appear in allowed_tables.",
    "unknown_column": "The column does not exist in its table. Check the schema for available columns with a similar name or meaning and replace the reference.",
    "semantic_contradiction": "The intent contains contradictory operations. Keep only the aggregation or pattern that matches the question.",
    "expression_type": "Arithmetic expressions require numeric columns. Ensure all operands in arithmetic and all comparison sides have compatible types.",
    "where_aggregation": "Conditions with aggregation functions (COUNT, SUM, AVG, MIN, MAX) belong in having, not where. Move the condition.",
    "having_aggregation": "HAVING conditions must have an aggregation function in left_expr. Conditions without aggregation belong in where.",
    "where_semantic": "Fix the filter comparison: remove self-comparisons and ensure type compatibility between left and right expressions.",
    "having_semantic": "Fix the HAVING comparison: remove self-comparisons and ensure type compatibility between aggregated expressions.",
    "nested_aggregation": "Nested aggregation is not allowed. Use a CTE: compute the inner aggregation in a CTE step, then aggregate the CTE output in the main query.",
    "mixed_aggregation": "An expression cannot mix aggregated and bare column terms. Either wrap all terms in an aggregation function or add bare columns to group_by_cols.",
    "group_by_membership": "Every non-aggregated column in select_cols must appear in group_by_cols when grain is 'grouped'. Add the missing column to group_by_cols or wrap it in an aggregation.",
    "order_by_aggregation": "ORDER BY cannot contain aggregation when grain is 'row_level'. Change grain to 'grouped' or remove the aggregation from order_by_cols.",
    "aggregation_hint": "The question contains a quantity-comparison phrase that typically requires COUNT or SUM aggregation with GROUP BY and HAVING. Add aggregation in select_cols, group_by_cols on the entity, having with the threshold, and set grain to 'grouped'.",
    "column_schema": "A referenced column does not exist in its table. Check the schema for the correct column name.",
    "table_schema": "A referenced table does not exist in the schema. Use only tables from allowed_tables.",
    "filter_type_ops": "The filter operator is not compatible with the column data type. Use an appropriate operator for the column type.",
    "null_filter": "NULL checks must use 'is null' or 'is not null' operator with no value or value_type field.",
    "between_filter": (
        "Use op 'between' with value as a two-element array [lower, upper]. "
        "The pipeline decomposes it into >= and <= automatically."
    ),
    "date_window": (
        "For relative time-window filters, "
        'use value_type \'date_window\' with the date column in left_expr, op \'>=\' and value as {"unit": "day"|"week"|"month"|"year", "amount": N}. '
        "Use singular unit names. The amount is interpreted as an ISO half-open window: "
        "the start is N units before today, the end is exclusive. Prefer unit 'day' whenever the question phrases the window in days, weeks, or fortnights (convert weeks to days) so the start/end boundary matches the schema's daily granularity. "
        'For absolute calendar bounds use {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"} in ISO 8601 form only.'
    ),
    "date_diff": (
        "For date-difference filters comparing two date columns "
        f"({INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_OTHER_DATE_COLUMN_PLACEHOLDER} - "
        f"{INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_DATE_COLUMN_PLACEHOLDER} compared to a duration), use value_type 'date_diff' with "
        'left_expr as the date subtraction expression (later minus earlier per the question), op as the comparison operator, and value as {"unit": "day"|"week"|"month"|"year", "amount": N}. '
        "Use singular unit names. Prefer unit 'day' whenever the question phrases the duration in days or weeks. Do NOT use date_diff for relative date-window filters; use date_window instead."
    ),
    "date_integer_days": (
        "For date-shift arithmetic comparing a date column shifted by an integer day count to another date column, express "
        f"{INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_DATE_COLUMN_PLACEHOLDER} + "
        f"{INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_INTEGER_COLUMN_PLACEHOLDER} or "
        f"{INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_DATE_COLUMN_PLACEHOLDER} - "
        f"{INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_INTEGER_COLUMN_PLACEHOLDER} directly in left_expr/right_expr using "
        "+/- between the date column and the integer day count (literal or column). Do not use date_diff when comparing a shifted date to another date column."
    ),
    "agg_role": "SUM and AVG should only be applied to numeric measure columns. Use COUNT for non-measure columns, or select a numeric column.",
    "agg_type": "SUM and AVG require a numeric column. The referenced column is not numeric. Use COUNT instead or choose a numeric column.",
    "for_each_grouping": "The question contains a 'for each', 'per', or 'by' phrase implying a GROUP BY on the referenced entity. Add the entity's identifying column to group_by_cols, include it as a non-aggregated entry in select_cols, and set grain to 'grouped'.",
    "scalar_func_type": "The scalar function is applied to an incompatible column type. Ensure the column type matches what the function expects.",
    "threshold_missing_having": "The question contains a threshold phrase and the intent already has aggregation, but no HAVING condition is defined. Add a HAVING clause that compares the aggregated column to the numeric threshold in the question.",
    "cte_structure": "CTE steps require a cte_name string, an output_columns list of alias strings, and valid tables.",
    "cte_grain_consistency": "CTE grain must match its structure: same rules as the main query regarding grain, group_by_cols, and aggregation.",
    "cte_table_reference": "A CTE references an unknown table. A CTE can only reference schema tables or CTEs defined earlier in the same WITH list.",
    "cte_grain_compatibility": "A row_level query or CTE depends on an aggregated CTE. Ensure the grain is compatible with upstream CTE grains.",
    "cte_aggregation": "A CTE has HAVING conditions but no aggregation in its select_cols. Add aggregation or remove the HAVING.",
    "agg_keyword_missing": "The question asks for an aggregation (total, count, average, sum, etc.) but the intent has no aggregated column and no HAVING condition. Add the appropriate aggregation function to select_cols, include all tables needed to compute the aggregated value, and set grain to 'grouped' with the correct group_by_cols.",
    "wrong_join": (
        "The referenced tables cannot be joined through the schema. Do not resolve join errors by removing a table from tables — that answers a different question. If the tables are genuinely related, add foreign_keys_add or a semantic neighbour override. Otherwise keep both tables and let the turn refuse."
    ),
}

INTERPRET_NL_CONVENTIONS_BODY: dict[str, Any] = {
    "mandatory": [
        "Copy every literal that constrains rows or ordering into the matching prose field; the compose stage never sees the original question.",
        f"List only semantic base tables in top-level tables; omit {INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER} from tables when its columns appear in prose.",
        "Reference window and case output names from select when you define them in window or case using as <registry_name>.",
        "Never emit SQL set operators, EXISTS, NOT EXISTS, LATERAL, param_key, raw_value, wNN, cNN, where_group, or IR vocabulary.",
        "Never use as <name> inside select, filter, having, group_by, order_by, or limit; select output aliases are assigned later by the pipeline.",
        "Leave group_by, order_by, and limit empty unless the question explicitly asks for grouping, ordering, or a row cap; do not invent presentation-layer sorting or grouping for context.",
        "Project select prose to primary keys, primary human-readable label columns such as names or titles, and every column referenced by stated filters, ordering, grouping, having, or limits; never enumerate every physical column unless the question explicitly asks for all columns, every column, or complete row dumps.",
        f"After structural encoding, only tables with qualified column references in that scope remain in join scope; name every required table with explicit {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} in the appropriate prose field.",
        f"The tables list alone never keeps a table in join scope; qualified {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} tokens must appear in select, filter, group_by, having, order_by, or registry prose.",
        f"When membership or existence requires {INSTRUCTIONAL_LINK_TABLE_PLACEHOLDER} or {INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER}, "
        f"name {INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_OTHER_COLUMN_PLACEHOLDER} or "
        f"{INSTRUCTIONAL_BRIDGE_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_OTHER_COLUMN_PLACEHOLDER} in select prose, not only join-equality narration in filter prose.",
        "Per-entity breakdown phrasing maps to group_by, not row-level DISTINCT deduplication.",
        "Existence or membership conditions belong in filter prose with binding columns named.",
        "Copy relative time unit and amount into filter prose; copy two-date duration comparisons into filter prose naming both columns.",
        "For cte_steps, tables lists base schema tables and names of prior cte_steps this step reads from; do not use a separate dependency field.",
    ],
    "recommended": [
        f"Qualify columns as {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} when multiple listed tables share a column name.",
        f"Describe aggregates with phrases such as sum of {INSTRUCTIONAL_OTHER_QUALIFIED_COLUMN_PLACEHOLDER}.",
        f"Describe anti-existence with plain language or {INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER}.pk is null style wording.",
    ],
}

INTERPRET_NL_CONVENTIONS: Mapping[str, Any] = MappingProxyType(INTERPRET_NL_CONVENTIONS_BODY)

COMPOSE_NL_PHRASE_MAPPINGS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "like": ("like", "contains pattern", "matches pattern"),
        "not like": ("not like", "does not match pattern"),
        "in": ("in", "one of", "any of", "among"),
        "not in": ("not in", "none of", "not among"),
        "is null": ("is null", "is absent", "is missing"),
        "is not null": ("is not null", "is present", "present"),
        "ilike": ("ilike", "case-insensitive like", "case-insensitive pattern"),
        "not ilike": ("not ilike", "case-insensitive not like"),
        "contains": ("contains element", "array contains", "has element"),
        "date_window": ("last", "past", "within the last", "in the last", "recent", "since"),
        "date_diff": (
            "more than N days after",
            "less than N days before",
            "at least N days between",
        ),
    }
)

COMPOSE_NL_TO_IR_GUIDANCE: tuple[str, ...] = (
    "When Ground prose is ambiguous among allowlisted IR operators and aggregates, match phrases against nl_phrase_mappings case-insensitively to choose among those allowlisted forms only. Phrase mappings never authorize operators outside operator_reference and schema-gated capabilities.",
    "Infer value_type from literal form: single-quoted text is string; bare digits are integer or number; ISO dates are date; true or false are boolean; is null operator uses value_type null without a value.",
    "Emit aggregate expressions inside select_cols expressions; do not invent select-column alias keys beyond CTE output_columns rules.",
    "WindowRegistryStep.registry_id and CaseRegistryStep.registry_id are wNN / cNN tokens referenced as bare tokens from select_cols; CTE names are cte_steps[].cte_name.",
    "logical_intent is the sole semantic source; translate select, filter, group_by, having, and order_by prose mechanically without inventing or dropping columns named in prose.",
    f"Emit a qualified column reference in the IR slot matching each clause for every {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} named in that clause's prose at that scope.",
    'Phrases such as last N days, past N weeks, or within the last N months map to value_type date_window with {"unit", "amount"}.',
    "Elapsed time between two date columns maps to value_type date_diff with left_expr as a subtraction expression.",
    "CURRENT_DATE and CURRENT_TIMESTAMP map to keyword right_expr leaves, not string raw_value literals.",
    "Integer columns with schema role temporal and type integer are day-count durations; compare them to elapsed-day expressions, not as calendar dates.",
    f"Date shifted by an integer duration column uses {INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_DATE_COLUMN_PLACEHOLDER} + "
    f"{INSTRUCTIONAL_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_INTEGER_COLUMN_PLACEHOLDER} in left_expr or right_expr, not date_diff.",
    "Emit only operators and expression forms present in operator_reference and schema-gated capabilities.",
)

COMPOSE_IR_ASSEMBLY_RULES: tuple[str, ...] = (
    "where is a nested predicate tree (op, predicates, groups) parsed from interpret filter prose.",
    "Emit raw_value for every WhereParam and HavingParam literal; never emit param_key or param_values.",
    f"Every column_ref is {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} using only structural_schema_for_chosen_tables identifiers.",
    f"Downstream repair keeps only tables with column references in each scope's IR; emit refs only for {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} tokens named in the matching clause prose.",
    "Do not invent filters, grouping, or columns absent from logical_intent prose fields.",
    "For CONCAT expressions, place every concat argument into the same MulGroup.multiply list under scalar_func='concat'. Do not introduce divisors or coefficients for CONCAT groups. "
    "COUNT(DISTINCT CONCAT(a, b)) is Shape A: an outer MulGroup with agg_func='count', distinct=true, and a single multiply child whose add_groups[0] is the CONCAT MulGroup (scalar_func='concat'); do not set agg_func and scalar_func='concat' on the same MulGroup (Shape B). "
    "Use only COUNT as the outer aggregation wrapper for a CONCAT MulGroup; SUM, AVG, MIN, and MAX are not valid as that outer aggregation.",
    "Emit cte_steps[].emission semi_join when the step is a probe that keeps parent rows with at least one related match on the compared keys. Emit emission anti_join when the step is a probe that keeps parent rows with no related match on the compared keys. Emit join_table or scalar_subquery only as shape hints; the engine reclassifies those two from CTE structure. Omit emission when unsure of join_table vs scalar_subquery.",
    "Probe CTE steps project the compared keys needed by the parent. Do not place a probe as the left join anchor. Do not set preserve_tables or distinct_on on a probe step.",
    "Every bare wNN referenced from select_cols (or CTE select_cols) must have a matching window_registry.registry_id in the same scope; every bare cNN must have a matching case_registry.registry_id. Do not reference missing registry ids.",
    "Having must be a PredicateGroup tree (op and/or with predicates and groups), never a flat list of leaves that drops grouping structure.",
    "Each having leaf compares an aggregated expression (engine may demote); do not put only non-aggregate row filters in having.",
    "After assembly, comparison and temporal sides are column-bearing on the left and constants, keywords, or non-column literals on the right.",
    "Do not emit engine-owned fields: param_values, limit_param_key, param_key, param_key_hi, param_key_unit, column_map, chosen_join_candidate_id, chosen_join_path_signature, resolved_join_tables, distinct_select_index, sql_param, sql_shape, schema_invalid, interpret_cte_names, table_scope_repairs, output_column_metadata, comparison_only_tables.",
    "PredicateGroup nesting depth must not exceed 3; CTE steps must stay within configured MAX_CTE_STEPS.",
)

INTERPRET_CARDINALITY_RELATIONSHIP_RULE: str = (
    f"When {INSTRUCTIONAL_TABLE_PLACEHOLDER} links to {INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER} through both a direct FK and a junction table, use the direct FK when the "
    f"question concerns one {INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER} value per {INSTRUCTIONAL_TABLE_PLACEHOLDER} row; use the junction when it concerns the set of "
    f"{INSTRUCTIONAL_OTHER_TABLE_PLACEHOLDER} values per {INSTRUCTIONAL_TABLE_PLACEHOLDER} row."
)

INTERPRET_SHARED_PK_TABLE_SCOPE_RULE: str = "When referenced columns belong to a table that shares its primary key with another table you list, include every such table whose columns appear in the intent; do not assume same-key columns are reachable without listing that table."

LOGICAL_DECOMPOSITION_GUIDANCE: tuple[str, ...] = (
    "Describe each prose field thoroughly enough that a structural converter can build the IR without re-reading the question.",
    "Only tables may name real schema tables; every other field is natural language.",
    "Use cte_steps when the question needs a reusable intermediate; each step lists name, tables, and the same prose fields as the top level; tables may name base schema tables and prior step names.",
    "Put window definitions in the window prose field and case definitions in the case prose field; use as <registry_name> only inside those two fields.",
    "Never describe joins as explicit paths; the engine discovers FK paths.",
    f"After structural encoding, only tables with qualified column references in that scope remain in join scope; name every required table with explicit {INSTRUCTIONAL_QUALIFIED_COLUMN_PLACEHOLDER} in the appropriate prose field.",
    f"Omitting {INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER} from tables is correct when its columns appear in prose; the tables list alone never keeps a table in join scope.",
    f"When membership or existence requires {INSTRUCTIONAL_LINK_TABLE_PLACEHOLDER} or {INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER}, "
    f"name {INSTRUCTIONAL_JUNCTION_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_OTHER_COLUMN_PLACEHOLDER} or "
    f"{INSTRUCTIONAL_BRIDGE_TABLE_PLACEHOLDER}.{INSTRUCTIONAL_OTHER_COLUMN_PLACEHOLDER} in select prose, not only join-equality narration in filter prose.",
    "Per-entity breakdown phrasing maps to group_by, not row-level DISTINCT deduplication.",
    "Existence or membership conditions belong in filter prose with binding columns named.",
    INTERPRET_CARDINALITY_RELATIONSHIP_RULE,
    INTERPRET_SHARED_PK_TABLE_SCOPE_RULE,
    INTERPRET_SINGLE_OUTPUT_WINDOW_NO_CTE_RULE,
)

FORMAT_STRUCTURAL_GUIDANCE: tuple[str, ...] = (
    "Window semantics belong in window_registry only; map logical_intent.window and each cte_steps[].window prose into WindowRegistryStep rows with wNN ids.",
    "Case semantics belong in case_registry only; map logical_intent.case and each cte_steps[].case prose into CaseRegistryStep rows with cNN ids.",
    "Reference registry outputs from select_cols using window_ref or case_ref tokens alongside other projected columns.",
    "Do not encode row filters as CASE branches; use where for row membership.",
    "Never put AVG, SUM, COUNT, MIN, MAX calls or OVER (...) frames inside where predicate right_expr as raw_sql. "
    "Compare to aggregates using an extra cte_steps row or window_registry references. "
    "cte_steps[].emission may be semi_join, anti_join, join_table, or scalar_subquery; semi_join and anti_join are author-owned probes; join_table and scalar_subquery are reclassified by the engine from CTE shape.",
    "string_agg uses agg_func='string_agg', agg_sep_param_key for the delimiter, and optional agg_order_by for within-aggregate ordering.",
    "stddev, variance, and median use agg_func on a single numeric column; median is refused when the engine capability flag is off.",
    "cte_steps[].emission: semi_join for related-match probes; anti_join for no-related-match probes; join_table or scalar_subquery as optional shape hints; omit when unsure between those two hints.",
    "window_registry and select_cols wNN tokens must pair in the same scope; never emit a bare wNN without a registry row.",
)

INTENT_ANSWER_STYLE_GUIDANCE: tuple[str, ...] = LOGICAL_DECOMPOSITION_GUIDANCE + FORMAT_STRUCTURAL_GUIDANCE

NORMALIZATION_ALLOWED_INTRODUCED_TOKENS: frozenset[str] = frozenset(
    {"list", "count", "sum", "average", "max", "min", "total", "of"},
)

INTERPRET_FIELDS: frozenset[str] = frozenset({SCHEMA_FIELD_DESCRIPTION, SCHEMA_FIELD_ENUM})

GROUND_FIELDS: frozenset[str] = frozenset(
    {
        SCHEMA_FIELD_DESCRIPTION,
        SCHEMA_FIELD_ROLE,
        SCHEMA_FIELD_TYPE,
        SCHEMA_FIELD_TRUTH_VALUE,
        SCHEMA_FIELD_ENUM,
        SCHEMA_FIELD_SAMPLES,
        SCHEMA_FIELD_DERIVED,
    }
)

COMPOSE_FIELDS: frozenset[str] = frozenset(
    {
        SCHEMA_FIELD_ROLE,
        SCHEMA_FIELD_TYPE,
        SCHEMA_FIELD_TRUTH_VALUE,
        SCHEMA_FIELD_KEYS,
        SCHEMA_FIELD_ENUM,
        SCHEMA_FIELD_DERIVED,
    }
)

FULL_FIELDS: frozenset[str] = frozenset(
    {
        SCHEMA_FIELD_DESCRIPTION,
        SCHEMA_FIELD_ROLE,
        SCHEMA_FIELD_TYPE,
        SCHEMA_FIELD_TRUTH_VALUE,
        SCHEMA_FIELD_KEYS,
        SCHEMA_FIELD_ENUM,
        SCHEMA_FIELD_DERIVED,
    }
)

DATA_QUALITY_ISSUE_EMPTY_FILE: str = "empty_file"

DATA_QUALITY_ISSUE_DUPLICATE_HEADER: str = "duplicate_header"

DATA_QUALITY_ISSUE_BLANK_HEADER: str = "blank_header"

DATA_QUALITY_ISSUE_HEADER_NOT_ROW_ONE: str = "header_not_row_one"

DATA_QUALITY_ISSUE_MULTIPLE_TABLES: str = "multiple_tables"

DATA_QUALITY_ISSUE_REPEATED_HEADER: str = "repeated_header"

DATA_QUALITY_ISSUE_SECTION_HEADING: str = "section_heading"

DATA_QUALITY_ISSUE_OVERFULL_ROW: str = "overfull_row"

DATA_QUALITY_ISSUE_BLANK_ROW: str = "blank_row"

DATA_QUALITY_ISSUE_FOOTER_NOTE_ROW: str = "footer_note_row"

DATA_QUALITY_ISSUE_FORMULA_CELL: str = "formula_cell"

DATA_QUALITY_ISSUE_EMPTY_COLUMN: str = "empty_column"

DATA_QUALITY_ISSUE_SINGLE_COLUMN: str = "single_column"

DATA_QUALITY_ISSUE_MERGED_METADATA: str = "merged_metadata"

DATA_QUALITY_ISSUE_TOTAL_ROW: str = "total_row"

DATA_QUALITY_ISSUE_MERGED_CELLS: str = "merged_cells"

DATA_QUALITY_ISSUE_EMBEDDED_OBJECT: str = "embedded_object"

DATA_QUALITY_ISSUE_MIXED_TYPES: str = "mixed_types"

DATA_QUALITY_ISSUE_NUMBER_AS_TEXT: str = "number_as_text"

DATA_QUALITY_ISSUE_NULL_TOKEN: str = "null_token"

DATA_QUALITY_ISSUE_EXCEL_ERROR: str = "excel_error"

DATA_QUALITY_ISSUE_DUPLICATE_RELATION: str = "duplicate_relation"

DATA_QUALITY_ISSUE_EMPTY_SHEET: str = "empty_sheet"

DATA_QUALITY_ISSUE_WORKBOOK_ENCRYPTED: str = "workbook_encrypted"

DATA_QUALITY_ISSUE_WORKBOOK_CORRUPT: str = "workbook_corrupt"

DATA_QUALITY_ISSUE_UNSUPPORTED_TYPE: str = "unsupported_type"

DATA_QUALITY_ISSUE_MERGEABLE_REGIONS: str = "mergeable_regions"

DATA_QUALITY_ISSUE_MERGE_HEADER_MISMATCH: str = "merge_header_mismatch"

DATA_QUALITY_ISSUE_APPENDABLE_REGIONS: str = "appendable_regions"

DATA_QUALITY_ISSUE_APPEND_HEADER_MISMATCH: str = "append_header_mismatch"

DATA_QUALITY_SEVERITY_ADVISORY: str = "advisory"

DATA_QUALITY_SEVERITY_REVIEW: str = "review"

DATA_QUALITY_SEVERITY_FATAL: str = "fatal"

DATA_QUALITY_ISSUE_SEVERITY: dict[str, str] = {
    DATA_QUALITY_ISSUE_APPENDABLE_REGIONS: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_APPEND_HEADER_MISMATCH: DATA_QUALITY_SEVERITY_REVIEW,
    DATA_QUALITY_ISSUE_BLANK_HEADER: DATA_QUALITY_SEVERITY_BLOCKING,
    DATA_QUALITY_ISSUE_BLANK_ROW: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_DUPLICATE_HEADER: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_DUPLICATE_RELATION: DATA_QUALITY_SEVERITY_BLOCKING,
    DATA_QUALITY_ISSUE_EMBEDDED_OBJECT: DATA_QUALITY_SEVERITY_FATAL,
    DATA_QUALITY_ISSUE_EMPTY_COLUMN: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_EMPTY_FILE: DATA_QUALITY_SEVERITY_BLOCKING,
    DATA_QUALITY_ISSUE_EMPTY_SHEET: DATA_QUALITY_SEVERITY_BLOCKING,
    DATA_QUALITY_ISSUE_EXCEL_ERROR: DATA_QUALITY_SEVERITY_BLOCKING,
    DATA_QUALITY_ISSUE_FOOTER_NOTE_ROW: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_FORMULA_CELL: DATA_QUALITY_SEVERITY_BLOCKING,
    DATA_QUALITY_ISSUE_HEADER_NOT_ROW_ONE: DATA_QUALITY_SEVERITY_REVIEW,
    DATA_QUALITY_ISSUE_INVALID_MERGE_RANGE: DATA_QUALITY_SEVERITY_REVIEW,
    DATA_QUALITY_ISSUE_MERGEABLE_REGIONS: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_MERGE_HEADER_MISMATCH: DATA_QUALITY_SEVERITY_REVIEW,
    DATA_QUALITY_ISSUE_MERGED_CELLS: DATA_QUALITY_SEVERITY_BLOCKING,
    DATA_QUALITY_ISSUE_MERGED_METADATA: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_MIXED_TYPES: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_MULTIPLE_TABLES: DATA_QUALITY_SEVERITY_REVIEW,
    DATA_QUALITY_ISSUE_NULL_TOKEN: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_NUMBER_AS_TEXT: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_OVERFULL_ROW: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_RAGGED_ROW: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_REPEATED_HEADER: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_SECTION_HEADING: DATA_QUALITY_SEVERITY_REVIEW,
    DATA_QUALITY_ISSUE_SINGLE_COLUMN: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_TOTAL_ROW: DATA_QUALITY_SEVERITY_ADVISORY,
    DATA_QUALITY_ISSUE_UNSUPPORTED_TYPE: DATA_QUALITY_SEVERITY_FATAL,
    DATA_QUALITY_ISSUE_WORKBOOK_CORRUPT: DATA_QUALITY_SEVERITY_FATAL,
    DATA_QUALITY_ISSUE_WORKBOOK_ENCRYPTED: DATA_QUALITY_SEVERITY_FATAL,
}

DATA_QUALITY_NULL_TOKENS: tuple[str, ...] = (
    "",
    "na",
    "n/a",
    "null",
    "none",
    "-",
    "tbd",
    "#n/a",
)

DATA_QUALITY_EXCEL_ERROR_TOKENS: tuple[str, ...] = (
    "#ref!",
    "#div/0!",
    "#value!",
    "#name?",
    "#null!",
    "#num!",
)

DATA_QUALITY_FOOTER_LABEL_PREFIXES: tuple[str, ...] = (
    "note",
    "notes",
    "source",
    "disclaimer",
    "footer",
    "comment",
    "see ",
)

DATA_QUALITY_TOTAL_PREFIXES: tuple[str, ...] = (
    "total",
    "totals",
    "subtotal",
    "grand total",
    "sum",
    "average",
    "mean",
)

DATA_QUALITY_MOJIBAKE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("\u00e2\u20ac\u201d", "\u2014"),
    ("\u00e2\u20ac\u201c", "\u2013"),
    ("\u00e2\u20ac\u2122", "\u2019"),
    ("\u00e2\u20ac\u0153", "\u201c"),
    ("\u00e2\u20ac\u009d", "\u201d"),
    ("\u00e2\u20ac\u00a6", "\u2026"),
)

DATA_QUALITY_ZERO_WIDTH_CHARS: tuple[str, ...] = ("\ufeff", "\u200b", "\u200c", "\u200d", "\u2060")

DETERMINISTIC_REPAIR_TRACE_HEADINGS: dict[str, str] = {
    "dedup_extract_year_vs_column_literal": "intent_after_deterministic_repair.dedup_extract_year_vs_column_literal",
    "repair_intent_placeholder_tokens": "intent_after_deterministic_repair.repair_intent_placeholder_tokens",
    "normalize_count_star": "intent_after_deterministic_repair.normalize_count_star",
    "promote_temporal_keyword_rhs": "intent_after_deterministic_repair.promote_temporal_keyword_rhs",
    "dedup_value_vs_right_expr": "intent_after_deterministic_repair.dedup_value_vs_right_expr",
    "qualify_count_star_mulgroups": "intent_after_deterministic_repair.qualify_count_star_mulgroups",
    "lift_distinct_select_from_raw_sql": "intent_after_deterministic_repair.lift_distinct_select_from_raw_sql",
    "canonicalize_registry_ids": "intent_after_deterministic_repair.canonicalize_registry_ids",
    "encode_inline_self_join_as_cte": "intent_after_deterministic_repair.encode_inline_self_join_as_cte",
    "reorder_cte_steps_by_dag": "intent_after_deterministic_repair.reorder_cte_steps_by_dag",
    "normalize_cte_names": "intent_after_deterministic_repair.normalize_cte_names",
    "normalize_cte_output_aliases": "intent_after_deterministic_repair.normalize_cte_output_aliases",
    "rewrite_main_query_refs_to_final_cte_columns": (
        "intent_after_deterministic_repair.rewrite_main_query_refs_to_final_cte_columns"
    ),
    "ensure_cte_output_columns_exposure": "intent_after_deterministic_repair.ensure_cte_output_columns_exposure",
    "qualify_cte_output_columns": "intent_after_deterministic_repair.qualify_cte_output_columns",
    "derive_tables_from_intent": "intent_after_deterministic_repair.derive_tables_from_intent",
    "expand_shared_pk_tables_for_refs": "intent_after_deterministic_repair.expand_shared_pk_tables_for_refs",
    "sanitize_table_names": "intent_after_deterministic_repair.sanitize_table_names",
    "replace_unknown_scalar_funcs": "intent_after_deterministic_repair.replace_unknown_scalar_funcs",
    "enforce_grain_consistency": "intent_after_deterministic_repair.enforce_grain_consistency",
    "repair_window_partition_group_by_alignment": (
        "intent_after_deterministic_repair.repair_window_partition_group_by_alignment"
    ),
    "strip_redundant_identifier_group_by": "intent_after_deterministic_repair.strip_redundant_identifier_group_by",
    "strip_spurious_group_by": "intent_after_deterministic_repair.strip_spurious_group_by",
    "decompose_between_params": "intent_after_deterministic_repair.decompose_between_params",
    "auto_repair_where_having": "intent_after_deterministic_repair.auto_repair_where_having",
    "coerce_predicate_group_mode": "intent_after_deterministic_repair.coerce_predicate_group_mode",
    "simplify_predicate_semantics": "intent_after_deterministic_repair.simplify_predicate_semantics",
    "normalize_where_havings": "intent_after_deterministic_repair.normalize_where_havings",
    "repair_null_equality_where": "intent_after_deterministic_repair.repair_null_equality_where",
    "strip_join_conditions": "intent_after_deterministic_repair.strip_join_conditions",
    "cte_grain_consistency": "intent_after_deterministic_repair.cte_grain_consistency",
    "sort_select_and_order_by": "intent_after_deterministic_repair.sort_select_and_order_by",
    "lift_distinct_modifier_in_multiply": "intent_after_deterministic_repair.lift_distinct_modifier_in_multiply",
    "simplify_exprs": "intent_after_deterministic_repair.simplify_exprs",
    "normalize_in_raw_values": "intent_after_deterministic_repair.normalize_in_raw_values",
    "promote_date_subtraction_to_date_diff": "intent_after_deterministic_repair.promote_date_subtraction_to_date_diff",
    "repair_misclassified_date_diff": "intent_after_deterministic_repair.repair_misclassified_date_diff",
    "normalize_date_diff_raw_values": "intent_after_deterministic_repair.normalize_date_diff_raw_values",
    "canonicalize_temporal_unit_args": "intent_after_deterministic_repair.canonicalize_temporal_unit_args",
    "strip_impossible_having": "intent_after_deterministic_repair.strip_impossible_having",
    "repair_fk_where_type_mismatch": "intent_after_deterministic_repair.repair_fk_where_type_mismatch",
    "resolve_where_value_case": "intent_after_deterministic_repair.resolve_where_value_case",
    "normalize_in_where_types": "intent_after_deterministic_repair.normalize_in_where_types",
    "normalize_boolean_where_values": "intent_after_deterministic_repair.normalize_boolean_where_values",
    "normalize_null_where_values": "intent_after_deterministic_repair.normalize_null_where_values",
    "expand_fk_select_to_descriptive": "intent_after_deterministic_repair.expand_fk_select_to_descriptive",
    "dedup_contradictory_where": "intent_after_deterministic_repair.dedup_contradictory_where",
    "repair_case_when_intent": "intent_after_deterministic_repair.repair_case_when_intent",
    "drop_invalid_case_registry_entries": "intent_after_deterministic_repair.drop_invalid_case_registry_entries",
    "prune_unreferenced_registries": "intent_after_deterministic_repair.prune_unreferenced_registries",
    "repair_array_where_intent": "intent_after_deterministic_repair.repair_array_where_intent",
    "enforce_sensitivity_policy_intent": "intent_after_deterministic_repair.enforce_sensitivity_policy_intent",
    "tail_enforce_grain_consistency": "intent_after_deterministic_repair.tail_enforce_grain_consistency",
    "tail_normalize_where_havings": "intent_after_deterministic_repair.tail_normalize_where_havings",
}

BOOLEAN_NEGATION_PREFIXES: tuple[str, ...] = (
    "no ",
    "no_",
    "not ",
    "not_",
    "non ",
    "non_",
    "non-",
    "un",
    "in",
    "a",
    "dis",
    "de",
    "mis",
    "ir",
    "il",
    "im",
)

BOOLEAN_NEGATION_SUFFIXES: tuple[str, ...] = ("less",)

BOOLEAN_AFFIRMATIVE_STRIP_PREFIXES: tuple[str, ...] = ("a ", "an ")

DO_NOT_LEMMATIZE: frozenset[str] = frozenset(
    {
        "status",
        "address",
        "business",
        "process",
        "axis",
        "basis",
        "news",
        "analysis",
    },
)

STOPWORDS_GRAMMATICAL_PARTICLES: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "can",
        "could",
        "would",
        "should",
        "will",
        "shall",
        "may",
        "might",
        "and",
        "or",
        "as",
        "that",
        "this",
        "these",
        "those",
        "i",
        "we",
        "you",
        "they",
        "he",
        "she",
        "it",
        "me",
        "us",
        "my",
        "our",
        "your",
        "its",
        "please",
        "kindly",
        "just",
        "simply",
        "also",
        "then",
        "want",
        "wanted",
        "need",
        "needed",
    },
)

BOOLEAN_TRUTHY_VALUES: frozenset[str] = frozenset(
    {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
        "active",
        "enabled",
        "pass",
        "passed",
        "open",
        "opened",
        "success",
        "succeeded",
        "present",
    }
)

BOOLEAN_FALSY_VALUES: frozenset[str] = frozenset(
    {
        "0",
        "false",
        "f",
        "no",
        "n",
        "off",
        "inactive",
        "disabled",
        "fail",
        "failed",
        "closed",
        "close",
        "failure",
        "absent",
    }
)

QUESTION_STARTS_LIST: tuple[str, ...] = (
    "List all",
    "Show me",
    "What are",
    "Which",
    "Find",
    "Display",
    "Get",
    "Return",
    "Retrieve",
)

QUESTION_STARTS_GROUP: tuple[str, ...] = (
    "Show me",
    "What is",
    "Group",
    "Break down",
    "Summarize",
    "Calculate",
    "Find the",
    "Get the",
)

IRREGULAR_PLURALS_MAP: Mapping[str, str] = MappingProxyType(
    {
        "data": "datum",
        "criteria": "criterion",
        "phenomena": "phenomenon",
        "indices": "index",
        "matrices": "matrix",
        "people": "person",
        "children": "child",
        "feet": "foot",
        "teeth": "tooth",
        "mice": "mouse",
        "geese": "goose",
    }
)

SANDBOX_QUESTION_TIERS: tuple[str, ...] = (
    "questions",
    "validation_failures",
    "views_questions",
)

SANDBOX_FIXTURE_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "notes.txt": "rental_shop_notes.txt",
        "catalog_notes.txt": "rental_shop_notes.txt",
        "schema.sql": "rental_shop.sql",
    }
)

SANDBOX_DOCTOR_REQUIRED_MEMBERS: tuple[str, ...] = (
    *RENTAL_SHOP_BUNDLE_MEMBERS,
    "questions.txt",
    "artifacts_baseline/owner/schema_graph.json.gz",
    "schema_literals.json",
    "schema_structure_demo.json",
    "sandbox_catalog.json",
    "sandbox_expectations.json",
    "sandbox_scenarios.json",
    "sandbox_handcrafted_fixtures.json",
    "migration_demo/schema_migration_map.json",
    "federation_storefront_notes.txt",
    "federation_catalog_notes.txt",
    "federation_logistics_notes.txt",
    "federation_crm_notes.txt",
)

SANDBOX_BASELINE_CACHE_FILES: tuple[str, ...] = (
    "schema_graph.json.gz",
    "artifact_manifest.json",
    "schema_context.json",
)

SANDBOX_DOCTOR_OPTIONAL_BASELINE_DIRS: tuple[str, ...] = (
    "artifacts_baseline/owner_views",
    "artifacts_baseline/consumer_views",
)

SANDBOX_DOCTOR_OPTIONAL_BASELINE_MEMBERS: tuple[str, ...] = (
    "artifacts_baseline/owner_views/schema_graph.json.gz",
    "artifacts_baseline/consumer_views/schema_graph.json.gz",
)

SANDBOX_UNEXERCISED_PRODUCTION_STAGES: tuple[str, ...] = (
    "live_reflection_and_profiling",
    "probe_mismatch_partial_rebuild",
    "cold_build_descriptions_and_classification",
    "member_cold_reflect_profile_and_member_drift_migration_pending",
    "warmup_and_question_simulation",
    "model_turns_outside_recorded_fixtures",
)

SANDBOX_DEFAULT_DATASET_NAME: str = "main"

SANDBOX_DEFAULT_DATA_DIR: str = "rental_shop_data"

SANDBOX_BUNDLED_MEMBER_SCHEMAS: tuple[tuple[str, str], ...] = (
    ("storefront", "federation_storefront_schema.sql"),
    ("catalog", "federation_catalog_schema.sql"),
    ("logistics", "federation_logistics_schema.sql"),
    ("crm", "federation_crm_schema.sql"),
)

SANDBOX_BUNDLED_MEMBER_NOTES: tuple[tuple[str, str], ...] = (
    ("storefront", "federation_storefront_notes.txt"),
    ("catalog", "federation_catalog_notes.txt"),
    ("logistics", "federation_logistics_notes.txt"),
    ("crm", "federation_crm_notes.txt"),
)

SANDBOX_SHIPPED_FIXTURE_BASENAMES: frozenset[str] = frozenset(
    {
        "rental_shop_notes.txt",
        "rental_shop.sql",
        "rental_shop_views.sql",
        *(fname for _, fname in SANDBOX_BUNDLED_MEMBER_NOTES),
        *(fname for _, fname in SANDBOX_BUNDLED_MEMBER_SCHEMAS),
    }
    | frozenset(SANDBOX_FIXTURE_ALIASES)
    | frozenset(SANDBOX_FIXTURE_ALIASES.values())
)

SANDBOX_BUNDLED_MEMBER_DATA_DIRS: tuple[tuple[str, str], ...] = (
    ("storefront", "federation_storefront_data"),
    ("catalog", "federation_catalog_data"),
    ("logistics", "federation_logistics_data"),
    ("crm", "federation_crm_data"),
)

SANDBOX_BUNDLED_DATASET_NAMES: frozenset[str] = frozenset(
    {SANDBOX_DEFAULT_DATASET_NAME, *(name for name, _ in SANDBOX_BUNDLED_MEMBER_SCHEMAS)}
)

SANDBOX_CONNECTION_HOST_ATTR: str = "_aether_sandbox_host"

SANDBOX_MEMBER_SPACE_TABLES: dict[str, frozenset[str]] = {
    "storefront": frozenset(
        {
            "address",
            "city",
            "country",
            "customer",
            "payment",
            "rental",
            "reservation",
            "staff",
            "store",
        }
    ),
    "catalog": frozenset(
        {
            "actor",
            "author",
            "book",
            "category",
            "city",
            "country",
            "film",
            "film_actor",
            "game",
            "game_supported_language",
            "inventory",
            "item",
            "item_category",
            "item_feature",
            "language",
            "payment",
            "publisher",
        }
    ),
    "logistics": frozenset(
        {
            "courier",
            "damage_report",
            "delivery",
            "inventory_status_history",
            "purchase_line",
            "purchase_order",
            "receipts",
            "stock_transfer",
            "supplier",
            "warehouse",
        }
    ),
    "crm": frozenset(
        {
            "customer",
            "promotion",
            "promotion_redemption",
            "staff",
        }
    ),
}

SANDBOX_MEMBER_SPACE_NOTES_FILES: dict[str, str] = {
    "storefront": "federation_storefront_notes.txt",
    "catalog": "federation_catalog_notes.txt",
    "logistics": "federation_logistics_notes.txt",
    "crm": "federation_crm_notes.txt",
}

SANDBOX_MEMBER_SPACE_QUESTIONS: dict[str, tuple[str, ...]] = {
    "storefront": (
        "How many open reservations are there?",
        "Show active staff at each store.",
        "How many rentals happened in 2025?",
        "Who are our top 5 customers by total payment?",
        "What is the count of pending reservations by store?",
        "List all store locations by city.",
        "Which staff members work at store 1?",
        "List customers who have never rented an item.",
        "What is the total revenue by store?",
        "Which city has the most customers?",
        "How many customers are in each country?",
        "What is the average payment amount?",
        "How many rentals are currently overdue?",
    ),
    "catalog": (
        "How many items are in the catalog by item type?",
        "How many books do we have?",
        "How many games are in the catalog?",
        "Which films include trailers?",
        "Which games support English?",
        "What is the average rental duration?",
        "Which films have the highest replacement cost?",
        "How many actors are in the database?",
        "What are the film ratings available?",
        "How many films are in the Horror category?",
        "Which languages are available?",
        "Which actors appear in the most films?",
        "Which actors have the most film credits?",
        "Which films are in the Horror category?",
        "List films released in 2006.",
        "Which author has the most books?",
        "How many inventory items does each store have?",
        "List publishers with more than five books.",
        "What is the average page count by publisher?",
        "List all films in the catalog.",
    ),
    "logistics": (
        "What is the total delivery fee by courier?",
        "Show purchase orders still open by supplier.",
        "How many damage reports are open?",
        "List inventory status changes in the last 90 days.",
        "Which warehouse holds the most stock?",
        "Show delivery status counts by courier.",
        "List stock transfers between warehouses this year.",
        "Which suppliers have the most purchase lines?",
    ),
    "crm": ("List promotion redemptions by promotion name.",),
}

TASK_PROFILES: dict[str, dict[str, Any]] = {
    "intent": {
        "reasoning": {"effort": "medium", "summary": "concise"},
    },
    "intent_interpret": {
        "reasoning": {"effort": "medium", "summary": "concise"},
    },
    "intent_ground": {
        "reasoning": {"effort": "medium", "summary": "concise"},
    },
    "intent_compose": {
        "reasoning": {"effort": "medium", "summary": "concise"},
    },
    "intent_format": {
        "temperature": 0,
    },
    "intent_schema_repair": {
        "reasoning": {"effort": "low", "summary": "concise"},
    },
    "feedback": {
        "temperature": 0,
    },
    "intake_validate": {
        "temperature": 0,
    },
    "intake_normalize": {
        "temperature": 0,
    },
    "meta_dk": {
        "temperature": 0,
    },
    "meta_schema": {
        "temperature": 0,
    },
    "meta_both": {
        "temperature": 0,
    },
    "domain_knowledge": {
        "reasoning": {"effort": "medium", "summary": "concise"},
    },
    "schema": {
        "reasoning": {"effort": "low", "summary": "concise"},
    },
    "schema_base": {
        "temperature": 0,
    },
    "ddl": {
        "temperature": 0,
    },
    "join": {
        "reasoning": {"effort": "low", "summary": "concise"},
    },
    "synth": {
        "reasoning": {"effort": "low", "summary": "concise"},
    },
    "synth_variety": {
        "reasoning": {"effort": "minimal", "summary": "concise"},
    },
    "conversation": {
        "reasoning": {"effort": "low", "summary": "concise"},
    },
    "default": {
        "temperature": 0,
    },
}

STAGE_ATTRIBUTION_TABLE: Mapping[str, Literal["ground", "compose"]] = MappingProxyType(
    {
        "column_not_found_in_chosen_tables": "compose",
        "chosen_table_lacks_required_column": "compose",
        "filter_targets_missing_column": "compose",
        "joinpath_does_not_exist": "ground",
        "grain_inconsistent_with_chosen_tables": "ground",
        "cte_chosen_tables_inconsistent": "ground",
        "window_partition_column_missing": "compose",
        "compose_added_or_removed_tables": "compose",
        "json_schema_violation": "compose",
        "missing_required_field": "compose",
        "invalid_operator": "compose",
        "invalid_value_type": "compose",
        "unqualified_column_reference": "compose",
        "cte_dependency_cycle": "compose",
        "window_frame_syntax_invalid": "compose",
        "existence_where_encoded_as_subquery": "compose",
        "self_reference_encoded_as_inline_self_join": "compose",
        "correlated_lookup_encoded_as_lateral": "compose",
    }
)
