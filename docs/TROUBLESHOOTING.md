# Troubleshooting reference

Integrator-facing catalogue of session outcomes, diagnostic codes, refusal text, and audit events. Exact API signatures live in [API reference](API_REFERENCE.md); embedding patterns in [Integrator guide](INTEGRATOR_GUIDE.md).

**Reading order:** [README](../README.md) → [Getting started](GETTING_STARTED.md) → [User guide](USER_GUIDE.md) → [Integrator guide](INTEGRATOR_GUIDE.md) → [Sandbox guide](SANDBOX.md) → [API reference](API_REFERENCE.md) → this document → [How it works](HOW_IT_WORKS.md) → [Security](SECURITY.md) → [Support matrix](SUPPORT_MATRIX.md).

## Sections

| Section | Contents |
| --- | --- |
| [Reading terminal steps](#reading-terminal-steps) | Branching on `SessionStep` |
| [SessionOutcome](#sessionoutcome) | Closed terminal failure enum |
| [Diagnostic codes](#diagnostic-codes) | `DiagnosticCode` and `SqlDiagnosticCode` |
| [REFUSAL_CATALOGUE](#refusal-catalogue) | Built-in refusal prose |
| [SessionStep kind values](#sessionstep-kind-values) | Suspend and terminal `kind` strings |
| [Audit events](#audit-events) | `AuditEvent.event_type` inventory |

---

## Reading terminal steps

Every `session.ask(...)` and `session.step(...)` returns a [`SessionStep`](API_REFERENCE.md#sessionstep). Branch on `step.kind` and `step.done` first.

| Field | When set | How integrators use it |
| --- | --- | --- |
| `done` | Every step | `False` means call `step(...)`; `True` means the turn ended |
| `kind` | Every step | Stable stage identifier; see [kind values](#sessionstep-kind-values) |
| `reply_shape` | Suspend steps | `yes_no` or `free_text` input expectation |
| `prompt` | Suspend steps | Short instruction for the interactive layer |
| `answer` | Terminal metadata | Rendered metadata answer |
| `sql` / `data` | Terminal analytical | Executed SQL and result rows |
| `error` | Terminal failure | [`SessionError`](API_REFERENCE.md#sessionerror) with [`SessionOutcome`](API_REFERENCE.md#sessionoutcome) |
| `intent_summary` | Intent-related steps | Structured interpretation headline |
| `semantic_warnings` | Intent confirmation | Model-authored caveats |
| `diagnostics` | Most steps | Tuple of [`Diagnostic`](API_REFERENCE.md#diagnostic) rows for the turn |
| `llm_usage` | Terminal steps | Per-turn token and cost summary when available |

Terminal success has three shapes: populate `answer`, or `sql` with `data`, or neither with `error` set.

**Callbacks** on [`AetherEngine`](API_REFERENCE.md#aetherengine) / [`AetherFederation`](API_REFERENCE.md#aetherfederation):

- `phase_callback` — receives [`PhaseProgressEvent`](API_REFERENCE.md#phaseprogressevent) during construction and ask turns.
- `diagnostic_sink` — receives [`Diagnostic`](API_REFERENCE.md#diagnostic) rows on the diagnostic channel.
- `audit_sink` — receives [`AuditEvent`](API_REFERENCE.md#auditevent) records; see [Audit events](#audit-events).

Map `error.code` to product UX. Use `error.detail_code` for analytics and support playbooks. The [REFUSAL_CATALOGUE](#refusal-catalogue) table lists the built-in refusal prose integrators may optionally surface.

---

## SessionOutcome

Closed enum on [`SessionError.code`](API_REFERENCE.md#sessionerror). Every terminal failure maps to exactly one member.

| Value | Enum member | Typical integrator response |
| --- | --- | --- |
| `forbidden` | `FORBIDDEN` | Credential or engine-context scope gate; warehouse permission denial. |
| `unsupported_operation` | `UNSUPPORTED_OPERATION` | Write, DDL, or administrative operation requested. |
| `unanswerable` | `UNANSWERABLE` | Question reaches outside the active AetherSpace partition. |
| `insufficient_knowledge` | `INSUFFICIENT_KNOWLEDGE` | Metadata route lacked schema or glossary coverage. |
| `not_a_question` | `NOT_A_QUESTION` | Conversational input or unmappable question. |
| `parse_failed` | `PARSE_FAILED` | Intent structure could not be parsed. |
| `validation_failed` | `VALIDATION_FAILED` | Compile-time refusal with a REFUSAL detail code. |
| `execution_failed` | `EXECUTION_FAILED` | Warehouse or federation execution failure after validation. |
| `execution_timeout` | `EXECUTION_TIMEOUT` | Statement or federation timeout. |
| `cost_exceeded` | `COST_EXCEEDED` | Estimated cost cap exceeded. |
| `limit_exceeded` | `LIMIT_EXCEEDED` | Configured row, byte, or federation cap exceeded. |
| `declined` | `DECLINED` | Caller rejected intent or SQL; programmatic sessions terminate instead of restarting. |
| `cancelled` | `CANCELLED` | Turn cancelled cooperatively. |
| `migration_pending` | `MIGRATION_PENDING` | Schema migration skeleton must be applied before construction proceeds. |
| `internal_error` | `INTERNAL_ERROR` | Unexpected library failure. |

---

## Diagnostic codes

`Diagnostic.code` uses [`DiagnosticCode`](#diagnosticcode-catalogue) values. [`SqlDiagnosticCode`](#sqldiagnosticcode-catalogue) is the AST and EXPLAIN subset; each `SqlDiagnosticCode` value also appears as a `DiagnosticCode` member.

### DiagnosticCode catalogue

**Pipeline and reuse**

| Code |
| --- |
| `COMPOSE_REPAIR` |
| `FALLBACK_FRESH_RESTART` |
| `INTERPRET_GROUND_RETRY` |
| `LARGE_RESULT_WARNING` |
| `REUSE_HIT` |
| `REUSE_MISS` |
| `STRUCTURE_EDIT_SKIP` |
| `SENSITIVITY_GATE_HIT` |
| `SQL_PARSE_FAILED` |
| `ZERO_ROW_WHERE_AUTO_FIXED` |
| `ZERO_ROW_WHERE_SUGGESTION` |

**LLM usage**

| Code |
| --- |
| `DESCRIPTION_ENRICHMENT_FAILED` |
| `DESCRIPTION_ENRICHMENT_NOOP` |
| `ENUM_PROMPT_TRUNCATED` |
| `LLM_TURN_COST` |

**Configuration**

| Code |
| --- |
| `CONFIG_FILE_VALUE_APPLIED` |

**Artifacts and write queue**

| Code |
| --- |
| `ARTIFACTS_DIR_NOT_LOCAL` |
| `ARTIFACT_GROWTH` |
| `ARTIFACT_LIMIT_NEAR` |
| `MIGRATION_CHECKPOINT_ORPHANED` |
| `STALE_ARTIFACT_LOCK` |
| `WRITE_QUEUE_CORRUPT` |
| `WRITE_QUEUE_FULL` |

**Schema profiling**

| Code |
| --- |
| `COLUMN_CHARSET_MISMATCH` |
| `COLUMN_PROFILE_FAILED` |
| `COMPOSITE_DESCRIPTIVE_PROFILE_FAILED` |
| `MATERIALIZED_VIEW_ANSWER` |
| `PK_INFERENCE_PROMPT` |
| `PROFILE_TABLE_CLONE_FAILED` |
| `SCHEMA_FK_CATALOG_ABSENT` |
| `STRUCTURE_EDIT_SKIP` |
| `SCHEMA_ROLE_TYPE_COERCED` |
| `SCHEMA_UNKNOWN_TYPE_UNUSABLE` |

**Structure edits**

| Code |
| --- |
| `STRUCTURE_NEEDS_RECONFIRMATION` |

**Join semantics**

| Code |
| --- |
| `COMPARISON_JOIN_DETOUR` |
| `JOIN_CANDIDATE_CAP` |
| `JOIN_NULLABLE_KEY` |
| `JOIN_ORPHAN_RATE_HIGH` |
| `JOIN_PATH_TIE_CEILING_EXCEEDED` |
| `REDUNDANT_JOIN_WHERE_DROPPED` |
| `REDUNDANT_KEY_JOIN_CAP_REACHED` |
| `REDUNDANT_KEY_JOIN_ELIMINATED` |
| `SEMANTIC_PROFILE_WHERE_EDGE` |

**Template store**

| Code |
| --- |
| `TEMPLATE_REMAP_DIVERGED` |
| `TEMPLATE_STORE_ORPHANED` |

**Upload validation**

| Code |
| --- |
| `DATA_QUALITY_ADVISORY` |
| `DATA_QUALITY_AUTO_CORRECTED` |
| `DATA_QUALITY_AUTO_READ` |
| `DATA_QUALITY_BLOCKING` |
| `UPLOAD_TRANSFORM_APPLIED` |
| `UPLOAD_TRANSFORM_REJECTED` |
| `UPLOAD_UNIT_AFFIX_STRIPPED` |

**Federation**

| Code |
| --- |
| `COORDINATOR_LIMITS` |
| `FEDERATION_CAP_EXCEEDED` |
| `FEDERATION_COORDINATOR_ARROW_SPILL_FALLBACK` |
| `FEDERATION_COORDINATOR_DECIMAL_FALLBACK` |
| `FEDERATION_COORDINATOR_EXECUTED` |
| `FEDERATION_INELIGIBLE` |
| `FEDERATION_JOIN_CANDIDATE_CAP` |
| `FEDERATION_JOIN_FAN_OUT` |
| `FEDERATION_MALFORMED_MEMBER_ANSWER` |
| `FEDERATION_MAPPING_DRIFT` |
| `FEDERATION_MEMBER_EXECUTED` |
| `FEDERATION_MEMBER_FAILED` |
| `FEDERATION_MEMBER_GENERATED` |
| `FEDERATION_MEMBER_PROBE_FAILED` |
| `FEDERATION_MEMBER_REMOVED` |
| `FEDERATION_MEMBER_TIMEZONE_MISMATCH` |
| `FEDERATION_PARTIAL_FAILURE` |
| `FEDERATION_PLAN_REPLAY` |
| `FEDERATION_POOL_UNDERSIZED` |
| `FEDERATION_REDUCTION_NULL_KEYS` |
| `FEDERATION_SEMIJOIN_SKIPPED` |
| `FEDERATION_SOURCES_QUERIED` |
| `FEDERATION_TIMESTAMP_NORMALISED` |
| `FEDERATION_TIME_ANCHOR` |
| `FEDERATION_TURN_CANCELLED` |
| `MEMBER_LIMIT_NARROWED` |
| `ROUNDING_MODE_MIXED` |

**Catch-all**

| Code |
| --- |
| `ENGINE_INFO` |
| `CANCEL_NOT_SUPPORTED` |

**Terminal refusals (`REFUSAL_*`)**

| Code |
| --- |
| `REFUSAL_AGGREGATE_FAN_OUT` |
| `REFUSAL_AMBIGUOUS_DATE_LITERAL` |
| `REFUSAL_CAPABILITY_GAP` |
| `REFUSAL_CLAUSE_WIDENED_ROWSET` |
| `REFUSAL_CONVERSATIONAL_DENY` |
| `REFUSAL_CTE_CAP` |
| `REFUSAL_DECLINED_SCHEMA` |
| `REFUSAL_HOP_CEILING` |
| `REFUSAL_INSUFFICIENT_KNOWLEDGE` |
| `REFUSAL_INVALID_QUESTION` |
| `REFUSAL_JOIN_PATH_TIE_CAP` |
| `REFUSAL_JOIN_PATH_UNAVAILABLE` |
| `REFUSAL_NOT_AVAILABLE_IN_CONTEXT` |
| `REFUSAL_NULL_IN_NEGATED_LIST` |
| `REFUSAL_OPAQUE_EXPR` |
| `REFUSAL_OPERATION_NOT_SUPPORTED` |
| `REFUSAL_PARSE_FAILURE` |
| `REFUSAL_PERMISSION_DENIED` |
| `REFUSAL_PROBE_CTE_PLACEMENT` |
| `REFUSAL_SCOPE_VIOLATION` |
| `REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN` |
| `REFUSAL_UNION_COLUMN_MISSING` |
| `REFUSAL_UNMAPPABLE_QUESTION` |
| `REFUSAL_UNSUPPORTED_COLUMN_TYPE` |

### SqlDiagnosticCode catalogue

| Code | Meaning |
| --- | --- |
| `agg_in_where` | Aggregate appears in WHERE. |
| `ambiguous_column` | Column name matches more than one source. |
| `ast_parse_failed` | SQL text could not be parsed into an AST. |
| `cross_join_not_allowed` | CROSS join is refused. |
| `cte_unreferenced` | CTE is defined but never referenced. |
| `exists_not_allowed` | EXISTS subquery is refused. |
| `explain_cartesian_join` | EXPLAIN shows a cartesian join. |
| `explain_cost_exceeded` | Estimated cost exceeded configured cap. |
| `explain_other` | Other EXPLAIN-plan concern. |
| `explain_seq_scan_indexed` | EXPLAIN shows sequential scan on indexed column. |
| `explain_sort_spill` | EXPLAIN shows sort spill to disk. |
| `explain_temporary_table` | EXPLAIN shows a temporary table. |
| `explain_zero_estimate` | EXPLAIN returned a zero row estimate. |
| `forbidden_structure` | AST contains a forbidden construct. |
| `having_without_group` | HAVING without GROUP BY. |
| `lateral_not_allowed` | LATERAL join is refused. |
| `multiple_statements` | More than one statement in the batch. |
| `no_root` | No root SELECT could be identified. |
| `non_grouped_select_col` | SELECT column is not grouped or aggregated. |
| `not_select` | Statement is not a SELECT. |
| `param_unbound` | Bind placeholder has no value. |
| `param_undeclared` | Bind placeholder is not declared on the template. |
| `self_join_not_allowed` | Self-join on the same alias is refused. |
| `subquery_not_allowed` | Subquery shape is refused. |
| `unknown_column` | Column reference is not in scope. |
| `unknown_cte` | CTE name is not defined. |
| `unknown_table` | Table reference is not in scope. |
| `using_not_allowed` | USING clause is refused. |

### Common Diagnostic.details keys

| Key | Typical use |
| --- | --- |
| `phase` | Pipeline or federation phase. |
| `attempt` | Retry or parse attempt index. |
| `reason` | Short machine reason. |
| `sources_queried` | Federation members queried on a turn. |
| `cap` | Truncation cap. |
| `types` | Truncated enum type list. |
| `logical_column` | Logical column label in federation mapping drift. |
| `path` | Artifact or file path. |
| `issue_code` | Upload data-quality issue code. |
| `requests` | LLM request count on `LLM_TURN_COST`. |
| `input_tokens` | Input token count on `LLM_TURN_COST`. |
| `cached_input_tokens` | Cached input tokens on `LLM_TURN_COST`. |
| `output_tokens` | Output token count on `LLM_TURN_COST`. |
| `cost_usd` | Estimated USD cost on `LLM_TURN_COST`. |
| `price_table_as_of` | Pricing table date on `LLM_TURN_COST`. |
| `unpriced_models` | Unpriced logical models on `LLM_TURN_COST`. |

---

## REFUSAL_CATALOGUE

Exact `user_text` strings from `REFUSAL_CATALOGUE` in `_constants_runtime.py`. Placeholders such as `{tables}` are filled at runtime.

| Code | User-facing text |
| --- | --- |
| `REFUSAL_AGGREGATE_FAN_OUT` | This aggregate would duplicate parent rows because of how the tables connect. Try grouping at the parent grain first, or use a join path that does not multiply rows. |
| `REFUSAL_AMBIGUOUS_DATE_LITERAL` | This filter cannot be expressed: the date bound is ambiguous. Use ISO 8601 form such as 2020-01-15 or 2020-01-15T14:30:00. |
| `REFUSAL_CAPABILITY_GAP` | This question shape cannot be answered with the databases currently available. Try a simpler question or ask on each source individually. |
| `REFUSAL_CLAUSE_WIDENED_ROWSET` | A limit, sort, or distinct cannot be applied cleanly when a join multiplies rows. Group first or simplify joins before limiting or deduplicating results. |
| `REFUSAL_CONVERSATIONAL_DENY` | I can only help with questions about your data. |
| `REFUSAL_CTE_CAP` | This question needs too many intermediate query steps. Try splitting the question into smaller parts or simplifying the logic. |
| `REFUSAL_DECLINED_SCHEMA` | The proposed table and column mapping was declined. Try rephrasing using tables and columns that exist in this database, or ask about a related concept. |
| `REFUSAL_HOP_CEILING` | This comparison would require joining across too many tables. Try comparing values on tables that are closer together in the schema. |
| `REFUSAL_INSUFFICIENT_KNOWLEDGE` | The available schema descriptions and domain knowledge do not contain enough information to answer this question. |
| `REFUSAL_INVALID_QUESTION` | I could not pin this question to specific tables or columns. Try naming the entity you care about, the metric you want, and any filter such as a date range, status, or region. |
| `REFUSAL_JOIN_PATH_TIE_CAP` | Too many equally short join paths between {source_table} and {target_table} ({path_count} paths; limit {ceiling}). Narrow the tables in your question or declare which relationship to use. |
| `REFUSAL_JOIN_PATH_UNAVAILABLE` | These tables could not be connected: {tables}. Declare a foreign-key or semantic relationship between them, or ask using fewer tables. |
| `REFUSAL_NOT_AVAILABLE_IN_CONTEXT` | This question refers to information that is not available in this context. |
| `REFUSAL_NULL_IN_NEGATED_LIST` | This filter cannot be expressed: a NOT IN list cannot include null. Ask whether the column is null or not, or name only the non-null values to exclude. |
| `REFUSAL_OPAQUE_EXPR` | This question uses an expression structure that cannot be compiled safely. Rephrase using explicit columns, filters, and aggregates supported by the schema. |
| `REFUSAL_OPERATION_NOT_SUPPORTED` | This type of operation is not supported. I can only answer questions that read from your data. |
| `REFUSAL_PARSE_FAILURE` | I could not understand the structure of this question. Try naming specific tables or columns, keeping filters simple and references clear. |
| `REFUSAL_PERMISSION_DENIED` | Unable to locate the requested data. Please contact your administrator. |
| `REFUSAL_PROBE_CTE_PLACEMENT` | A filter step cannot be used in this join position. Restructure the question so filtering happens on the correct side of the join. |
| `REFUSAL_SCOPE_VIOLATION` | This question cannot be answered with the information currently available. Try rephrasing to ask about tables and columns you can see in the schema. |
| `REFUSAL_SUBDAY_DATE_WINDOW_ON_DATE_COLUMN` | This filter cannot be expressed: the column stores dates without time-of-day, so hour, minute, or second windows cannot be answered. Ask for a day-level window instead. |
| `REFUSAL_UNION_COLUMN_MISSING` | A column needed for this answer is missing from one or more databases in the group. Try asking over the sources that have the column, or declare a shared column mapping. |
| `REFUSAL_UNMAPPABLE_QUESTION` | I could not pin this question to specific tables or columns. Try naming the entity you care about, the metric you want, and any filter such as a date range, status, or region. |
| `REFUSAL_UNSUPPORTED_COLUMN_TYPE` | This question cannot be answered: the {column} column has an unsupported data type and cannot be filtered or aggregated. |

---

## SessionStep kind values

| `kind` | `done` | `reply_shape` | Meaning |
| --- | --- | --- | --- |
| `idle` | `—` | `—` | Session reset; no active turn. |
| `awaiting_intent_confirm` | `False` | `yes_no` | Confirm interpreted plan. |
| `awaiting_intent_feedback` | `False` | `free_text` | Supply reason intent is wrong. |
| `awaiting_reuse_confirm` | `False` | `yes_no` | Confirm a template reuse hit. |
| `awaiting_sql_confirm` | `False` | `yes_no` | Confirm SQL after execution preview. |
| `execute` | `False` | `yes_no` | Confirm running stored SQL. |
| `awaiting_sql_feedback` | `False` | `free_text` | Supply reason SQL is wrong. |
| `result` | `True` | `—` | Terminal analytical success (`sql` and `data`). |
| `meta` | `True` | `—` | Terminal metadata answer (`answer`). |
| `error` | `True` | `—` | Terminal failure (`error`). |

---

## Audit events

`audit_sink` receives [`AuditEvent`](API_REFERENCE.md#auditevent) records.

| `event_type` | When emitted |
| --- | --- |
| `init` | After successful `AetherEngine` / `AetherFederation` construction. |
| `data_quality` | After construction when the file engine validated uploads. |
| `domain_knowledge_ingest` | Domain knowledge loaded from artifacts or notes during construction. |
| `refresh` | After `refresh` reconciles artifacts and reopens the connection. |
| `ask_begin` | Start of `session.ask(...)`. |
| `ask_suspend` | Pipeline returned a deferred prompt (`session.step` required). |
| `ask_cancelled` | Turn ended by cooperative `session.cancel()`. |
| `ask_done` | Turn completed (`details` includes `outcome`, `kind`). |
| `ask_error` | Terminal failure or fatal guard error during a turn. |
| `ask_blocked` | `ask` rejected before pipeline start (non-`str` question or active turn). |
| `llm_call` | One model request during a turn (token details in `details`). |
| `llm_turn` | Turn-level LLM usage summary. |
| `sql_execution` | After guarded SQL execution (statement hash, row count, elapsed time). |
| `federation_semijoin_key_transfer` | Semijoin key reduction transferred keys between federation members. |
| `write_queue_feedback_record` | Writer applied a queued `feedback_record`. |
| `write_queue_template_reject` | Writer applied a queued `template_reject`. |
| `write_queue_template_accept` | Writer applied a queued `template_accept`. |
| `write_queue_structure_proposal` | Writer materialised a queued `structure_proposal`. |
| `apply_structure` | After `apply_structure` persists. |
| `clear_template_store` | After template tree removal and reload. |
| `clear_simulation_caches` | After QSim or seed-warmup cache deletion. |
| `clear_all_learning` | After combined learning clears. |
| `close` | After `close()` disposes connections. |
| `export_federation` | After `export_federation` returns. |
| `export_knowledge` | After `export_knowledge` returns. |
| `export_structure` | After `export_structure` returns. |

### Common AuditEvent.details keys

| Key | Typical use |
| --- | --- |
| `engine` | Engine dialect on `init`. |
| `federation` | Federation name on `init`. |
| `members` | Member count on federation `init`. |
| `status` | Skipped ingest status. |
| `kept` | Rows kept during ingest. |
| `space` | Aetherspace uid for export or clear events. |
| `removed` | Count removed during structure or template clears. |
| `existed` | Whether a template store existed before clear. |
| `outcome` | Terminal ask outcome on `ask_done`. |
| `kind` | Terminal `SessionStep.kind` on `ask_done`. |
| `result_columns` | Column names on successful `ask_done`. |
| `sources_queried` | Federation member sources on `ask_done`. |
| `message` | Error or suspend message on `ask_error` / `ask_suspend`. |
| `source_id` | Federation member attribution on suspend or error. |
| `phase` | Federation execute phase on suspend or error. |
| `limit_key` | Federation limit key on suspend or error. |
| `scope` | LLM call scope (`build`, `question`, `run`). |
| `task` | LLM task name on `llm_call`. |
| `logical_model` | Logical model tier on `llm_call`. |
| `api_model` | Provider model id on `llm_call`. |
| `input_tokens` | Input tokens on `llm_call` / `llm_turn`. |
| `cached_input_tokens` | Cached prefix tokens on `llm_call` / `llm_turn`. |
| `output_tokens` | Output tokens on `llm_call` / `llm_turn`. |
| `attempt` | Retry attempt index on `llm_call`. |
| `elapsed_ms` | Wall time for one `llm_call` or SQL execution. |
| `statement_hash` | SHA-256 of the executed SQL on `sql_execution`. |
| `row_count` | Rows returned on `sql_execution`. |
| `schema_hash` | Schema fingerprint on `sql_execution` when available. |
| `cache_write_tokens` | Cache-write tokens when reported by provider. |
| `cost_usd` | Estimated USD cost when priced. |
| `requests` | Request count on `llm_turn`. |
| `removed_files` | Files removed during cache clears. |
| `keep_structure` | Whether applied structure was kept on `clear_all_learning`. |
| `issues_json` | Serialised upload issues on `data_quality`. |
| `ok` | Upload validation pass or fail on `data_quality`. |
| `issue_count` | Upload issue count on `data_quality`. |
| `path` | Corrupt artifact path in write-queue recovery. |
| `migration_tier` | Migration tier on `refresh`. |
| `schema_changed` | Whether schema changed on `refresh`. |
| `orphans_removed` | Orphan directories removed on `refresh`. |
| `bytes_reclaimed` | Bytes reclaimed on `refresh`. |
| `table_edits` | Table edits on `apply_structure`. |
| `column_edits` | Column edits on `apply_structure`. |

---

**See also:** [API reference](API_REFERENCE.md) | [Integrator guide](INTEGRATOR_GUIDE.md) | [Support matrix](SUPPORT_MATRIX.md)
