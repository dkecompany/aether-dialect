# Security model

Threat model, credential inheritance, why SQL injection is impossible, what schema construction learns and discloses, the per-phase LLM disclosure inventory, sensitivity tiers, deny lists, and operational local-machine disclosure. Pipeline narrative and operator day-to-day semantics live in [How it works](HOW_IT_WORKS.md) and the [User guide](USER_GUIDE.md). Per-engine SQL capability limits live in the [Support matrix](SUPPORT_MATRIX.md).

**Reading order:** [README](../README.md) → [Getting started](GETTING_STARTED.md) → [User guide](USER_GUIDE.md) → [Integrator guide](INTEGRATOR_GUIDE.md) → [Sandbox guide](SANDBOX.md) → [API reference](API_REFERENCE.md) → [Troubleshooting](TROUBLESHOOTING.md) → [How it works](HOW_IT_WORKS.md) → this document → [Support matrix](SUPPORT_MATRIX.md).

## Sections

| Section | Contents |
| --- | --- |
| [1. Threat model](#1-threat-model) | Trust boundaries and mitigations |
| [2. Execution boundary and credentials](#2-execution-boundary-and-credentials) | Database grants vs engine / federation / space scope |
| [3. Schema profiling, roles, and classification](#3-schema-profiling-roles-and-classification) | What construction reflects, profiles, and classifies |
| [4. Why SQL injection is not possible](#4-why-sql-injection-is-not-possible) | Intent IR, deterministic SQL, AST, scope, EXPLAIN |
| [5. LLM context inventory](#5-llm-context-inventory) | Per-phase prompt disclosure |
| [6. Sensitivity tags](#6-sensitivity-tags) | Canonical tier definitions |
| [7. Deny lists](#7-deny-lists) | Graph removal vs sensitivity |
| [8. Operational logging and observability](#8-operational-logging-and-observability) | Debug logs, audit sink, session diagnostics |
| [9. Federation coordinator spill](#9-federation-coordinator-spill) | Temporary parquet when member frames exceed memory budgets |

---

## 1. Threat model

Nothing in the engine generates executable SQL from free-form user or model text. On the questioning surface, **context assembly** (schema payloads, closed candidate sets, refusal conditions, and post-provider validation gates) is **deterministic** for a fixed schema graph, question, and policy configuration — the same inputs produce the same assembled context and the same validation outcome. **Provider output is not deterministic**; provider-mediated stages (intent slots, join disambiguation, display labels, and similar) may vary between runs even when context assembly is identical. After provider output arrives, refusal and validation remain deterministic. SQL materialization from legal IR is **deterministic SQL generation**. Regex forbidden-SQL checks, dialect AST validation, schema/scope/sensitivity gates, and an EXPLAIN-class check sit between provider-mediated intent and the database.

The engine is built for the case where:

- A trusted operator configures the engine with database credentials and an LLM provider.
- A semi-trusted user asks natural-language questions; the user may read only what those credentials authorize and must not bypass the analytical `SELECT` surface.
- The LLM provider sees the prompts the engine sends. Whatever the operator considers acceptable to send to the configured provider must include the schema metadata listed in [Section 5](#5-llm-context-inventory).
- The local filesystem under the engine storage directory (and, for federations, the federation tree including coordinator spill) is operator-controlled.

| Risk | Mitigation |
| --- | --- |
| User crafts a question that runs arbitrary SQL | Structured intent IR, deterministic SQL generation, `PolicyConfig.FORBIDDEN_SQL` regex list, dialect AST structural rejection, schema/scope/sensitivity gates, EXPLAIN gate before execution ([Section 4](#4-why-sql-injection-is-not-possible)). |
| LLM output escapes the analytical subset | Same gates; constructs the IR cannot represent are refused or reformulated ([Support matrix](SUPPORT_MATRIX.md)). |
| Sensitive column data leaks through prompts or artifacts | Sensitivity tiers (`none`, `restricted`, `hidden`), deny lists, visibility gates, capped enum heads (unioned across members on a composite graph), and the rule that cross-source join keys must be `sensitivity == none`. |
| User assumes AetherSpace replaces database RBAC | A space narrows which objects a turn may reference and refuses questions that reach past it, and it is not a permission boundary because it can neither be defined nor entered beyond what credentials already permit ([Section 2](#2-execution-boundary-and-credentials)). |
| Cross-source join or semi-join keys transmit restricted values | Declared cross-source keys that are not `none` fail federation validation; mapping suggestions exclude non-`none` columns; semi-join reduction at execution raises if a reducing key is not allowed. |
| Semi-join reduction copies key values across members | At execution, distinct key values from one member's result frame are bound as filters in another member's SQL. This is not an LLM disclosure - it is cross-database data movement under the connected roles. Keys must be `sensitivity == none`. |

## 2. Execution boundary and credentials

**Database credentials are the real boundary.** Every query runs with the grants of the database role the operator connected with. The library never elevates privileges, never impersonates another database user, and never bypasses server-side `GRANT` / `REVOKE`. If the connected role cannot `SELECT` a table, the engine cannot either - no matter what allow lists or sensitivity tags say.

**`EngineContext` and `FederationContext` are a small additive layer on top of those inherited permissions - not a replacement for them.** Their allow/deny object and column lists remove tables and columns from the in-memory schema graph and block references at validation and execution time. They tighten what the analytical surface can reach among objects the role already permits. They can never grant access the database would refuse.

**`SpaceContext` narrows knowledge and template partitions at question time.** A space narrows which objects a turn may reference and refuses questions that reach past it, and it is not a permission boundary because it can neither be defined nor entered beyond what credentials already permit. Effective SQL and meta gates use **context ∩ credential reflection ∩ non-HIDDEN** ([User guide — AetherSpace](USER_GUIDE.md#aetherspace)).

Three-way summary:

| Scope type | Tightens the in-memory analytical surface? | Refuses out-of-scope questions? |
| --- | --- | --- |
| Connected database role | Indirectly (reflection only sees what the role can see) | **Yes — warehouse grants are the hard boundary** |
| `EngineContext` / `FederationContext` | Yes (graph removal + execution-time reference checks) | Yes — further restriction only; no elevation |
| `SpaceContext` | Yes — knowledge, prompts, and template partition | Yes — `unanswerable` when the question reaches past the active space |

**Consumer deployments** pair restricted database logins with named `EngineContext` subsets and optional AetherSpaces. Align context allow lists with each login's grants ([Integrator guide - Multi-user deployment](INTEGRATOR_GUIDE.md#multi-user-deployment)).

**Credential reflection snapshot.** When a consumer role reflects a subset of the owner-published schema, the engine stores that visible object set and intersects it with context allow/deny and sensitivity for SQL gates, meta answers, and `export_structure`. Prefer refusing before EXPLAIN/execute when the intent references objects outside that intersection. If grants drift after the snapshot and the warehouse still rejects the statement, the user-facing message asks them to contact an administrator — by then warehouse query history may already record the SQL, so that path is drift fallback only, not the primary authorization control.

## 3. Schema profiling, roles, and classification

At first construction (and on cache invalidation), the engine builds a compiled schema graph. Security-relevant detail:

1. **Reflect.** Catalog reflection loads table and column names, data types, primary keys, and foreign keys where the dialect exposes them (for example InnoDB / `information_schema`, PostgreSQL, Unity Catalog, Snowflake, `PRAGMA foreign_key_list` on SQLite). BigQuery returns no live FK metadata from the catalog; join edges then come from `EngineContext.sql_file`, `export_structure` / `apply_structure`, or inference.

2. **Profile (read-only aggregates only).** For each column still in scope the profiler runs aggregate queries: row counts, distinct counts, null counts / null rates, min/max for numerics and dates, mode-frequency ratios, and capped frequent-value samples for categorical columns (`PolicyConfig.CATEGORICAL_SAMPLE_SIZE`, default 20). Bounded distinct-value samples also feed foreign-key overlap probes. Profiling never bulk-exports row payloads to the LLM. Progress prints as `Profiling [i/n] <table>` on stdout.

3. **Usability gates.** Columns that fail statistical gates are marked unusable and omitted from LLM-facing schema literals (unless they are primary or foreign keys). Gates include: distinct count <= 1; null ratio at or above `UNUSABLE_NULL_RATIO_THRESHOLD` (0.99); mode frequency ratio at or above `SENTINEL_MODE_FREQUENCY_THRESHOLD` (0.99). Structure documents may set `usable: true` to re-enable a profiler-rejected column for classification and visibility. `apply_structure` rejects `usable: false` - withhold a column with sensitivity tiers or deny lists instead.

4. **Role and sensitivity classification.** A construction-time LLM pass receives per-column profiling summaries (name, type, PK/FK flags, and profile hints such as distinct count, distinct ratio, null ratio) plus optional domain notes - not arbitrary row dumps. It assigns column **roles** (identifier, measure, dimension / categorical, temporal, free text, boolean, audit, and similar) and **sensitivity** tiers (`none`, `restricted`, `hidden`). Assignments are metadata that steer intent validation and prompt shaping.

5. **Foreign-key inference.** When the catalog graph is disconnected, name-overlap heuristics and bounded value-containment / overlap probes may add inferred edges. Inference never requires shipping full tables to the model.

**Federation mapping suggestions (no provider call).** When operators build or extend cross-source mappings, `suggest_cross_source_mappings` proposes candidate column and table equivalences from identifier overlap and bounded value-overlap comparison on join-key columns. Scoring and ranking are entirely deterministic; **no provider call**. Proposals are advisory only and are **never applied without explicit confirmation** through the mappings editor workflow. Candidate cross-source join-key columns must be `sensitivity == none`; restricted or hidden columns are excluded from scoring.

## 4. Why SQL injection is not possible

User questions and model output never become executable SQL strings directly. The chain is mechanical:

1. **Bounded intent slots - the model never emits SQL.** Interpret, ground, and compose stages fill structured JSON (tables, columns, filters, aggregates, CTE steps). Free-form SQL fragments are not a legal intent shape; parse and schema validators reject them.

2. **Typed intermediate representation.** Intent JSON is parsed into a typed runtime IR. Slot values are schema-bound identifiers or bound parameter handles, not statement text.

3. **Deterministic SQL generation.** The SQL generator renders the IR through dialect adapters. Filter and having literals from the question become bind parameters (`:param` / dialect equivalents). User text can influence parameter *values*; it cannot rewrite statement structure, introduce a second statement, or concatenate predicate text into the SQL string.

4. **Forbidden-SQL regex gate.** Before AST checks, the engine scans the rendered statement against a fixed list of disallowed keywords and shapes: DML and DDL verbs (`UPDATE`, `DELETE`, `INSERT`, `MERGE`, `ALTER`, `DROP`, and similar), multi-statement batches, set operators (`UNION`, `INTERSECT`, `EXCEPT`), lateral joins, row-skipping `OFFSET` / `FETCH FIRST`, bare `DISTINCT ON`, array literals, existential subqueries (`EXISTS (`), and related patterns.

5. **Dialect AST structural rejection.** The parsed statement tree is walked (PostgreSQL via `pglast`; other engines via sqlglot). Structural rules reject existential subqueries, lateral joins, `USING` joins (except scalar CTE names explicitly allowed for scalar-subquery emission), disallowed cross joins, nested subqueries beyond the analytical subset, top-level set operations, direct self-joins on the same physical table, recursive CTEs, and set operations inside CTEs. Window functions and `CASE` are allowed within the analytical subset.

6. **Schema, scope, usability, and sensitivity gates.** Every referenced table and column must exist in the compiled graph, survive `EngineContext` / `FederationContext` allow/deny scope, pass usability and selectability rules, and satisfy sensitivity validators (bare restricted/hidden projection dropped or rejected; literal-value filters on non-selectable columns rejected; sensitive `GROUP BY` / `ORDER BY` rejected). Denied columns are treated as absent.

**Federation replica projections.** The CRM `staff` mirror is the worked example: the partition seed exports only `staff_id`, `first_name`, `last_name`, and `store_id` - `password` and government-id columns never enter the CRM database file, so reflection cannot surface them even before mappings apply. The declaration's column map on the CRM member declares that same column subset; the authoritative `storefront` member retains the full table. Post-composition re-redaction removes profile samples for any column marked `hidden` on any member, so a sensitive field hidden on one side stays hidden and unsampled on the composite graph.

**Semi-join key gating at execution:** Reducing and semi-join keys must be `sensitivity == none`. When a reducing or semi-join edge references a restricted or hidden key, federation execution **raises** `FederationRuntimeError` (rejected - not silently skipped). Separately, plan-time eligibility omits non-`none` joins from the eligible cross-source set, and declaration-time validation rejects non-`none` declared keys. Low distinct-count keys fail with a cardinality diagnostic and fall back to a different plan shape when an alternate shape is available; the decision is deterministic for a fixed profile.

7. **EXPLAIN-class gate.** When the dialect still has a usable EXPLAIN backend, validation runs an EXPLAIN-class diagnose before execution (warehouse `EXPLAIN` / `EXPLAIN COST` / `SHOWPLAN_*`, or BigQuery dry-run). Permission-denied on EXPLAIN disables further EXPLAIN for that dialect instance rather than elevating privileges. Cost caps can reject plans as `EXPLAIN_COST_EXCEEDED`.

Even a fully compromised or adversarial model cannot emit DML/DDL, multi-statement batches, or concatenated predicate text that reaches the database. Those shapes are not produced by the deterministic generator from legal IR, and they do not survive the forbidden-SQL list, AST structural rules, schema/scope/sensitivity gates, or the EXPLAIN gate.

## 5. LLM context inventory

Each subsection below states the task, exactly what content is sent to the configured provider, and which data category leaves the process.

**Scoped determinism.** **Context assembly** for every provider call is **deterministic**: for a fixed schema graph, question text, template state, and policy flags, the engine assembles the same user payload every time. **Refusal conditions** (scope denial, sensitivity gates, forbidden-SQL checks, AST structural rejection) and **validation** after provider output are also **deterministic**. **Provider output is not deterministic** — two calls with identical assembled context may return different slot values, join choices, or labels. The inventory below separates deterministic assembly from provider-mediated decisions.

### 5.1 Question gate (pre-interpret)

- **Task:** Classify whether the question is an analytical query and normalize canonical wording before the interpret stage.
- **Calls:** `validate_question`, then `normalize_question_via_llm` on the questioning path (`session.ask` / `session.step`).
- **Deterministic context assembly:** The raw question string (and corrected text from the validation step) is the payload; same question → same assembled context.
- **Provider-mediated:** Validity classification and canonical wording; **provider output is not deterministic**.
- **Content:** Question text only.
- **Data sent:** User question strings; no schema metadata and no warehouse row data.

### 5.2 Interpret phase

- **Task:** Read the question against domain table/column descriptions; emit a natural-language solution plan (not SQL, not IR).
- **Content:** Question, Interpret-stage schema payload (descriptions and enum heads), optional prior-question feedback, supported-capability prose.
- **Data sent:** Metadata and capped enum heads only; no row data. On a federated composite graph, enum heads are the capped union of distinct labels contributed by every member for each logical enum - not one member's domain alone.

### 5.3 Ground phase

- **Task:** Convert the interpret plan into logical intent JSON bound to schema identifiers.
- **Deterministic context assembly:** Question, interpret plan, and Ground-stage schema payload (identifiers, types, PK/FK markers, column **roles** and descriptions, value types, enum heads).
- **Provider-mediated:** Logical intent slot values; **provider output is not deterministic**.
- **Content:** Same as assembled context above.
- **Data sent:** Metadata and enum heads only. Column roles steer validation and join disambiguation but are not row data. Federated graphs union member enum labels under the same per-enum cap as single-engine payloads.

### 5.4 Compose phase

- **Task:** Convert logical intent into the runtime IR.
- **Deterministic context assembly:** Logical intent plus Compose-stage structural schema payload (identifiers, types, PK/FK markers, declared cross-source relationships as ordinary edges).
- **Provider-mediated:** Runtime IR slot filling and repairs; **provider output is not deterministic**.
- **Content:** Same as assembled context above.
- **Data sent:** Structural metadata only - no source ids, no member physical names beyond the unified composite identifiers.

### 5.5 Join selection (generation-time)

Join selection is the canonical example of **deterministic selection, then configured provider**:

1. **Deterministic candidate enumeration.** The engine computes eligible join paths or declared cross-source keys from the compiled graph, scope gates, and sensitivity rules. The resulting candidate list is a function of schema state and intent only — not provider output.
2. **Deterministic context assembly.** For each scope with more than one candidate, the engine builds a fixed JSON payload: question text, SQL preview, and the closed candidate list with path signatures (`candidate_id`, path signature, edge kinds — no row values). For federation, each unordered source pair lists only declared cross-source join keys; non-`none` sensitivity keys are excluded before the payload is built. The same intent and graph always produce the same payload.
3. **Configured provider disambiguation.** The assembled payload is sent to the configured provider, which returns one `candidate_id` per scope from the closed set only. **Provider output is not deterministic** — two calls with identical assembled context may choose different `candidate_id` values.
4. **Post-selection validation** — schema binding, sensitivity, and SQL gates remain **deterministic**.

Single-candidate scopes skip step 3 entirely.

### 5.6 Repair and validation

When validation fails, the same Compose system prompt is reused with error rows and the structural payload for implicated tables. The engine never sends raw database warehouse rows to the provider for repair. Assembly of error rows and structural payloads is **deterministic**; repair slot values are **provider-mediated** and **provider output is not deterministic**. Diagnostic codes for retries and repairs: [Troubleshooting — Diagnostic codes](TROUBLESHOOTING.md#diagnostic-codes).

### 5.7 Upload inspection (CSV file engine)

Called from `inspect_tabular_upload` and `validate_upload_sources` during CSV/Excel engine construction.

- **Identifier naming:** Propose a SQL identifier for one messy upload label when deterministic normalization is insufficient.
  - **Deterministic context assembly:** Single column or table label text and the identifier JSON schema.
  - **Provider-mediated:** Proposed identifier string; **provider output is not deterministic**.
  - **Calls:** `validate_upload_sources` / `inspect_tabular_upload` when a label fails deterministic identifier normalization.
  - **Content:** Label text only; no row data.

- **Upload summary (when `TABULAR_LLM_ASSIST` enabled):** Compress structured inspection findings into one operator-facing explanation.
  - **Content:** Issue codes, locations, counts, and any `suggested_selections` payload.
  - **Data sent:** Structural findings only; no row values.

- **Upload interpretation (ambiguous layouts only, when `TABULAR_LLM_ASSIST` enabled):** Propose header row, table ranges, append regions, or orientation when deterministic scoring is inconclusive and the caller supplied no `source_selections`.
  - **Content:** A bounded sample of cell text (up to 25 rows × 40 columns from the upload grid) plus per-row/column structural statistics.
  - **Data sent:** A limited slice of **actual cell values** from the uploaded file (not full-database row data). Verified against the grid before use. This is the questioning-surface path that may ship real cell contents.

- **Upload column transforms (when `TABULAR_LLM_ASSIST` enabled):** Propose closed-vocabulary column transforms (`parse_temporal`, `strip_numeric_affix`, `band_bounds`, `band_value_map`, `keep_canonical_columns`, `derive_by_pattern`, `drop_empty_columns`, `null_tokens`, `unpivot_columns`) for one relation at a time.
  - **Deterministic context assembly:** Up to five data rows shuffled with a stable seed derived from upload content hash; header or export labels; for columns with at most 25 distinct non-empty values, the full distinct-value set for that column.
  - **Provider-mediated:** Transform proposals (`transform_id`, target column label, params, `requires_review`); **provider output is not deterministic**.
  - **Calls:** `prepare_relations_for_paths` and `validate_upload_sources` when no accepted `column_transforms` are supplied in `source_selections`.
  - **Apply path:** Uses accepted or auto-applied proposals plus **full-column deterministic verification** only; bounded samples are not re-sent at apply time. Failed verification emits `UPLOAD_TRANSFORM_REJECTED` and leaves data unchanged. Shape-changing transforms require review via `suggested_selections` / accepted `column_transforms`.

**Policy flag — `TABULAR_LLM_ASSIST`:** Upload interpretation, upload summary, and upload column-transform proposals are the questioning-surface call sites that may send **sampled cell content** to the configured provider. `PolicyConfig.TABULAR_LLM_ASSIST` defaults to enabled. Set environment variable `AETHERDIALECT_TABULAR_LLM_ASSIST=false` (or assign `PolicyConfig.TABULAR_LLM_ASSIST = False` before calling `inspect_tabular_upload`) to turn off model-assisted upload interpretation, the provider-written upload summary, and column-transform proposal shipping. With the flag disabled, inspection context assembly for layout scoring remains **deterministic**: structural scoring resolves header rows and table regions without shipping cell text; issue severities (**Advisory**, **Review**, **Blocking**, **Fatal**), `suggested_selections` from deterministic rules, and auto-correct reshaping still run unchanged. Native Excel temporal typing, deterministic scalar-affix heuristics, and empty-column detection still run at ingest without model assistance. Identifier naming for messy labels (label text only, no row data) is a separate call and is unaffected by this flag.

### 5.8 Provider call site inventory

Every questioning-surface path that calls the configured provider appears below. Corpus generators, qsim fill, and other offline synth tools use separate `synth` / `synth_variety` tasks and are not part of the analyst questioning surface.

| Call site | Function / trigger | Task id | Data leaving the process |
| --- | --- | --- | --- |
| Question gate | `validate_question` | `default` | Question text only |
| Question canonicalize | `normalize_question_via_llm` | `default` | Question text only |
| Interpret | intent interpret stage | `intent` | Schema metadata, enum heads, feedback prose |
| Ground | intent ground stage | `intent` | Schema metadata, roles, enum heads |
| Compose | intent compose stage | `intent` | Structural schema metadata |
| Intent repair | compose / schema repair loops | `intent`, `intent_format`, `intent_schema_repair` | Error rows + structural metadata |
| Join selection | join disambiguation when multiple candidates | `join` | Question, SQL preview, path signatures |
| Template reuse params | fuzzy template parameter extraction | `default` | Question, template slot metadata |
| Display aliases | `enriched_display_alias_map` | `default` | Question, SQL expression signatures |
| Template display names | template param display labels | `default` | Question, slot metadata |
| Question feedback | user rejection / validation-failure summaries | `feedback` | Intent structure JSON, feedback text |
| Upload identifier naming | `inspect_tabular_upload` / `validate_upload_sources` | `default` | Single label text |
| Upload summary | `inspect_tabular_upload` when `TABULAR_LLM_ASSIST` | `upload_summary` | Structural findings only |
| Upload interpretation | `inspect_tabular_upload` when `TABULAR_LLM_ASSIST` | `upload_interpret` | Bounded cell sample + statistics |
| Upload column transforms | `prepare_relations_for_paths` / `validate_upload_sources` when `TABULAR_LLM_ASSIST` | `upload_column_transforms` | Up to five shuffled data rows, header labels, optional distinct sets (≤25 values) |
| Role / sensitivity classification | `apply_column_roles_llm` at schema build | `schema_base`, `schema` | Per-column profiling summaries |
| Schema notes refinement | schema catalog notes pass | `schema_base`, `schema` | Table/column description drafts |
| Description refinement | `refine_descriptions` on `apply_structure` | `default` | Operator-edited description text |
| DDL assistance | schema catalog DDL helper | `ddl` | Structural DDL context |
| Seed question clarify | seed warmup normalization (offline) | `default` | Seed question text |

Assembly of every row's **Content** column is **deterministic** for fixed inputs. **Provider output is not deterministic** for any row.

### 5.9 Role and sensitivity classification (construction-time)

During schema build, `apply_column_roles_llm` sends per-column profiling summaries (name, type, PK/FK flags, distinct/null ratios) and optional domain notes to the configured provider - not arbitrary row dumps. Assembly of profiling summaries is **deterministic**; assigned **table and column roles** and **sensitivity tiers** are **provider-mediated** and **provider output is not deterministic**. Assignments persist in the schema graph and structure layer. Domain notes may tighten sensitivity to `restricted` or `hidden` only when they explicitly mark sensitive data.

## 6. Sensitivity tags

Tier definitions (`none`, `restricted`, `hidden`): [User guide — Sensitivity classification](USER_GUIDE.md#sensitivity-classification). This section covers enforcement consequences only.

- **Restricted and hidden columns** — bare row projection is not selectable; literal-value filters in `WHERE` / `HAVING` are rejected (null-checks alone are exempt); `GROUP BY` and `ORDER BY` on the column are rejected. The repair pass may drop offending `SELECT` entries (`SENSITIVITY_GATE_HIT`).
- **Hidden columns** — omitted from all LLM-facing schema literals; remain in the compiled graph unless removed by `deny_columns`.
- **Cross-source join keys and semi-join reduction keys** — must be `sensitivity == none`; declaration-time validation rejects non-`none` declared keys; plan-time eligibility omits them; execution raises `FederationRuntimeError` when a reducing key is not allowed.

Assign tiers through `export_structure` / `apply_structure` or domain notes. Demo hidden columns: [Sandbox guide — Column security](SANDBOX.md#column-security).

**Cross-source semi-join keys at execution:** Reducing and semi-join keys must be `sensitivity == none`. When a reducing or semi-join edge references a restricted or hidden key, federation execution **raises** `FederationRuntimeError` (rejected - not silently skipped). Separately, plan-time eligibility omits non-`none` joins from the eligible cross-source set, and declaration-time validation rejects non-`none` declared keys.

## 7. Deny lists

`EngineContext.deny_columns` and `allow_columns` accept qualified `table.column` tokens (and `*.column` where supported). Denied columns are **removed from the graph** before profiling and classification - they do not exist while the deny is active. This is a stronger removal than **hidden** sensitivity, which omits a column from prompts but keeps it in the graph.

`FederationContext` deny lists apply the same rules on the composite namespace, including `source.table.column` when names collide across members.

`SpaceContext.deny_objects` and `deny_columns` apply the same token shapes at AetherSpace knowledge scope only ([User guide - SpaceContext](USER_GUIDE.md#spacecontext)).

## 8. Operational logging and observability

This section covers debug log output, the optional audit sink, and turn-level diagnostics. It does not describe federation coordinator spill files - see [Section 9](#9-federation-coordinator-spill).

### Debug bind-parameter logs

When debug logging is enabled, the engine may log resolved bind maps during SQL rendering and execution. Those logs can contain literal filter values extracted from user questions. Disable debug logging in production deployments that must not persist user-supplied literals on disk.

### Audit sink callback

When the operator supplies an `audit_sink` callback on `AetherEngine` or `AetherFederation`, the library emits structured `AuditEvent` records at lifecycle boundaries. Each event carries `event_type`, `timestamp_iso`, the normalised question (when applicable), `schema_hash`, `provider`, and a `details` tuple of key/value pairs.

These events are **not** LLM disclosures and are **not** written unless a sink is configured. Typical `event_type` values include `init`, `ask_begin`, `ask_done`, `ask_error`, `ask_blocked`, structure and cache maintenance operations, and federation-specific events that name the member sources touched on a federated turn. Integrators use the sink for operator audit trails, SIEM forwarding, or compliance logging on the application side.

The sink receives metadata about what ran (question text, schema identity, diagnostic codes in `details`) - not warehouse row payloads.

### Session diagnostics

Every `session.ask` / `session.step` returns `SessionStep.diagnostics`: turn-level tracing rows (`REUSE_HIT`, `COMPOSE_REPAIR`, `FEDERATION_*`, `LLM_TURN_COST`, and similar). These are returned to the embedding application, not written to disk unless you forward them. Full code catalog: [Troubleshooting — Diagnostic codes](TROUBLESHOOTING.md#diagnostic-codes).

## 9. Federation coordinator spill

When member result frames exceed in-memory coordinator budgets (`spill_row_threshold` and related caps in the federation manifest), the in-process DuckDB coordinator writes temporary **parquet** files instead of holding entire frames in RAM.

**Location:** `<artifacts_root>/aetherdialect/fed_<federation_id>/coordinator_spill/` (directory mode `0700`). Spill lives under the federation artifact tree inside the operator-controlled artifacts root - not in the system temp directory.

**Contents:** Member row values for join keys and projected columns the coordinator must retain to finish cross-source joins and residual combine work. Spill files never contain prompt text, model payloads, or audit events.

**Lifecycle:** Spill is triggered only during federated execution - not during schema build or single-engine turns. The directory is removed after each federated execution completes (success or terminal failure). A failed cleanup raises a typed runtime error rather than leaving files silently.

**Sensitivity:** Treat the federation tree - including any spill directory present mid-turn - as sensitive when member data is restricted. Restrict filesystem access to the artifacts root accordingly. Spill is separate from operational logging and observability ([Section 8](#8-operational-logging-and-observability)).

---

**See also:** [Getting started](GETTING_STARTED.md) | [User guide](USER_GUIDE.md) | [Integrator guide](INTEGRATOR_GUIDE.md) | [Sandbox guide](SANDBOX.md) | [API reference](API_REFERENCE.md) | [Troubleshooting](TROUBLESHOOTING.md) | [How it works](HOW_IT_WORKS.md) | [Support matrix](SUPPORT_MATRIX.md) | [README](../README.md)
