# User guide

Operator and analyst manual for day-to-day use of `aetherdialect`: the three public facades, the three scope types, notes, overrides, asking questions, migration, warmup, resets, uploads, and quality troubleshooting. First-time install and wiring live in [Getting started](GETTING_STARTED.md). Programmatic embedding contracts live in the [Integrator guide](INTEGRATOR_GUIDE.md). Exact signatures and TOML keys live in the [API reference](API_REFERENCE.md).

**Reading order:** [README](../README.md) -> [Getting started](GETTING_STARTED.md) -> this document -> [Integrator guide](INTEGRATOR_GUIDE.md) -> [Sandbox guide](SANDBOX.md) -> [API reference](API_REFERENCE.md) -> [How it works](HOW_IT_WORKS.md) -> [Security](SECURITY.md) -> [Support matrix](SUPPORT_MATRIX.md).

## Sections

| Section | Contents |
| --- | --- |
| [Configure](#configure) | Credentials skim; scope is not a permission system |
| [First run](#first-run) | Initial profiling wait |
| [AetherEngine](#aetherengine) | Single-database facade |
| [EngineContext](#enginecontext) | Per-connection scope object |
| [AetherFederation](#aetherfederation) | Multi-database composite facade |
| [FederationContext](#federationcontext) | Composite scope object |
| [AetherSpace](#aetherspace) | Named knowledge partition |
| [SpaceContext](#spacecontext) | Knowledge allow/deny shape |
| [SessionStep](#sessionstep) | One observable point per ask/step |
| [Sensitivity classification](#sensitivity-classification) | Skim - full detail in Security |
| [Asking a question](#asking-a-question) | Confirmations, reuse, feedback |
| [Notes file](#notes-file) | Domain vocabulary |
| [Schema overrides](#schema-overrides) | Descriptions, roles, sensitivity, usability |
| [Migration](#migration) | Catalog drift workflow |
| [Seed warmup](#seed-warmup) | Pre-populating templates |
| [Partition and cluster pruning](#partition-and-cluster-pruning) | Warehouse pruning hints |
| [Resetting learning](#resetting-learning) | Scoped clears |
| [CSV and Excel uploads](#csv-and-excel-uploads) | Validation, naming, and multi-sheet tables |
| [Common pitfalls](#common-pitfalls) | Quality troubleshooting |

---

## Configure

Point your application at LLM and database credentials via TOML or process environment. Engine selection and connection fields vary by engine. Azure OpenAI uses two deployment slots (`light` / `heavy`); mapping detail: [Integrator guide - LLM provider wiring](INTEGRATOR_GUIDE.md#llm-provider-wiring). Full key lists: [API reference - Configuration](API_REFERENCE.md#configuration). First-time wiring: [Getting started - Connect your warehouse](GETTING_STARTED.md#connect-your-warehouse).

Scope types are not a permission system. The database role's grants are the real boundary. `EngineContext` and `FederationContext` only narrow further on top of those grants; `SpaceContext` affects knowledge and template partitioning only and never execution. Full treatment: [Security - Execution boundary](SECURITY.md#2-execution-boundary-and-credentials).

## First run

The first time the engine connects to a database, it reflects the catalog, profiles every column in scope, classifies roles on usable columns, and writes a versioned snapshot under the engine storage directory. Later constructions reuse this cache when fingerprints match.

| Schema size | Typical first construction |
| --- | --- |
| Small (~15 tables) | Tens of seconds to a few minutes |
| Moderate (~50 tables) | Several minutes |
| Large (hundreds of tables) | Much longer - narrow with `allow_objects` |

Progress prints as `Profiling [i/n] <table>` on stdout during construction. Profiling and role classification detail: [Security - Schema profiling and roles](SECURITY.md#3-schema-profiling-roles-and-classification).

## AetherEngine

**AetherEngine** is the public facade for one database connection. Construction reflects the catalog (or loads cache), builds or loads the master schema graph, and stamps artifact fingerprints. One instance binds exactly one database; artifacts live under `<artifacts_dir>/aetherdialect/<connection_slug>/`.

- **Sessions** - `engine.session(...)` returns a `PipelineSession` (or `engine.asession()` for async). Each turn yields a `SessionStep`: one observable point with `kind`, `done`, `prompt`, and optional `sql` / `data` / `error`. Suspend and terminal `kind` values: [Integrator guide - The session contract](INTEGRATOR_GUIDE.md#the-session-contract-suspend-and-terminal-steps).
- **Roles** - `role="owner"` (default) builds and mutates artifacts; `role="consumer"` pins to the owner's published snapshot.
- **Named engine contexts** - pass a saved scope-preset name string instead of `EngineContext(...)` to load a stored subset spec ([EngineContext](#enginecontext)).
- **Named spaces** - pass `space="name"` at session open to narrow knowledge ([AetherSpace](#aetherspace)).

## EngineContext

**EngineContext** is the frozen scope object passed into `AetherEngine`. It controls which relations enter the graph, supplies domain notes and optional DDL, and enforces allow/deny lists at **execution time** (table and column visibility for the connection). It has no `name` field - the name identifies the engine.

- **include** - `"tables"` (default) or `"views"`. Only the master context (the `EngineContext` passed at owner construction) may set include mode. `"both"` is rejected - reflect tables and views in separate passes when both are needed.
- **allow_objects** / **deny_objects** - optional table or view names to include or exclude from the graph.
- **allow_columns** / **deny_columns** - qualified `table.column` or `*.column` entries. Denied columns are removed from the in-memory graph entirely (stronger than sensitivity tiers).
- **notes_file** / **notes** - path to domain notes or inline notes text on the master context only (set at most one; [Notes file](#notes-file)). Saved scope presets cannot set notes.
- **sql_file** - optional DDL or annotated SQL on the master context only. Saved scope presets cannot set `sql_file`.

The master context is implicit: pass an `EngineContext` at owner construction (or reload a cached master). Register saved scope presets with `engine.engine_context(name, context)` on a master-bound owner; consumers bind by that preset name string only. The master name cannot be created or overwritten as a named sidecar.

Each database connection gets its own engine storage directory keyed by a **connection slug** derived from database identity (host, database name, and similar) - not from user or role names. That keeps shared learning and consumer roles meaningful across logins to the same warehouse.

## AetherFederation

**AetherFederation** composes several `AetherEngine` instances into one unified schema graph and session surface. Register members by connection name, declare cross-source joins in the **federation declaration** (`federation_declaration.json` passed as `declaration_file=` at construction), and optionally map logical tables (`union` stacks members; `replica` reads one authority member).

- **Artifact trees** - Each member keeps `conn_<slug>/`; the composite persists under `fed_<federation_id>/` beside the member trees.
- **Sessions** - Open `fed.session(...)` and ask normally. The turn uses the same suspend kinds and confirmations as a single engine (including terminal `kind="meta"`). `SessionStep.federated_bundle` holds the executed plan; **`step.sql`** is a member `source_id` → dialect SQL mapping for federated analytical turns.
- **Reuse** - Federated questions replay stored plan templates through `prepare_federated_sql_plan`; member templates are stamped `federation_plan_only` and are not reused as standalone SQL. Prefer `fed.execute_template(step.template_id, params)` for agent re-runs after accept.
- **Ineligible plans** - Cross-source joins are allowed only when declared. Unsupported shapes surface a diagnostic and rephrase hint instead of a wrong answer. When one member fails during execution, the turn ends with `federation_partial_failure` rather than a partial result.
- **File engines** - A database may be federated with uploaded files, but a federation whose members are all file (`csv`) engines is refused at registration; load several uploads into one CSV engine instead.
- **Clock consistency** - On a federated turn, when the parent intent uses relative date-window filters or clock keywords (`current_date`, `current_timestamp`, and similar), the coordinator resolves one UTC anchor at turn start (`AnchoredTemporalBind`) so every member statement uses the same clock instant instead of re-evaluating per-member clock functions.
- **Re-entry** - Federation artifacts are library-owned ([Integrator guide - Artifacts are library-owned](INTEGRATOR_GUIDE.md#artifacts-are-library-owned)). Use `export_federation_declaration()` / `apply_federation_declaration()` for authored joins and mappings. Persisted sidecar shapes: [API reference - Federation and migration JSON](API_REFERENCE.md#federation-and-migration-json). Member catalog drift uses each source's `schema_migration_map.json`; cross-source identifier changes use `federation_migration_map.json`.
- **Warmup** - `run_seed_warmup`, `run_seed_warmup_from_history`, and `run_seed_warmup_from_query_log` raise `ConfigError` (`warmup is not supported on AetherFederation`). Run those on each member `AetherEngine` instead.
- **Exports** - `export_knowledge()` wraps engine/federation plus per-space business knowledge; `export_space_knowledge(space=...)` is per-space BK only; `export_metadata(space=...)` is deterministic table/column inventory (plus federation members when present).
Operator embedding detail: [Integrator guide - Embedding a federation](INTEGRATOR_GUIDE.md#embedding-a-federation).

## FederationContext

**FederationContext** scopes the composite graph the same way `EngineContext` scopes one connection. It is optional on `AetherFederation(..., context=...)` and applies at composition and execution time over the unified member graph. It has no `name` field - the name identifies the federation.

- **include** - `"tables"` (default) or `"views"`. `"both"` is rejected.
- **allow_objects** / **deny_objects** - table or view names in the composite namespace.
- **allow_columns** / **deny_columns** - qualified `table.column` or `*.column` entries (three-part `source.table.column` inputs are accepted by the same normalizer). Deny-list semantics: [Security - Deny lists](SECURITY.md#7-deny-lists).
- **notes_file** / **notes** - optional domain notes for the composite (set at most one; [Notes file](#notes-file)). Member engines may still carry their own `EngineContext` notes for per-source vocabulary.

`FederationContext` does not replace database grants or member-level `EngineContext` scope. It narrows what the composite graph exposes and what federated execution may target on top of each member's credentials.

## AetherSpace

An **AetherSpace** is a named knowledge partition over the master graph on an engine or federation. It improves model focus and partitions template learning by space name. It is **not** a permission boundary - database grants and engine/federation context scope remain the execution boundary ([Security - Execution boundary](SECURITY.md#2-execution-boundary-and-credentials)).

- **Master space** - implicit default (`space="master"`). Reflects the full master engine or federation context graph. Cannot be deleted or redefined.
- **Named spaces** - defined on the owner with `engine.aetherspace(name, space_context=...)` (or the federation equivalent). The space **name** identifies the AetherSpace; put notes on `SpaceContext(notes=...)` or `SpaceContext(notes_file=...)` ([SpaceContext](#spacecontext)). Snapshots persist under the engine tree or `fed_<federation_id>/aetherspaces/` on a federation.
- **Per-space notes** - optional `SpaceContext.notes` or `SpaceContext.notes_file` at define time; content is baked into the aetherspace snapshot (not a catalog-rebuild fingerprint). The engine merges inherited master descriptions with space-specific refinements ([Sandbox guide - Named AetherSpaces](SANDBOX.md#named-aetherspaces)).

## SpaceContext

**SpaceContext** supplies the allow/deny shape for a named AetherSpace: `tables`, `columns`, `deny_objects`, `deny_columns`, and optional `notes` or `notes_file` (set at most one). It uses the same token shapes as `EngineContext` column specs (`table.column` or `*.column`) but applies at **question time** for knowledge and template partitioning only - not at SQL execution. It has no `name` field - the name identifies the AetherSpace. Parallel to `EngineContext` / `FederationContext` for notes: space notes land in the snapshot, not a catalog fingerprint.

Named spaces are always created against the master engine or federation context, never as an intersection with a non-master engine context. On a federation, space snapshots live under the federation artifact tree.

## SessionStep

Every `session.ask(...)` or `session.step(...)` returns a **`SessionStep`** - one observable point in a turn. Branch embedded UIs on **`step.kind`** and **`step.done`**, not on parsing `step.prompt` text.

| State | `step.done` | What you see |
| --- | --- | --- |
| Suspended | `False` | `prompt`, `reply_shape` (`yes_no` or `free_text`); collect input and call `step(reply)`. |
| Terminal success | `True` | `sql`, `data`, optional `message`. For analytical success, `step.template_id` identifies the stored template when one was matched or accepted. |
| Metadata answer | `True` | `kind="meta"`; `sql` is `None`; read `message` and optional `meta_payload` (schema inventory or business-knowledge prose). No confirm loop. |
| Terminal failure | `True` | `error`, optional `status` and federation attribution fields. |

On a single engine, **`step.sql`** is a dialect SQL string. On federated turns, **`step.sql`** is a `dict` mapping member `source_id` → that member's dialect SQL (also inspect **`step.federated_bundle`** for the executed plan). Turn-level tracing rows (reuse hits, repairs, `LLM_TURN_COST`, federation codes, `meta.*` route codes) are on **`step.diagnostics`**.

Full field list and suspend `kind` values: [Integrator guide - The session contract](INTEGRATOR_GUIDE.md#the-session-contract-suspend-and-terminal-steps) | [API reference - SessionStep](API_REFERENCE.md#sessionstep).

## Sensitivity classification

Each column carries a sensitivity tier (`none`, `restricted`, or `hidden`). Full definitions, query-time rules, and LLM disclosure: [Security - Sensitivity tags](SECURITY.md#6-sensitivity-tags).

## Asking a question

When the engine is not confident about a translation, it suspends the turn and asks for confirmation - intent readback, SQL preview, or execute confirmation depending on confidence and reuse path. Accept builds template learning; reject with a reason steers future turns away from the same mistake.

Some questions never enter the SQL path: schema inventory/count questions and business-knowledge definitions return a terminal **`kind="meta"`** step instead of Intent/SQL confirmations.

**Template reuse** is automatic - there is no separate "warm reuse" API:

1. **Direct reuse** - when new wording is within a small normalized token edit distance (at most two, `FUZZY_MATCH_MAX_DISTANCE`) of a stored question, the engine replays the stored SQL path with no full parse LLM calls.
2. **Intent-level match** - larger wording changes still require LLM calls even when structure is similar.

For programmatic re-execution with new bind values on a known stored question, use `session.reuse_saved_question(question_old, question_new, new_values)` instead of a full ask loop. After an accepted analytical turn, agents may also re-run with `engine.execute_template(step.template_id, {b.handle: b.current_value for b in step.parameters})` (same surface on `AetherFederation`).

Ask normally; change wording and observe which path runs via diagnostics or audit events ([Integrator guide - Observability](INTEGRATOR_GUIDE.md#observability)).

Schema descriptions and business knowledge for agent context (without asking) come from `export_knowledge()`, `export_space_knowledge(space=...)`, and `export_metadata(space=...)` on the engine or federation.

The offline sandbox supports the same accept/reject mechanism as production; bundled mock coverage is finite ([Sandbox guide](SANDBOX.md)). Federated date-window anchoring lives under [AetherFederation](#aetherfederation).

## Notes file

Domain notes refine descriptions and guide role classification. Use one or two sentences per important table, business definitions, join hints, and explicit sensitivity statements. Supply notes as a path (`notes_file`) or inline text (`notes`) on the master context — set at most one. The master context reads `EngineContext` notes; a federation may also supply `FederationContext` notes. Each named AetherSpace may supply notes via `SpaceContext(notes=...)` or `SpaceContext(notes_file=...)` at define time.

## Schema overrides

Overrides live in one JSON file beside your working directory. Export a starter with the public API, edit descriptions, roles, sensitivity, foreign keys, or primary keys, then apply. Every override replays on cache invalidation.

- **usable** - statistical gates mark low-signal columns unusable (for example extreme null rate, single-value, or sentinel-dominated distributions). Overrides accept `"usable": true` to force a column usable; `"usable": false` is rejected - use sensitivity tiers or deny lists to hide a column instead.
- **sensitivity** - assign `none`, `restricted`, or `hidden` per column ([Security](SECURITY.md#6-sensitivity-tags)).

Workflow detail: [API reference - Schema overrides JSON](API_REFERENCE.md#schema-overrides-json-schema_overridesjson).

## Migration

When the catalog changes structurally, the engine writes `schema_migration_map.json` and stops. Edit the map with one of:

- **remap** - rewrite identifiers in stored templates.
- **destructive** - clear templates and learning for a fresh slate.
- **abort** - stop initialization to investigate drift.

Federation variation: member-source drift still uses each source's own `schema_migration_map.json`. Cross-source identifier changes that span the composite graph use `federation_migration_map.json` beside the federation tree with the same remap/destructive/abort actions. See [AetherFederation](#aetherfederation).

## Seed warmup

Warmup paths populate templates so common questions hit the cache sooner:

1. **Seed-question warmup** - parses and paraphrases a list of natural-language questions.
2. **SQL-history warmup** - reverse-engineers historical `SELECT` statements into intents.
3. **Query-log warmup** - reads warehouse system logs for historical queries.

Federation variation: `run_seed_warmup`, `run_seed_warmup_from_history`, and `run_seed_warmup_from_query_log` raise `ConfigError` on `AetherFederation` - run them on each member engine. See [AetherFederation](#aetherfederation).

Warmup and QSim are unavailable inside the offline sandbox (`ConfigError`).

## Partition and cluster pruning

When the compiled schema graph carries partition or clustering metadata and the parsed intent includes matching date or key filters, the engine may append `WHERE` predicates on those columns at SQL finalization (`inject_pruning_predicates`). This reinforces warehouse pruning without inventing data ranges the user did not ask for. Engine-specific behavior: [Support matrix - Engine capabilities](SUPPORT_MATRIX.md#engine-capabilities).

## Resetting learning

Scoped resets clear persisted overrides, the template store, or simulation caches through public API methods on `AetherEngine` (`clear_persisted_overrides`, `clear_template_store`, `clear_simulation_caches`, `clear_all_learning`). See [API reference - AetherEngine](API_REFERENCE.md#aetherengine).

## CSV and Excel uploads

The `csv` engine uses the in-memory DuckDB backend and `CSV_*` environment keys. It accepts `.csv` and `.xlsx` uploads (not `.xls`). There is no separate `excel` engine - Excel workbooks go through the same `csv` engine.

Every upload is classified by severity before load:

| Severity | Meaning | What you see |
| --- | --- | --- |
| **Advisory** | One correct interpretation exists and was applied | Inspection succeeds; changes are listed; you may proceed |
| **Review** | More than one defensible interpretation exists | Inspection flags choices; you must confirm before construction |
| **Blocking** | File is readable but no coherent table can be derived | Inspection returns `ok=False`; construction raises `ConfigError` with the report |
| **Fatal** | File cannot be read, or the format is unsupported | Inspection raises `ConfigError` |

**Inspect first, then construct:**

1. Call `inspect_tabular_upload(path)` on the raw file (nothing is written to DuckDB yet).
2. When `report.requires_review` is true, read `report.narrative`, `report.issues`, and `report.suggested_selections`, then pass your accepted interpretation as `source_selections` on `AetherEngine(...)`.
3. Construction applies your selections without re-deciding layout. When **Review** issues remain and no selections were supplied, construction raises `ConfigError` with the report attached.

During inspection the engine auto-reads encoding, removes layout scaffolding, normalizes values, and records **Advisory** fixes (ragged rows padded, duplicate headers suffixed, and similar). **Review** cases - multiple tables on one sheet, uncertain header rows, appendable regions - require your choice via `source_selections` (`header_row`, `table_range`, `append_regions`, and similar). Only **Blocking** and **Fatal** cases prevent a usable table.

Each worksheet loads as its own table; workbooks with multiple data sheets use `filename__sheetname` table names. Blank-separated blocks with identical headers load as separate tables by default; the report may suggest `append_regions` when combining them is appropriate. Upload column labels are preserved as `original_name` on the schema graph while SQL identifiers use normalized `name` values.

Spreadsheet cell sampling for model-assisted interpretation is optional and controlled by `PolicyConfig.TABULAR_LLM_ASSIST` ([Security - Upload inspection](SECURITY.md#58-upload-inspection-csv-file-engine)).

## Common pitfalls

- **Wrong table chosen:** ensure your notes file clearly distinguishes similar business entities.
- **Related tables not joined:** confirm foreign keys exist or add them through overrides.
- **Column never appears:** check statistical usability, deny lists, and sensitivity tags.
- **AetherSpace confused with permissions:** narrowing a space changes model focus only; tighten `EngineContext`, `FederationContext`, or database grants for access control.
- **CSV rejected at construction:** open the `ConfigError` message - it lists exact `file!sheet!cell` locations. Fix the spreadsheet and retry; do not rely on pandas alone to validate structure.

---

**See also:** [Getting started](GETTING_STARTED.md) | [Integrator guide](INTEGRATOR_GUIDE.md) | [Sandbox guide](SANDBOX.md) | [API reference](API_REFERENCE.md) | [How it works](HOW_IT_WORKS.md) | [Security](SECURITY.md) | [Support matrix](SUPPORT_MATRIX.md) | [README](../README.md)
