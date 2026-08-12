# User guide

Operator and analyst manual for day-to-day use of `aetherdialect`: the three public facades, the three scope types, notes, structure documents, asking questions, migration, warmup, resets, uploads, and quality troubleshooting. First-time install and wiring live in [Getting started](GETTING_STARTED.md). Programmatic embedding contracts live in the [Integrator guide](INTEGRATOR_GUIDE.md). Exact signatures and TOML keys live in the [API reference](API_REFERENCE.md).

**Reading order:** [README](../README.md) → [Getting started](GETTING_STARTED.md) → this document → [Integrator guide](INTEGRATOR_GUIDE.md) → [Sandbox guide](SANDBOX.md) → [API reference](API_REFERENCE.md) → [Troubleshooting](TROUBLESHOOTING.md) → [How it works](HOW_IT_WORKS.md) → [Security](SECURITY.md) → [Support matrix](SUPPORT_MATRIX.md).

## Sections

| Section | Contents |
| --- | --- |
| [Configure](#configure) | Credentials skim; scope is not a permission system |
| [First run](#first-run) | Initial profiling wait |
| [AetherEngine](#aetherengine) | Single-database facade |
| [EngineContext](#enginecontext) | Per-connection scope object |
| [Sensitivity classification](#sensitivity-classification) | Column tiers and query behaviour |
| [AetherFederation](#aetherfederation) | Multi-database composite facade |
| [FederationContext](#federationcontext) | Composite scope object |
| [AetherSpace](#aetherspace) | Named knowledge partition |
| [SpaceContext](#spacecontext) | Knowledge allow/deny shape |
| [Knowledge layers](#knowledge-layers) | Structure document vs knowledge document |
| [Notes file](#notes-file) | Domain vocabulary |
| [Structure document](#structure-document) | Roles, sensitivity, usability, keys |
| [Asking a question](#asking-a-question) | Confirmations, reuse, feedback |
| [SessionStep](#sessionstep) | One observable point per ask/step |
| [Migration](#migration) | Catalog drift workflow |
| [Seed warmup](#seed-warmup) | Pre-populating templates |
| [Partition and cluster pruning](#partition-and-cluster-pruning) | Warehouse pruning hints |
| [Resetting learning](#resetting-learning) | Scoped clears |
| [CSV and Excel uploads](#csv-and-excel-uploads) | Validation, naming, and multi-sheet tables |
| [Common pitfalls](#common-pitfalls) | Quality troubleshooting |

---

## Configure

Point your application at LLM and database credentials via TOML or process environment. Engine selection and connection fields vary by engine. Azure OpenAI uses two deployment slots (`light` / `heavy`); mapping detail: [Integrator guide — LLM provider wiring](INTEGRATOR_GUIDE.md#llm-provider-wiring). Full key lists: [API reference — Configuration](API_REFERENCE.md#configuration). First-time wiring: [Getting started — Connect your warehouse](GETTING_STARTED.md#connect-your-warehouse).

Scope types are not a permission system. The database role's grants are the real boundary. `EngineContext` and `FederationContext` narrow further on top of those grants. A space narrows which objects a turn may reference and refuses questions that reach past it, and it is not a permission boundary because it can neither be defined nor entered beyond what credentials already permit. Full treatment: [Security — Execution boundary](SECURITY.md#2-execution-boundary-and-credentials).

## First run

The first time the engine connects to a database, it reflects the catalog, profiles every column in scope, classifies roles on usable columns, and writes a versioned snapshot under the engine storage directory. Later constructions reuse this cache when fingerprints match.

| Schema size | Typical first construction |
| --- | --- |
| Small (~15 tables) | Tens of seconds to a few minutes |
| Moderate (~50 tables) | Several minutes |
| Large (hundreds of tables) | Much longer — narrow with `allow_objects` |

Progress prints as `Profiling [i/n] <table>` on stdout during construction. Profiling and role classification detail: [Security — Schema profiling and roles](SECURITY.md#3-schema-profiling-roles-and-classification).

## AetherEngine

**AetherEngine** is the public facade for one database connection. Construction reflects the catalog (or loads cache), builds or loads the default schema graph, and stamps artifact fingerprints. One instance binds exactly one database; artifacts live under `<artifacts_dir>/aetherdialect/<connection_slug>/`.

- **Sessions** — `engine.session(...)` returns a `PipelineSession` (or `engine.asession()` for async). Each turn yields a `SessionStep`: one observable point with `kind`, `done`, `prompt`, and optional `sql` / `data` / `error`. Suspend and terminal contract: [Integrator guide — The session contract](INTEGRATOR_GUIDE.md#the-session-contract-suspend-and-terminal-steps).
- **Roles** — `role="owner"` (default) builds and mutates artifacts; `role="consumer"` pins to the owner's published snapshot.
- **Named engine contexts** — pass a saved scope-preset name string instead of `EngineContext(...)` to load a stored subset spec ([EngineContext](#enginecontext)).
- **Named spaces** — pass `space=<uid>` at session open to narrow knowledge ([AetherSpace](#aetherspace)). Omit `space` for the default space (`default_space_uid`).

## EngineContext

**EngineContext** is the frozen scope object passed into `AetherEngine`. It controls which relations enter the graph, supplies domain notes and optional DDL, and enforces allow/deny lists at **execution time** (table and column visibility for the connection). It has no `name` field — the name identifies the engine.

- **include** — `"tables"` (default) or `"views"`. Honoured only when both `allow_objects` and `deny_objects` are empty. `"both"` is rejected.
- **allow_objects** — when non-empty, resolves each name against the catalog without filtering by `include` (tables and views may mix).
- **deny_objects** — when non-empty and `allow_objects` is empty, reflects both tables and views, then removes each denied name (unknown deny names raise `SchemaAccessError`).
- **allow_columns** / **deny_columns** — qualified `table.column` or `*.column` entries. Denied columns are removed from the in-memory graph entirely (stronger than sensitivity tiers).
- **notes_file** / **notes** — path to domain notes or inline notes text on the default context only (set at most one; [Notes file](#notes-file)). Saved scope presets cannot set notes.
- **sql_file** — optional DDL or annotated SQL on the default context only. Saved scope presets cannot set `sql_file`.

The default context is implicit: pass an `EngineContext` at owner construction (or reload a cached default). Register saved scope presets with `engine.engine_context(name, context)` on an owner-bound instance; consumers bind by that preset name string only. The default context name cannot be created or overwritten as a named sidecar.

Each database connection gets its own engine storage directory keyed by a **connection slug** derived from database location keys ([API reference — Artifact storage slug](API_REFERENCE.md#artifact-storage-slug)) — credentials never appear in the slug. That keeps shared learning and consumer roles meaningful across logins to the same warehouse.

## Sensitivity classification

Each column carries a sensitivity tier. Enforcement consequences (prompt disclosure, query-time rules, cross-source key requirements): [Security — Sensitivity tags](SECURITY.md#6-sensitivity-tags).

| Tier | Model visibility | Query behaviour |
| --- | --- | --- |
| **none** | Full visibility in schema payloads when usable and not denied. | Normal `SELECT`, filter, group, and order behaviour. Required for cross-source join keys and semi-join reduction keys. |
| **restricted** | Column name, type, role, and description appear in LLM-facing schema literals when usable. | Bare row projection is not selectable; literal-value filters on the column in `WHERE` / `HAVING` are rejected (null-checks alone are exempt). `GROUP BY` and `ORDER BY` on the column are rejected. Aggregations that wrap the column may still be emitted when role policy allows. |
| **hidden** | Column is omitted from all LLM-facing schema literals. | Invisible to the questioning surface. Remains in the compiled graph for profiling and structure bookkeeping unless removed by `deny_columns`. Same non-selectable predicate / group / order rules as restricted when somehow referenced. |

Assign tiers through `export_structure` / `apply_structure` or domain notes. Demo hidden columns in the offline sandbox: [Sandbox guide — Column security](SANDBOX.md#column-security).

## AetherFederation

**AetherFederation** composes several `AetherEngine` instances into one unified schema graph and session surface. Each member's `source_id` is its connection name. Declare cross-source joins in the **federation declaration** (dict passed as `declaration=` at construction; suggested persistence name: `federation_declaration.json`), and optionally map logical tables (`union` stacks members; `replica` reads one authority member).

- **Artifact trees** — Each member keeps `conn_<slug>/`; the composite persists under `fed_<federation_id>/` beside the member trees.
- **Sessions** — Open `fed.session(...)` and ask normally. The turn uses the same suspend kinds and confirmations as a single engine. On multi-member analytical turns, `step.sql` is a `dict` mapping member `source_id` → dialect SQL.
- **Reuse** — Federated questions replay stored plan templates. Member templates are stamped `federation_plan_only` and are not reused as standalone SQL. Prefer `fed.execute_template(step.template_id, params)` for programmatic re-runs after accept.
- **Ineligible plans** — Cross-source joins are allowed only when declared. Unsupported shapes surface a diagnostic and rephrase hint. When one member fails during execution, the turn ends with a structured federation error rather than a partial result.
- **File engines** — A database may be federated with uploaded files, but a federation whose members are all file (`csv`) engines is refused at registration; load several uploads into one CSV engine instead.
- **Clock consistency** — On a federated turn, when the parent intent uses relative date-window filters or clock keywords, the coordinator resolves one UTC anchor at turn start so every member statement uses the same clock instant.
- **Re-entry** — Federation artifacts are library-owned ([Integrator guide — Artifacts are library-owned](INTEGRATOR_GUIDE.md#artifacts-are-library-owned)). Use `export_federation()` / `apply_federation()` for authored joins and mappings. Document shapes: [API reference — Federation documents](API_REFERENCE.md#federation-documents). Member catalog drift uses each source's migration map; cross-source identifier changes use the federation migration map.
- **Warmup** — `run_seed_warmup`, `run_seed_warmup_from_history`, and `run_seed_warmup_from_query_log` raise `ConfigError` on `AetherFederation`. Run those on each member `AetherEngine` instead.
- **Exports** — `export_structure(space=...)` is the read-only visible catalog. `export_knowledge(space=...)` / `apply_knowledge(space, document)` are the space-tier domain knowledge and description pair. Connection-tier structural edits use `export_structure` / `apply_structure` (owner).

Operator embedding detail: [Integrator guide — Embedding a federation](INTEGRATOR_GUIDE.md#embedding-a-federation).

## FederationContext

**FederationContext** scopes the composite graph the same way `EngineContext` scopes one connection. It is optional on `AetherFederation(..., context=...)` and applies at composition and execution time over the unified member graph. It has no `name` field — the name identifies the federation.

- **include** — `"tables"` (default) or `"views"`. `"both"` is rejected.
- **allow_objects** / **deny_objects** — table or view names in the composite namespace.
- **allow_columns** / **deny_columns** — qualified `table.column` or `*.column` entries (three-part `source.table.column` inputs are accepted by the same normalizer). Deny-list semantics: [Security — Deny lists](SECURITY.md#7-deny-lists).
- **notes_file** / **notes** — optional domain notes for the composite (set at most one; [Notes file](#notes-file)). Member engines may still carry their own `EngineContext` notes for per-source vocabulary.

`FederationContext` does not replace database grants or member-level `EngineContext` scope. It narrows what the composite graph exposes and what federated execution may target on top of each member's credentials.

## AetherSpace

An **AetherSpace** is a knowledge partition over the default graph on an engine or federation. It improves model focus and partitions template learning by space **uid**. A space narrows which objects a turn may reference and refuses questions that reach past it, and it is not a permission boundary because it can neither be defined nor entered beyond what credentials already permit. SQL execution still follows database grants and engine/federation context ([Security — Execution boundary](SECURITY.md#2-execution-boundary-and-credentials)).

**Effective visibility** is the intersection of the active engine or federation context, the credential reflection subset, and non-hidden sensitivity. Catalog APIs (`list_aetherspaces`, read, `session(..., space=...)`, delete) gate on effective visibility so callers never learn identities of wider spaces; missing and out-of-visibility identities raise the same `ConfigError("unknown aetherspace …")`.

- **Default space** — implicit identity for owners (`default_space_uid`). Reflects the full default engine or federation context graph. Cannot be deleted or redefined. Owners always see the default space in `list_aetherspaces()`. Consumers omit it from listings; an omitted `space` argument resolves to `default_space_uid` ([Integrator guide — Multi-user deployment](INTEGRATOR_GUIDE.md#multi-user-deployment)).
- **Named spaces** — created with `engine.aetherspace(name, space_context=...)` (or the federation equivalent), **owner-only**. Each create mints a stable **`S####` uid**; the **name** is a display label. Duplicate visible names raise on create; update with `aetherspace(uid=..., space_context=...)`, also owner-only. Prefer `session(..., space=uid)` after create. Define scope must be ⊆ effective visibility; a define that names tables or columns outside visibility raises a `ConfigError` naming those objects. Put notes on `SpaceContext(notes=...)` or `SpaceContext(notes_file=...)` ([SpaceContext](#spacecontext)). Snapshots persist under the shared artifacts tree (`aetherspaces/{uid}.json`).
- **Listing** — `list_aetherspaces(*, include_system=False)` returns visible `AetherSpace` descriptors with `uid`, `name`, `tables`, `columns`, and `notes`. System spaces (including the credential-default space) are omitted unless `include_system=True`.
- **Deleting a space** — `delete_aetherspace(uid=...)` **(owner-only)** removes the snapshot and that space's learning partition when the space is ⊆ the caller's visibility. Returns `AetherspaceDeleteResult` with `deleted` and `merge_counts`. By default (`persist_learning=True`), templates and feedback unique to the space are promoted into the default space before removal; when the default space already has the same question or intent, the default space wins. Pass `persist_learning=False` to discard the space's learning without merging.
- **Per-space notes** — optional `SpaceContext.notes` or `SpaceContext.notes_file` at define time; content is baked into the aetherspace snapshot. The engine merges inherited default descriptions with space-specific refinements ([Sandbox guide — Named AetherSpaces](SANDBOX.md#named-aetherspaces)).

## SpaceContext

**SpaceContext** supplies the allow/deny shape for an AetherSpace: `tables`, `columns`, `deny_objects`, `deny_columns`, and optional `notes` or `notes_file` (set at most one). It uses the same token shapes as `EngineContext` column specs (`table.column` or `*.column`) but applies at **question time** for knowledge and template partitioning. It has no identity field — pass the space **name** (create) or **uid** (update) on `aetherspace(...)`.

Named spaces are always created against the default engine or federation context. On a federation, space snapshots live under the federation artifact tree.

## Knowledge layers

Two editable document layers stay separate from notes input:

| Layer | What it carries | Read / write |
| --- | --- | --- |
| **Structure** | Tables, columns, types, keys, relationships, roles, sensitivity, usability | `export_structure(space=...)` / `apply_structure(document)` (owner) |
| **Knowledge** | Domain-knowledge entries plus table/column **descriptions** for the default or one named space | `export_knowledge` / `apply_knowledge(space, document)` (owner) |

**Domain knowledge** is keyed glossary/policy/concept text used in prompts and metadata answers. **Structural knowledge** is fact-shaped notes (join hints, role/sensitivity cues) that drive description enrichment at build time. Engine and federation overlays share one merge/security path: notes extract, space-over-base merge, visibility key filter, then drop entries that name HIDDEN columns. On an engine or federation, named AetherSpaces take a subset of that facade's compiled graph as their parent.

Mechanism detail: [How it works — Knowledge pipeline](HOW_IT_WORKS.md#knowledge-pipeline).

## Notes file

Domain notes refine descriptions and guide role classification. Use one or two sentences per important table, domain definitions, join hints, and explicit sensitivity statements. Supply notes as a path (`notes_file`) or inline text (`notes`) on the default context — set at most one. The default context reads `EngineContext` notes; a federation may also supply `FederationContext` notes. Each named AetherSpace may supply notes via `SpaceContext(notes=...)` or `SpaceContext(notes_file=...)` at define time.

## Structure document

Structural edits live in the structure document returned by `export_structure()`. Export, edit roles, sensitivity, foreign keys, or primary keys, then `apply_structure(document)`. Descriptions belong in the knowledge layer (`export_knowledge` / `apply_knowledge`). Every structure edit replays on cache invalidation. Suggested caller-owned persistence name: `schema_structure.json`. After `apply_structure`, the library writes `applied_structure.json` under the engine artifact tree.

- **usable** — statistical gates mark low-signal columns unusable. Structure documents accept `"usable": true` to force a column usable; `"usable": false` is rejected — use sensitivity tiers or deny lists to withhold a column instead.
- **sensitivity** — assign `none`, `restricted`, or `hidden` per column ([Sensitivity classification](#sensitivity-classification)).

Workflow detail: [API reference — Structure document](API_REFERENCE.md#structure-document).

## Asking a question

When the engine is not confident about a translation, it suspends the turn and asks for confirmation — intent readback, SQL preview, or execute confirmation depending on confidence and reuse path. Accept builds template learning; reject with a reason steers future turns away from the same mistake. Drafts awaiting accept are stored as pending templates so a repeated question can resume confirmation without regenerating SQL.

Some questions never enter the SQL path: schema inventory/count questions, domain-knowledge definitions, and combined structure+domain questions return a terminal step with `answer` set instead of Intent/SQL confirmations. Branch on terminal shape before reading `sql`/`data`.

**Template reuse** is automatic:

1. **Direct reuse** — when new wording is within a small normalized token edit distance (at most two) of a stored question, the engine replays the stored SQL path with no full parse LLM calls.
2. **Intent-level match** — larger wording changes still require LLM calls even when structure is similar.

For programmatic re-execution with new bind values on a known stored template, use `engine.execute_template(step.template_id, {b.handle: b.current_value for b in step.parameters})` (same surface on `AetherFederation`).

Ask normally; change wording and observe which path runs via diagnostics or audit events ([Integrator guide — Observability](INTEGRATOR_GUIDE.md#observability)).

The offline sandbox supports the same accept/reject mechanism as production; bundled mock coverage is finite ([Sandbox guide](SANDBOX.md)). Federated date-window anchoring lives under [AetherFederation](#aetherfederation).

## SessionStep

Every `session.ask(...)` or `session.step(...)` returns a **`SessionStep`** — one observable point in a turn. Branch embedded UIs on **`step.kind`** and **`step.done`**.

Terminal success has three shapes: `answer` for metadata, `sql` with `data` for analytical success, or `error` for failure. On federated analytical turns, `step.sql` is a `dict` mapping member `source_id` → SQL; on single-engine turns it is a `str`.

Full field list, suspend `kind` values, and outcome mapping: [Integrator guide — The session contract](INTEGRATOR_GUIDE.md#the-session-contract-suspend-and-terminal-steps) | [Troubleshooting — Reading terminal steps](TROUBLESHOOTING.md#reading-terminal-steps) | [API reference — SessionStep](API_REFERENCE.md#sessionstep).

## Migration

When the catalog changes structurally, the engine raises `MigrationPendingError` with the migration skeleton document. Edit the map with one of:

- **remap** — rewrite identifiers in stored templates.
- **destructive** — clear templates and learning for a fresh slate.
- **abort** — stop initialization to investigate drift.

Federation variation: member-source drift still uses each source's own migration map. Cross-source identifier changes that span the composite graph use the federation migration map beside the federation tree with the same remap/destructive/abort actions. See [AetherFederation](#aetherfederation).

## Seed warmup

Warmup paths populate templates so common questions hit the cache sooner:

1. **Seed-question warmup** — parses and paraphrases a list of natural-language questions.
2. **SQL-history warmup** — reverse-engineers historical `SELECT` statements into intents.
3. **Query-log warmup** — reads warehouse system logs for historical queries.

Federation variation: warmup methods raise `ConfigError` on `AetherFederation` — run them on each member engine. See [AetherFederation](#aetherfederation).

Warmup and QSim are unavailable inside the offline sandbox (`ConfigError`).

## Partition and cluster pruning

When the compiled schema graph carries partition or clustering metadata and the parsed intent includes matching date or key filters, the engine may append `WHERE` predicates on those columns at SQL finalization (`inject_pruning_predicates`). This reinforces warehouse pruning without inventing data ranges the user did not ask for. Engine-specific behavior: [Support matrix — Engine capabilities](SUPPORT_MATRIX.md#engine-capabilities).

## Resetting learning

Scoped resets clear the template store or simulation caches through public API methods on `AetherEngine` (`clear_template_store(*, space=…)`, `clear_simulation_caches`, `clear_all_learning(*, space=…)`). To clear applied structure, export an empty structure document and `apply_structure`. Template/learning clears accept `space=None`/`"all"` (every partition) or a space uid. Domain knowledge is cleared by exporting an empty knowledge document and `apply_knowledge`. See [API reference — AetherEngine](API_REFERENCE.md#aetherengine).

## CSV and Excel uploads

The `csv` engine uses a DuckDB backend and `CSV_*` environment keys. With an engine `artifacts_dir`, row data is cached in `upload_store.duckdb` under that directory and reused when the source probe matches; without artifacts it stays in-memory. It accepts `.csv` and `.xlsx` uploads (not `.xls`).

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

During inspection the engine auto-reads encoding, removes title and header chrome, normalizes values, and records **Advisory** fixes (ragged rows padded, duplicate headers suffixed, and similar). **Review** cases — multiple tables on one sheet, uncertain header rows, appendable regions — require your choice via `source_selections` (`header_row`, `table_range`, `append_regions`, and similar). Only **Blocking** and **Fatal** cases prevent a usable table.

Each worksheet loads as its own table; workbooks with multiple data sheets use `filename__sheetname` table names. Spreadsheet cell sampling for model-assisted interpretation is optional and controlled by `PolicyConfig.TABULAR_LLM_ASSIST` ([Security — Upload inspection](SECURITY.md#57-upload-inspection-csv-file-engine)).

## Common pitfalls

- **Wrong table chosen:** ensure your notes file clearly distinguishes similar domain entities.
- **Related tables not joined:** confirm foreign keys exist or add them through the structure document.
- **Column never appears:** check statistical usability, deny lists, and sensitivity tags.
- **AetherSpace confused with permissions:** a space narrows model focus and refuses out-of-scope questions; tighten `EngineContext`, `FederationContext`, or database grants for access control.
- **CSV rejected at construction:** open the `ConfigError` message — it lists exact `file!sheet!cell` locations. Fix the spreadsheet and retry.

---

**See also:** [Getting started](GETTING_STARTED.md) | [Integrator guide](INTEGRATOR_GUIDE.md) | [Sandbox guide](SANDBOX.md) | [API reference](API_REFERENCE.md) | [Troubleshooting](TROUBLESHOOTING.md) | [How it works](HOW_IT_WORKS.md) | [Security](SECURITY.md) | [Support matrix](SUPPORT_MATRIX.md) | [README](../README.md)
