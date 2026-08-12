# How it works

Conceptual end-to-end narrative for the `aetherdialect` engine: why it is built the way it is, and what happens between a natural-language question and a returned table. Public signatures: [API reference](API_REFERENCE.md). Operator semantics: [User guide](USER_GUIDE.md). Embedding loops: [Integrator guide](INTEGRATOR_GUIDE.md). Session outcomes and diagnostic codes: [Troubleshooting](TROUBLESHOOTING.md).

**Reading order:** [README](../README.md) → [Getting started](GETTING_STARTED.md) → [User guide](USER_GUIDE.md) → [Integrator guide](INTEGRATOR_GUIDE.md) → [Sandbox guide](SANDBOX.md) → [API reference](API_REFERENCE.md) → [Troubleshooting](TROUBLESHOOTING.md) → this document → [Security](SECURITY.md) → [Support matrix](SUPPORT_MATRIX.md).

## Sections

| Section | Contents |
| --- | --- |
| [Design philosophy](#design-philosophy) | Reuse, bounded LLM jobs, cheap checks first |
| [1. Configuration](#1-configuration) | Effective settings without mutating the process environment |
| [2. Schema compilation](#2-schema-compilation) | Reflect, profile, gate, classify, connect — and federation composition |
| [3. Engine storage and fingerprints](#3-engine-storage-and-fingerprints) | Member trees, federation tree, cache keys |
| [4. Catalog drift and migration](#4-catalog-drift-and-migration) | Why drift stops construction |
| [5. AetherEngine, AetherFederation, and AetherSpace](#5-aetherengine-aetherfederation-and-aetherspace) | The three facades and their scope types |
| [6. Question pipeline](#6-question-pipeline) | Routing, intake through validated SQL — single-engine and federated |
| [7. Learning model](#7-learning-model) | Templates, spaces, feedback, and parameter replay |
| [8. Concurrent sessions and durability](#8-concurrent-sessions-and-durability) | Reader vs writer modes, write queue, locks, cancellation |
| [9. Tabular ingestion](#9-tabular-ingestion) | Inspect, selections, and embedded load |

---

## Design philosophy

The engine prefers **reuse and deterministic rules** over open-ended generation. A language model (LLM) does small, bounded jobs inside constraints supplied by the compiled schema graph and by validators. A turn is a sequence of **cheap checks before any expensive one**: normalize and look for a trusted template before calling the model; validate a typed intermediate representation (IR) before rendering SQL; run dialect and catalog checks before execution.

- **SQL is rendered from a typed IR**, not invented as free-form text. The model fills structured slots (tables, filters, aggregates, and similar). Deterministic code materializes dialect SQL only after schema and semantic gates pass. Free-form SQL generation would make injection defenses and catalog alignment probabilistic; the IR makes them mechanical.
- **Scope is fingerprinted** together with catalog structure. Changing allow/deny lists, notes, or DDL changes the cache and template keys, so learning from one visibility boundary cannot silently satisfy another.
- **Learning is partitioned** by AetherSpace and by on-disk template shards. Concurrent readers and large histories stay tractable, and knowledge narrowing keeps its own reuse surface.
- **Federation decomposes per source** instead of pushing one federated query into warehouses that do not share a planner. Each member runs under its own dialect, template store, and execution gates; an in-process DuckDB coordinator combines member result frames.

The same philosophy applies at every facade: single engine, federation, or named space.

## 1. Configuration

Effective settings are built for each process **without mutating `os.environ`**. When a `config_file` is provided, every flattened field that the file claims is authoritative for its mapped environment key: non-empty TOML values replace the process environment in the effective map, and empty claimed keys remove the variable so ambient defaults cannot leak past an explicit TOML field. A `connection=` mapping overlays per-instance credentials without writing to `os.environ`.

Behavioural limits (`statement_timeout_ms`, row caps, pool sizing, write-queue byte caps, federation coordinator caps) arrive through `limits=` on the constructor or through `[limits]` / `[federation_limits]` tables loaded with `EngineLimits.from_config_file` / `FederationLimits.from_config_file`. Limits are never flattened into the environment map.

## 2. Schema compilation

When an engine is constructed, it compiles a **schema graph** from the live database (or from optional DDL on `EngineContext.sql_file`). The graph is built once per fingerprint and cached on disk.

1. **Reflect** — introspect the warehouse catalog (or parse DDL) into tables, columns, keys, and declared foreign keys.
2. **Profile** — run read queries against live data for counts, value distributions, and samples that later roles and join inference use.
3. **Usability gating** — columns that fail statistical gates (for example distinct count ≤ 1, extreme null rate, or sentinel-dominated mode frequency) are marked unusable and omitted from LLM-facing schema literals. Primary and foreign keys stay usable.
4. **Classification** — the LLM assigns table and column roles and sensitivity tiers from profiles and optional domain notes, only for usable columns.
5. **Foreign-key inference** — when catalog foreign keys leave disconnected islands, name matches and value-containment checks add inferred edges so multi-table questions have a join graph.

### Knowledge pipeline

Domain notes feed a construction-time extraction pass that produces two artifacts: **domain knowledge entries** (glossary, policy, concept prose keyed by `kind` and `referenced_entities`) and **structural facts** (join hints, role cues) that drive description enrichment. Every fact anchors to schema entities through `referenced_entities`; claim validation at build time accepts, rejects, or flags each entry. Merge identity is `(kind, referenced_entities)`; conflicting duplicates fail rather than silently resolve. Staleness keys on notes content hash plus scope fingerprint.

At ask time, knowledge reaches prompts through per-entity enrichment filtered to the caller's effective visibility. Space snapshots carry their own domain knowledge and description overlays; applying knowledge for an object outside the active space raises a scope error.

Operator-facing layer split: [User guide — Knowledge layers](USER_GUIDE.md#knowledge-layers). Sensitivity enforcement: [Security — Sensitivity tags](SECURITY.md#6-sensitivity-tags).

### Federation composition

For `AetherFederation`, each member `AetherEngine` runs steps 1–5 on its own connection and artifact tree. Composition then merges those member graphs into one **unified graph** the model sees as a single schema:

- **Namespace and collisions** — table names from every member are stamped with internal source metadata. Colliding names must be resolved through logical mappings or explicit aliases before construction succeeds.
- **Logical tables** — a `union` mapping stacks members into one relation; a `replica` mapping reads one authoritative member.
- **Logical columns** — differing spellings across members can unify under one composite column name.
- **Cross-source joins** — declared edges connect members where no catalog foreign key exists. Undeclared spans are ineligible rather than guessed.
- **Classification reconcile** — descriptions, roles, and sensitivity on collapsed or unified columns are reconciled so the composite graph describes itself consistently.
- **Prompt neutrality** — interpret, ground, and compose prompts describe one unified schema. They do not claim a single warehouse, disclose multiple databases, or carry source identifiers; union and replica semantics appear only through neutral capability wording.

The model never sees source identifiers — only the unified catalog.

## 3. Engine storage and fingerprints

Persisted state lives under an **artifacts root** you choose (or the platform user-data directory when omitted):

```
<artifacts_root>/aetherdialect/
  conn_<engine>_<slug>/     # one member engine tree per connection
  fed_<federation_id>/      # one federation tree (sibling to member trees)
```

- **Member slug** — `conn_<engine>_<slug>/` derives from database location keys only (host, port, database, schema, and the per-engine key order in [API reference — Artifact storage slug](API_REFERENCE.md#artifact-storage-slug)). Credentials never appear in the slug. Each `AetherEngine` owns exactly one tree.
- **Federation slug** — `fed_<federation_id>/` uses the stable federation name from the manifest (`federation_id` must match the `AetherFederation` name argument). Member `conn_*` trees are **parallel siblings** under `<artifacts_root>/aetherdialect/`. The federation tree holds the composite graph, manifest sidecar, mappings, plan templates, AetherSpace snapshots, and coordinator spill directories; member engines keep their own `conn_*` trees with member-scoped template stores.

Fingerprints in each tree's `artifact_manifest.json` decide whether cache and learning remain valid:

- **structural** — DDL-stable shape (kinds, columns, keys, foreign-key edges).
- **profiling** — profile-only payloads (counts, roles, top values).
- **scope** — include mode, allow/deny lists, and content hashes of notes and DDL files.
- **effective structural** — structural combined with scope; this is the template-store key so reuse is scoped to the same visibility boundary.

On disk you also find `schema_graph.json.gz`, the partitioned `intent_templates/` tree, and persisted structure documents. Plan templates and composite learning live under `fed_<federation_id>/`; per-member intent templates stay under each `conn_<slug>/` tree.

Artifact format versions are exact. A mismatched aetherspace snapshot version raises `ConfigError` (path, found version, expected version). A mismatched federation artifact format version raises `FederationConfigError` the same way. Delete the stale artifact and redefine or re-initialize.

## 4. Catalog drift and migration

Structural catalog drift stops initialization with `MigrationPendingError`, which carries the migration skeleton document. Operators choose **remap**, **destructive**, or **abort** before construction may continue ([User guide — Migration](USER_GUIDE.md#migration)). Silent resume would let templates and structure edits target the wrong identifiers.

Federation drift uses `federation_migration_map.json` on the federation tree; per-member drift still uses each member's `schema_migration_map.json`.

## 5. AetherEngine, AetherFederation, and AetherSpace

Three facades compose in order: engine (or federation) first, then optional named space.

| Facade | Scope type | Role |
| --- | --- | --- |
| **AetherEngine** | `EngineContext` | One database connection, one `conn_*` tree, one default graph. Saved scope presets register with `engine_context(name, context)`. |
| **AetherFederation** | `FederationContext` | Named member engines, manifest-driven joins and mappings, one `fed_<federation_id>/` tree beside member `conn_*` trees. |
| **AetherSpace** | `SpaceContext` | Named knowledge subset over the default graph on an engine or federation — partitions prompts and template learning by space **uid**. A space narrows which objects a turn may reference and refuses questions that reach past it, and it is not a permission boundary because it can neither be defined nor entered beyond what credentials already permit ([User guide — AetherSpace](USER_GUIDE.md#aetherspace)). |

**Engine role and session mode:** pass **`role="owner"`** or **`role="consumer"`** on construction (engine role). Pass **`mode="writer"`** or **`mode="reader"`** on `session()` (session mode). `role="owner"` builds and mutates shared artifacts; `role="consumer"` pins to the owner's published snapshot. `session(mode="writer")` persists learning under the artifacts lock; `session(mode="reader")` keeps learning session-local. Role and connection credentials stay independent.

`SpaceContext` narrows knowledge and template partitions at question time. A space narrows which objects a turn may reference and refuses questions that reach past it, and it is not a permission boundary because it can neither be defined nor entered beyond what credentials already permit. Space create, update, and delete are owner-only. Catalog read surfaces (`list_aetherspaces`, read by uid or unique visible name, `session(..., space=...)`) gate on effective visibility. Structural catalog read-back is `export_structure`; space domain knowledge and description overlays use `export_knowledge` / `apply_knowledge`; connection-tier structural edits use `export_structure` / `apply_structure` (owner). `space=None` means the default space; use `default_space_uid` for the uid. Operator-facing layer split: [User guide — Knowledge layers](USER_GUIDE.md#knowledge-layers).

## 6. Question pipeline

### Question routing

Before interpret, the engine classifies each question into a route:

| Route | Outcome shape |
| --- | --- |
| Analytical | SQL suspend/confirm loop; terminal `sql` + `data` |
| Schema catalog | Terminal `answer` (metadata) |
| Domain knowledge | Terminal `answer` |
| Schema and knowledge | Terminal `answer` |
| Conversational / unsupported | Terminal `error` with a closed outcome code |
| Insufficient knowledge | Terminal `error` with `insufficient_knowledge` |

Integrators never pass a route flag — branch on the returned [`SessionStep`](API_REFERENCE.md#sessionstep). Outcome codes: [Troubleshooting — SessionOutcome](TROUBLESHOOTING.md#sessionoutcome).

### Analytical turn order

1. **Intake** — validate and normalize question text (`q_norm`).
2. **Reuse** — consult the template store. Direct reuse (normalized token edit distance within a small fixed budget, default two) can replay a trusted SQL path with at most a bounded parameter-extraction call. Larger wording changes continue.
3. **Interpret** — the LLM produces a natural-language analytical plan against a compact schema payload.
4. **Ground** — the LLM binds that plan to schema identifiers.
5. **Compose** — the LLM lowers logical intent into the typed IR.
6. **Repair and validate** — see [Validation cascade](#validation-cascade) and [Retry model](#retry-model).
7. **Finalize** — join resolution and parameter binding for executable SQL.
8. **Execute** — generated SQL passes dialect AST validation, schema alignment, EXPLAIN, and cost gates before any warehouse run.

Programmatic callers observe **suspend** steps (`step.done == false`) and **terminal** steps (`step.done == true`) via `SessionStep.kind` ([Integrator guide — The session contract](INTEGRATOR_GUIDE.md#the-session-contract-suspend-and-terminal-steps)). Terminal success has three shapes: `answer` for metadata, `sql` with `data` for analytical success, or `error` for failure.

### Metadata answer path

Schema inventory, domain glossary, and combined structure+knowledge questions take a separate path: one model call with JSON-schema-validated output, rendered to `answer`, cached on a visibility fingerprint. No SQL pipeline runs. Meta scoping happens before the model sees schema payloads, so no separate meta-route guard is required.

### Join resolution

Join resolution enumerates candidate paths from the compiled graph with tie caps. Notes-sourced foreign-key hints admit edges the catalog omitted. Declared multi-hop paths pin when the operator authored them. Grain and cardinality drive fan-out guards; aggregate fan-out beyond policy refuses with a diagnostic. Orphan-rate diagnostics flag weak edges. When several candidates remain in one scope, one bounded LLM disambiguation call chooses among the closed candidate list; single-candidate scopes skip the call.

### Validation cascade

Validation runs in tiers, each deterministic for fixed inputs:

1. **IR shape** — parse intent JSON into typed structures; reject illegal slot shapes.
2. **Semantic rules** — grain, aggregation, having, select-list, where, and CTE constraints.
3. **Dialect AST** — structural walk of rendered SQL.
4. **Catalog alignment** — tables, columns, usability, sensitivity, and scope gates.
5. **EXPLAIN** — warehouse plan smoke check and cost cap when supported.
6. **Cost** — row/byte estimates against configured limits.

Bounded **automatic repair** rounds re-invoke compose with error rows between tiers. Exhaustion terminates the turn with a structured outcome rather than open-ended regeneration.

### Retry model

| Path | Programmatic `PipelineSession` | Interactive `run_interactive` |
| --- | --- | --- |
| Automatic post-validation repair | Yes | Yes |
| Intent rejection with free-text feedback | Terminates with `declined` | Restarts the turn (refinement loop) |
| SQL rejection with free-text feedback | Terminates with `declined` | Restarts the turn (refinement loop) |

### LLM interaction

The provider abstraction routes task classes to light/heavy model slots. Each JSON-producing stage validates responses against a JSON Schema with bounded retries before `LlmJsonExhausted`. Timeouts and batching are configurable. Per-turn token and cost accounting surfaces as `llm_usage` on terminal steps. Disclosure inventory: [Security — LLM context inventory](SECURITY.md#5-llm-context-inventory).

### Federated turns

After a validated intent against the unified graph, a multi-source question adds stages:

1. **Decompose** (`plan.decompose`) — split the intent into per-source sub-intents, each validated against its member schema. Capability is the **intersection** of member dialect surfaces. Join enumeration caps scale with member count on the composite graph.
2. **Stage** — build a dependency graph: member stages, optional cross-member CTE stage, coordinator combine stage.
3. **Generate** — render member SQL under each source's dialect and template store using the same deterministic renderer as a single-engine turn (a one-member federated plan is byte-identical to direct single-engine SQL).
4. **Execute** — run members with optional semi-join reduction, per-source caps, and cooperative cancellation between stages.
5. **Combine** (`plan.combine`) — the in-process DuckDB coordinator registers member frames, streams them into join and residual operators, and releases each member frame once consumed so resident bytes stay under the configured cap.
6. **Return** — `SessionStep.sql` carries per-member dialect SQL (`dict[str, str]` on multi-member turns, `str` on single-engine or degenerate one-member plans).

Prepared federated SQL is session-owned until execution or `PipelineSession.reset()`. Invalid declarations fail at `AetherFederation` construction.

A federated answer reads each member at instants that are close but not identical; treat results as consistent only within the reported read window.

## 7. Learning model

Accepted turns fold into the partitioned template store under the active AetherSpace (keyed by normalized question text and space **uid**). Future similar questions hit reuse before a full interpret/ground/compose cycle. Negative feedback is summarized and steers retry and template selection away from past mistakes.

On federation accepts, per-source templates land in member stores and a versioned **plan template** on the federation tree records how to replay the turn. Member templates are stamped `federation_plan_only` so they replay only through the plan record, not as standalone single-engine SQL reuse.

### Parameter binding and replay

Each bind slot is a `ParameterBinding` with `handle` (the bind token), `current_value`, `display_name`, and `column_expr`. Stored templates record the slot schema; replay rebinds literal and structural parameters against it. `execute_template(template_id, values)` replays without learning. `session(mode="writer")` saves the approved question-to-template mapping so the same question later replays deterministically; `session(mode="reader")` answers without persisting shared learning.

Deleting a named AetherSpace removes its snapshot and per-space learning partition. With the default `persist_learning=True`, templates and feedback that exist only in that space are copied into the default space first; when both namespaces already hold the same normalized question or intent fingerprint, the default space wins. With `persist_learning=False`, the space partition is removed without merging.

## 8. Concurrent sessions and durability

### Reader and writer modes

Reader sessions do not persist shared learning; writer sessions persist templates and feedback into the active space partition under that partition's advisory lock. Engine-root mutations (schema, structure, space catalog) use the engine artifacts lock. Readers never mutate the partitioned template files directly.

### Write queue

When multiple processes share one artifacts directory, writer-mode turns **drain `write_queue.jsonl` at turn start** under the engine artifacts lock before applying new learning. The queue carries template accepts, rejections, and structure proposals. Consumer writers apply only learning events for their active space and leave structure proposals and foreign-space events on the queue. Reader sessions do **not** enqueue durable write-queue events — reader learning stays session-local.

### Locks, atomic writes, and retention

Advisory locks serialize cooperating processes on a local filesystem: one lock domain for the engine (or federation) root, and one lock domain per space learning partition. Writes use atomic replace patterns; orphan pruning and retention policies cap template store growth. Federation `clear_all_learning()` clears federation and member template stores.

### Cancellation

`session.cancel()` on the owning `PipelineSession` (or `await session.cancel()` on `AsyncPipelineSession`) cooperatively stops an in-flight turn. On federation turns, cancellation is observed between member stages and batches and cancels in-flight database statements on members.

## 9. Tabular ingestion

The `csv` engine path runs before or during construction:

1. **`inspect_tabular_upload`** — analyse grid layout, encoding, and candidate regions on raw files (no engine required). Returns a `DataQualityReport` with `suggested_selections` shaped for `source_selections`.
2. **Caller confirmation** — pass accepted `source_selections` on `AetherEngine(...)` when review issues remain.
3. **Construction / ingest** — relations load into embedded DuckDB; `engine.data_quality_report` reflects confirmed post-construction state.
4. **Re-inspection** — `ingest_upload_sources` on an existing `csv` or `duckdb` engine validates and materialises additional uploads.

Operator walkthrough: [User guide — CSV and Excel uploads](USER_GUIDE.md#csv-and-excel-uploads). API shapes: [API reference — Tabular upload](API_REFERENCE.md#tabular-upload).

---

**See also:** [Getting started](GETTING_STARTED.md) | [User guide](USER_GUIDE.md) | [Integrator guide](INTEGRATOR_GUIDE.md) | [Sandbox guide](SANDBOX.md) | [API reference](API_REFERENCE.md) | [Troubleshooting](TROUBLESHOOTING.md) | [Security](SECURITY.md) | [Support matrix](SUPPORT_MATRIX.md) | [README](../README.md)
