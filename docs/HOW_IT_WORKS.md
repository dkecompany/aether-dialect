# How it works

Conceptual end-to-end narrative for the `aetherdialect` engine: why it is built the way it is, and what happens between a natural-language question and a returned table. Public signatures: [API reference](API_REFERENCE.md). Operator semantics: [User guide](USER_GUIDE.md). Embedding loops: [Integrator guide](INTEGRATOR_GUIDE.md).

**Reading order:** [README](../README.md) -> [Getting started](GETTING_STARTED.md) -> [User guide](USER_GUIDE.md) -> [Integrator guide](INTEGRATOR_GUIDE.md) -> [Sandbox guide](SANDBOX.md) -> [API reference](API_REFERENCE.md) -> this document -> [Security](SECURITY.md) -> [Support matrix](SUPPORT_MATRIX.md).

## Sections

| Section | Contents |
| --- | --- |
| [Design philosophy](#design-philosophy) | Reuse, bounded LLM jobs, cheap checks first |
| [1. Configuration](#1-configuration) | Effective settings without mutating the process environment |
| [2. Schema compilation](#2-schema-compilation) | Reflect, profile, gate, classify, connect - and federation composition |
| [3. Engine storage and fingerprints](#3-engine-storage-and-fingerprints) | Member trees, federation tree, cache keys |
| [4. Catalog drift and migration](#4-catalog-drift-and-migration) | Why drift never resumes silently |
| [5. AetherEngine, AetherFederation, and AetherSpace](#5-aetherengine-aetherfederation-and-aetherspace) | The three facades and their scope types |
| [6. Question pipeline](#6-question-pipeline) | Intake through validated SQL - single-engine and federated |
| [7. Learning model](#7-learning-model) | Templates, spaces, and feedback |
| [8. Concurrent sessions and the write queue](#8-concurrent-sessions-and-the-write-queue) | Reader enqueue, writer drain |

---

## Design philosophy

The engine prefers **reuse and deterministic rules** over open-ended generation. A language model (LLM) does small, bounded jobs inside constraints supplied by the compiled schema graph and by validators. A turn is a sequence of **cheap checks before any expensive one**: normalize and look for a trusted template before calling the model; validate a typed intermediate representation (IR) before rendering SQL; run dialect and catalog checks before execution.

That order is deliberate:

- **SQL is rendered from a typed IR**, not invented as free-form text. The model fills structured slots (tables, filters, aggregates, and similar). Deterministic code materializes dialect SQL only after schema and semantic gates pass. Free-form SQL generation would make injection defenses and catalog alignment probabilistic; the IR makes them mechanical.
- **Scope is fingerprinted** together with catalog structure. Changing allow/deny lists, notes, or DDL changes the cache and template keys, so learning from one visibility boundary cannot silently satisfy another.
- **Learning is partitioned** by AetherSpace and by on-disk template shards. Concurrent readers and large histories stay tractable, and knowledge narrowing keeps its own reuse surface.
- **Federation decomposes per source** instead of pushing one federated query into warehouses that do not share a planner. Each member runs under its own dialect, template store, and execution gates; an in-process DuckDB coordinator combines member result frames.

The same philosophy applies at every facade: single engine, federation, or named space.

## 1. Configuration

Effective settings are built for each process **without mutating `os.environ`**. When a `config_file` is provided, every flattened field that the file claims is authoritative for its mapped environment key: non-empty TOML values replace the process environment in the effective map, and empty claimed keys remove the variable so ambient defaults cannot leak past an explicit TOML field. That merged map drives database connection, LLM provider selection, timeouts, and execution limits.

## 2. Schema compilation

When an engine is constructed, it compiles a **schema graph** from the live database (or from optional DDL on `EngineContext.sql_file`). The graph is built once per fingerprint and cached on disk.

1. **Reflect** - introspect the warehouse catalog (or parse DDL) into tables, columns, keys, and declared foreign keys.
2. **Profile** - run read queries against live data for counts, value distributions, and samples that later roles and join inference use.
3. **Usability gating** - columns that fail statistical gates (for example distinct count <= 1, extreme null rate, or sentinel-dominated mode frequency) are marked unusable and omitted from LLM-facing schema literals. Primary and foreign keys stay usable.
4. **Classification** - the LLM assigns table and column roles and sensitivity tiers from profiles and optional domain notes, only for usable columns.
5. **Foreign-key inference** - when catalog foreign keys leave disconnected islands, name matches and value-containment checks add inferred edges so multi-table questions have a join graph.

### Federation composition

For `AetherFederation`, each member `AetherEngine` runs steps 1-5 on its own connection and artifact tree. Composition then merges those member graphs into one **unified graph** the model sees as a single schema:

- **Namespace and collisions** - table names from every member are stamped with internal source metadata. Colliding names must be resolved through logical mappings or explicit aliases before construction succeeds.
- **Logical tables** - a `union` mapping stacks members into one relation; a `replica` mapping reads one authoritative member.
- **Logical columns** - differing spellings across members can unify under one composite column name.
- **Cross-source joins** - declared edges connect members where no catalog foreign key exists. Undeclared spans are ineligible rather than guessed.
- **Classification reconcile** - descriptions, roles, and sensitivity on collapsed or unified columns are reconciled so the composite graph describes itself consistently.
- **Prompt neutrality** - interpret, ground, and compose prompts describe one unified schema. They do not claim a single warehouse, disclose multiple databases, or carry source identifiers; union and replica semantics appear only through neutral capability wording.

The model never sees source identifiers - only the unified catalog.

## 3. Engine storage and fingerprints

Persisted state lives under an **artifacts root** you choose (or the platform user-data directory when omitted):

```
<artifacts_root>/aetherdialect/
  conn_<engine>_<slug>/     # one member engine tree per connection
  fed_<federation_id>/      # one federation tree (sibling to member trees)
```

- **Member slug** - `conn_<engine>_<slug>/` comes from database identity fields (engine type, host, database name, and similar). Each `AetherEngine` owns exactly one tree.
- **Federation slug** - `fed_<federation_id>/` uses the `FEDERATION_STORAGE_PREFIX` (`fed_`) plus the stable federation name from the manifest (`federation_id` must match the `AetherFederation` name argument). Member `conn_*` trees are **parallel siblings** under `<artifacts_root>/aetherdialect/` - they are **not** nested inside `fed_<federation_id>/`. The federation tree holds the composite graph, manifest sidecar, mappings, plan templates, AetherSpace snapshots, and coordinator spill directories; member engines keep their own `conn_*` trees with member-scoped template stores.

Fingerprints in each tree's `artifact_manifest.json` decide whether cache and learning remain valid:

- **structural** - DDL-stable shape (kinds, columns, keys, foreign-key edges).
- **profiling** - profile-only payloads (counts, roles, top values).
- **scope** - include mode, allow/deny lists, and content hashes of notes and DDL files.
- **effective structural** - structural combined with scope; this is the template-store key so reuse is scoped to the same visibility boundary.

On disk you also find `schema_graph.json.gz`, the partitioned `intent_templates/` tree, and `applied_overrides.json` for operator corrections. Plan templates and composite learning live under `fed_<federation_id>/`; per-member intent templates stay under each `conn_<slug>/` tree.

Artifact format versions are exact. A mismatched aetherspace snapshot version raises `ConfigError` (path, found version, expected version) rather than returning nothing; a mismatched federation artifact format version raises `FederationConfigError` the same way. There is no silent rebuild and no legacy-version shim - delete the stale artifact and redefine or re-initialize.

## 4. Catalog drift and migration

Structural catalog drift stops initialization with `MigrationPendingError`. The engine writes a `schema_migration_map.json` skeleton; operators choose **remap**, **destructive**, or **abort** before construction may continue ([User guide - Migration](USER_GUIDE.md#migration)). Silent resume would let templates and overrides target the wrong identifiers.

Federation drift uses `federation_migration_map.json` on the federation tree; per-member drift still uses each member's `schema_migration_map.json`.

## 5. AetherEngine, AetherFederation, and AetherSpace

Three facades compose in order: engine (or federation) first, then optional named space.

| Facade | Scope type | Role |
| --- | --- | --- |
| **AetherEngine** | `EngineContext` | One database connection, one `conn_*` tree, one master graph. Saved scope presets register with `engine_context(name, context)`. |
| **AetherFederation** | `FederationContext` | Named member engines, manifest-driven joins and mappings, one `fed_<federation_id>/` tree beside member `conn_*` trees. |
| **AetherSpace** | `SpaceContext` | Named knowledge subset over the master graph on an engine or federation - partitions prompts and template learning by space name. |

**Engine role and session mode:** pass **`role="owner"`** or **`role="consumer"`** on construction (engine role). Pass **`mode="writer"`** or **`mode="reader"`** on `session()` (session mode). `role="owner"` builds and mutates shared artifacts; `role="consumer"` pins to the owner's published snapshot. `session(mode="writer")` persists learning under the artifacts lock; `session(mode="reader")` enqueues to `write_queue.jsonl` for a writer on the same artifacts directory to drain.

`SpaceContext` narrows knowledge and template partitions at question time only - not SQL execution scope ([Security - Execution boundary](SECURITY.md#2-execution-boundary-and-credentials)).

## 6. Question pipeline

Turn order for a single-engine question:

1. **Intake** - validate and normalize question text (`q_norm`).
2. **Reuse** - consult the template store. Direct reuse (normalized token edit distance within `FUZZY_MATCH_MAX_DISTANCE`, default 2) can replay a trusted SQL path with at most a bounded parameter-extraction call. Larger wording changes continue.
3. **Interpret** - the LLM produces a natural-language analytical plan against a compact schema payload.
4. **Ground** - the LLM binds that plan to schema identifiers.
5. **Compose** - the LLM lowers logical intent into the typed IR.
6. **Repair and validate** - deterministic repair plus multi-tier schema and semantic validation (the "funnel"). Failures may trigger bounded format or semantic repair, not unconstrained regeneration.
7. **Finalize** - join resolution (enumerated candidates, with one bounded LLM disambiguation call when several paths remain) and parameter binding for executable SQL.
8. **Execute** - generated SQL passes dialect AST validation, schema alignment, EXPLAIN, and cost gates before any warehouse run.

Programmatic callers observe **suspend** steps (`step.done == false`) and **terminal** steps (`step.done == true`) via `SessionStep.kind` ([Integrator guide - The session contract](INTEGRATOR_GUIDE.md#the-session-contract-suspend-and-terminal-steps)).

### Federated turns

After a validated intent against the unified graph, a multi-source question adds stages instead of a different pipeline:

1. **Decompose** (`plan.decompose`) - split the intent into per-source sub-intents, each validated against its member schema. Capability is the **intersection** of member dialect surfaces. One-source plans stay degenerate (no coordinator branch). Join enumeration caps differ by graph role: the composite scales shortest-path tie storage and merged candidate cross-products by member count, because more tables mean more reachable paths. Each member slice recomputed during decomposition uses the single-source defaults. A parent intent may therefore see join candidates the member slice does not; execution and validation use the member slice, which stays aligned with a standalone single-source plan for the same tables.
2. **Stage** - build a dependency graph: member stages, optional cross-member CTE stage, coordinator combine stage. Topological order governs execution; independent members may run in parallel waves.
3. **Generate** - render member SQL under each source's dialect and template store using the same deterministic renderer as a single-engine turn (a one-member federated plan is byte-identical to direct single-engine SQL). Cross-member CTE bodies compose through declared join edges, not ad hoc wrappers.
4. **Execute** - run members with optional semi-join reduction, per-source caps, and cooperative cancellation between stages. Member results arrive as Arrow or pandas frames at the coordinator.
5. **Combine** (`plan.combine`) - the in-process DuckDB coordinator registers member frames, streams them into join and residual operators, and releases each member frame once it is no longer needed so total resident bytes stay under the configured cap.
6. **Return** - structured **FederatedSqlBundle** on `SessionStep.federated_bundle`; `SessionStep.sql` is display glue only.

Prepared federated SQL is session-owned until execution or `PipelineSession.reset()`. Invalid declarations fail at `AetherFederation` construction - there is no separate dry-run entry point.

A federated answer reads each member at instants that are close but not identical; treat results as consistent only within the reported read window.

## 7. Learning model

Accepted turns fold into the partitioned template store under the active AetherSpace (keyed by normalized question text and space name). Future similar questions hit reuse before a full interpret/ground/compose cycle. Negative feedback is summarized and steers retry and template selection away from past mistakes.

On federation accepts, per-source templates land in member stores and a versioned **plan template** on the federation tree records how to replay the turn. Member templates are stamped `federation_plan_only` so they replay only through the plan record, not as standalone single-engine SQL reuse.

## 8. Concurrent sessions and the write queue

Reader sessions defer durable learning by appending structured events to `write_queue.jsonl`. A writer session on the same artifacts directory drains the queue automatically at the **start** of each writer turn under the artifacts lock. Readers never mutate the partitioned template files directly. Federation `clear_all_learning()` drains member queues before clearing federation and member template stores.

---

**See also:** [Getting started](GETTING_STARTED.md) | [User guide](USER_GUIDE.md) | [Integrator guide](INTEGRATOR_GUIDE.md) | [Sandbox guide](SANDBOX.md) | [API reference](API_REFERENCE.md) | [Security](SECURITY.md) | [Support matrix](SUPPORT_MATRIX.md) | [README](../README.md)
