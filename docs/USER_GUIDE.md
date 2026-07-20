# User guide

**Reading order:** [README — Documentation](../README.md#documentation) → [Getting started](GETTING_STARTED.md) → this guide → [Integrator guide](INTEGRATOR_GUIDE.md) → [Sandbox guide](SANDBOX.md) → [API reference](API_REFERENCE.md) → [How it works](HOW_IT_WORKS.md) → [Security](SECURITY.md) → [Support matrix](SUPPORT_MATRIX.md).

Operator manual for analysts and application owners: scope, notes, overrides, asking questions, migration, and warmup. Setup walkthroughs live in [Getting started](GETTING_STARTED.md); programmatic embedding in the [Integrator guide](INTEGRATOR_GUIDE.md).

## Sections

| Section | Contents |
| --- | --- |
| [Configure](#configure) | Credentials and engine selection |
| [First run](#first-run) | Initial profiling wait |
| [EngineContext](#enginecontext) | Scope, allow/deny lists, execution boundary |
| [AetherSpace](#aetherspace) | Knowledge narrowing and per-space notes |
| [Sensitivity classification](#sensitivity-classification) | Skim — full detail in Security |
| [Asking a question](#asking-a-question) | Confirmations, reuse, feedback |
| [Notes file](#notes-file) | Domain vocabulary |
| [Schema overrides](#schema-overrides) | Descriptions, roles, sensitivity, usability |
| [Migration](#migration) | Catalog drift workflow |
| [Seed warmup](#seed-warmup) | Pre-populating templates |
| [Partition and cluster pruning](#partition-and-cluster-pruning) | Warehouse pruning hints |
| [Resetting learning](#resetting-learning) | Scoped clears |
| [Common pitfalls](#common-pitfalls) | Quality troubleshooting |

---

## Configure

Point your application at LLM and database credentials via TOML or process environment. The engine selection and connection fields vary by engine. Detailed key lists: [API reference — Configuration](API_REFERENCE.md#configuration). First-time wiring: [Getting started — Connect your warehouse](GETTING_STARTED.md#connect-your-warehouse).

## First run

The first time the engine connects to a database, it reflects the catalog, profiles every column in scope, classifies roles on usable columns, and writes a versioned snapshot under the engine storage directory. Subsequent constructions reuse this cache when fingerprints match.

| Schema size | Typical first construction |
| --- | --- |
| Small (~15 tables) | Tens of seconds to a few minutes |
| Moderate (~50 tables) | Several minutes |
| Large (hundreds of tables) | Much longer — use `allow_objects` |

Progress prints as `Profiling [i/n] <table>` on stdout during construction.

## EngineContext

`EngineContext` is the frozen scope object passed into `AetherEngine`. It controls which relations enter the graph, supplies domain notes and optional DDL, and enforces allow/deny lists at **execution time** (RBAC-like table and column visibility for the connection).

- **name** — `"master"` by default. The first context is always master; the master name cannot be removed or renamed away. Named contexts are subset specs stored beside the master graph; consumers load them by name string only.
- **include** — `"tables"` (default), `"views"`, or `"both"`. Only the master context may set include mode.
- **allow_objects** / **deny_objects** — optional table or view names to include or exclude from the graph.
- **allow_columns** / **deny_columns** — qualified `table.column` or `*.column` entries. Denied columns are removed from the in-memory graph entirely (stronger than sensitivity tiers).
- **notes_file** — path to domain notes on the master context only ([Notes file](#notes-file)).
- **sql_file** — optional DDL or annotated SQL on the master context only.

Each database connection gets its own engine storage directory keyed by a **connection slug** derived from database identity (host, database name, and similar) — not from user or role names. That keeps shared learning and consumer roles meaningful across logins to the same warehouse.

## AetherSpace

An **AetherSpace** narrows which tables and columns contribute **knowledge** for a session — an accuracy and focus aid, not a permission boundary. Database grants and `EngineContext` scope remain the execution boundary.

- **Master space** — implicit default (`space="master"`). Reflects the full master engine context graph. Cannot be deleted or redefined.
- **Named spaces** — defined on the owner engine with a `SpaceContext` (`tables`, `columns`, `deny_objects`, `deny_columns`). All named spaces are subsets of the master graph; they inherit shared schema learning from master construction and partition template learning by space name.
- **SpaceContext vs EngineContext** — `SpaceContext` applies the same allow/deny shape as engine scope but only at question time for knowledge and template partitioning. Named spaces are always created against the master engine context, never as an intersection with a non-master engine context.
- **Per-space notes** — when defining a named space, you may supply a separate notes file. The engine merges inherited master descriptions with space-specific refinements (see [Sandbox guide — Named AetherSpaces](SANDBOX.md#named-aetherspaces) for an offline demo).

## Sensitivity classification

Each column carries a sensitivity tier (`none`, `restricted`, or `hidden`). Full definitions, query-time rules, and LLM disclosure: [Security — Sensitivity tags](SECURITY.md#3-sensitivity-tags). In brief: **restricted** columns stay visible to the model but block bare row projections and literal-value filters; aggregations may still use them. **Hidden** columns are omitted from the questioning surface entirely.

## Asking a question

When the engine is not confident about a translation, it suspends the turn and asks for confirmation — intent readback, SQL preview, or execute confirmation depending on confidence and reuse path. Accept builds template learning; reject with a reason steers future turns away from the same mistake.

**Template reuse** is automatic — there is no separate “warm reuse” API:

1. **Direct reuse** — when new wording is within a small normalized token edit distance (at most two characters) of a stored question, the engine replays the stored SQL path with no full parse LLM calls.
2. **Intent-level match** — larger wording changes still require LLM calls even when structure is similar.

Ask normally; change wording and observe which path runs via diagnostics or audit events ([Integrator guide — Observability](INTEGRATOR_GUIDE.md#observability)).

The offline sandbox supports the same accept/reject mechanism as production; bundled mock coverage is finite ([Sandbox guide](SANDBOX.md)).

## Notes file

Domain notes refine descriptions and guide role classification. Use one or two sentences per important table, business definitions, join hints, and explicit sensitivity statements. The master context reads `EngineContext.notes_file`; each named AetherSpace may also supply its own notes file at define time to refine knowledge context.

## Schema overrides

Overrides live in one JSON file beside your working directory. Export a starter with the public API, edit descriptions, roles, sensitivity, foreign keys, or primary keys, then apply. Every override replays on cache invalidation.

- **usable** — you may set `"usable": false` to mark a column unusable. **Statistical omission** automatically marks columns unusable if they fail profiling gates (for example extreme null rate or sentinel-dominated distributions). You cannot flip usability back on for columns the profiler already marked unusable.
- **sensitivity** — assign `none`, `restricted`, or `hidden` per column ([Security](SECURITY.md#3-sensitivity-tags)).

Workflow detail: [API reference — Schema overrides JSON](API_REFERENCE.md#schema-overrides-json-schema_overridesjson).

## Migration

When the catalog changes structurally, the engine writes `schema_migration_map.json` and stops. Edit the map with one of:

- **remap** — rewrite identifiers in stored templates.
- **destructive** — clear templates and learning for a fresh slate.
- **abort** — stop initialization to investigate drift.

## Seed warmup

Warmup paths populate templates so common questions hit the cache sooner:

1. **Seed-question warmup** — parses and paraphrases a list of natural-language questions.
2. **SQL-history warmup** — reverse-engineers historical `SELECT` statements into intents.
3. **Query-log warmup** — reads warehouse system logs for historical queries.

Warmup and QSim are unavailable inside the offline sandbox (`ConfigError`).

## Partition and cluster pruning

When the compiled schema graph carries partition or clustering metadata and the parsed intent includes matching date or key filters, the engine may append `WHERE` predicates on those columns at SQL finalization. This reinforces warehouse pruning without inventing data ranges the user did not ask for. Engine-specific behavior: [Support matrix — Engine capabilities](SUPPORT_MATRIX.md#engine-capabilities).

## Resetting learning

Scoped resets clear persisted overrides, the template store, or simulation caches through public API methods on `AetherEngine`. See [API reference — AetherEngine methods](API_REFERENCE.md#aetherengine-methods).

## Common pitfalls

- **Wrong table chosen:** ensure your notes file clearly distinguishes similar business entities.
- **Related tables not joined:** confirm foreign keys exist or add them through overrides.
- **Column never appears:** check statistical usability, deny lists, and sensitivity tags.
- **AetherSpace confused with permissions:** narrowing a space changes model focus only; tighten `EngineContext` or database grants for access control.

---

**See also:** [Getting started](GETTING_STARTED.md) · [Integrator guide](INTEGRATOR_GUIDE.md) · [Sandbox guide](SANDBOX.md) · [API reference](API_REFERENCE.md) · [How it works](HOW_IT_WORKS.md) · [Security](SECURITY.md) · [Support matrix](SUPPORT_MATRIX.md) · [README](../README.md#documentation)
