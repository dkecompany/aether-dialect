# Integrator guide

How an engineer embeds `aetherdialect` in a service, job runner, notebook, or automation: the six core objects, the `SessionStep` contract, sync and async loops, reader/writer and owner/consumer deployment, federation and file-engine embedding, LLM wiring, and observability. Operator-facing semantics (notes, overrides, migration, warmup): [User guide](USER_GUIDE.md). Exact signatures, TOML keys, JSON shapes, and exception catalogue: [API reference](API_REFERENCE.md).

**Reading order:** [README](../README.md) -> [Getting started](GETTING_STARTED.md) -> [User guide](USER_GUIDE.md) -> this document -> [Sandbox guide](SANDBOX.md) -> [API reference](API_REFERENCE.md) -> [How it works](HOW_IT_WORKS.md) -> [Security](SECURITY.md) -> [Support matrix](SUPPORT_MATRIX.md).

## Sections

| Section | Contents |
| --- | --- |
| [Core objects](#core-objects) | Six constructible types and role/mode knobs |
| [Engine storage](#engine-storage) | Artifacts directory and connection slug |
| [The session contract: suspend and terminal steps](#the-session-contract-suspend-and-terminal-steps) | `SessionStep`, `done`, `kind`, `reply_shape` |
| [Minimal embedding (sync)](#minimal-embedding-sync) | Smallest correct `ask` / `step` loop |
| [Async embedding](#async-embedding) | Same loop with `AsyncPipelineSession` |
| [Reader and writer split](#reader-and-writer-split) | Multi-process learning queue |
| [Multi-user deployment](#multi-user-deployment) | Owner/consumer on shared artifacts |
| [Embedding a federation](#embedding-a-federation) | Composite `AetherFederation` embedding |
| [Embedding file uploads (CSV/Excel)](#embedding-file-uploads-csvexcel) | File-engine construction and validation |
| [LLM provider wiring](#llm-provider-wiring) | Azure two-slot mapping and OpenAI note |
| [Observability](#observability) | Audit, diagnostics, config snapshot |

---

## Core objects

Meet the six public types in composition order: each scopes the next. Operator semantics for allow/deny lists, notes, and spaces live in the [User guide](USER_GUIDE.md); this section is what an integrator constructs and wires.

### AetherEngine

Root facade for one database connection. Construction reflects the catalog (or loads cache), builds or loads the master schema graph, and stamps artifact fingerprints. One instance binds exactly one database. Operator overview: [User guide - AetherEngine](USER_GUIDE.md#aetherengine).

### EngineContext

Frozen scope passed into `AetherEngine(...)`. Controls which relations enter the graph, master notes/DDL paths, and execution-time allow/deny lists (`allow_objects`, `deny_objects`, `allow_columns`, `deny_columns`). The implicit first scope is always **master**. Saved scope presets are registered with `engine.engine_context(name, context)` and later selected by passing the **preset name string** to `AetherEngine(...)`. Operator detail: [User guide - EngineContext](USER_GUIDE.md#enginecontext).

### AetherFederation

Composite facade over named member `AetherEngine` instances. Each member keeps its own artifact tree; the federation tree lives beside them. Construct with member engines plus `declaration_file=` pointing at authored `federation_declaration.json` - there are no inline `joins=` or `mappings=` constructor arguments. Use `fed.session(...)` for the same `PipelineSession` / `AsyncPipelineSession` contract as a single engine. Declaration format: [Sandbox - Federation declaration format](SANDBOX.md#federation-declaration-format). Operator overview: [User guide - AetherFederation](USER_GUIDE.md#aetherfederation).

### FederationContext

Optional composite scope on `AetherFederation(..., context=...)`. Same allow/deny shape as `EngineContext`, plus optional `notes_file` on the master federation context. Qualified `source.table.column` tokens resolve member ownership; bare names that match more than one member raise. Operator detail: [User guide - FederationContext](USER_GUIDE.md#federationcontext).

### AetherSpace

Named knowledge partition over the master graph on an engine or federation. Define with `engine.aetherspace(name, space_context=...)` or the federation equivalent - put notes on `SpaceContext(notes_file=...)`, not a loose `notes_file=` kwarg. Select at session open: `engine.session(space="catalog")` or `fed.session(space="catalog")`. Template learning is partitioned by space name. Operator overview: [User guide - AetherSpace](USER_GUIDE.md#aetherspace).

### SpaceContext

Frozen allow/deny lists for a named space: `tables`, `columns`, `deny_objects`, `deny_columns`, and optional `notes_file`. Applied at question time for knowledge and template partitioning only - not at SQL execution. All `SpaceContext` instances are created on the master engine or federation context. Operator detail: [User guide - SpaceContext](USER_GUIDE.md#spacecontext).

### Engine role and session mode

| Knob | Values | Effect |
| --- | --- | --- |
| `AetherEngine(..., role=...)` / `AetherFederation(..., role=...)` | `owner` (default), `consumer` | **Engine role:** owner builds/mutates artifacts; consumer pins the owner snapshot. |
| `engine.session(mode=...)` / `fed.session(mode=...)` | `writer` (default), `reader` | **Session mode:** writer persists learning; reader enqueues to `write_queue.jsonl`. |

---

## Engine storage

The objects above persist under a shared root so owner, consumer, reader, and writer processes can point at the same learning:

`<artifacts_parent>/aetherdialect/<connection_slug>/`

- **`<artifacts_parent>`** - expanded `artifacts_dir` argument, or the platform user-data directory when omitted.
- **`<connection_slug>`** - deterministic from **database connection parameters** (engine, host, database, and similar). User names and role names do not appear in the slug, so shared learning and consumer deployments stay aligned.

Each database connection gets its own directory and its own master context snapshot. Named engine contexts and AetherSpaces persist as JSON sidecars under the same tree. Federation composites use `fed_<federation_id>/` beside member `conn_<slug>/` trees as parallel siblings under the same artifacts root. Layout detail: [How it works - Engine storage](HOW_IT_WORKS.md#3-engine-storage-and-fingerprints).

### Artifacts are library-owned

The artifacts directory is **library-owned**: never edit, move, or partially restore files under `<artifacts_parent>/aetherdialect/` by hand. User-facing changes go through export, edit in the working directory, then apply:

| Change | Export | Apply |
| --- | --- | --- |
| Schema overrides | `export_schema_overrides()` | `apply_schema_overrides()` |
| Federation declaration | `export_federation_declaration()` | `apply_federation_declaration()` |
| Aetherspace snapshot | `export_aetherspace(name)` | `apply_aetherspace(name)` |
| Migration decisions | `preview_migration_map()` | `apply_migration_map()` |

On a federation, member schema overrides use `export_schema_overrides(connection_name)` / `apply_schema_overrides(connection_name)`. Cross-source identifier drift uses `apply_migration_map(path="federation_migration_map.json")` after editing the map the engine wrote to the working directory.

---

## The session contract: suspend and terminal steps

Every `session.ask(...)` or `session.step(...)` returns a **`SessionStep`**. Your middle layer should branch on **`step.kind`** and **`step.done`** - not on parsing `step.prompt` text.

### Terminal vs suspended

| State | `step.done` | What to do |
| --- | --- | --- |
| **Suspended** | `False` | Pipeline waits for user input. Render `step.prompt` (and optional `step.message` / SQL preview), collect a reply, call `session.step(reply)`. |
| **Terminal** | `True` | Turn finished. Read `step.sql`, `step.data`, or `step.error`. No further `step` call for this turn unless you `ask` a new question. |

Use **`step.reply_shape`** (`"yes_no"` or `"free_text"`) to choose UI controls while suspended.

### Public `kind` values

Inside the pipeline, suspend points carry an internal `state_id`. Before returning `SessionStep`, the engine maps each internal id to a stable public **`kind`** string (via `SUSPEND_ID_TO_SESSION_KIND`). Integrators should use **`kind` only** - the internal ids are not part of the public contract.

| Public `kind` | Typical suspend | `reply_shape` | Meaning |
| --- | --- | --- | --- |
| `awaiting_intent_confirm` | intent confirmation | `yes_no` | Confirm interpreted plan. |
| `awaiting_intent_feedback` | intent rejection | `free_text` | Supply reason intent is wrong. |
| `awaiting_sql_confirm` | SQL preview / direct reuse | `yes_no` | Confirm generated or reused SQL. |
| `execute` | execute gate | `yes_no` | Confirm running stored SQL. |
| `awaiting_sql_feedback` | SQL rejection | `free_text` | Supply reason SQL is wrong. |
| `result` | - | - | Terminal success (`step.done == True`). |
| `error` | - | - | Terminal failure (`step.done == True`, `step.error` set). |
| `idle` | - | - | Session reset / no active turn. |

Direct template reuse suspends with `kind="awaiting_sql_confirm"` (same as a normal SQL preview) - there is no separate public reuse API or kind.

### SessionStep fields (embedding checklist)

| Field | Use |
| --- | --- |
| `done` | Loop until `True`. |
| `kind` | Branch UI logic ([table above](#public-kind-values)). |
| `prompt` | Short line to show before collecting input. |
| `reply_shape` | `"yes_no"` vs `"free_text"` while suspended. |
| `message` | Optional narrative (also printed by `run_interactive`). |
| `sql` / `data` | Preview or final result. |
| `federated_bundle` | Structured per-member statements on federated turns (display `sql` is glue only). |
| `federation_source_id` / `federation_phase` / `federation_limit_key` / `federation_succeeded` | Federation error attribution on terminal failure steps. |
| `error` | Terminal error text. |
| `diagnostics` | Turn-level tracing rows. |

Full field list: [API reference - SessionStep](API_REFERENCE.md#sessionstep).

---

## Minimal embedding (sync)

The smallest correct loop uses the contract above:

```python
from aetherdialect import EngineContext, AetherEngine

engine = AetherEngine(
    EngineContext(),
    artifacts_dir="./my_run",
    config_file="./aetherdialect.toml",
)

with engine.session() as session:
    step = session.ask("Top 5 customers by total payment?")
    while not step.done:
        if step.reply_shape == "yes_no":
            reply = "y"
        else:
            reply = "count distinct customers only"
        step = session.step(reply)

    if not step.error:
        print(step.sql)
        print(step.data)
```

Helper methods:

- `session.ask_until_done(question, on_confirm="y")` - auto-answers **yes/no** suspends only; raises on free-text suspends.
- `session.accept_until_done(question, ...)` - auto-answers both yes/no and free-text suspends until the turn ends.

Pass `on_confirm` as `"y"` or `"n"` (string, not Python `or`).

---

## Async embedding

`AsyncPipelineSession` (from `engine.asession()`) runs the same contract on worker threads:

```python
async with engine.asession() as session:
    step = await session.ask("...")
    while not step.done:
        reply = await get_user_input(step)
        step = await session.step(reply)
```

Method table: [API reference - AsyncPipelineSession methods](API_REFERENCE.md#asyncpipelinesession-methods).

### Cancelling a federated turn

Call `session.cancel_active_federation_turn()` on the **same** `PipelineSession` (or `await session.cancel_active_federation_turn()` on `AsyncPipelineSession`) that owns the in-flight turn. It returns `True` only when that session has an active federated turn. Cancellation is cooperative: the coordinator observes it **between member stages or batches**. It does **not** interrupt an already-running database statement on a member.

---

## Reader and writer split

More than one process can share one artifacts directory when you split write responsibility:

- **Writer** (`mode="writer"`, default): persists learning directly. One active writer per artifacts directory is recommended.
- **Reader** (`mode="reader"`): enqueues learning to `write_queue.jsonl`. Many readers can overlap on the same artifacts path.

A writer turn drains the queue at start under the engine's writer lock. Queue path: `engine.write_queue_path`.

Offline demo: [Sandbox guide - Reader/writer queue](SANDBOX.md#readerwriter-queue).

---

## Multi-user deployment

Owner/consumer builds directly on the reader/writer split and the shared artifacts directory.

Pattern for **any supported database engine** (not PostgreSQL-specific):

1. **Owner writer** - one process with `role="owner"`, `mode="writer"`, full `EngineContext`, and durable `artifacts_dir` on shared storage. Builds the master graph and drains the write queue.
2. **Consumer readers** - processes with `role="consumer"`, a saved scope-preset **name string**, restricted database credentials matching their allow lists, and `mode="reader"` (or `"writer"` only on the owner).
3. **Align scope** - `EngineContext.allow_objects` / `allow_columns` must match each consumer's database grants.
4. **Optional AetherSpaces** - further narrow model focus per team or product surface without changing warehouse roles.

Readers share the owner's artifacts directory but never mutate it directly; they append queue events the owner drains.

```python
# Owner (any engine - fill config_file for yours)
owner = AetherEngine(
    EngineContext(allow_objects=frozenset({"orders", "customers"})),
    artifacts_dir="/shared/aether_artifacts",
    config_file="./aetherdialect.toml",
    role="owner",
)

# Consumer - saved scope-preset name only, matching DB login scope
consumer = AetherEngine(
    "analyst_scope",
    artifacts_dir="/shared/aether_artifacts",
    config_file="./aetherdialect.toml",
    role="consumer",
)
with consumer.session(mode="reader", space="master") as session:
    session.accept_until_done("How many orders last month?")
```

Practice the same pattern offline: [Sandbox guide - Owner vs consumer presets](SANDBOX.md#owner-vs-consumer-presets).

---

## Embedding a federation

Once a single-engine embedding loop works, the composite case is the same session contract over several member engines.

Register members with `AetherFederation` (preferred). Each member keeps its own artifact tree under `conn_<connection>/`; the federation tree lives at `fed_<federation_id>/`. On re-entry, member graphs reload from disk when hashes match before composition runs.

`clear_all_learning()` drains member write queues before clearing federation and member template stores.

```python
from aetherdialect import AetherEngine, AetherFederation

orders_db = AetherEngine(..., artifacts_dir="/artifacts")
catalog_db = AetherEngine(..., artifacts_dir="/artifacts")
fed = AetherFederation(
    "commerce",
    members={"orders": orders_db, "catalog": catalog_db},
    declaration_file="/path/to/federation_declaration.json",
    artifacts_dir="/artifacts",
)
with fed.session() as session:
    step = session.ask("...")
    while not step.done:
        step = session.step(reply_for(step))
```

Cross-source joins and logical mappings are declared in `federation_declaration.json` passed to `declaration_file=` at construction. The planner decomposes multi-source intents into per-source sub-intents (each reuses that source's template store) and combines frames in a DuckDB coordinator. Execution attaches a structured bundle on `SessionStep.federated_bundle`; display SQL on `SessionStep.sql` is glue only.

### End-to-end walkthrough

1. **Named connections** - in `aetherdialect.toml`, give each database its own connection block (or use separate env profiles). Set `[engine] connection` per `AetherEngine(..., connection="name")` when multiple databases share one engine type.
2. **Member engines** - construct one `AetherEngine` per connection with a shared `artifacts_dir`. Each member gets `conn_<slug>/` under that root. Wait for first construction (profiling) on every member before composing.
3. **Author declaration** - write `federation_declaration.json` (`federation_id`, `cross_source_joins`, `coordinator` caps, optional `logical_tables` / `logical_columns`). Resolve table-name collisions with `aliases` or logical mappings before construction - colliding bare names raise at compose time.
4. **Compose** - `AetherFederation(name, members={...}, declaration_file=..., artifacts_dir=...)`. The federation `name` must equal `federation_id` in the declaration. Export the authored shape with `export_federation_declaration()` when you need to review or edit it.
5. **First question** - `with fed.session() as session:` then the same `ask` / `step` loop as a single engine. Confirm intent, then execute. Inspect `step.federated_bundle` for per-member SQL; `step.sql` is display-only.
6. **Re-entry** - use the export/apply pairs above; never edit persisted sidecars directly. For federation logical tables and joins, `export_federation_declaration()` into the working directory, edit `federation_declaration.json`, then `apply_federation_declaration()`. Member catalog drift uses `preview_migration_map()` / `apply_migration_map()` on each member engine; cross-source identifier drift uses `apply_migration_map(path="federation_migration_map.json")` on the federation after editing the map in the working directory.
7. **Add or remove a member** - register a new `AetherEngine`, update the declaration JSON, delete the federation tree if `artifact_format_version` mismatches, and reconstruct. Plan templates prune when topology changes.

Offline recipe with bundled seeds: [Sandbox guide - Federation recipe](SANDBOX.md#federation-recipe).

### Federation errors

Import the exported federation exceptions from `aetherdialect`. Constructor keyword arguments and the inheritance chain: [API reference - Exceptions](API_REFERENCE.md#exceptions).

| Exception | When raised | Catch notes |
| --- | --- | --- |
| `FederationDeclarationError` | Manifest or mappings JSON is structurally invalid, contains unknown keys, or member references do not resolve at build time. | Subclass of `FederationConfigError`. |
| `FederationIneligibleError` | A validated intent cannot be decomposed into a federated plan (unsupported shape). | Often surfaces as a turn diagnostic without raising to the session loop. |
| `FederationRuntimeError` | Coordinator or member execution fails after planning (row caps, missing frames, glue errors). | Base for runtime failures. |
| `FederationPartialFailureError` | One member failed after others succeeded. | kwargs `source_id`, `phase`, `succeeded`, `retryable`. |
| `FederationInvariantError` | Prepared plan pins no longer match the live composite graph or plan id on session resume. | Subclass of `FederationConfigError`. |
| `FederationConfigError` | Base for declaration and invariant failures; also raised for corrupt federation plan template files. | Base type for configuration failures. |
| `FederationMemberExecutionError` | One member's query failed during execution (`source_id`, `phase`). | Catch by type. Attribute the failure to that database; the turn may be retryable when the cause is a `RetryableError`. |
| `FederationCapExceededError` | A federated row, byte, or timeout cap was exceeded (`limit_key`, optional `source_id`). | Catch by type. Narrow or re-scope the question - retrying the same turn will not help. |

Per-source `limits.row_cap` is enforced after each member fetch (exceeding the cap raises `FederationCapExceededError` with `limit_key="row_cap"`). Per-source `limits.timeout_ms` is passed through to the member result backend on fetch (timeouts surface as `StatementTimeoutError` / retryable failures depending on the driver). Coordinator `semijoin_key_cap` bounds semi-join reduction. The execute confirmation gate states how many member databases a federated plan spans. Member template feedback is keyed with member-scoped `q_norm` (`source_id::question`) so federation learning does not collide with standalone reuse on the same member store.

### Degenerate single-member plans

When a federated plan touches only one member source, the coordinator is bypassed and the member SQL is executed directly. Rendered SQL is byte-identical to the standalone path for the same intent on that member schema. Two deliberate differences remain on the prepare path:

| Aspect | Standalone | Degenerate federation prepare |
| --- | --- | --- |
| `member_source_id` | `None` (unscoped) | `None` (unscoped; same as standalone) |
| `persist_template_learning` | caller-controlled (default `True`) | always `False` during prepare; persistence is deferred until the federated turn completes successfully |

Multi-member plans pass `member_source_id=<source_id>` so template and feedback keys are scoped with `source_id::question`. Degenerate plans omit that scope because only one member is involved and the SQL is equivalent to standalone. Integrators that compare template stores across standalone and degenerate federation turns should expect prepare-time learning to be deferred, not absent.

Signatures and JSON shapes: [API reference - AetherFederation](API_REFERENCE.md#aetherfederation).

---

## Embedding file uploads (CSV/Excel)

When `AETHERDIALECT_ENGINE` is `csv` (the TOML `[excel]` section is an alias that also selects the CSV file engine), uploads are validated before DuckDB load. **Inspect first, then construct** — do not rely on construction alone to surface layout choices.

| Severity | Meaning | Outcome |
| --- | --- | --- |
| **Advisory** | One correct interpretation exists and was applied | Inspection succeeds; issue is recorded; construction may proceed |
| **Review** | More than one defensible interpretation exists | Construction does not proceed on a guess; caller must choose |
| **Blocking** | File is readable but no coherent table can be derived | `inspect_tabular_upload` returns `ok=False`; construction raises `ConfigError` with the report |
| **Fatal** | File cannot be read, or the format is unsupported | `inspect_tabular_upload` raises `ConfigError` |

Ragged rows, duplicate headers, blank leading rows, and mixed column types are **Advisory**. Multiple candidate tables on one sheet, uncertain header rows, and append-region mismatches are **Review**. Empty files and sheets without a usable header are **Blocking**. Corrupt, encrypted, embedded-object, and unsupported extensions are **Fatal**.

```python
from aetherdialect import AetherEngine, EngineContext, inspect_tabular_upload

path = "/uploads/items.xlsx"
report = inspect_tabular_upload(path)  # file object or bytes also accepted
if report.requires_review:
    # Present report.narrative, report.issues, and report.suggested_selections.
    selections = {"items.xlsx": report.suggested_selections.get("items.xlsx", {})}
else:
    selections = {}

engine = AetherEngine(
    EngineContext(),
    artifacts_dir="./my_run",
    config_file="./aetherdialect.toml",
    source_selections=selections,
)
```

`inspect_tabular_upload` reads and analyses the grid only — it does not open DuckDB or construct an engine. When **Review** issues remain and you pass no `source_selections`, construction raises `ConfigError` with the inspection report attached rather than silently picking a region.

Successful construction emits `audit_sink` event `data_quality` and exposes `engine.data_quality_report` for post-hoc **Advisory** display. Each worksheet becomes its own table; multi-sheet workbooks use `file_stem__sheet_name` relation names. The CSV file engine accepts `.csv` and `.xlsx` (not `.xls`).

Model-assisted upload interpretation (sampled cell content) is gated by `PolicyConfig.TABULAR_LLM_ASSIST` — disable flag and deterministic-only behaviour: [Security - Upload inspection](SECURITY.md#58-upload-inspection-csv-file-engine). Operator detail: [User guide - CSV and Excel uploads](USER_GUIDE.md#csv-and-excel-uploads).

---

## LLM provider wiring

When `AETHERDIALECT_LLM_PROVIDER` is `azure`, provision exactly two deployments and map them through TOML or environment variables:

| Slot | Env var | Deploy this model |
| --- | --- | --- |
| `light` | `AZURE_OPENAI_DEPLOYMENT_LIGHT` | `gpt-5-mini` |
| `heavy` | `AZURE_OPENAI_DEPLOYMENT_HEAVY` | `gpt-5.4-mini` |

The library routes internal task classes to these slots automatically. OpenAI direct mode selects finer-grained models per task without operator configuration.

Optional offline corpus generation on OpenAI can use the Batch API at half token cost when `AETHERDIALECT_LLM_BATCH_ENABLED=true` (seed clarification, QSim intent fill, warmup paraphrase generation).

Full key tables: [API reference - Configuration](API_REFERENCE.md#configuration).

---

## Observability

| Channel | Access | Granularity |
| --- | --- | --- |
| **`audit_sink`** | Constructor callback on `AetherEngine` / `AetherFederation` | Coarse lifecycle (`init`, `ask_begin`, `ask_done`, queue drain events). |
| **`SessionStep.diagnostics`** | Every `ask` / `step` return | Turn-level codes (`REUSE_HIT`, `COMPOSE_REPAIR`, `SENSITIVITY_GATE_HIT`, ...). |
| **`engine.show_config()`** / **`fed.show_config()`** | Method call | Redacted config snapshot. |

Diagnostic catalog: [API reference - Observability](API_REFERENCE.md#observability).

---

**See also:** [Getting started](GETTING_STARTED.md) | [User guide](USER_GUIDE.md) | [Sandbox guide](SANDBOX.md) | [API reference](API_REFERENCE.md) | [How it works](HOW_IT_WORKS.md) | [Security](SECURITY.md) | [Support matrix](SUPPORT_MATRIX.md) | [README](../README.md)
