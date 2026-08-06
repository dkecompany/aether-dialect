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
| [Reader and writer split](#reader-and-writer-split) | Multi-process reader vs writer sessions |
| [Multi-user deployment](#multi-user-deployment) | Owner/consumer on shared artifacts |
| [Embedding a federation](#embedding-a-federation) | Composite `AetherFederation` embedding |
| [Embedding file uploads (CSV/Excel)](#embedding-file-uploads-csvexcel) | File-engine construction and validation |
| [LLM provider wiring](#llm-provider-wiring) | Azure two-slot mapping and OpenAI note |
| [Observability](#observability) | Audit, diagnostics, config snapshot |
| [Guarantees](#guarantees) | Determinism, replay limits, federation numerics, cost caps, configuration boundary |

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

Optional composite scope on `AetherFederation(..., context=...)`. Same allow/deny shape as `EngineContext`, plus optional `notes` or `notes_file` on the master federation context (set at most one). Qualified `source.table.column` tokens resolve member ownership; bare names that match more than one member raise. Operator detail: [User guide - FederationContext](USER_GUIDE.md#federationcontext).

### AetherSpace

Named knowledge partition over the master graph on an engine or federation. Define with `engine.aetherspace(name, space_context=...)` or the federation equivalent - put notes on `SpaceContext(notes=...)` or `SpaceContext(notes_file=...)`, not a loose unpaired kwarg unless you intentionally override. Select at session open: `engine.session(space="catalog")` or `fed.session(space="catalog")`. Template learning is partitioned by space name. Operator overview: [User guide - AetherSpace](USER_GUIDE.md#aetherspace).

### SpaceContext

Frozen allow/deny lists for a named space: `tables`, `columns`, `deny_objects`, `deny_columns`, and optional `notes` or `notes_file` (set at most one). Applied at question time for knowledge and template partitioning only - not at SQL execution. All `SpaceContext` instances are created on the master engine or federation context. Operator detail: [User guide - SpaceContext](USER_GUIDE.md#spacecontext).

### Engine role and session mode

| Knob | Values | Effect |
| --- | --- | --- |
| `AetherEngine(..., role=...)` / `AetherFederation(..., role=...)` | `owner` (default), `consumer` | **Engine role:** owner builds/mutates artifacts; consumer pins the owner snapshot. |
| `engine.session(mode=...)` / `fed.session(mode=...)` | `writer` (default), `reader` | **Session mode:** writer persists learning; reader does not persist shared learning (session-local only). |

---

## Engine storage

The objects above persist under a shared root so owner, consumer, reader, and writer processes can point at the same learning:

`<artifacts_parent>/aetherdialect/<connection_slug>/`

- **`<artifacts_parent>`** - expanded `artifacts_dir` argument, or the platform user-data directory when omitted.
- **`<connection_slug>`** - deterministic from **database connection parameters** (engine, host, database, and similar). User names and role names do not appear in the slug, so shared learning and consumer deployments stay aligned.

Each database connection gets its own directory and its own master context snapshot. Named engine contexts and AetherSpaces persist as JSON sidecars under the same tree. Federation composites use `fed_<federation_id>/` beside member `conn_<slug>/` trees as parallel siblings under the same artifacts root. Layout detail: [How it works - Engine storage](HOW_IT_WORKS.md#3-engine-storage-and-fingerprints).

The artifacts parent directory must reside on a **local filesystem** so advisory artifact locks can serialize cooperating processes reliably; network-mounted paths may emit an `ARTIFACTS_DIR_NOT_LOCAL` diagnostic at construction.

### Artifacts are library-owned

The artifacts directory is **library-owned**: never edit, move, or partially restore files under `<artifacts_parent>/aetherdialect/` by hand. User-facing changes go through export, edit in the working directory, then apply:

| Change | Export | Apply |
| --- | --- | --- |
| Schema overrides | `export_overrides()` | `apply_overrides()` |
| Federation declaration | `export_federation_declaration()` | `apply_federation_declaration()` |
| Aetherspace snapshot | `export_aetherspace(name)` | `apply_aetherspace(name)` |
| Migration decisions | `preview_migration_map()` | `apply_migration_map()` |

On a federation, member schema overrides use `export_overrides(connection_name)` / `apply_overrides(connection_name)`. Cross-source identifier drift uses `apply_migration_map(path="federation_migration_map.json")` after editing the map the engine wrote to the working directory.

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
| `awaiting_reuse_confirm` | direct template reuse | `yes_no` | Confirm a matched template reuse hit. |
| `awaiting_sql_confirm` | post-execution SQL preview | `yes_no` | Confirm generated SQL after execution. |
| `execute` | execute gate | `yes_no` | Confirm running stored SQL. |
| `awaiting_sql_feedback` | SQL rejection | `free_text` | Supply reason SQL is wrong. |
| `result` | - | - | Terminal success (`step.done == True`). |
| `meta` | - | - | Terminal metadata answer (`step.done == True`); inspect `step.meta_payload`. No confirm loop. |
| `error` | - | - | Terminal failure (`step.done == True`, `step.error` set). |
| `idle` | - | - | Session reset / no active turn. |

Direct template reuse suspends with `kind="awaiting_reuse_confirm"`; post-execution SQL preview uses `kind="awaiting_sql_confirm"`.

### Template replay vs fresh generation

Generation paths **1** (exact question reuse) and **2.x** (fuzzy question reuse with parameter extraction) pin the stored template SQL and its recorded join signature (`chosen_join_candidate_id` / `chosen_join_path_signature`). They rebind literal and structural parameters against the template's stored `param_values` schema; missing or wrongly typed extractions abort reuse and fall through to fresh generation.

**Equality with a cold generation path is not promised.** A replayed turn may differ in SQL text, bind map, or join choice from what the full interpret/ground/compose pipeline would produce today for the same question. Integrators should treat replay as "execute this trusted template with new binds," not as a deterministic duplicate of path 3+ generation.

After a successful analytical turn, store `step.template_id` and re-run later with `engine.execute_template(step.template_id, {b.handle: b.current_value for b in step.parameters})` — no second `ask()`. Template ids stay stable across additive schema changes when the template footprint still survives (see template footprint migration). `AetherFederation` exposes the same `list_templates` / `fetch_template` / `execute_template` trio against the federation artifact store.

### SessionStep fields (embedding checklist)

| Field | Use |
| --- | --- |
| `done` | Loop until `True`. Meta turns finish in one step (`done=True`) with no confirm loop. |
| `kind` | Branch UI logic ([table above](#public-kind-values)). |
| `prompt` | Short line to show before collecting input. |
| `reply_shape` | `"yes_no"` vs `"free_text"` while suspended. |
| `message` | Optional narrative (also printed by `run_interactive`). |
| `sql` | Dialect parameterized SQL: `str` on single-engine / one-member federation; `dict[str, str]` (`source_id` → SQL) on multi-member federation; `None` when absent. Branch with `isinstance(step.sql, dict)`. |
| `data` | Preview or final result frame. |
| `parameters` | P-params only (`p1`, `p2`, …) with `handle`, `current_value`, `display_name`, `column_expr`, `upper_handle`, `unit_handle`. Present whenever `sql` is set and a template supplies slots. |
| `template_id` | Stable template id for agent cache → `execute_template`. |
| `meta_payload` | Structured meta answer when `kind="meta"`. Schema catalog includes `counts` (and may omit detail `tables`); business knowledge is `{response_kind: "business_knowledge"}` with prose in `message`. |
| `federated_bundle` | Optional timings/row counts; prefer `step.sql` for member SQL text. |
| `federation_source_id` / `federation_phase` / `federation_limit_key` / `federation_succeeded` | Federation error attribution on terminal failure steps. |
| `error` | Terminal error text. |
| `status` | On terminal steps: coarse outcome (`permission_denied`, `restricted`, `validation_failed`, ...). Default owner warehouse failures do not surface contact-admin text; RBAC scope denials do. Learning/reuse is role-agnostic; scope gates run at execute. |
| `refusal_code` | Back-compat alias for the primary refusal code; inspect `diagnostics` for the full refusal catalogue. |
| `retryable` | On terminal failure steps: whether the same question may be retried. |
| `notices` | Structured bookkeeping notices (`turn_saved`, `feedback_noted`, ...). |
| `data_truncated` | `True` when `data` was trimmed to the configured row cap. |
| `diagnostics` | Turn-level tracing rows, including structured refusal codes. |

Full field list: [API reference - SessionStep](API_REFERENCE.md#sessionstep).

### Ask routes (caller still only calls `ask`)

Validation classifies the question internally. Integrators never pass a route flag — branch on the returned step:

| Outcome | How it shows up |
| --- | --- |
| Analytical | Existing SQL suspend/confirm loop; terminal `kind="result"` with `sql` / `data` / `template_id`. |
| Schema catalog / business knowledge | One terminal `kind="meta"` step (`done=True`); no confirm loop. Read `message` and `meta_payload`. |
| Restricted / invalid | Terminal refuse (`kind="error"` / validation status); no SQL pipeline. |

### Knowledge export (agent context without asking)

Use `export_knowledge()` for the engine/federation plus per-space business-knowledge wrapper. Use `export_space_knowledge(space=...)` for one space’s BK only. Use `export_metadata(space=...)` for deterministic table/column inventory (and federation `members` when present). Prefer `ask()` meta turns when the user asked in natural language; prefer export when the agent only needs structured inventory.

### Permission denial

When the database or execution scope blocks a turn, detect denial without importing message constants:

| Signal | How to detect | Action |
| --- | --- | --- |
| Synchronous API | `except AccessError as exc` during `preview_table`, validation, or direct execute helpers | Map to your product's access-denied UX; read `exc.operation`. |
| Session turn | `step.done` and `step.status == "permission_denied"` | Show `step.message` to the user; branch on `step.refusal_code == "refusal_not_available_in_context"` for structured analytics. |

Do not compare against exported string constants for denial text. `SchemaAccessError` remains the init-time access surface; runtime execute/EXPLAIN denial is `AccessError`.

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

Call `session.cancel()` on the **same** `PipelineSession` (or `await session.cancel()` on `AsyncPipelineSession`) that owns the in-flight turn. It returns `True` only when that session has an active turn. Cancellation is cooperative: the coordinator observes it **between member stages or batches** and also cancels any in-flight database statement on a member. `cancel_active_federation_turn()` remains as a deprecated alias.

---

## Reader and writer split

More than one process can share one artifacts directory when you split write responsibility:

- **Writer** (`mode="writer"`, default): persists learning directly. One active writer per artifacts directory is recommended.
- **Reader** (`mode="reader"`): does not persist shared learning (session-local only). Many readers can overlap on the same artifacts path.

Offline demo: [Sandbox guide - Reader/writer sessions](SANDBOX.md#readerwriter-sessions).

---

## Multi-user deployment

Owner/consumer builds directly on the reader/writer split and the shared artifacts directory.

Pattern for **any supported database engine** (not PostgreSQL-specific):

1. **Owner writer** - one process with `role="owner"`, `mode="writer"`, full `EngineContext`, and durable `artifacts_dir` on shared storage. Builds the master graph and persists learning.
2. **Consumer readers** - processes with `role="consumer"`, a saved scope-preset **name string**, restricted database credentials matching their allow lists, and `mode="reader"` (or `"writer"` only on the owner).
3. **Align scope** - `EngineContext.allow_objects` / `allow_columns` must match each consumer's database grants.
4. **Optional AetherSpaces** - further narrow model focus per team or product surface without changing warehouse roles.

Readers share the owner's artifacts directory but never mutate it directly.

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

Practice the same pattern offline: [Sandbox guide - Owner vs consumer roles](SANDBOX.md#owner-vs-consumer-roles).

### Restricted environments

When `artifacts_dir` is omitted, the library uses the platform user-data directory as the artifacts parent. If that default directory is not writable, construction raises `ConfigError` naming the attempted path and stating that an explicit `artifacts_dir` is required:

```text
default artifacts directory '<path>' is not writable; set an explicit artifacts_dir
```

Federation coordinator spill and DuckDB temp files are created under the federation artifact tree when one exists, or under the system temporary directory otherwise. If that coordinator temporary directory is not writable, construction raises `ConfigError` naming the directory:

```text
coordinator temporary directory '<path>' is not writable; set FederationLimits.coordinator_temp_dir or ensure the system temporary directory is writable
```

Set `artifacts_dir` and, when needed, `FederationLimits.coordinator_temp_dir` to writable local paths before embedding in locked-down service accounts, containers, or CI sandboxes.

---

## Embedding a federation

Once a single-engine embedding loop works, the composite case is the same session contract over several member engines.

Register members with `AetherFederation` (preferred). Each member keeps its own artifact tree under `conn_<connection>/`; the federation tree lives at `fed_<federation_id>/`. On re-entry, member graphs reload from disk when hashes match before composition runs.

`clear_all_learning()` clears federation and member template stores (and related learning artifacts).

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

Cross-source joins and logical mappings are declared in `federation_declaration.json` passed to `declaration_file=` at construction. The planner decomposes multi-source intents into per-source sub-intents (each reuses that source's template store) and combines frames in a DuckDB coordinator. Multi-member turns set `SessionStep.sql` to a `dict` of member SQL; prefer that over reading `federated_bundle` for text.

### End-to-end walkthrough

1. **Named connections** - in `aetherdialect.toml`, give each database its own connection block (or use separate env profiles). Set `[engine] connection` per `AetherEngine(..., connection="name")` when multiple databases share one engine type.
2. **Member engines** - construct one `AetherEngine` per connection with a shared `artifacts_dir`. Each member gets `conn_<slug>/` under that root. Wait for first construction (profiling) on every member before composing.
3. **Author declaration** - write `federation_declaration.json` (`federation_id`, `cross_source_joins`, `coordinator` caps, optional `logical_tables` / `logical_columns`). Resolve table-name collisions with `aliases` or logical mappings before construction - colliding bare names raise at compose time.
4. **Compose** - `AetherFederation(name, members={...}, declaration_file=..., artifacts_dir=...)`. The federation `name` must equal `federation_id` in the declaration. Export the authored shape with `export_federation_declaration()` when you need to review or edit it.
5. **First question** - `with fed.session() as session:` then the same `ask` / `step` loop as a single engine. Confirm intent, then execute. On multi-member turns `isinstance(step.sql, dict)` is true (`source_id` → member parameterized SQL); one-member degenerate turns keep `step.sql` as `str`. `step.federated_bundle` remains available for timings/row counts.
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

## Configuration boundary

The TOML config file and environment variables carry **connection identity** only: engine selection, hosts, ports, databases, schemas, warehouses, catalogs, roles, users, passwords, tokens, key paths, LLM provider and API keys, named connection blocks, artifacts roots, and file-engine source paths.

**Behaviour** — pool sizing, timeouts, caps, retention counts, and non-security policy flags — is set only through `EngineLimits` and `FederationLimits` constructor arguments, or explicitly via `EngineLimits.from_config_file(path)` and `FederationLimits.from_config_file(path)` reading `[limits]` and `[federation_limits]` from the same TOML document. A limit value you pass to the constructor is never overlaid by the file or the environment.

Legacy behaviour environment variables (`AETHERDIALECT_STATEMENT_TIMEOUT_MS`, `AETHERDIALECT_MAX_QUERY_COST_ROWS`, and the other keys listed in [API reference - Configuration](API_REFERENCE.md#configuration)) are ignored; when still set they emit diagnostic `CONFIGURATION_KEY_IGNORED` naming the replacement field.

| Side | Examples |
| --- | --- |
| Connection identity (env / TOML) | `[postgresql].host`, `PGHOST`, `[engine].selected`, `OPENAI_API_KEY`, `[llm].provider`, `CSV_DIRECTORY` |
| Behaviour (`EngineLimits` / `FederationLimits`) | `pool_size`, `statement_timeout_ms`, `max_result_rows`, `max_members`, `member_row_cap` |

```python
from aetherdialect import AetherEngine, EngineContext, EngineLimits

limits = EngineLimits.from_config_file("./aetherdialect.toml")
engine = AetherEngine(EngineContext(), artifacts_dir="./run", config_file="./aetherdialect.toml", limits=limits)
```

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
| **`audit_sink`** | Constructor callback on `AetherEngine` / `AetherFederation` | Coarse lifecycle (`init`, `ask_begin`, `ask_done`, `close`, and related admin events). |
| **`construction_phase_callback` / `ask_phase_callback`** | Constructor callbacks on `AetherEngine` / `AetherFederation` | Coarse `PhaseProgressEvent` transitions during construction and ask turns. |
| **`SessionStep.diagnostics`** | Every `ask` / `step` return | Turn-level codes (`REUSE_HIT`, `COMPOSE_REPAIR`, `SENSITIVITY_GATE_HIT`, structured refusals, ...). |
| **`engine.show_config()`** / **`fed.show_config()`** | Method call | Redacted config snapshot. |

Diagnostic catalog: [API reference - Observability](API_REFERENCE.md#observability).

---

## Guarantees

What the library promises integrators, and what it deliberately does not. For refused question shapes and per-engine capability limits, see the [Support matrix](SUPPORT_MATRIX.md).

### Deterministic ask rebuild

When the artifacts tree is unchanged and the resolved intent is the same, the **ask rebuild** path produces byte-identical SQL. This applies to fresh generation after intent confirmation — not to template replay (below).

### Template replay boundaries

**template reuse** on paths **1** and **2.x** pins the stored template SQL and its recorded **join signature**. Parameters rebind against the template's stored schema; replay is **not** promised to match a cold regenerate after schema enrichment or graph changes. Treat replay as executing a trusted template, not as a duplicate of today's full pipeline output.

### Federation numeric exactness

Exact numeric types remain exact through **federation egress** (coordinator glue and member fetch). Approximate numeric types may be widened to float at federation boundaries.

### EXPLAIN cost caps

Query cost caps are enforced only where the [Support matrix](SUPPORT_MATRIX.md) shows an active EXPLAIN cost gate for your engine. When the warehouse returns no row estimate, the gate is **fail-open**: the turn proceeds and a named diagnostic records the missing estimate.

### Host configuration vs library-owned artifacts

| You configure | Library-owned (do not hand-edit) |
| --- | --- |
| `EngineLimits` / `FederationLimits` | Files under `<artifacts_parent>/aetherdialect/` |
| Engine and federation contexts, AetherSpaces | Template stores, fingerprint sidecars |
| `audit_sink`, phase callbacks | Internal graph snapshots and learning partitions |

Export/apply pairs ([Artifacts are library-owned](#artifacts-are-library-owned)) are the supported edit surface.

### Unsupported constructs

Question shapes the engine refuses, and how to reformulate them, are listed in [SUPPORT_MATRIX](SUPPORT_MATRIX.md). Route user-facing capability questions there rather than inferring support from successful turns on other shapes.

---

**See also:** [Getting started](GETTING_STARTED.md) | [User guide](USER_GUIDE.md) | [Sandbox guide](SANDBOX.md) | [API reference](API_REFERENCE.md) | [How it works](HOW_IT_WORKS.md) | [Security](SECURITY.md) | [Support matrix](SUPPORT_MATRIX.md) | [README](../README.md)
