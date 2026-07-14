# Integrator guide

Engineers embedding the engine in a service, job runner, notebook, or automation. Stable programmatic surfaces: `AetherEngine`, `PipelineSession`, and `AsyncPipelineSession`.

**Reading order:** [README — Documentation](../README.md#documentation) → [Getting started](GETTING_STARTED.md) → [User guide](USER_GUIDE.md) → this guide → [Sandbox guide](SANDBOX.md) → [API reference](API_REFERENCE.md) → [How it works](HOW_IT_WORKS.md) → [Security](SECURITY.md) → [Support matrix](SUPPORT_MATRIX.md).

## Sections

| Section | Contents |
| --- | --- |
| [Core objects](#core-objects) | Engine, contexts, spaces, roles, modes |
| [Engine storage](#engine-storage) | Artifacts layout and connection slug |
| [Suspend and terminal steps](#suspend-and-terminal-steps) | SessionStep contract and kind mapping |
| [Minimal embedding (sync)](#minimal-embedding-sync) | ask/step loop |
| [Async support](#async-support) | AsyncPipelineSession |
| [Reader and writer split](#reader-and-writer-split) | Queue semantics |
| [Multi-user deployment](#multi-user-deployment) | Owner/consumer on any engine |
| [Observability](#observability) | Audit, diagnostics, config snapshot |

Operator-facing semantics (notes, overrides, migration): [User guide](USER_GUIDE.md). Method signatures and JSON shapes: [API reference](API_REFERENCE.md).

---

## Core objects

Read this section before the session method tables below.

### AetherEngine

Root facade for one database connection. Construction reflects the catalog (or loads cache), builds or loads the master schema graph, and stamps artifact fingerprints. One instance binds exactly one database; artifacts live under `<artifacts_dir>/aetherdialect/<connection_slug>/`.

### EngineContext

Frozen scope at construction. Controls which relations enter the graph, master notes/DDL paths, and execution-time allow/deny lists (`allow_objects`, `deny_objects`, `allow_columns`, `deny_columns`). The implicit first context is always **`master`**; that name cannot be removed. Named engine contexts are subset specs stored beside the master graph — consumers pass a context **name string** only.

### SpaceContext and AetherSpace

**SpaceContext** defines a knowledge subset: `tables`, `columns`, `deny_objects`, and `deny_columns` (same RBAC-like shape as engine scope, but applied at question time only). **AetherSpace** is the persisted named instance. Define with `engine.aetherspace(name, space_context=..., notes_file=...)`. Select at session open: `engine.session(space="catalog")`.

Named spaces inherit master graph learning; template stores partition by space name. All SpaceContexts are created on the master engine context — never as an intersection with a non-master engine context.

### Roles and session modes

| Knob | Values | Effect |
| --- | --- | --- |
| `AetherEngine(..., role=...)` | `owner` (default), `consumer` | Owner builds/mutates artifacts; consumer pins owner snapshot. |
| `engine.session(mode=...)` | `writer` (default), `reader` | Writer persists learning; reader enqueues to `write_queue.jsonl`. |

---

## Engine storage

Versioned artifacts live under:

`<artifacts_parent>/aetherdialect/<connection_slug>/`

- **`<artifacts_parent>`** — expanded `artifacts_dir` argument, or platform user-data when omitted.
- **`<connection_slug>`** — deterministic from **database connection parameters** (engine, host, database, and similar). User names and role names do not appear in the slug so shared learning and consumer deployments stay aligned.

Each database connection gets its own directory and its own master context snapshot. Named engine contexts and AetherSpaces persist as JSON sidecars under the same tree. Layout detail: [How it works — Engine storage](HOW_IT_WORKS.md#3-engine-storage-and-artifact-lifecycle).

---

## Suspend and terminal steps

Every `session.ask(...)` or `session.step(...)` returns a **`SessionStep`**. Your middle layer should branch on **`step.kind`** and **`step.done`** — not on parsing `step.prompt` text.

### Terminal vs suspended

| State | `step.done` | What to do |
| --- | --- | --- |
| **Suspended** | `False` | Pipeline waits for user input. Render `step.prompt` (and optional `step.message` / SQL preview), collect a reply, call `session.step(reply)`. |
| **Terminal** | `True` | Turn finished. Read `step.sql`, `step.data`, or `step.error`. No further `step` call for this turn unless you `ask` a new question. |

Use **`step.reply_shape`** (`"yes_no"` or `"free_text"`) to choose UI controls while suspended.

### Public `kind` values

Inside the pipeline, suspend points carry an internal `state_id`. Before returning `SessionStep`, the engine maps each internal id to a stable public **`kind`** string (via `SUSPEND_ID_TO_SESSION_KIND`). Integrators should use **`kind` only** — the internal ids are not part of the public contract.

| Public `kind` | Typical suspend | `reply_shape` | Meaning |
| --- | --- | --- | --- |
| `awaiting_intent_confirm` | intent confirmation | `yes_no` | Confirm interpreted plan. |
| `awaiting_intent_feedback` | intent rejection | `free_text` | Supply reason intent is wrong. |
| `awaiting_sql_confirm` | SQL preview / direct reuse | `yes_no` | Confirm generated or reused SQL. |
| `execute` | execute gate | `yes_no` | Confirm running stored SQL. |
| `awaiting_sql_feedback` | SQL rejection | `free_text` | Supply reason SQL is wrong. |
| `result` | — | — | Terminal success (`step.done == True`). |
| `error` | — | — | Terminal failure (`step.done == True`, `step.error` set). |
| `idle` | — | — | Session reset / no active turn. |

Direct template reuse during a normal `ask` turn suspends with `kind="awaiting_sql_confirm"` (same as a SQL preview). For orchestrators that already resolved new bind values, `reuse_saved_question` executes the stored template directly without fuzzy matching or parameter extraction.

### SessionStep fields (embedding checklist)

| Field | Use |
| --- | --- |
| `done` | Loop until `True`. |
| `kind` | Branch UI logic ([table above](#public-kind-values)). |
| `prompt` | Short line to show before collecting input. |
| `reply_shape` | `"yes_no"` vs `"free_text"` while suspended. |
| `message` | Optional narrative (also printed by `run_interactive`). |
| `sql` / `data` | Preview or final result. |
| `parameters` | Tuple of `ParameterBinding` on terminal success (`handle`, `current_value`, `display_name`). |
| `error` | Terminal error text. |
| `diagnostics` | Turn-level tracing rows. |

Full type definitions: [API reference — PipelineSession methods](API_REFERENCE.md#pipelinesession-methods).

---

## Minimal embedding (sync)

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

- `session.ask_until_done(question, on_confirm="y")` — auto-answers **yes/no** suspends only; raises on free-text suspends.
- `session.accept_until_done(question, ...)` — auto-answers both yes/no and free-text suspends until the turn ends.
- `session.reuse_saved_question(question_old, question_new, new_values)` — re-executes a stored template when your orchestrator already knows the new bind values (`{"p1": ...}`). Returns a terminal step with SQL, data, and `parameters`.

Pass `on_confirm` as `"y"` or `"n"` (string, not Python `or`).

### Forced template reuse

When an outer orchestrator has already rewritten natural-language questions and resolved new parameter values, call `reuse_saved_question` instead of `ask`:

```python
step = session.reuse_saved_question(
    "count of item in category alpha",
    "count of item in category beta",
    {"p1": "beta"},
)
assert step.done and step.parameters[0].handle == "p1"
```

The engine locates the template that owns `question_old`, overlays `new_values` on the stored bind map, validates shape, executes, records the new question row, and returns bindings with display names. It does not run fuzzy reuse detection or LLM parameter extraction on this path.

---

## Async support

`AsyncPipelineSession` delegates blocking work to worker threads:

```python
async with engine.asession() as session:
    step = await session.ask("...")
    while not step.done:
        reply = await get_user_input(step)
        step = await session.step(reply)
```

---

## Reader and writer split

- **Writer** (`mode="writer"`, default): persists learning directly. One active writer per artifacts directory recommended.
- **Reader** (`mode="reader"`): enqueues learning to `write_queue.jsonl`. Many readers can overlap on the same artifacts path.

A writer turn drains the queue at start under the engine's writer lock. Queue path: `engine.write_queue_path`.

Offline demo: [Sandbox guide — Reader/writer queue](SANDBOX.md#readerwriter-queue).

---

## Multi-user deployment

Pattern for **any supported database engine** (not PostgreSQL-specific):

1. **Owner writer** — one process with `role="owner"`, `mode="writer"`, full `EngineContext`, and durable `artifacts_dir` on shared storage. Builds the master graph and drains the write queue.
2. **Consumer readers** — processes with `role="consumer"`, `engine_context="context_name"`, restricted database credentials matching their allow lists, and `mode="reader"` (or `"writer"` only on the owner).
3. **Align scope** — `EngineContext.allow_objects` / `allow_columns` must match each consumer's database grants.
4. **Optional AetherSpaces** — further narrow model focus per team or product surface without changing warehouse roles.

Readers share the owner's artifacts directory but never mutate it directly; they append queue events the owner drains.

```python
# Owner (any engine — fill config_file for yours)
owner = AetherEngine(
    EngineContext(allow_objects=frozenset({"orders", "customers"})),
    artifacts_dir="/shared/aether_artifacts",
    config_file="./aetherdialect.toml",
    role="owner",
)

# Consumer — context name only, matching DB login scope
consumer = AetherEngine(
    "analyst_scope",
    artifacts_dir="/shared/aether_artifacts",
    config_file="./aetherdialect.toml",
    role="consumer",
)
with consumer.session(mode="reader", space="master") as session:
    session.accept_until_done("How many orders last month?")
```

Practice the same pattern offline: [Sandbox guide — Owner vs consumer presets](SANDBOX.md#owner-vs-consumer-presets).

---

## Observability

| Channel | Access | Granularity |
| --- | --- | --- |
| **`audit_sink`** | Constructor callback | Coarse lifecycle (`init`, `ask_begin`, `ask_done`, queue drain events). |
| **`SessionStep.diagnostics`** | Every `ask` / `step` return | Turn-level codes (`REUSE_HIT`, `COMPOSE_REPAIR`, `SENSITIVITY_GATE_HIT`, …). |
| **`engine.show_config()`** | Method call | Redacted config snapshot. |

Diagnostic catalog: [API reference — Observability](API_REFERENCE.md#observability).

---

**See also:** [User guide](USER_GUIDE.md) · [Sandbox guide](SANDBOX.md) · [API reference](API_REFERENCE.md) · [How it works](HOW_IT_WORKS.md) · [Security](SECURITY.md) · [Support matrix](SUPPORT_MATRIX.md) · [README](../README.md#documentation)
