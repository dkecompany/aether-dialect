# Security model

Nothing in the engine generates raw SQL from free-form text. The LLM fills bounded slots in a structured intent JSON. That intent is parsed, validated against the schema graph, and materialized into SQL by a deterministic generator. Four guards (intent JSON shape, dialect AST validation, schema/catalog alignment, dialect EXPLAIN) sit between the LLM and the database.

**Reading order:** [README — Documentation](../README.md#documentation) → [Getting started](GETTING_STARTED.md) → [User guide](USER_GUIDE.md) → [Integrator guide](INTEGRATOR_GUIDE.md) → [Sandbox guide](SANDBOX.md) → [API reference](API_REFERENCE.md) → [How it works](HOW_IT_WORKS.md) → this file → [Support matrix](SUPPORT_MATRIX.md).

## Sections

| Section | Contents |
| --- | --- |
| [1. Threat model](#1-threat-model) | Trust boundaries and mitigations |
| [2. LLM context inventory](#2-llm-context-inventory) | Per-phase prompts |
| [3. Sensitivity tags](#3-sensitivity-tags) | Canonical tier definitions |
| [4. Deny lists](#4-deny-lists) | Graph removal vs sensitivity |

---

## 1. Threat model

The engine is built for the case where:

- A trusted operator configures the engine with database credentials and an LLM provider.
- A semi-trusted user asks natural-language questions; the user is permitted to read the data the credentials authorize but should not be able to bypass the analytical SQL surface.
- The LLM provider sees the prompts the engine sends. Whatever the operator considers acceptable to send to the configured provider must include the schema metadata listed in Section 2 below.
- The local filesystem under the **engine storage directory** is operator-controlled.

| Risk | Mitigation |
| --- | --- |
| User crafts a question that runs arbitrary SQL | `SELECT`-only enforcement, forbidden-SQL regex list, dialect AST validation, structured intent IR, EXPLAIN gate before execution. |
| LLM output escapes the analytical subset | Same gates plus validators that reject constructs the IR cannot represent safely. |
| Sensitive column data leaks through prompts or artifacts | Sensitivity tiers (`none`, `restricted`, `hidden`), deny lists, visibility gates, top-k caps on enum heads. |
| User assumes AetherSpace replaces RBAC | AetherSpace narrows knowledge context only; database grants and `EngineContext` scope remain the execution boundary. |

## 2. LLM context inventory

### 2.1 Interpret phase

- **Task:** Read the question against domain table/column descriptions; emit a natural-language solution plan.
- **Content:** Question, Interpret-stage schema payload (descriptions and enum heads), optional prior-question feedback.
- **Data sent:** Metadata and enum heads only; no row data.

### 2.2 Ground phase

- **Task:** Convert the interpret plan into logical intent JSON bound to schema identifiers.
- **Content:** Question, interpret plan, Ground-stage schema payload (descriptions, roles, value types, enum heads).
- **Data sent:** Metadata and enum heads only.

### 2.3 Compose phase

- **Task:** Convert logical intent into the runtime intermediate representation (IR).
- **Content:** Logical intent, Compose-stage structural schema payload (identifiers, types, PK/FK markers).
- **Data sent:** Structural metadata only.

### 2.4 Repair and validation

When validation fails, the same Compose system prompt is reused with error rows and the structural payload for implicated tables. The engine never sends raw database rows to the LLM.

Diagnostic codes for retries and repairs include `INTERPRET_GROUND_RETRY`, `COMPOSE_REPAIR`, `SENSITIVITY_GATE_HIT`, and `FALLBACK_FRESH_RESTART` ([API reference — Diagnostic code catalog](API_REFERENCE.md#diagnostic-code-catalog)).

## 3. Sensitivity tags

Canonical definitions — repeated elsewhere only as a one-line skim with a pointer here.

| Tier | Model visibility | Query behavior |
| --- | --- | --- |
| **none** | Full visibility in schema payloads. | Normal SELECT, filter, group, and order behavior. |
| **restricted** | Column name, type, role, and description appear in LLM-facing schema literals. | Users cannot project individual row values from the column (bare SELECT entries are dropped). Literal-value filters on the column are blocked. GROUP BY and ORDER BY on the column are blocked. Aggregations that wrap the column (COUNT, SUM, AVG, and similar) may still be emitted when policy allows. |
| **hidden** | Column is omitted from all LLM-facing schema literals (`is_visible == false`). | The column is invisible to the questioning surface — analysts and the model cannot target it by name. It remains in the compiled graph for profiling and override bookkeeping unless removed by `deny_columns`. |

The intent repair pass may drop individual SELECT or GROUP BY entries that reference restricted columns. Semantic validation terminates the turn when WHERE or HAVING references a non-selectable column.

Assign tiers through schema overrides or domain notes; demo hidden columns in the offline sandbox: [Sandbox guide — Sensitivity](SANDBOX.md#sensitivity).

## 4. Deny lists

`EngineContext.deny_columns` and `allow_columns` accept qualified `table.column` tokens. Denied columns are **removed from the graph** before profiling and classification — they do not exist while the deny is active. This is a stronger removal than **hidden** sensitivity, which omits a column from prompts but keeps it in the graph.

`SpaceContext.deny_objects` and `deny_columns` apply the same shape at AetherSpace knowledge scope only ([User guide — AetherSpace](USER_GUIDE.md#aetherspace)).

---

**See also:** [Getting started](GETTING_STARTED.md) · [User guide](USER_GUIDE.md) · [Integrator guide](INTEGRATOR_GUIDE.md) · [Sandbox guide](SANDBOX.md) · [API reference](API_REFERENCE.md) · [How it works](HOW_IT_WORKS.md) · [Support matrix](SUPPORT_MATRIX.md) · [README](../README.md#documentation)
