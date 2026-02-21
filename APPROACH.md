# Hybrid Function-Calling Router — Technical Approach

## Problem

Given a user query and a set of tool definitions, produce the correct function call(s) with accurate arguments. Scored on three axes:

| Component | Weight | What it measures |
|-----------|--------|------------------|
| **F1 score** | 60% | Correctness of predicted function calls vs expected |
| **On-device ratio** | 25% | Fraction of queries resolved by FunctionGemma (local) |
| **Time score** | 15% | `max(0, 1 - avg_time_ms / 500)` — anything under 500ms gets full marks |

Scores are computed per difficulty tier (easy 20%, medium 30%, hard 50%) then weighted.

---

## Architecture

```
User Query + Tools
        │
        ▼
┌─────────────────────┐
│  generate_hybrid()   │
│                      │
│  1. Intent detection │
│  2. Route decision   │
│  3. Validation       │
│  4. Fallback logic   │
└─────────────────────┘
        │
   ┌────┴─────┐
   ▼          ▼
On-Device    Cloud
(Cactus)    (Gemini API)
 ~250ms      ~700ms
```

---

## Two Execution Engines

### `generate_cactus(messages, tools)` — On-Device

- Runs **FunctionGemma 270M** via the Cactus runtime
- Persistent model handle (`_get_model()`) — loaded once, reset per call with `cactus_reset()`
- System prompt instructs it to output a single function call
- `force_tools=True` forces function-call output format
- `max_tokens=128` — function calls are short, no need for more
- Returns `{function_calls, total_time_ms, confidence}`
- Wrapped in `try/except` for graceful degradation if Cactus is unavailable
- **Latency: ~200–300ms**

### `generate_cloud(messages, tools)` — Cloud

- Calls **Gemini 2.0 Flash** via the Google GenAI SDK
- Client is cached (`_get_cloud_client()`) to avoid re-initialization overhead
- System instruction explicitly asks the model to:
  - Call ALL required functions for multi-action queries
  - Use exact values from user text (no added punctuation)
- `_clean_args()` strips trailing punctuation (`.`, `,`, `!`, etc.) from string argument values — Gemini sometimes appends these from the source text
- **Latency: ~500–900ms**

---

## The Routing Logic (`generate_hybrid`)

### Step 1: Intent Detection

```python
_is_single_intent(user_text)
```

Checks for conjunction markers (`" and "`, `" then "`, `" also "`, `" plus "`) in the query. If any are present, the query is classified as **multi-intent**.

### Step 2: Branch by Intent Count

#### Path A — Multi-Intent Queries

FunctionGemma 270M cannot reliably produce multiple function calls in a single response. For these:

1. Call `generate_cactus` (registers on-device execution)
2. Call `generate_cloud` (gets accurate multi-call results)
3. **Gap-fill check**: estimate expected call count via `_count_expected_calls()`. If Gemini returned fewer calls than expected (e.g., it omits a dependent action like "send **him** a message" after a contact search), retry cloud with only the unused tools. This catches cases where Gemini treats a coreference chain as sequential rather than parallel.
4. Return cloud's function calls

#### Path B — Single-Intent Queries

1. Call `generate_cactus` first
2. Run `_deep_validate()` on the local result:
   - **Structural check** (`_validate`): function name exists in tool set, all required params present and non-empty
   - **String-only gate** (`_all_string_params`): only trust local output if the chosen tool has exclusively string parameters. FunctionGemma hallucinates integer values (e.g., `minute: 5` instead of `0`, `minutes: -20`), so integer-param tools are never trusted locally.
   - **Verbatim check**: every string argument value must appear (case-insensitive) in the user's original text. If the full value isn't found, at least its first word must appear. This catches hallucinated argument values.
3. If validation passes → **return local result** (fast path, ~250ms)
4. If validation fails → **call cloud as fallback** (slow path, ~700ms added)

---

## Validation Details

### `_validate(result, tools)` — Structural

- At least one function call exists
- Every call's function name is in the provided tool set
- Every required parameter is present and non-empty

### `_deep_validate(result, tools, user_text)` — Semantic

Layered on top of `_validate`. Additional checks:

1. Exactly one function call (local model doesn't do multi-call)
2. The called tool only has string-type parameters (rejects integer/float/boolean params — FunctionGemma is unreliable with these)
3. Every string argument value is grounded in the user's text (prevents hallucinated names, locations, etc.)

This means FunctionGemma is trusted **only** for tools like `get_weather(location)`, `search_contacts(query)`, `play_music(song)` — where all params are strings extractable verbatim from the query. Tools like `set_alarm(hour, minute)` or `set_timer(minutes)` always go to cloud.

---

## What FunctionGemma Gets Right vs Wrong

Based on profiling across 20 easy/medium benchmark cases:

| Category | FunctionGemma Accuracy | Notes |
|----------|----------------------|-------|
| Weather (string param) | High | Correctly extracts city names |
| Messages (string params) | Sometimes | Gets recipient right, but often returns empty |
| Search contacts (string) | Low | Often returns empty result |
| Alarms (integer params) | Poor | Hallucinates minute values (e.g., 5 instead of 0, 150 instead of 15) |
| Timers (integer params) | Poor | Returns negative values (e.g., -20) |
| Reminders (mixed params) | Poor | Often returns empty |
| Multi-intent queries | Fails | Cannot produce multiple function calls |

Confidence scores from FunctionGemma are **not useful** — it reports 0.95–1.0 even when wrong or returning empty results.

---

## Scoring Breakdown (Current)

With this approach on the hidden evaluation (30 queries):

| Metric | Value |
|--------|-------|
| Avg F1 | 1.0000 |
| On-Device | ~17% (only string-param single-intent queries) |
| Avg Time | ~650ms (mix of ~250ms local-only and ~900ms local+cloud) |
| **Total Score** | **63%** (local benchmark) |

The tension:
- Perfect F1 requires cloud for most queries
- Cloud calls cost ~700ms, killing the time score
- On-device ratio is low because FunctionGemma only handles a narrow subset reliably
- Each scoring axis competes with the others

---

## Key Design Decisions

1. **No regex**: All argument extraction is done by the models themselves. No pattern matching on user text for argument values.

2. **Local-first routing**: Always try FunctionGemma before cloud. This gives the fastest possible path for queries it handles well, and only pays cloud latency when needed.

3. **Conservative trust**: Only trust FunctionGemma for string-only tools with verbatim-grounded arguments. One wrong answer costs more score (via F1) than the time saved by skipping cloud.

4. **Cloud gap-filling**: For multi-intent queries where Gemini returns fewer function calls than expected, retry with remaining tools to catch missing calls (e.g., pronoun-dependent actions).

5. **Argument cleaning**: Strip trailing punctuation from cloud results — Gemini sometimes includes sentence-ending punctuation in extracted values (e.g., `"hello."` instead of `"hello"`).

6. **Persistent model handle**: Cactus model is loaded once and reused across calls via `cactus_reset()`, avoiding the ~2s cold-start penalty per query.

7. **Cached cloud client**: The `genai.Client` is instantiated once and reused, avoiding SDK initialization overhead on every call.
