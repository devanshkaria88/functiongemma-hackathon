# EdgeAgent — Cactus × Google DeepMind Hackathon Project

## Context & Overview

This is a submission for the **Cactus × Google DeepMind Global Hackathon** (Feb 22, 2026). It's a one-day hackathon at UCL East, London. Hacking window: 10:00 AM – 5:30 PM (~7.5 hours). Solo builder.

**The hackathon challenge**: Build a smart **hybrid routing algorithm** that decides when to run AI tool-calling **locally on-device** (FunctionGemma 270M via Cactus Engine on Mac) vs **falling back to the cloud** (Gemini 2.0 Flash API). The goal is to maximize tool-call correctness while keeping as much processing on-device as possible and minimizing end-to-end latency.

**Getting started repo**: https://github.com/cactus-compute/functiongemma-hackathon

---

## What We're Building

**"EdgeAgent"** — A voice-first hybrid AI assistant with a web UI that:

1. **Core routing engine** (`main.py`) — Smart `generate_hybrid()` function that classifies query complexity and routes to local/cloud intelligently. This is what gets submitted to the leaderboard.
2. **Voice-first web UI** — Browser-based interface using `cactus_transcribe` (on-device Whisper) for voice input, showing real-time routing decisions, latency, and executed actions.

---

## The Technical Stack

### Provided by Hackathon (already set up)

- **Cactus Engine** — On-device inference runtime for Mac (Apple Silicon). Installed via `cactus build --python`.
- **FunctionGemma 270M** — Google's tiny function-calling model, runs locally via Cactus at ~3000 tok/s prefill, ~200 tok/s decode. Located at `cactus/weights/functiongemma-270m-it`.
- **Gemini 2.0 Flash** — Cloud API fallback via `google-genai` Python package. API key in env var `GEMINI_API_KEY`.
- **Whisper (on-device)** — Speech-to-text via `cactus_transcribe`. Model at `cactus/weights/whisper-small`.

### Cactus Python API (Key Functions)

```python
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_transcribe, cactus_reset

# Init model
model = cactus_init("cactus/weights/functiongemma-270m-it")

# Tool-calling completion
response = cactus_complete(
    model, messages,
    tools=tools,           # List of tool definitions
    force_tools=True,      # Constrain output to tool call format
    max_tokens=256,
    tool_rag_top_k=2,      # Select top-k relevant tools (0 = use all)
    confidence_threshold=0.7,  # Below this triggers cloud_handoff flag
    stop_sequences=["<|im_end|>", "<end_of_turn>"],
)

# Response format:
{
    "success": true,
    "cloud_handoff": false,
    "response": "...",
    "function_calls": [{"name": "get_weather", "arguments": {"location": "London"}}],
    "confidence": 0.85,
    "total_time_ms": 163.7,
    ...
}

# Voice transcription
whisper = cactus_init("cactus/weights/whisper-small")
prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
response = cactus_transcribe(whisper, "audio.wav", prompt=prompt)

# Cleanup
cactus_destroy(model)
```

### What We're Adding

- **FastAPI** backend — Serves the routing engine + transcription as HTTP endpoints
- **Web frontend** — HTML/JS/CSS with mic input, routing visualization, results display
- **Query classifier** — Pre-routes queries by analyzing complexity before calling any model
- **Output validator** — Validates FunctionGemma's local output before accepting it
- **Query decomposer** — Splits multi-intent queries into simpler sub-queries for local execution

---

## How Scoring Works

### Phase 1: Objective Leaderboard (Top 10 per city advance to judging)

Submit via `python submit.py --team "YourTeamName" --location "London"` (max 1x/hour). Evaluated on a **hidden** eval dataset (harder than public benchmarks).

**Exact scoring formula** (from `benchmark.py`):

```python
Total Score = Σ (difficulty_weight × level_score) × 100

level_score = (0.60 × avg_f1) + (0.15 × time_score) + (0.25 × on_device_ratio)
time_score  = max(0, 1 - avg_time_ms / 500)

Difficulty weights:
  easy:   0.20   # 1 tool, direct request
  medium: 0.30   # 2-5 tools, must pick right one
  hard:   0.50   # multiple tools needed, multi-call
```

**Point distribution** (what matters most):
- Hard F1 accuracy: 50% × 60% = **30 points** ← biggest single lever
- Medium F1 accuracy: 30% × 60% = **18 points**
- On-device ratio: ~**25 points** total across all tiers
- Time score: ~**15 points** total (anything under 500ms is full marks)
- Easy F1: 20% × 60% = **12 points** (should be near-free locally)

**F1 Scoring**: Matches predicted function calls against expected ones. Compares function names exactly, argument values case-insensitive and trimmed.

### Phase 2: Qualitative Judging (Top 10 per city)

- **Rubric 1**: Quality of hybrid routing algorithm — depth and cleverness
- **Rubric 2**: End-to-end products that execute function calls to solve real-world problems
- **Rubric 3**: Building low-latency voice-to-action products leveraging `cactus_transcribe`

---

## Architecture

### System Diagram

```
┌─────────────────────────────────────────────────┐
│              EdgeAgent Web UI                    │
│  [🎤 Mic Button] → [Routing Viz] → [Results]    │
│  Shows: transcript, local/cloud path, latency   │
└────────────────────┬────────────────────────────┘
                     │ HTTP
         ┌───────────▼───────────┐
         │   FastAPI Backend      │
         │   POST /transcribe     │
         │   POST /execute        │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────────────┐
         │      generate_hybrid()         │
         │                                │
         │  1. CLASSIFY query complexity  │
         │     ├─ num_tools              │
         │     ├─ num_intents            │
         │     └─ complexity level       │
         │                                │
         │  2. ROUTE based on complexity  │
         │     ├─ simple  → local only   │
         │     ├─ medium  → local+validate│
         │     ├─ hard_decomposable →    │
         │     │    split into sub-queries│
         │     │    run each locally      │
         │     └─ hard_complex → cloud   │
         │                                │
         │  3. VALIDATE local output      │
         │     ├─ valid JSON?            │
         │     ├─ function name exists?  │
         │     ├─ required params present?│
         │     └─ call count matches?    │
         │                                │
         │  4. FALLBACK to cloud if needed│
         └───────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         │                       │
    ┌────▼────┐           ┌─────▼─────┐
    │ LOCAL   │           │  CLOUD    │
    │FuncGemma│           │ Gemini 2.0│
    │via Cactus│          │  Flash    │
    └─────────┘           └───────────┘
```

### Routing Logic (The Core Innovation)

```
Input: messages, tools
  │
  ├─ Step 1: CLASSIFY QUERY
  │   ├─ num_tools = len(tools)
  │   ├─ num_intents = detect intents via NLP signals:
  │   │     - conjunctions: "and", "then", "also", "plus"
  │   │     - commas separating clauses
  │   │     - multiple action verbs
  │   ├─ estimated_calls = max(1, num_intents)
  │   └─ complexity = classify(num_tools, estimated_calls)
  │
  ├─ Step 2: ROUTE
  │   ├─ SIMPLE (1 tool OR 1 intent + ≤2 tools):
  │   │     → Local with optimized prompt → Validate → Return or fallback
  │   │
  │   ├─ MEDIUM (1 intent + 3-5 tools):
  │   │     → Local with tool_rag_top_k filtering → Validate → Return or fallback
  │   │
  │   ├─ HARD_DECOMPOSABLE (multi-intent, decomposable):
  │   │     → Split into sub-queries
  │   │     → Assign minimal tool subset per sub-query
  │   │     → Run each sub-query locally (each becomes a SIMPLE case)
  │   │     → Validate each result
  │   │     → If ALL valid: merge function_calls → Return as local
  │   │     → If ANY fails: fallback ENTIRE query to cloud
  │   │
  │   └─ HARD_COMPLEX (ambiguous, not decomposable):
  │         → Skip local, go straight to cloud (saves time)
  │
  └─ Step 3: RETURN
      └─ {function_calls, total_time_ms, source, routing_reason}
```

### Query Decomposition Example

```python
# Input: "Text Emma good night and check weather in Chicago"
# Detected intents: 2 (split on "and")
# Decomposed into:

sub_queries = [
    {
        "messages": [{"role": "user", "content": "Text Emma good night"}],
        "tools": [TOOL_SEND_MESSAGE]  # only relevant tool assigned
    },
    {
        "messages": [{"role": "user", "content": "check weather in Chicago"}],
        "tools": [TOOL_GET_WEATHER]   # only relevant tool assigned
    },
]

# Each sub-query is now a SIMPLE case → FunctionGemma handles it locally
# Merge all function_calls from sub-results into one response
# This turns "hard" cases into "easy" cases → massive on-device ratio boost
```

### Tool Matching for Decomposition

When decomposing sub-queries, we need to assign the right tool(s) to each sub-query. Approach:

1. For each sub-query text, compute keyword overlap with each tool's name + description
2. Assign the top-1 or top-2 matching tools to that sub-query
3. This reduces the tool selection problem from "pick 1 of 5" to "pick 1 of 1-2"

---

## Files Structure

```
project_root/
├── cactus/                    # Cactus SDK (cloned, pre-built)
│   ├── weights/
│   │   ├── functiongemma-270m-it/   # FunctionGemma model
│   │   └── whisper-small/           # Whisper model
│   └── python/src/                  # Cactus Python bindings
│
├── main.py                    # ⭐ SUBMISSION FILE — contains generate_hybrid()
│                              #    Also contains generate_cactus() and generate_cloud()
│                              #    ONLY modify internals of generate_hybrid()
│                              #    DO NOT change its input/output signature
│
├── benchmark.py               # Local benchmark (30 test cases: 10 easy, 10 medium, 10 hard)
│                              #    Run: python benchmark.py
│
├── submit.py                  # Leaderboard submission
│                              #    Run: python submit.py --team "EdgeAgent" --location "London"
│
├── classifier.py              # 🔨 TO BUILD — Query complexity classifier
│                              #    - count_intents(text) → int
│                              #    - classify_complexity(messages, tools) → str
│                              #    - decompose_query(text, tools) → list of sub-queries
│
├── validator.py               # 🔨 TO BUILD — Output validation
│                              #    - validate_local_result(result, tools, expected_calls) → bool
│                              #    - Checks: valid JSON, function name exists in tools,
│                              #      required params present, call count sanity
│
├── server.py                  # 🔨 TO BUILD — FastAPI backend
│                              #    POST /transcribe — voice → text via cactus_transcribe
│                              #    POST /execute — text + tools → hybrid routing → result
│                              #    Serves static frontend files
│
├── static/                    # 🔨 TO BUILD — Web frontend
│   └── index.html             #    - Mic button (MediaRecorder API → WAV)
│                              #    - Routing visualization (local vs cloud path animation)
│                              #    - Results display (function calls, timing, source)
│                              #    - History of queries with routing decisions
│
└── demo_tools.py              # 🔨 TO BUILD — Simulated tool execution for demo
                               #    - Fake weather responses, alarm confirmations, etc.
                               #    - Makes the demo feel like a real end-to-end product
```

---

## Build Sequence (Time-Boxed)

| Block | Time | What to Build | Success Criteria |
|-------|------|---------------|------------------|
| 1 | 0:00–1:30 | `classifier.py` + prompt engineering in `main.py` | Smart routing for easy/medium cases, easy cases always local |
| 2 | 1:30–3:00 | `validator.py` + query decomposition in `classifier.py` | Hard multi-call cases decomposed and run locally |
| 3 | 3:00–3:30 | Run `benchmark.py`, iterate, first `submit.py` | Score on leaderboard |
| 4 | 3:30–5:00 | `server.py` + `static/index.html` + `demo_tools.py` | Working voice-first web demo |
| 5 | 5:00–6:00 | Polish routing, re-submit, tune thresholds | Higher leaderboard score |
| 6 | 6:00–6:30 | Demo script + pitch prep | Ready to present |

**Each block has a standalone checkpoint** — if Block 2 runs long, skip to Block 3 and still have a valid submission.

---

## Key Constraints & Rules

1. **`generate_hybrid()` signature is FROZEN** — Do not modify its input args (`messages`, `tools`, `confidence_threshold`) or return format. Only modify internal logic. It must remain compatible with `benchmark.py`.
2. **Leaderboard submissions max 1x per hour** via `submit.py`.
3. **Hidden eval dataset** is harder than the public 30-case benchmark. Design for generalization, not overfitting.
4. **FunctionGemma needs a Mac** — Cactus runs on Apple Silicon. Already set up locally.
5. **Gemini API key** in `GEMINI_API_KEY` env var. Free credits provided by hackathon.
6. **Model init is expensive** — `cactus_init()` / `cactus_destroy()` are slow. Consider keeping models alive across calls in the server, or batch operations.

---

## Existing Code Reference

### main.py — Current Baseline (What We're Improving)

The current `generate_hybrid()` is naive:

```python
def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Baseline: fall back to cloud if confidence below threshold."""
    local = generate_cactus(messages, tools)
    if local["confidence"] >= confidence_threshold:
        local["source"] = "on-device"
        return local
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local["confidence"]
    cloud["total_time_ms"] += local["total_time_ms"]
    return cloud
```

Problems with baseline:
- Always runs local first (wastes time on hard cases that will always fail)
- Only checks confidence score (README says "you will NOT rely on this")
- No query analysis, no output validation, no decomposition
- No prompt optimization for FunctionGemma

### generate_cactus() — Local Inference (Don't Modify Signature)

```python
def generate_cactus(messages, tools):
    model = cactus_init(functiongemma_path)
    cactus_tools = [{"type": "function", "function": t} for t in tools]
    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )
    cactus_destroy(model)
    raw = json.loads(raw_str)
    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }
```

### generate_cloud() — Cloud Fallback (Don't Modify Signature)

Uses `google-genai` to call Gemini 2.0 Flash with tool definitions.

### benchmark.py — Public Test Cases

30 cases across 3 difficulty levels:
- **Easy** (10): Single tool, direct request. E.g., "What is the weather in San Francisco?" with only `get_weather` tool.
- **Medium** (10): 2-5 tools available, must pick correct one. E.g., "Set an alarm for 8:15 AM" with `send_message`, `set_alarm`, `get_weather` tools.
- **Hard** (10): Multi-intent, multi-call. E.g., "Text Emma good night, check weather in Chicago, and set alarm for 5 AM" with 5 tools, expects 3 function calls.

Tools used: `get_weather`, `set_alarm`, `send_message`, `create_reminder`, `search_contacts`, `play_music`, `set_timer`.

---

## Optimization Strategies to Implement

### 1. Prompt Engineering (in generate_cactus or wrapper)

- Better system prompt: "You are a function-calling assistant. Always respond with the correct tool call. Match the user's request to the most relevant available function."
- FunctionGemma performs better with explicit, distinctive tool descriptions
- Use `tool_rag_top_k` parameter to pre-filter relevant tools

### 2. Pre-Classification Signals

Detect query complexity BEFORE calling any model:

```python
MULTI_INTENT_SIGNALS = ["and", "then", "also", "plus", ",", ";"]
ACTION_VERBS = ["set", "send", "check", "get", "play", "find", "remind", "create", "text", "look up"]

def count_intents(text):
    # Count conjunction-separated clauses containing action verbs
    # "Text Emma and check weather" → 2 intents
    # "What's the weather in London?" → 1 intent
    ...
```

### 3. Output Validation

After local inference, validate before accepting:

```python
def validate_result(result, tools, expected_num_calls):
    # 1. Has function_calls?
    # 2. Each function name is in available tools?
    # 3. Required parameters present for each call?
    # 4. Number of calls is reasonable given query?
    # Returns True if all checks pass
    ...
```

### 4. Query Decomposition (Key Differentiator)

For multi-intent queries, split and run each independently:

```python
def decompose_query(text, tools):
    # Split "Do X and Y and Z" into ["Do X", "Y", "Z"]
    # For each sub-query, match to most relevant tool(s)
    # Return list of (sub_message, sub_tools) pairs
    ...
```

### 5. Tool Matching for Sub-queries

Simple keyword overlap between sub-query text and tool name/description:

```python
def match_tools(sub_query_text, tools):
    # Score each tool by keyword overlap with sub-query
    # Return top-1 or top-2 matching tools
    ...
```

---

## Demo & Pitch Narrative

**Pitch**: "EdgeAgent keeps AI fast, private, and smart. Simple commands execute instantly on your Mac — no cloud, no latency, no data leaving your device. When queries get complex, EdgeAgent intelligently decomposes them into simpler on-device tasks. Only truly ambiguous requests escalate to Gemini in the cloud. The user never sees the seam."

**Demo flow**:
1. Voice: "What's the weather in London?" → Shows LOCAL path, ~50ms, instant response
2. Voice: "Set an alarm for 8 AM" → Shows LOCAL path with tool selection from 5 options
3. Voice: "Text Bob hello and check the weather in Paris" → Shows DECOMPOSITION into 2 local calls
4. Voice: [Something genuinely ambiguous] → Shows CLOUD fallback with reasoning

**Key visual**: An animated routing diagram that lights up the local or cloud path in real-time, showing the classifier's decision and why.

---

## Environment Setup Checklist

- [x] Cactus Engine installed and built (`cactus build --python`)
- [x] FunctionGemma 270M downloaded (`cactus download google/functiongemma-270m-it --reconvert`)
- [x] Cactus auth token set (`cactus auth`)
- [x] Gemini API key set (`export GEMINI_API_KEY="..."`)
- [x] `google-genai` installed (`pip install google-genai`)
- [ ] Whisper model downloaded for voice (`cactus download` whisper-small)
- [ ] FastAPI installed (`pip install fastapi uvicorn python-multipart`)
- [ ] Test `python benchmark.py` runs successfully

---

## Progress Tracker

- [x] Phase 1: Research — hackathon theme, sponsors, constraints
- [x] Phase 2: Team-Topic Match — solo builder, routing + voice UI focus
- [x] Phase 3: Judging Criteria Analysis — exact scoring formula extracted
- [x] Phase 4: Ideation — EdgeAgent concept, 3-layer routing strategy
- [x] Phase 5: Architecture — system design, file structure, build sequence
- [ ] Phase 6: Build — implementation begins at hackathon
  - [ ] Block 1: Query classifier + prompt engineering
  - [ ] Block 2: Output validator + query decomposition
  - [ ] Block 3: Benchmark + first leaderboard submission
  - [ ] Block 4: Voice-first web UI
  - [ ] Block 5: Polish + re-submit
  - [ ] Block 6: Demo script + pitch