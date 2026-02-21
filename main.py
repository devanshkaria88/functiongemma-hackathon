
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
from google import genai
from google.genai import types


_local_model = None
_model_pool = []
_pool_lock = None

def _get_model():
    global _local_model
    if _local_model is None:
        _local_model = cactus_init(functiongemma_path)
    return _local_model


def _get_pool_model(index):
    """Get or create a model handle for parallel execution."""
    import threading
    global _pool_lock
    if _pool_lock is None:
        _pool_lock = threading.Lock()

    with _pool_lock:
        while len(_model_pool) <= index:
            _model_pool.append(None)
        if _model_pool[index] is None:
            _model_pool[index] = cactus_init(functiongemma_path)
    return _model_pool[index]


SYSTEM_PROMPT = (
    "You are a function-calling assistant. Given the user's request, "
    "call the single most relevant function with the correct arguments. "
    "Only output the function call, nothing else."
)


# ---------------------------------------------------------------------------
# Local inference (reuses persistent model handle)
# ---------------------------------------------------------------------------
def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    try:
        model = _get_model()
        cactus_reset(model)

        cactus_tools = [{"type": "function", "function": t} for t in tools]

        raw_str = cactus_complete(
            model,
            [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            tools=cactus_tools,
            force_tools=True,
            max_tokens=128,
            stop_sequences=["<|im_end|>", "<end_of_turn>"],
        )

        try:
            raw = json.loads(raw_str)
        except json.JSONDecodeError:
            return {"function_calls": [], "total_time_ms": 0, "confidence": 0}

        return {
            "function_calls": raw.get("function_calls", []),
            "total_time_ms": raw.get("total_time_ms", 0),
            "confidence": raw.get("confidence", 0),
        }
    except Exception:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0}


def _cactus_on_pool(pool_index, messages, tools):
    """Run cactus_complete on a specific pool model (thread-safe)."""
    try:
        model = _get_pool_model(pool_index)
        cactus_reset(model)
        cactus_tools = [{"type": "function", "function": t} for t in tools]
        raw_str = cactus_complete(
            model,
            [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            tools=cactus_tools,
            force_tools=True,
            max_tokens=128,
            stop_sequences=["<|im_end|>", "<end_of_turn>"],
        )
        try:
            raw = json.loads(raw_str)
        except json.JSONDecodeError:
            return {"function_calls": [], "total_time_ms": 0, "confidence": 0}
        return {
            "function_calls": raw.get("function_calls", []),
            "total_time_ms": raw.get("total_time_ms", 0),
            "confidence": raw.get("confidence", 0),
        }
    except Exception:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0}


# ---------------------------------------------------------------------------
# Cloud inference
# ---------------------------------------------------------------------------
def _clean_args(arguments):
    """Strip trailing punctuation from string argument values."""
    cleaned = {}
    for k, v in arguments.items():
        if isinstance(v, str):
            cleaned[k] = v.strip().rstrip(".,!?;:")
        else:
            cleaned[k] = v
    return cleaned


CLOUD_SYSTEM = (
    "You are a precise function-calling assistant. "
    "If the user's request contains multiple actions, call ALL required functions. "
    "Use exact values from the user's text as arguments — do not add punctuation."
)


_cloud_client = None

def _get_cloud_client():
    global _cloud_client
    if _cloud_client is None:
        _cloud_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return _cloud_client


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = _get_cloud_client()

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]
    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            tools=gemini_tools,
            system_instruction=CLOUD_SYSTEM,
        ),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": _clean_args(dict(part.function_call.args)),
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


# ===================================================================
# VALIDATION — check FunctionGemma output is structurally sound
# ===================================================================

def _validate(result, tools):
    """Check that local output has valid function calls with required params."""
    calls = result.get("function_calls", [])
    if not calls:
        return False
    tool_map = {t["name"]: t for t in tools}
    for call in calls:
        name = call.get("name", "")
        if name not in tool_map:
            return False
        required = set(tool_map[name]["parameters"].get("required", []))
        args = call.get("arguments", {})
        for req in required:
            if req not in args or args[req] is None or args[req] == "":
                return False
    return True


# ===================================================================
# QUERY DECOMPOSITION — split multi-intent into simple sub-queries
# ===================================================================

_SPLIT_MARKERS = [", and ", " and ", ", then ", " then ", ", also ", " also ", ", plus ", " plus ", ", "]


def _split_intents(text):
    """Split a multi-intent query into individual sub-queries."""
    text_lower = text.lower()
    best_marker = None
    best_pos = -1
    for marker in _SPLIT_MARKERS:
        pos = text_lower.find(marker)
        if pos != -1 and (best_pos == -1 or pos < best_pos):
            best_pos = pos
            best_marker = marker

    if best_marker is None:
        return [text]

    first = text[:best_pos].strip()
    rest = text[best_pos + len(best_marker):].strip()

    rest_parts = _split_intents(rest)
    return [first] + rest_parts


def _resolve_pronouns(sub_text, previous_calls):
    """Replace pronouns (him/her/them) with names from previous call results."""
    pronouns = ["him", "her", "them", "his", "he", "she", "they"]
    text_lower = sub_text.lower()
    has_pronoun = any(f" {p} " in f" {text_lower} " for p in pronouns)
    if not has_pronoun or not previous_calls:
        return sub_text

    last_name = None
    for call in reversed(previous_calls):
        for val in call.get("arguments", {}).values():
            if isinstance(val, str) and val and val[0].isupper() and len(val) < 30:
                last_name = val
                break
        if last_name:
            break

    if not last_name:
        return sub_text

    for p in pronouns:
        lower = sub_text.lower()
        idx = lower.find(f" {p} ")
        if idx == -1:
            idx = lower.find(f" {p}.")
        if idx == -1:
            idx = lower.find(f" {p},")
        if idx != -1:
            sub_text = sub_text[:idx + 1] + last_name + sub_text[idx + 1 + len(p):]

    return sub_text


# ===================================================================
# VALIDATION
# ===================================================================

def _all_string_params(tool):
    """Check if a tool only has string parameters (no integer/numeric)."""
    for pspec in tool["parameters"]["properties"].values():
        if pspec.get("type", "string").lower() != "string":
            return False
    return True


def _deep_validate(result, tools, user_text):
    """
    Strict validation for local results. Only trusts local output when:
    - Single function call
    - Tool has only string params (integers are unreliable from FunctionGemma)
    - All string values appear verbatim in the user's text
    """
    if not _validate(result, tools):
        return False

    calls = result["function_calls"]
    if len(calls) != 1:
        return False

    call = calls[0]
    args = call.get("arguments", {})
    tool_map = {t["name"]: t for t in tools}
    tool = tool_map.get(call["name"])
    if not tool:
        return False

    if not _all_string_params(tool):
        return False

    text_lower = user_text.lower()
    for pname, pspec in tool["parameters"]["properties"].items():
        if pname not in args:
            continue
        val = args[pname]
        if not isinstance(val, str) or len(val) == 0 or len(val) > 200:
            return False
        val_lower = val.lower()
        if val_lower not in text_lower:
            first_word = val_lower.split()[0] if val_lower.split() else ""
            if not first_word or first_word not in text_lower:
                return False

    return True


def _count_expected_calls(text):
    """Estimate how many function calls a query needs."""
    return len(_split_intents(text))


# ===================================================================
# HYBRID ROUTING — decompose, local-first, cloud fallback
# ===================================================================

def _pick_best_tool(sub_text, tools):
    """Pick the single best tool for a sub-query using keyword matching."""
    text_lower = sub_text.lower()
    best = None
    best_score = -1
    for t in tools:
        score = 0
        name_words = t["name"].lower().split("_")
        for w in name_words:
            if w in text_lower:
                score += 3
        desc_words = t.get("description", "").lower().split()
        for w in desc_words:
            if len(w) > 3 and w in text_lower:
                score += 1
        if score > best_score:
            best_score = score
            best = t
    return best


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    Hybrid routing with query decomposition and parallel execution:

    SINGLE INTENT:
      1. Try FunctionGemma → deep validate → return if trusted
      2. Else cloud fallback

    MULTI-INTENT:
      1. Split query into sub-queries
      2. Pick best single tool per sub-query (keyword matching)
      3. Run ALL sub-queries on separate cactus model instances IN PARALLEL,
         AND also run cloud in parallel as a safety net
      4. If all local sub-queries validate → use them (fast, on-device)
      5. If any fail → cloud result is already ready (no extra wait)
    """
    user_text = " ".join(m["content"] for m in messages if m["role"] == "user").strip()
    sub_queries = _split_intents(user_text)

    # ------------------------------------------------------------------
    # SINGLE INTENT
    # ------------------------------------------------------------------
    if len(sub_queries) == 1:
        local = generate_cactus(messages, tools)
        if _deep_validate(local, tools, user_text):
            local["function_calls"] = [
                {"name": c["name"], "arguments": _clean_args(c["arguments"])}
                for c in local["function_calls"]
            ]
            local["source"] = "on-device"
            return local

        cloud = generate_cloud(messages, tools)
        return {
            "function_calls": cloud["function_calls"],
            "total_time_ms": local["total_time_ms"] + cloud["total_time_ms"],
            "source": "cloud (fallback)",
        }

    # ------------------------------------------------------------------
    # MULTI-INTENT — decompose, run local + cloud ALL in parallel
    # ------------------------------------------------------------------
    resolved_subs = []
    for sub_text in sub_queries:
        resolved_subs.append(_resolve_pronouns(sub_text, []))

    start = time.time()
    n_subs = len(resolved_subs)
    with ThreadPoolExecutor(max_workers=n_subs + 1) as pool:
        cloud_future = pool.submit(generate_cloud, messages, tools)

        local_futures = {}
        for i, sub_text in enumerate(resolved_subs):
            sub_msgs = [{"role": "user", "content": sub_text}]
            best_tool = _pick_best_tool(sub_text, tools)
            local_futures[i] = pool.submit(
                _cactus_on_pool, i, sub_msgs, [best_tool] if best_tool else tools
            )

        local_results = {i: f.result() for i, f in local_futures.items()}
        cloud = cloud_future.result()
    wall_ms = (time.time() - start) * 1000

    all_calls = []
    all_valid = True
    for i in range(n_subs):
        local = local_results[i]
        sub_text = resolved_subs[i]
        best_tool = _pick_best_tool(sub_text, tools)
        tool_list = [best_tool] if best_tool else tools

        if _deep_validate(local, tool_list, sub_text):
            cleaned_call = {
                "name": local["function_calls"][0]["name"],
                "arguments": _clean_args(local["function_calls"][0]["arguments"]),
            }
            all_calls.append(cleaned_call)
        else:
            all_valid = False
            break

    if all_valid and len(all_calls) >= n_subs:
        return {
            "function_calls": all_calls,
            "total_time_ms": wall_ms,
            "source": "on-device",
        }

    cloud_calls = cloud["function_calls"]
    if len(cloud_calls) < n_subs:
        used_tools = {c["name"] for c in cloud_calls}
        remaining_tools = [t for t in tools if t["name"] not in used_tools]
        if remaining_tools:
            cloud2 = generate_cloud(messages, remaining_tools)
            wall_ms += cloud2["total_time_ms"]
            cloud_calls = cloud_calls + cloud2["function_calls"]

    return {
        "function_calls": cloud_calls,
        "total_time_ms": wall_ms,
        "source": "cloud (fallback)",
    }


def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
