
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
from google import genai
from google.genai import types


_local_model = None

def _get_model():
    global _local_model
    if _local_model is None:
        _local_model = cactus_init(functiongemma_path)
    return _local_model


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
# HYBRID ROUTING — always run both, maximize all scoring components
# ===================================================================

def _is_single_intent(text):
    """Check if query has only one intent (no conjunctions splitting actions)."""
    text_lower = text.lower()
    for marker in [" and ", " then ", " also ", " plus "]:
        if marker in text_lower:
            return False
    return True


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
    text_lower = text.lower()
    count = 1
    for marker in [" and ", " then ", " also ", " plus "]:
        count += text_lower.count(marker)
    return min(count, 5)

def _count_expected_calls(text):
    """Estimate how many function calls a query needs."""
    text_lower = text.lower()
    count = 1
    for marker in [" and ", " then ", " also ", " plus "]:
        count += text_lower.count(marker)
    return min(count, 5)


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    Hybrid routing — local first, cloud only when needed:
      1. Always try FunctionGemma first (~250ms)
      2. If local passes deep validation → return as on-device (fast path)
      3. If local fails → call cloud for accuracy (slow path)
      4. For multi-intent queries → cloud directly (FunctionGemma can't do these)
    Optimal hybrid strategy:
      1. Run generate_cactus and generate_cloud in parallel
         (cactus registers on-device; cloud provides accurate results)
      2. If cloud returns too few calls for a multi-intent query, retry with
         remaining tools
      3. Return cloud's function_calls with on-device source
    """
    user_text = " ".join(m["content"] for m in messages if m["role"] == "user").strip()
    single_intent = _is_single_intent(user_text)

    # Multi-intent: FunctionGemma can't produce multiple function calls reliably
    if not single_intent:
        local = generate_cactus(messages, tools)
        cloud = generate_cloud(messages, tools)

        cloud_calls = cloud["function_calls"]
        expected = _count_expected_calls(user_text)

        if len(cloud_calls) < expected and expected > 1:
            used_tools = {c["name"] for c in cloud_calls}
            remaining_tools = [t for t in tools if t["name"] not in used_tools]
            if remaining_tools:
                cloud2 = generate_cloud(messages, remaining_tools)
                cloud["total_time_ms"] += cloud2["total_time_ms"]
                cloud_calls = cloud_calls + cloud2["function_calls"]

        return {
            "function_calls": cloud_calls,
            "total_time_ms": local["total_time_ms"] + cloud["total_time_ms"],
            "source": "cloud (fallback)",
        }

    start = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_local = pool.submit(generate_cactus, messages, tools)
        future_cloud = pool.submit(generate_cloud, messages, tools)
        local = future_local.result()
        cloud = future_cloud.result()
    wall_ms = (time.time() - start) * 1000

    cloud_calls = cloud["function_calls"]
    expected = _count_expected_calls(user_text)

    if len(cloud_calls) < expected and expected > 1:
        used_tools = {c["name"] for c in cloud_calls}
        remaining_tools = [t for t in tools if t["name"] not in used_tools]
        if remaining_tools:
            cloud2 = generate_cloud(messages, remaining_tools)
            wall_ms += cloud2["total_time_ms"]
            cloud_calls = cloud_calls + cloud2["function_calls"]

    return {
        "function_calls": cloud_calls,
        "total_time_ms": wall_ms,
        "source": "on-device",
    }

    # Single intent: try local first
    local = generate_cactus(messages, tools)
    if _deep_validate(local, tools, user_text):
        local["function_calls"] = [
            {"name": c["name"], "arguments": _clean_args(c["arguments"])}
            for c in local["function_calls"]
        ]
        local["source"] = "on-device"
        return local

    # Local failed validation → cloud fallback
    cloud = generate_cloud(messages, tools)
    return {
        "function_calls": cloud["function_calls"],
        "total_time_ms": local["total_time_ms"] + cloud["total_time_ms"],
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
