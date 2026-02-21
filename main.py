
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
            max_tokens=256,
            stop_sequences=["<|im_end|>", "<end_of_turn>"],
            tool_rag_top_k=2,
        )

        try:
            raw = json.loads(raw_str)
        except json.JSONDecodeError:
            return {"function_calls": [], "total_time_ms": 0, "confidence": 0, "cloud_handoff": True}

        return {
            "function_calls": raw.get("function_calls", []),
            "total_time_ms": raw.get("total_time_ms", 0),
            "confidence": raw.get("confidence", 0),
            "cloud_handoff": raw.get("cloud_handoff", False),
        }
    except Exception:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0, "cloud_handoff": True}


def _cactus_on_pool(pool_index, messages, tools, system_prompt=None):
    """Run cactus_complete on a specific pool model (thread-safe)."""
    try:
        model = _get_pool_model(pool_index)
        cactus_reset(model)
        cactus_tools = [{"type": "function", "function": t} for t in tools]
        sys_msg = system_prompt or SYSTEM_PROMPT
        raw_str = cactus_complete(
            model,
            [{"role": "system", "content": sys_msg}] + messages,
            tools=cactus_tools,
            force_tools=True,
            max_tokens=256,
            stop_sequences=["<|im_end|>", "<end_of_turn>"],
            tool_rag_top_k=2,
        )
        try:
            raw = json.loads(raw_str)
        except json.JSONDecodeError:
            return {"function_calls": [], "total_time_ms": 0, "confidence": 0, "cloud_handoff": True}
        return {
            "function_calls": raw.get("function_calls", []),
            "total_time_ms": raw.get("total_time_ms", 0),
            "confidence": raw.get("confidence", 0),
            "cloud_handoff": raw.get("cloud_handoff", False),
        }
    except Exception:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0, "cloud_handoff": True}


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
# QUERY PRE-PROCESSING — normalize before passing to FunctionGemma
# ===================================================================

_PHRASE_REWRITES = [
    (r"\btext\s+(\w+)\s+saying\b", r"send a message to \1 saying"),
    (r"\btext\s+(\w+)\b", r"send a message to \1"),
    (r"\blook up\b", "search for"),
    (r"\bfind\s+(\w+)\s+in my contacts\b", r"search for \1 in contacts"),
    (r"(\d+)\s+minute\s+timer\b", r"\1 minutes timer"),
    (r"(\d+)\s+minute\s+", r"\1 minutes "),
]


def _preprocess_query(text):
    """
    Normalize query before passing to FunctionGemma to improve model comprehension.
    - Rewrite colloquial phrases to explicit tool-matching forms
    - Normalize time: "5 AM" -> "5:00 AM"
    - Strip trailing punctuation
    """
    t = text.strip().rstrip(".,!?;:")
    for pattern, repl in _PHRASE_REWRITES:
        t = re.sub(pattern, repl, t, flags=re.IGNORECASE)
    t = re.sub(r"(?<!\d)(?<!:)(\b\d{1,2})\s+(AM|PM|am|pm)\b", r"\1:00 \2", t)
    return t.strip()


# ===================================================================
# QUERY DECOMPOSITION — split multi-intent into simple sub-queries
# ===================================================================

_SPLIT_MARKERS = [", and ", " and ", ", then ", " then ", ", also ", " also ", ", plus ", " plus ", ", "]

_LEADING_CONJUNCTIONS = ["and ", "then ", "also ", "plus "]


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


def _clean_sub_query(text):
    """Strip leading conjunctions and normalize capitalization."""
    t = text.strip()
    t_lower = t.lower()
    for conj in _LEADING_CONJUNCTIONS:
        if t_lower.startswith(conj):
            t = t[len(conj):].strip()
            if t and t[0].islower():
                t = t[0].upper() + t[1:]
            break
    return t


def _extract_names_from_text(text):
    """Extract likely person names (capitalized words) from text."""
    names = []
    for m in re.finditer(r'\b([A-Z][a-z]+)\b', text):
        name = m.group(1)
        if len(name) > 1 and len(name) < 25 and name.lower() not in ("am", "pm", "i", "me"):
            names.append(name)
    return names


def _resolve_pronouns(sub_text, previous_calls, previous_sub_texts=None):
    """Replace pronouns (him/her/them) with names from previous context.
    Uses previous_calls if available, else extracts names from previous_sub_texts.
    """
    pronouns = ["him", "her", "them", "his", "he", "she", "they"]
    text_lower = sub_text.lower()
    has_pronoun = any(f" {p} " in f" {text_lower} " for p in pronouns)
    if not has_pronoun:
        return sub_text

    last_name = None
    if previous_calls:
        for call in reversed(previous_calls):
            for val in call.get("arguments", {}).values():
                if isinstance(val, str) and val and val[0].isupper() and len(val) < 30:
                    last_name = val
                    break
            if last_name:
                break

    if not last_name and previous_sub_texts:
        for prev in reversed(previous_sub_texts):
            names = _extract_names_from_text(prev)
            if names:
                last_name = names[-1]
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

import re


def _extract_time_tuples(text):
    """Extract (hour, minute) tuples from time expressions in text."""
    times = []
    for m in re.finditer(r'(\d{1,2}):(\d{2})\s*(?:AM|PM|am|pm)?\b', text):
        h, mn = int(m.group(1)), int(m.group(2))
        times.append((h, mn))
    for m in re.finditer(r'(\d{1,2})\s*(?:AM|PM|am|pm)\b', text):
        h = int(m.group(1))
        if not any(t[0] == h for t in times):
            times.append((h, 0))
    return times


def _extract_standalone_numbers(text):
    """Extract plain numbers from text (not already part of time expressions)."""
    nums = set()
    for m in re.finditer(r'\d+', text):
        nums.add(int(m.group()))
    return nums


def _normalize_model_output(result, tools, user_text):
    """
    Fix common FunctionGemma output errors before validation:
    - Negative integers: use abs() when the positive value appears in text
    - Nested dicts: unpack {"minutes": {"minutes": 15}} -> {"minutes": 15}
    Mutates result in place.
    """
    calls = result.get("function_calls", [])
    if not calls:
        return

    time_tuples = _extract_time_tuples(user_text)
    standalone_nums = _extract_standalone_numbers(user_text)
    tool_map = {t["name"]: t for t in tools}

    for call in calls:
        args = call.get("arguments", {})
        tool = tool_map.get(call.get("name"))
        if not tool:
            continue

        props = tool["parameters"]["properties"]
        int_params = {
            p for p, s in props.items()
            if s.get("type", "string").lower() in ("integer", "number")
        }
        has_hour_minute = "hour" in int_params and "minute" in int_params

        for pname, pspec in props.items():
            if pname not in args:
                continue
            val = args[pname]
            ptype = pspec.get("type", "string").lower()

            if ptype in ("integer", "number"):
                if isinstance(val, dict):
                    inner = val.get(pname)
                    if isinstance(inner, (int, float)):
                        args[pname] = int(inner)
                        val = args[pname]
                if isinstance(val, (int, float)) and val < 0:
                    abs_val = int(abs(val))
                    if has_hour_minute and pname in ("hour", "minute"):
                        h = int(args.get("hour", 0)) if isinstance(args.get("hour"), (int, float)) else 0
                        m = int(args.get("minute", 0)) if isinstance(args.get("minute"), (int, float)) else 0
                        if pname == "hour":
                            h = abs_val
                        else:
                            m = abs_val
                        if (abs(h), abs(m)) in time_tuples:
                            args["hour"], args["minute"] = abs(h), abs(m)
                    elif pname not in ("hour", "minute") and abs_val in standalone_nums:
                        args[pname] = abs_val

        call["arguments"] = args


def _validate_integer_args(call_args, tool, user_text):
    """Validate integer arguments using context-aware strategies:
    - If the tool has both 'hour' and 'minute' params, validate as a time tuple
    - Otherwise validate each integer individually against numbers in the text
    """
    props = tool["parameters"]["properties"]
    int_params = {
        pname for pname, pspec in props.items()
        if pspec.get("type", "string").lower() in ("integer", "number")
    }

    has_hour = "hour" in int_params and "hour" in call_args
    has_minute = "minute" in int_params and "minute" in call_args

    if has_hour and has_minute:
        time_tuples = _extract_time_tuples(user_text)
        pair = (int(call_args["hour"]), int(call_args["minute"]))
        if pair not in time_tuples:
            return False
        remaining = int_params - {"hour", "minute"}
    else:
        remaining = int_params

    standalone_nums = _extract_standalone_numbers(user_text)
    for pname in remaining:
        if pname not in call_args:
            continue
        val = call_args[pname]
        if not isinstance(val, (int, float)) or val < 0:
            return False
        if int(val) not in standalone_nums:
            return False

    return True


def _deep_validate(result, tools, user_text):
    """
    Validate local results by checking every argument value is grounded
    in the user's text — works for ALL param types:
      - strings: value (or first word) must appear verbatim in text
      - integer pairs (hour+minute): validated as a time tuple from text
      - other integers: value must appear as a number in text
      - negative numbers: rejected outright
      - empty/missing required args: caught by _validate
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

    if not _validate_integer_args(args, tool, user_text):
        return False

    text_lower = user_text.lower()
    for pname, pspec in tool["parameters"]["properties"].items():
        if pname not in args:
            continue
        val = args[pname]
        ptype = pspec.get("type", "string").lower()

        if ptype == "string":
            if not isinstance(val, str) or len(val) == 0 or len(val) > 200:
                return False
            val_lower = val.lower().replace("'", "")
            text_norm = text_lower.replace("'", "")
            if val_lower in text_norm:
                continue
            val_words = val_lower.split()
            if not val_words:
                continue
            ungrounded = [w for w in val_words if w not in text_norm]
            if ungrounded:
                return False

        elif ptype == "boolean":
            if not isinstance(val, bool):
                return False

    return True


def _debug_validation(result, tools, user_text):
    """Return a string explaining why _deep_validate failed."""
    calls = result.get("function_calls", [])
    if not calls:
        return "no function calls returned"
    if not _validate(result, tools):
        call_names = [c.get("name", "?") for c in calls]
        tool_names = {t["name"] for t in tools}
        bad = [n for n in call_names if n not in tool_names]
        if bad:
            return f"unknown tool(s): {bad}"
        for c in calls:
            tool = next((t for t in tools if t["name"] == c.get("name")), None)
            if tool:
                required = set(tool["parameters"].get("required", []))
                args = c.get("arguments", {})
                missing = [r for r in required if r not in args or args[r] is None or args[r] == ""]
                if missing:
                    return f"missing required args {missing} for {c['name']}"
        return "structural validation failed"
    if len(calls) != 1:
        return f"expected 1 call, got {len(calls)}"
    call = calls[0]
    args = call.get("arguments", {})
    tool = next((t for t in tools if t["name"] == call["name"]), None)
    if not tool:
        return f"unknown tool: {call['name']}"
    if not _validate_integer_args(args, tool, user_text):
        int_args = {k: v for k, v in args.items()
                    if tool["parameters"]["properties"].get(k, {}).get("type", "").lower() in ("integer", "number")}
        return f"integer args failed: {int_args} vs text '{user_text}'"
    text_lower = user_text.lower()
    for pname, pspec in tool["parameters"]["properties"].items():
        if pname not in args:
            continue
        val = args[pname]
        ptype = pspec.get("type", "string").lower()
        if ptype == "string":
            if not isinstance(val, str) or len(val) == 0:
                return f"string param '{pname}' empty or non-string: {val!r}"
            val_norm = val.lower().replace("'", "")
            text_norm = text_lower.replace("'", "")
            if val_norm not in text_norm:
                val_words = val_norm.split()
                ungrounded = [w for w in val_words if w not in text_norm]
                if ungrounded:
                    return f"string param '{pname}'={val!r} has ungrounded words: {ungrounded}"
    return "unknown reason"


def _count_expected_calls(text):
    """Estimate how many function calls a query needs."""
    return len(_split_intents(text))


def _pick_best_pass(results, tools, user_text):
    """Given a list of cactus results from multiple passes, return the first
    one that passes _deep_validate. Skips results where cloud_handoff is True.
    Returns (result, pass_index) or (None, -1)."""
    for i, r in enumerate(results):
        if r.get("cloud_handoff", False):
            continue
        _normalize_model_output(r, tools, user_text)
        if _deep_validate(r, tools, user_text):
            return r, i
    return None, -1


# ===================================================================
# HYBRID ROUTING — double-pass local, cloud fallback
# ===================================================================

def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    Hybrid routing — double-pass FunctionGemma with probabilistic validation.

    SINGLE INTENT:
      1. Run FunctionGemma TWICE (different prompts) + Cloud ALL in parallel
      2. If either local pass validates via _deep_validate → return it (on-device)
      3. Else return cloud result (already computed, no extra wait)

    MULTI-INTENT:
      1. Split query into sub-queries
      2. Each sub-query gets TWO local passes (parallel) + cloud safety net
      3. For each sub, pick the first validated pass
      4. If all subs have a valid local result → combine (on-device)
      5. If any fail → use cloud
    """
    raw_text = " ".join(m["content"] for m in messages if m["role"] == "user").strip()
    user_text = _preprocess_query(raw_text)
    sub_queries = _split_intents(user_text)

    messages_for_model = [{"role": "user", "content": user_text}]

    # ------------------------------------------------------------------
    # SINGLE INTENT — local pass 1, validate, optional pass 2, cloud
    # background. Only ONE cactus model active at a time to avoid
    # resource contention that causes empty/garbage outputs.
    # ------------------------------------------------------------------
    if len(sub_queries) == 1:
        start = time.time()

        with ThreadPoolExecutor(max_workers=1) as bg:
            cloud_future = bg.submit(generate_cloud, messages_for_model, tools)

            local1 = generate_cactus(messages_for_model, tools)
            if _deep_validate(local1, tools, user_text):
                local_done_ms = (time.time() - start) * 1000
                local1["function_calls"] = [
                    {"name": c["name"], "arguments": _clean_args(c["arguments"])}
                    for c in local1["function_calls"]
                ]
                local1["source"] = "on-device"
                local1["total_time_ms"] = local_done_ms
                local1["_debug"] = {
                    "path": "single/local-pass1",
                    "local_calls": local1["function_calls"],
                    "validation": "passed",
                }
                return local1

            local2 = generate_cactus(messages_for_model, tools)
            if _deep_validate(local2, tools, user_text):
                local_done_ms = (time.time() - start) * 1000
                local2["function_calls"] = [
                    {"name": c["name"], "arguments": _clean_args(c["arguments"])}
                    for c in local2["function_calls"]
                ]
                local2["source"] = "on-device"
                local2["total_time_ms"] = local_done_ms
                local2["_debug"] = {
                    "path": "single/local-pass2",
                    "local_calls": local2["function_calls"],
                    "validation": "passed",
                }
                return local2

            cloud = cloud_future.result()

        wall_ms = (time.time() - start) * 1000
        return {
            "function_calls": cloud["function_calls"],
            "total_time_ms": wall_ms,
            "source": "cloud (fallback)",
            "_debug": {
                "path": "single/cloud-fallback",
                "local1_calls": local1.get("function_calls", []),
                "local2_calls": local2.get("function_calls", []),
                "validation_p1": _debug_validation(local1, tools, user_text),
                "validation_p2": _debug_validation(local2, tools, user_text),
            },
        }

    # ------------------------------------------------------------------
    # MULTI-INTENT — split into sub-queries, clean, resolve pronouns,
    # run each through local sequentially (2 passes each), cloud in background
    # ------------------------------------------------------------------
    cleaned_subs = [_clean_sub_query(s) for s in sub_queries]
    resolved_subs = []
    for i, sub_text in enumerate(cleaned_subs):
        prev_texts = cleaned_subs[:i]
        resolved_subs.append(_resolve_pronouns(sub_text, [], previous_sub_texts=prev_texts))

    start = time.time()
    n_subs = len(resolved_subs)

    with ThreadPoolExecutor(max_workers=1) as bg:
        cloud_future = bg.submit(generate_cloud, messages_for_model, tools)

        all_calls = []
        all_valid = True
        sub_debug = []
        failed_idx = -1

        for i in range(n_subs):
            sub_text = resolved_subs[i]
            sub_msgs = [{"role": "user", "content": sub_text}]

            p1 = generate_cactus(sub_msgs, tools)
            _normalize_model_output(p1, tools, sub_text)
            if not p1.get("cloud_handoff", False) and _deep_validate(p1, tools, sub_text):
                cleaned_call = {
                    "name": p1["function_calls"][0]["name"],
                    "arguments": _clean_args(p1["function_calls"][0]["arguments"]),
                }
                all_calls.append(cleaned_call)
                sub_debug.append({
                    "sub": sub_text, "status": "passed", "pass": 1,
                    "local_calls": p1.get("function_calls", []),
                })
                continue

            p2 = generate_cactus(sub_msgs, tools)
            _normalize_model_output(p2, tools, sub_text)
            if not p2.get("cloud_handoff", False) and _deep_validate(p2, tools, sub_text):
                cleaned_call = {
                    "name": p2["function_calls"][0]["name"],
                    "arguments": _clean_args(p2["function_calls"][0]["arguments"]),
                }
                all_calls.append(cleaned_call)
                sub_debug.append({
                    "sub": sub_text, "status": "passed", "pass": 2,
                    "local_calls": p2.get("function_calls", []),
                })
                continue

            all_valid = False
            failed_idx = i
            sub_debug.append({
                "sub": sub_text,
                "status": _debug_validation(p1, tools, sub_text),
                "local_calls_p1": p1.get("function_calls", []),
                "local_calls_p2": p2.get("function_calls", []),
            })
            break

        if all_valid and len(all_calls) >= n_subs:
            local_done_ms = (time.time() - start) * 1000
            return {
                "function_calls": all_calls,
                "total_time_ms": local_done_ms,
                "source": "on-device",
                "_debug": {"path": "multi/all-local", "subs": sub_debug},
            }

        cloud = cloud_future.result()

    cloud_calls = cloud["function_calls"]
    if len(cloud_calls) < n_subs:
        used_tools = {c["name"] for c in cloud_calls}
        remaining_tools = [t for t in tools if t["name"] not in used_tools]
        if remaining_tools:
            cloud2 = generate_cloud(messages_for_model, remaining_tools)
            cloud_calls = cloud_calls + cloud2["function_calls"]

    wall_ms = (time.time() - start) * 1000
    return {
        "function_calls": cloud_calls,
        "total_time_ms": wall_ms,
        "source": "cloud (fallback)",
        "_debug": {"path": "multi/cloud-fallback", "failed_sub": failed_idx, "subs": sub_debug},
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
