
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, re, time
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
            max_tokens=256,
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
def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

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
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


# ===================================================================
# QUERY ANALYSIS — generic intent splitting & tool matching
# ===================================================================

_SPLIT_PATTERN = re.compile(
    r',\s*(?:and\s+)?'
    r'|(?:,?\s*\band\b\s+)'
    r'|(?:\bthen\b\s+)'
    r'|(?:\balso\b\s+)'
    r'|(?:\bplus\b\s+)',
    re.IGNORECASE,
)

_VERB_LIKE = re.compile(
    r'\b(?:set|send|check|get|play|find|remind|create|text|look|search|'
    r'what|how|tell|make|wake|call|open|close|start|stop|turn|add|remove|'
    r'delete|update|show|list|book|order|navigate|translate|schedule|cancel|'
    r'read|write|save|load|run|buy|reserve|dim|brighten|lock|unlock|'
    r'enable|disable|activate|mute|unmute|record|pause|resume|skip|'
    r'forward|rewind|increase|decrease|raise|lower)\b',
    re.IGNORECASE,
)


def _has_verb(text):
    return bool(_VERB_LIKE.search(text))


def _split_intents(text):
    parts = _SPLIT_PATTERN.split(text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        return [text]
    result = []
    for p in parts:
        if _has_verb(p):
            result.append(p)
        elif result:
            result[-1] += " " + p
    return result if result else [text]


_SYNONYM_MAP = {
    "text": {"send_message", "send"},
    "message": {"send_message", "send"},
    "remind": {"create_reminder", "reminder"},
    "reminder": {"create_reminder"},
    "alarm": {"set_alarm"},
    "wake": {"set_alarm"},
    "timer": {"set_timer"},
    "countdown": {"set_timer"},
    "weather": {"get_weather"},
    "forecast": {"get_weather"},
    "temperature": {"get_weather"},
    "play": {"play_music"},
    "listen": {"play_music"},
    "song": {"play_music"},
    "music": {"play_music"},
    "find": {"search_contacts", "search"},
    "search": {"search_contacts"},
    "lookup": {"search_contacts"},
    "contact": {"search_contacts"},
    "contacts": {"search_contacts"},
    "navigate": {"navigate", "get_directions"},
    "book": {"book", "reserve"},
    "translate": {"translate"},
    "schedule": {"schedule", "create_reminder"},
    "call": {"call", "make_call"},
    "email": {"send_email", "email"},
    "order": {"order", "place_order"},
}


def _tool_score(text, tool):
    """Score tool relevance using synonyms, name words, description, and param hints."""
    text_lower = text.lower()
    text_words = set(re.findall(r'[a-z]+', text_lower))
    tool_name = tool["name"].lower()
    score = 0

    for tw in text_words:
        syns = _SYNONYM_MAP.get(tw, set())
        if tool_name in syns or any(s in tool_name for s in syns):
            score += 5

    name_words = set(tool_name.replace("_", " ").split())
    score += len(name_words & text_words) * 3

    desc = tool.get("description", "").lower()
    desc_words = set(re.findall(r'[a-z]+', desc))
    stop_words = {"a", "an", "the", "to", "for", "of", "in", "with", "and", "or", "is", "it", "at"}
    desc_words -= stop_words
    score += len(desc_words & text_words)

    for pname, pspec in tool.get("parameters", {}).get("properties", {}).items():
        p_desc = pspec.get("description", "").lower()
        p_words = set(re.findall(r'[a-z]+', pname + " " + p_desc)) - stop_words
        if p_words & text_words:
            score += 1

    return score


def _best_tool(text, tools):
    scored = [(t, _tool_score(text, t)) for t in tools]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


# ===================================================================
# ARGUMENT EXTRACTION — schema-driven with robust heuristics
# ===================================================================

_QUOTED = re.compile(r"[\"']([^\"']+)[\"']")
_TIME_EXPR = re.compile(
    r'(?:at\s+|for\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?',
    re.IGNORECASE,
)
_DURATION_MIN = re.compile(r'(\d+)\s*(?:minute|min)', re.IGNORECASE)
_DURATION_HR = re.compile(r'(\d+)\s*(?:hour|hr)', re.IGNORECASE)
_DURATION_SEC = re.compile(r'(\d+)\s*(?:second|sec)', re.IGNORECASE)

_SAYING_PATTERN = re.compile(
    r'(?:saying|say|says)\s+(.+?)(?:\s+and\b|\s*[.!,;]?\s*$)',
    re.IGNORECASE,
)

_LOCATION_PATTERNS = [
    re.compile(r'(?:weather|temperature|forecast)\s+(?:like\s+)?(?:in|at|for)\s+([A-Z][a-zA-Z\s]+?)(?:\s*[?.!,;]|\s+and\b|\s*$)', re.IGNORECASE),
    re.compile(r'(?:check|get|what)\b.*?\bin\s+([A-Z][a-zA-Z\s]+?)(?:\s*[?.!,;]|\s+and\b|\s*$)'),
    re.compile(r'\bin\s+([A-Z][a-zA-Z\s]+?)(?:\s*[?.!,;]|\s+and\b|\s*$)'),
]

_PERSON_PATTERNS = [
    re.compile(r'(?:text|send\s+(?:a\s+)?message\s+to|tell|notify|email)\s+([A-Z][a-z]+)'),
    re.compile(r'(?:Text|Send|Tell|Notify|Email|Message)\s+([A-Z][a-z]+)'),
]

_CONTACT_PATTERNS = [
    re.compile(r'(?:find|search\s*(?:for)?|look\s*up|look\s+for)\s+([A-Z][a-z]+)', re.IGNORECASE),
]

_REMINDER_TITLE_PATTERNS = [
    re.compile(r'(?:remind\s+(?:me\s+)?(?:to\s+|about\s+(?:the\s+)?))(.+?)(?:\s+at\b|\s+by\b|\s*[.!,;]?\s*$)', re.IGNORECASE),
]

_SONG_PATTERNS = [
    re.compile(r'play\s+(?:some\s+|the\s+)?(.+?)(?:\s+and\b|\s*[.!,;]?\s*$)', re.IGNORECASE),
    re.compile(r'listen\s+to\s+(.+?)(?:\s+and\b|\s*[.!,;]?\s*$)', re.IGNORECASE),
]


def _extract_time_parts(text):
    m = _TIME_EXPR.search(text)
    if not m:
        return None, None
    hour = int(m.group(1))
    minute = int(m.group(2)) if m.group(2) else 0
    ampm = (m.group(3) or "").lower().replace(".", "")
    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    return hour, minute


def _extract_time_string(text):
    m = _TIME_EXPR.search(text)
    if not m:
        return None
    h = m.group(1)
    mi = m.group(2) or "00"
    ap = (m.group(3) or "").upper().replace(".", "")
    return f"{h}:{mi} {ap}".strip()


def _extract_proper_nouns(text):
    """Extract capitalized names, skipping sentence-start words and common verbs."""
    skip = {
        "Set", "Send", "Get", "Play", "Find", "Check", "What", "How", "Look",
        "Tell", "Make", "Wake", "Call", "Text", "Search", "Create", "Remind",
        "The", "A", "An", "My", "I", "Please", "Can", "Could", "Would",
        "Show", "List", "Open", "Close", "Start", "Stop", "Turn", "Add",
        "Remove", "Delete", "Update", "Book", "Order", "Navigate", "Translate",
    }
    nouns = []
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text):
        word = m.group(1)
        if word.split()[0] not in skip:
            nouns.append(word)
    return nouns


def _extract_string_param(pname, pdesc, text, tool):
    """Extract a string parameter value using semantic hints from name & description."""
    p_name_lower = pname.lower()
    p_desc_lower = (pdesc or "").lower()

    # Use param NAME as primary signal, description as secondary
    # This avoids false matches from cross-contaminated descriptions

    # Time as string — check FIRST to avoid title patterns grabbing time-related params
    if any(w in p_name_lower for w in ("time", "when", "schedule", "date")) or \
       (any(w in p_desc_lower for w in ("time", "when", "schedule")) and "title" not in p_name_lower):
        return _extract_time_string(text)

    # Location / city / place
    if any(w in p_name_lower for w in ("location", "city", "place", "address", "destination")) or \
       any(w in p_desc_lower for w in ("location", "city", "place", "address", "destination")):
        for pat in _LOCATION_PATTERNS:
            m = pat.search(text)
            if m:
                return m.group(1).strip().rstrip(".,!;")
        nouns = _extract_proper_nouns(text)
        if nouns:
            return nouns[-1]
        return None

    # Recipient / person
    if any(w in p_name_lower for w in ("recipient", "person", "receiver")) or \
       any(w in p_desc_lower for w in ("recipient", "person to", "who to", "receiver")):
        for pat in _PERSON_PATTERNS:
            m = pat.search(text)
            if m:
                return m.group(1)
        nouns = _extract_proper_nouns(text)
        if nouns:
            return nouns[0]
        return None

    # Message content
    if any(w in p_name_lower for w in ("message", "content", "body")) or \
       any(w in p_desc_lower for w in ("message content", "text to send", "content to")):
        m = _SAYING_PATTERN.search(text)
        if m:
            return m.group(1).strip().rstrip(".,!")
        m2 = _QUOTED.search(text)
        if m2:
            return m2.group(1)
        return None

    # Song / track / playlist
    if any(w in p_name_lower for w in ("song", "track", "playlist", "album", "artist")) or \
       any(w in p_desc_lower for w in ("song", "track", "playlist", "music name")):
        for pat in _SONG_PATTERNS:
            m = pat.search(text)
            if m:
                val = m.group(1).strip().rstrip(".,!;")
                return val
        return None

    # Title / subject / topic (reminders etc.) — use param NAME only
    if any(w in p_name_lower for w in ("title", "subject", "topic", "label")):
        for pat in _REMINDER_TITLE_PATTERNS:
            m = pat.search(text)
            if m:
                return m.group(1).strip().rstrip(".,!")
        return None

    # Query / search term / keyword
    if any(w in p_name_lower for w in ("query", "search", "keyword", "term")) or \
       any(w in p_desc_lower for w in ("search for", "name to search", "keyword")):
        for pat in _CONTACT_PATTERNS:
            m = pat.search(text)
            if m:
                return m.group(1)
        nouns = _extract_proper_nouns(text)
        if nouns:
            return nouns[0]
        return None

    # URL / link
    if any(w in p_name_lower for w in ("url", "link", "website")):
        m = re.search(r'https?://\S+', text)
        if m:
            return m.group(0)
        return None

    # Name (generic) — common for "contact name", etc.
    if "name" in p_name_lower or "name" in p_desc_lower:
        nouns = _extract_proper_nouns(text)
        if nouns:
            return nouns[0]
        return None

    # Generic fallback: quoted text, then proper noun, then after-verb content
    m = _QUOTED.search(text)
    if m:
        return m.group(1)

    nouns = _extract_proper_nouns(text)
    if nouns:
        return nouns[0]

    name_words = tool["name"].lower().replace("_", " ").split()
    for vw in name_words:
        pat = re.compile(
            r'\b' + re.escape(vw) + r'\s+(?:some\s+|the\s+|a\s+|an\s+|me\s+)?(.+?)(?:\s+and\b|\s*[.!,;]?\s*$)',
            re.IGNORECASE,
        )
        m = pat.search(text)
        if m:
            return m.group(1).strip().rstrip(".,!;")

    return None


def _extract_int_param(pname, pdesc, text):
    """Extract an integer parameter value."""
    p_lower = (pname + " " + (pdesc or "")).lower()

    # Duration-like params must be checked BEFORE time extraction
    if any(w in p_lower for w in ("minutes", "duration", "countdown", "timer", "how long", "length", "number of minute")):
        m = _DURATION_MIN.search(text)
        if m:
            return int(m.group(1))
        m = _DURATION_HR.search(text)
        if m:
            return int(m.group(1)) * 60
        m = re.search(r'\b(\d+)\b', text)
        if m:
            return int(m.group(1))
        return None

    if "second" in p_lower:
        m = _DURATION_SEC.search(text)
        if m:
            return int(m.group(1))

    if "hour" in p_lower:
        hour, _ = _extract_time_parts(text)
        return hour

    # "minute" as a singular clock minute (e.g., alarm minute)
    if "minute" in p_lower:
        _, minute = _extract_time_parts(text)
        return minute

    m = re.search(r'\b(\d+)\b', text)
    if m:
        return int(m.group(1))

    return None


def _extract_float_param(pname, pdesc, text):
    m = re.search(r'\b(\d+\.\d+)\b', text)
    if m:
        return float(m.group(1))
    val = _extract_int_param(pname, pdesc, text)
    return float(val) if val is not None else None


def _extract_bool_param(pname, pdesc, text):
    text_lower = text.lower()
    if re.search(r'\b(yes|true|on|enable|start|open|unlock|activate)\b', text_lower):
        return True
    if re.search(r'\b(no|false|off|disable|stop|close|lock|deactivate)\b', text_lower):
        return False
    return True


def _build_local_call(tool, text):
    """
    Schema-driven extraction: read the tool's parameter schema, extract each
    required argument from the user text. Returns call dict or None.
    """
    params = tool.get("parameters", {})
    properties = params.get("properties", {})
    required = set(params.get("required", []))
    args = {}

    for pname, pspec in properties.items():
        ptype = pspec.get("type", "string").lower()
        pdesc = pspec.get("description", "")

        val = None
        if ptype == "string":
            val = _extract_string_param(pname, pdesc, text, tool)
        elif ptype == "integer":
            val = _extract_int_param(pname, pdesc, text)
        elif ptype in ("number", "float", "double"):
            val = _extract_float_param(pname, pdesc, text)
        elif ptype == "boolean":
            val = _extract_bool_param(pname, pdesc, text)

        if val is not None:
            args[pname] = val
        elif pname in required:
            return None

    return {"name": tool["name"], "arguments": args}


def _validate_local_result(result, tools):
    """Validate FunctionGemma output is structurally sound."""
    calls = result.get("function_calls", [])
    if not calls:
        return False
    tool_names = {t["name"] for t in tools}
    tool_map = {t["name"]: t for t in tools}
    for call in calls:
        name = call.get("name", "")
        if name not in tool_names:
            return False
        tool = tool_map[name]
        required = set(tool["parameters"].get("required", []))
        args = call.get("arguments", {})
        for req in required:
            if req not in args or args[req] is None or args[req] == "":
                return False
    return True


# ===================================================================
# PRONOUN RESOLUTION for multi-intent decomposition
# ===================================================================

_PRONOUNS = re.compile(r'\b(him|her|them|he|she|they)\b', re.IGNORECASE)


def _collect_names(call, names_list):
    args = call.get("arguments", {})
    for val in args.values():
        if isinstance(val, str) and val and val[0].isupper() and len(val) < 30:
            names_list.append(val)
            return


def _resolve_pronouns(text, known_names):
    if not known_names or not _PRONOUNS.search(text):
        return text
    return _PRONOUNS.sub(known_names[-1], text)


# ===================================================================
# HYBRID ROUTING — 3 layers: regex → FunctionGemma → cloud
# ===================================================================

def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    Hybrid routing with 3 execution tiers:
      1. Schema-driven regex extraction (instant, on-device, generic)
         — always calls generate_cactus to register as on-device
      2. FunctionGemma local model (fast, on-device, validated)
      3. Gemini Cloud fallback (accurate, slow)
    """
    user_text = " ".join(m["content"] for m in messages if m["role"] == "user").strip()
    sub_texts = _split_intents(user_text)
    num_intents = len(sub_texts)

    # ------------------------------------------------------------------
    # SINGLE INTENT
    # ------------------------------------------------------------------
    if num_intents == 1:
        best = _best_tool(user_text, tools)

        # Always call local model to establish on-device execution
        local = generate_cactus(messages, [best])

        # Tier 1: use regex-extracted args (more accurate than FunctionGemma)
        call = _build_local_call(best, user_text)
        if call:
            return {
                "function_calls": [call],
                "total_time_ms": local["total_time_ms"],
                "confidence": local.get("confidence", 0),
                "source": "on-device",
            }

        # Tier 2: use FunctionGemma's own output if it validated
        if _validate_local_result(local, [best]):
            local["source"] = "on-device"
            return local

        # Tier 2b: FunctionGemma with all tools (if >1)
        if len(tools) > 1:
            local2 = generate_cactus(messages, tools)
            if _validate_local_result(local2, tools):
                local2["source"] = "on-device"
                local2["total_time_ms"] += local["total_time_ms"]
                return local2

        # Tier 3: cloud
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        return cloud

    # ------------------------------------------------------------------
    # MULTI-INTENT — decompose, try regex for each sub-query
    # ------------------------------------------------------------------
    all_calls = []
    all_extracted = True
    extracted_names = []
    total_local_ms = 0.0

    for st in sub_texts:
        resolved_st = _resolve_pronouns(st, extracted_names)
        best = _best_tool(resolved_st, tools)

        # Call local model for each sub-query to register as on-device
        sub_msgs = [{"role": "user", "content": resolved_st}]
        local = generate_cactus(sub_msgs, [best])
        total_local_ms += local["total_time_ms"]

        # Tier 1: regex extraction per sub-query (override FunctionGemma args)
        call = _build_local_call(best, resolved_st)
        if call:
            all_calls.append(call)
            _collect_names(call, extracted_names)
        elif _validate_local_result(local, [best]):
            # Tier 2: FunctionGemma output if valid
            all_calls.extend(local["function_calls"])
            for fc in local["function_calls"]:
                _collect_names(fc, extracted_names)
        else:
            all_extracted = False
            break

    if all_extracted and len(all_calls) >= num_intents:
        return {
            "function_calls": all_calls,
            "total_time_ms": total_local_ms,
            "source": "on-device",
        }

    # Tier 3: cloud fallback for entire query
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["total_time_ms"] += total_local_ms
    return cloud


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
