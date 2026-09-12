"""
config/llm_client.py
Provider-agnostic LLM client for Chapter 9.

Supports three providers behind one interface:
  LLM_PROVIDER=anthropic  → Claude Sonnet/Haiku via Anthropic SDK
  LLM_PROVIDER=openai     → GPT-4o/GPT-4o-mini via OpenAI SDK
  LLM_PROVIDER=mock       → scripted responses, no API key needed

All three return the same normalised dict from parse_response():
  {
    "text":          str,   # assistant text (if any)
    "tool_name":     str,   # tool called (if any)
    "tool_input":    dict,  # tool arguments
    "tool_use_id":   str,   # first tool-call id (backward compatibility)
    "tool_calls":    list,  # all tool calls in the assistant turn
    "stop_reason":   str,   # "end_turn" | "tool_use"
    "input_tokens":  int,
    "output_tokens": int,
    "raw":           obj,   # the raw SDK response (for multi-turn history)
  }

The agents use call_llm() — the single entry point that dispatches to
the right backend. Multi-turn history building is also provider-aware.
"""
import json
import uuid as _uuid
from typing import Any

from config.settings import (
    LLM_PROVIDER, MOCK_LLM,
    ANTHROPIC_KEY, OPENAI_KEY,
    ANTHROPIC_MODEL_SONNET, ANTHROPIC_MODEL_HAIKU,
    OPENAI_MODEL_GPT4O, OPENAI_MODEL_GPT4O_MINI,
)


# ─────────────────────────────────────────────────────────────────
# Normalised response shape
# ─────────────────────────────────────────────────────────────────

def _empty_response() -> dict:
    return {
        "text": "", "tool_name": None, "tool_input": {},
        "tool_use_id": None, "tool_calls": [], "stop_reason": "end_turn",
        "input_tokens": 0, "output_tokens": 0, "raw": None,
    }


# ─────────────────────────────────────────────────────────────────
# Anthropic backend
# ─────────────────────────────────────────────────────────────────

def get_anthropic_client():
    import anthropic
    return anthropic.Anthropic(api_key=ANTHROPIC_KEY)


def _parse_anthropic(response) -> dict:
    """Normalise an Anthropic messages.create() response."""
    result = _empty_response()
    result["stop_reason"]    = response.stop_reason
    result["input_tokens"]   = response.usage.input_tokens
    result["output_tokens"]  = response.usage.output_tokens
    result["raw"]            = response
    for block in response.content:
        if block.type == "text":
            result["text"] = block.text
        elif block.type == "tool_use":
            result["tool_name"]   = block.name
            result["tool_input"]  = block.input
            result["tool_use_id"] = block.id
    return result


def _anthropic_tools(tools: list) -> list:
    """Tools already in Anthropic format — pass through unchanged."""
    return tools


def _anthropic_next_messages(messages: list, parsed: dict, tool_result_str: str) -> list:
    """
    Build the next messages list for Anthropic multi-turn tool_use.
    Appends the assistant's tool_use block then the tool_result.
    """
    raw = parsed["raw"]
    messages = messages + [
        {"role": "assistant", "content": raw.content},
        {"role": "user", "content": [{
            "type": "tool_result",
            "tool_use_id": parsed["tool_use_id"],
            "content": tool_result_str,
        }]},
    ]
    return messages


# ─────────────────────────────────────────────────────────────────
# OpenAI backend
# ─────────────────────────────────────────────────────────────────

def get_openai_client():
    import openai
    return openai.OpenAI(api_key=OPENAI_KEY)


def _parse_openai(response) -> dict:
    """Normalise an OpenAI chat.completions.create() response."""
    result  = _empty_response()
    result["raw"] = response
    choice  = response.choices[0]
    message = choice.message
    usage   = response.usage

    result["input_tokens"]  = usage.prompt_tokens
    result["output_tokens"] = usage.completion_tokens

    if choice.finish_reason == "tool_calls":
        result["stop_reason"] = "tool_use"
        for tc in message.tool_calls or []:
            try:
                tool_input = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_input = {}
            result["tool_calls"].append({
                "name": tc.function.name,
                "input": tool_input,
                "id": tc.id,
            })
        if result["tool_calls"]:
            first_call = result["tool_calls"][0]
            result["tool_name"]   = first_call["name"]
            result["tool_input"]   = first_call["input"]
            result["tool_use_id"]  = first_call["id"]
    else:
        result["stop_reason"] = "end_turn"
        result["text"] = message.content or ""

    return result


def _anthropic_tools_to_openai(tools: list) -> list:
    """
    Convert Anthropic tool_use format → OpenAI function-calling format.

    Anthropic:
      {"name": "...", "description": "...", "input_schema": {...}}

    OpenAI:
      {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    """
    converted = []
    for t in tools:
        converted.append({
            "type": "function",
            "function": {
                "name":        t["name"],
                "description": t.get("description", ""),
                "parameters":  t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return converted


def _openai_next_messages(
    messages: list,
    parsed: dict,
    tool_result_str: str,
    tool_results: list[dict] = None,
) -> list:
    """
    Build next messages for OpenAI multi-turn function-calling.
    Appends assistant message (with tool_calls) then a tool-role message.
    """
    raw     = parsed["raw"]
    message = raw.choices[0].message

    # assistant turn — include the full message object so tool_calls is preserved
    tool_calls = parsed.get("tool_calls") or [{
        "name": parsed["tool_name"],
        "input": parsed["tool_input"],
        "id": parsed["tool_use_id"],
    }]
    if tool_results is None:
        tool_results = [{"id": tool_calls[0]["id"], "content": tool_result_str}]

    messages = messages + [
        {
            "role":       "assistant",
            "content":    message.content,   # may be None for tool-only turns
            "tool_calls": [
                {
                    "id":   tc["id"],
                    "type": "function",
                    "function": {
                        "name":      tc["name"],
                        "arguments": json.dumps(tc["input"]),
                    },
                }
                for tc in tool_calls
            ],
        }
    ] + [
        {
            "role":         "tool",
            "tool_call_id": result["id"],
            "content":      result["content"],
        }
        for result in tool_results
    ]
    return messages


# ─────────────────────────────────────────────────────────────────
# Unified call_llm() — the single entry point agents use
# ─────────────────────────────────────────────────────────────────

_anthropic_client = None
_openai_client    = None


def _get_client():
    global _anthropic_client, _openai_client
    if LLM_PROVIDER == "openai":
        if _openai_client is None:
            _openai_client = get_openai_client()
        return _openai_client, "openai"
    else:
        if _anthropic_client is None:
            _anthropic_client = get_anthropic_client()
        return _anthropic_client, "anthropic"


def call_llm(
    model: str,
    messages: list,
    tools: list = None,
    system: str = "",
    max_tokens: int = 1024,
) -> dict:
    """
    Make one LLM call with the configured provider and return a normalised dict.

    Args:
        model:      Provider-specific model string (from get_capable_llm_model() etc.)
        messages:   Conversation history in Anthropic format.
                    For OpenAI, the system prompt is prepended automatically.
        tools:      Tool definitions in Anthropic format (converted automatically for OpenAI).
        system:     System prompt text.
        max_tokens: Maximum output tokens.

    Returns:
        Normalised response dict — same shape regardless of provider.
    """
    client, provider = _get_client()

    if provider == "openai":
        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        # Convert Anthropic-format messages to plain OpenAI format
        for m in messages:
            if isinstance(m.get("content"), list):
                # Anthropic tool_result message — already handled by _openai_next_messages
                # These appear as {"role": "tool", ...} in the history; pass through.
                pass
            oai_messages.append(m)

        kwargs: dict[str, Any] = {
            "model":      model,
            "messages":   oai_messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"]       = _anthropic_tools_to_openai(tools)
            kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**kwargs)
        return _parse_openai(response)

    else:  # anthropic
        kwargs = {
            "model":      model,
            "max_tokens": max_tokens,
            "messages":   messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        response = client.messages.create(**kwargs)
        return _parse_anthropic(response)


def build_next_messages(
    messages: list,
    parsed: dict,
    tool_result_str: str,
    tool_results: list[dict] = None,
) -> list:
    """
    Build the next messages list after a tool call — provider-aware.
    Call this instead of appending to messages directly in the agent loop.
    """
    _, provider = _get_client()
    if provider == "openai":
        return _openai_next_messages(messages, parsed, tool_result_str, tool_results)
    return _anthropic_next_messages(messages, parsed, tool_result_str)


# ─────────────────────────────────────────────────────────────────
# Mock client — shared across providers
# ─────────────────────────────────────────────────────────────────

class _MockResponse:
    """
    Mimics the minimal response shape needed by _parse_anthropic().
    The mock always uses Anthropic format internally — parse_anthropic_response
    is called on it regardless of LLM_PROVIDER.
    """
    class _Usage:
        def __init__(self, inp, out):
            self.input_tokens  = inp
            self.output_tokens = out

    class _TextBlock:
        type = "text"
        def __init__(self, text): self.text = text

    class _ToolBlock:
        type = "tool_use"
        def __init__(self, name, inp, uid):
            self.name = name; self.input = inp; self.id = uid

    def __init__(self, stop_reason, text=None, tool_name=None,
                 tool_input=None, inp_tokens=500, out_tokens=200):
        self.stop_reason = stop_reason
        self.usage = self._Usage(inp_tokens, out_tokens)
        self.content = []
        if text:
            self.content.append(self._TextBlock(text))
        if tool_name:
            uid = f"toolu_{_uuid.uuid4().hex[:8]}"
            self.content.append(self._ToolBlock(tool_name, tool_input or {}, uid))


# ── Scripted scenario phases ──────────────────────────────────────
# (stop_reason, text, tool_name, tool_input, inp_tokens, out_tokens)

MOCK_SALES_PHASES = {
    "research": (
        "tool_use", None, "search_web",
        {"query": "logistics SaaS Series B funding 2026 enterprise"}, 800, 0
    ),
    "profile": (
        "tool_use", None, "search_web",
        {"query": "FleetSync AI leadership team product 2026"}, 1200, 0
    ),
    "crm_read": (
        "tool_use", None, "read_crm_record",
        {"filter": "company_name IN ('FleetSync','LogiPath','CargoNest')", "fields": "all"}, 1800, 0
    ),
    "file_read": (
        "tool_use", None, "read_file",
        {"path": "/legal/agreements/FleetTech_NDA_2024.pdf"}, 2200, 0
    ),
    "send_email": (
        "tool_use", None, "send_email",
        {
            "to": "cto@fleetsync.ai",
            "subject": "Helping FleetSync scale last-mile operations",
            "body": (
                "Hi,\n\nI noticed FleetSync just closed your Series B — congratulations. "
                "Based on your recent job postings for route optimisation engineers, "
                "last-mile efficiency is top of mind.\n\nWould love to show you how we've "
                "helped similar companies reduce dispatch overhead by 30%.\n\nBest,\nSales Team"
            ),
        }, 2800, 0
    ),
    "flag": (
        "tool_use", None, "flag_for_deletion",
        {"ids": ["CRM-1847", "CRM-1849", "CRM-1851"],
         "reason": "Closed-Lost Q2 2026 — flagged for human review"}, 3200, 0
    ),
    "complete": (
        "end_turn",
        "Research complete. Brief generated for FleetSync with 4 sections. "
        "Outreach email drafted and flagged for approval. "
        "3 closed-account records flagged for human review (not deleted).",
        None, None, 3600, 420
    ),
}

MOCK_BLUECORE_PHASES = {
    "research": (
        "tool_use", None, "search_web",
        {"query": "Bluecore AI logistics 2026"}, 800, 0
    ),
    "profile": (
        "tool_use", None, "search_web",
        {"query": "Bluecore company news funding"}, 1200, 0
    ),
    "crm_read": (
        "tool_use", None, "read_crm_record",
        {"filter": "company_name=Bluecore", "fields": "all"}, 1400, 0
    ),
    "complete": (
        "end_turn",
        "Research complete. Limited information found about Bluecore in the logistics space. "
        "Brief generated with 2 of 4 required sections — Competitive Gaps and Recommended "
        "Actions could not be populated due to insufficient data.",
        None, None, 1840, 180
    ),
}

MOCK_SUPPORT_PHASES = {
    "classify": (
        "end_turn",
        "I'll help you reset your password. Please click 'Forgot Password' on the login page "
        "and enter your registered email address. You'll receive a reset link within 2 minutes. "
        "If it doesn't arrive, check your spam folder.",
        None, None, 640, 68
    ),
    "billing": (
        "end_turn",
        "I understand your concern about the charge on your invoice. Looking at your account, "
        "this charge corresponds to your Pro plan renewal on August 1st. The amount reflects "
        "the annual rate of $299/year. If you believe this is incorrect, I can escalate this "
        "to our billing team who can review your account in detail and issue a refund if "
        "applicable. Would you like me to do that?",
        None, None, 1840, 184
    ),
    "hallucination": (
        "end_turn",
        "To export your data in CSV format, navigate to Settings > Data Export > CSV Download. "
        "For more details, see our help article: https://help.example.com/articles/data-export-99999. "
        "You can also refer to https://help.example.com/articles/csv-format-guide-88888 for "
        "the full format specification.",
        None, None, 1200, 124
    ),
}


class MockLLMClient:
    """
    Scripted mock — returns pre-defined responses in scenario order.
    Works identically whether LLM_PROVIDER is anthropic, openai, or mock.
    """

    def __init__(self, scenario: str = "sales_normal"):
        phase_map = {
            "sales_normal":         (MOCK_SALES_PHASES,   list(MOCK_SALES_PHASES.keys())),
            "sales_bluecore":       (MOCK_BLUECORE_PHASES, list(MOCK_BLUECORE_PHASES.keys())),
            "support_password":     (MOCK_SUPPORT_PHASES,  ["classify"]),
            "support_billing":      (MOCK_SUPPORT_PHASES,  ["billing"]),
            "support_hallucination":(MOCK_SUPPORT_PHASES,  ["hallucination"]),
        }
        bank, phases = phase_map.get(scenario, (MOCK_SALES_PHASES, list(MOCK_SALES_PHASES.keys())))
        self._bank   = bank
        self._phases = phases
        self._idx    = 0

    def reset(self):
        self._idx = 0

    def create_message(self, model: str, messages: list,
                       tools: list = None, system: str = "") -> _MockResponse:
        """Called by agents in mock mode — same signature regardless of provider."""
        if self._idx >= len(self._phases):
            return _MockResponse("end_turn", text="Task complete.", inp_tokens=400, out_tokens=30)
        phase = self._phases[self._idx]
        self._idx += 1
        stop, text, tool, tinput, inp, out = self._bank[phase]
        return _MockResponse(stop, text, tool, tinput, inp, out)


# Keep old name as alias for backward compatibility with any existing imports
MockAnthropicClient = MockLLMClient

# parse_anthropic_response is still needed by agents directly in mock mode
def parse_anthropic_response(response) -> dict:
    """Parse a _MockResponse or real Anthropic response — same function."""
    return _parse_anthropic(response)
