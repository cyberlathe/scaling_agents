"""
agents/sales_agent.py
Sales Intelligence Agent — Chapter 9.

ReAct loop (Thought → Tool → Observation) using tool_use / function-calling.
Works with Anthropic (Claude) or OpenAI (GPT-4o) — set LLM_PROVIDER in .env.

Every tool call is wrapped with @traceable for LangSmith.
Anthropic SDK calls are auto-captured by Phoenix via AnthropicInstrumentor.
OpenAI SDK calls are auto-captured via OpenAIInstrumentor (if installed).
"""
import json
import time

from config.settings import (
    MOCK_LLM, LLM_PROVIDER,
    get_capable_llm_model, estimate_cost_usd,
)
from config.llm_client import (
    call_llm, build_next_messages,
    MockLLMClient, parse_anthropic_response,
)
from observability.langsmith_setup import agent_trace, wrap_tool
from tools.mock_crm        import read_crm_record, update_crm_record, flag_for_deletion, log_activity, delete_crm_record
from tools.mock_filestore  import read_file, list_files
from tools.mock_email      import send_email, draft_email
from tools.mock_web_search import search_web

# ── Tool definitions (Anthropic format — auto-converted for OpenAI) ──
TOOLS = [
    {
        "name": "search_web",
        "description": "Search the web for prospect research, news, and company information.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
    {
        "name": "read_crm_record",
        "description": "Query CRM records. Use filter to narrow results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filter": {"type": "string"},
                "fields": {"type": "string", "default": "all"},
            },
            "required": ["filter"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from storage by path.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "list_files",
        "description": "List files in a folder.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":  {"type": "string"},
                "folder": {"type": "string"},
            },
        },
    },
    {
        "name": "draft_email",
        "description": "Create an unsent email draft for human review.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to":      {"type": "string"},
                "subject": {"type": "string"},
                "body":    {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an outreach email. Only when explicitly instructed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to":      {"type": "string"},
                "subject": {"type": "string"},
                "body":    {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "flag_for_deletion",
        "description": (
            "Flag CRM records for human review. "
            "Does NOT delete — use instead of deleting directly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ids":    {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["ids"],
        },
    },
    {
        "name": "log_activity",
        "description": "Log a sales activity against a company record.",
        "input_schema": {
            "type": "object",
            "properties": {
                "company":       {"type": "string"},
                "activity_type": {"type": "string"},
                "notes":         {"type": "string"},
            },
            "required": ["company", "activity_type"],
        },
    },
]

TOOLS_SALES_HARDENED = [
    {
        "name": "search_web",
        "description": "Search the web for prospect research, news, and company information.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
    {
        "name": "read_crm_record",
        "description": "Query CRM records. Use filter to narrow results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filter": {"type": "string"},
                "fields": {"type": "string", "default": "all"},
            },
            "required": ["filter"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from storage by path.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "list_files",
        "description": "List files in a folder.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":  {"type": "string"},
                "folder": {"type": "string"},
            },
        },
    },
    {
        "name": "draft_email",
        "description": "Create an unsent email draft for human review.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to":      {"type": "string"},
                "subject": {"type": "string"},
                "body":    {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "flag_for_deletion",
        "description": (
            "Flag CRM records for human review. "
            "Does NOT delete — use instead of deleting directly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ids":    {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["ids"],
        },
    },
    {
        "name": "log_activity",
        "description": "Log a sales activity against a company record.",
        "input_schema": {
            "type": "object",
            "properties": {
                "company":       {"type": "string"},
                "activity_type": {"type": "string"},
                "notes":         {"type": "string"},
            },
            "required": ["company", "activity_type"],
        },
    },
]

TOOLS_SALES_NORMAL = [
    {
        "name": "search_web",
        "description": "Search the web for prospect research, news, and company information.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
    {
        "name": "read_crm_record",
        "description": "Query CRM records. Use filter to narrow results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filter": {"type": "string"},
                "fields": {"type": "string", "default": "all"},
            },
            "required": ["filter"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from storage by path.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "list_files",
        "description": "List files in a folder.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":  {"type": "string"},
                "folder": {"type": "string"},
            },
        },
    },
    {
        "name": "send_email",
        "description": "Send an outreach email",
        "input_schema": {
            "type": "object",
            "properties": {
                "to":      {"type": "string"},
                "subject": {"type": "string"},
                "body":    {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "delete_crm_record",
        "description": (
            "Delete CRM records."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ids":    {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["ids"],
        },
    },
    {
        "name": "log_activity",
        "description": "Log a sales activity against a company record.",
        "input_schema": {
            "type": "object",
            "properties": {
                "company":       {"type": "string"},
                "activity_type": {"type": "string"},
                "notes":         {"type": "string"},
            },
            "required": ["company", "activity_type"],
        },
    },
]

SYSTEM_PROMPT = """
You are the Sales Intelligence Agent for a B2B SaaS company. You help the
sales team research prospects, check account history, and prepare outreach.

Your tools:
- search_web: research prospects, find funding news, understand market
- read_crm_record: check existing account data
- read_file / list_files: read legal agreements and templates
- draft_email: create outreach drafts
- send_email: send outreach email
- delete_crm_record: delete stale CRM records
- flag_for_deletion: flag stale CRM records for human review (never delete directly)
- log_activity: record outreach activity

Rules:
- Always be explicit about what you found and what you did.
""".strip()
# - Never delete records or files. Use flag_for_deletion instead.
# - Prefer draft_email over send_email unless the task explicitly says to send.
# - If a company is not in the expected sector, note the mismatch and stop.


# ── Traced tool dispatch ──────────────────────────────────────────
_TOOL_MAP = {
    "search_web":        wrap_tool(search_web),
    "read_crm_record":   wrap_tool(read_crm_record),
    "read_file":         wrap_tool(read_file),
    "list_files":        wrap_tool(list_files),
    "send_email":        wrap_tool(send_email),
    "draft_email":       wrap_tool(draft_email),
    "flag_for_deletion": wrap_tool(flag_for_deletion),
    "log_activity":      wrap_tool(log_activity),
    "delete_crm_record": wrap_tool(delete_crm_record),
}


def _call_tool(name: str, args: dict) -> str:
    fn = _TOOL_MAP.get(name)
    if not fn:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return json.dumps(fn(**args), default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


class SalesAgent:
    """
    Sales Intelligence Agent — provider-agnostic.

    Usage:
        agent = SalesAgent(scenario="sales_normal")
        result = agent.run("Research FleetSync and prepare an outreach brief.")
        print(agent.get_run_summary())
    """

    def __init__(self, scenario: str = "sales_normal"):
        self.scenario  = scenario
        self.model     = get_capable_llm_model()
        self._trace:   list[dict] = []
        self._total_in  = 0
        self._total_out = 0
        self._run_start = 0.0
        self._tools = []

        if MOCK_LLM:
            self._mock  = MockLLMClient(scenario=scenario)
            self._mode  = "mock"
        else:
            self._mock  = None
            self._mode  = LLM_PROVIDER  # "anthropic" | "openai"

        if(scenario == "sales_normal"):
            self._tools = TOOLS_SALES_NORMAL
        elif(scenario == "sales_bluecore"):
            self._tools = TOOLS
        elif(scenario == "sales_hardened"):
            self._tools = TOOLS_SALES_HARDENED

    @agent_trace(name="SalesAgent.run", run_type="chain")
    def run(self, task: str) -> str:
        self._run_start = time.time()
        self._trace     = []
        self._total_in  = 0
        self._total_out = 0

        messages  = [{"role": "user", "content": task}]
        max_steps = 10

        for step in range(max_steps):
            # ── LLM call ──────────────────────────────────────────
            if self._mock:
                raw    = self._mock.create_message(
                    model=self.model, messages=messages,
                    tools=self._tools, system=SYSTEM_PROMPT
                )
                parsed = parse_anthropic_response(raw)
            else:
                parsed = call_llm(
                    model=self.model, messages=messages,
                    tools=self._tools, system=SYSTEM_PROMPT,
                )

            self._total_in  += parsed["input_tokens"]
            self._total_out += parsed["output_tokens"]
            self._record("llm", {
                "step": step + 1,
                "stop_reason":   parsed["stop_reason"],
                "text":          (parsed["text"] or "")[:200],
                "tool_called":   parsed["tool_name"],
                "input_tokens":  parsed["input_tokens"],
                "output_tokens": parsed["output_tokens"],
            })

            # ── Terminal ──────────────────────────────────────────
            if parsed["stop_reason"] == "end_turn" or not parsed["tool_name"]:
                final = parsed["text"] or "Task complete."
                self._record("done", {"summary": final[:300]})
                return final

            # ── Tool call ─────────────────────────────────────────
            tool_calls = parsed.get("tool_calls") or [{
                "name": parsed["tool_name"],
                "input": parsed["tool_input"],
                "id": parsed["tool_use_id"],
            }]
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_input = tool_call["input"]
                self._record("tool", {"name": tool_name, "input": tool_input})

                tool_result = _call_tool(tool_name, tool_input)
                tool_results.append({"id": tool_call["id"], "content": tool_result})
                self._record("observation", {
                    "tool": tool_name,
                    "result_preview": tool_result[:300],
                })

            # ── Build next turn (provider-aware) ──────────────────
            if self._mock:
                messages = messages + [
                    {"role": "assistant",
                     "content": f"Calling {tool_name}"},
                    {"role": "user",
                     "content": f"Tool result: {tool_result[:400]}"},
                ]
            else:
                messages = build_next_messages(
                    messages,
                    parsed,
                    tool_results[0]["content"],
                    tool_results=tool_results,
                )

        return "Agent reached max steps."

    def _record(self, kind: str, data: dict):
        self._trace.append({
            "kind": kind,
            "ts":   round(time.time() - self._run_start, 2),
            **data,
        })

    def get_run_summary(self) -> dict:
        elapsed = round(time.time() - self._run_start, 1)
        cost    = estimate_cost_usd(self.model, self._total_in, self._total_out)
        return {
            "scenario":      self.scenario,
            "provider":      self._mode,
            "model":         self.model,
            "elapsed_s":     elapsed,
            "input_tokens":  self._total_in,
            "output_tokens": self._total_out,
            "cost_usd":      round(cost, 4),
            "tools_called":  [e["name"] for e in self._trace if e["kind"] == "tool"],
            "steps":         len([e for e in self._trace if e["kind"] == "llm"]),
        }

    def print_trace(self, verbose: bool = False):
        icons = {"llm": "🧠", "tool": "⚙️ ", "observation": "👁 ", "done": "✅"}
        print("\n" + "─" * 60)
        print(f"  AGENT TRACE  [{self._mode.upper()}]")
        print("─" * 60)
        for entry in self._trace:
            icon = icons.get(entry["kind"], "·")
            ts   = f"+{entry['ts']:.1f}s"
            if entry["kind"] == "llm":
                print(f"\n  {icon} [{ts}] LLM · {entry.get('input_tokens',0)}→{entry.get('output_tokens',0)} tok")
                if entry.get("text"):
                    print(f"     {entry['text']}")
                if entry.get("tool_called"):
                    print(f"     → calls: {entry['tool_called']}")
            elif entry["kind"] == "tool":
                args_str = json.dumps(entry.get("input", {}))
                print(f"\n  {icon} [{ts}] {entry['name']}({args_str})")
            elif entry["kind"] == "observation":
                print(f"     ↳ {entry.get('result_preview','')}")
            elif entry["kind"] == "done":
                print(f"\n  {icon} [{ts}] {entry.get('summary','')}")
        s = self.get_run_summary()
        print(f"\n  Tokens: {s['input_tokens']} in / {s['output_tokens']} out")
        print(f"  Cost: ${s['cost_usd']:.4f} · {s['elapsed_s']}s · {len(s['tools_called'])} tool calls")
        print("─" * 60)
