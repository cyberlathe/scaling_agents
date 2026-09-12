# Chapter 9 — From Prototype to Production: Companion Code

This codebase accompanies **Chapter 9: From Prototype to Production**.

It contains two production-ready agents wired to real observability (LangSmith)
and evaluation (Arize Phoenix) infrastructure. Run the examples and watch traces
and evaluation scores appear in your dashboards automatically.

---

## What you'll see after running

| Dashboard | What appears |
|-----------|-------------|
| **LangSmith** | Full traces for every agent run — tool calls, LLM calls, token counts, latency, per-span cost |
| **Arize Phoenix** | Hallucination scores, QA correctness, relevance scores for every Support Agent scenario |

---

## Quickstart

### 1. Install dependencies

```bash
cd ch09
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env — add your API keys (see table below)
```

| Variable | Required for | Where to get it |
|----------|-------------|-----------------|
| `LLM_PROVIDER` | Choosing provider | `anthropic`, `openai`, or `mock` (default) |
| `ANTHROPIC_API_KEY` | Anthropic provider | console.anthropic.com |
| `OPENAI_API_KEY` | OpenAI provider | platform.openai.com → API keys |
| `LANGSMITH_API_KEY` | LangSmith traces | smith.langchain.com → Settings → API Keys |
| `LANGSMITH_PROJECT` | LangSmith project name | any string, e.g. `chapter-09` |
| `PHOENIX_API_KEY` | Arize Phoenix (cloud) | app.phoenix.arize.com → Settings |

**Don't have all keys?** The agents run in mock mode without any API key.
LangSmith and Phoenix are optional — traces just won't appear in the dashboards.

**Switching providers:** change `LLM_PROVIDER` and set the matching key. The
agents, tools, and observability stack work identically for both providers.

### 3. Run the examples

```bash
# Run all pre-configured scenarios for both agents (recommended first run)
python examples/run_all_scenarios.py

# Run just the Sales Intelligence Agent scenarios
python examples/example_01_sales_agent.py

# Run just the Customer Support Agent scenarios
python examples/example_02_support_agent.py

# See the observability stack in isolation (no agent required)
python examples/example_03_observability_demo.py

# See evaluation scoring in isolation
python examples/example_04_evaluation_demo.py
```

### 4. View your results

After running, open:
- **LangSmith**: https://smith.langchain.com → your project → Traces
- **Arize Phoenix**: https://app.phoenix.arize.com → your project → Evals

If running Phoenix locally (`PHOENIX_MODE=local`):
```bash
# Phoenix UI starts automatically when you run any example.
# Open: http://localhost:6006
```

---

## Agents in this chapter

### Sales Intelligence Agent
A research and outreach agent for an enterprise sales team. Three pre-configured scenarios:
1. **Normal run** — FleetSync brief: full research, CRM reads, email drafted
2. **Sector mismatch** — Bluecore brief: wrong-sector search → thin output → LangSmith shows why
3. **Hardened run** — Same task with Irreversibility Classifier from Chapter 8 installed

### Customer Support Agent
A ticket-handling agent with model routing (Haiku for simple tickets, Sonnet for complex). Three scenarios:
1. **Password reset** → routed to Haiku (tier-0, $0.036/ticket)
2. **Billing dispute** → routed to Sonnet (tier-2, $0.29/ticket)  
3. **Hallucination trigger** — agent cites a non-existent help article → Phoenix flags it

---

## Codebase structure

```
ch09/
├── config/
│   ├── settings.py          # All env vars and model settings
│   └── llm_client.py        # Anthropic client (real + mock)
│
├── agents/
│   ├── sales_agent.py       # Sales Intelligence Agent — ReAct loop with tool_use
│   └── support_agent.py     # Customer Support Agent — classification + response
│
├── tools/
│   ├── mock_crm.py          # Mock CRM (read, update, flag)
│   ├── mock_filestore.py    # Mock file storage (read, list)
│   ├── mock_email.py        # Mock email (draft, send)
│   ├── mock_web_search.py   # Mock web search (prospect research)
│   └── mock_helpdesk.py     # Mock help article DB for Support Agent
│
├── observability/
│   ├── langsmith_setup.py   # LangSmith initialisation + @traceable wrappers
│   └── phoenix_setup.py     # Arize Phoenix initialisation + instrumentation
│
├── evaluation/
│   └── phoenix_evals.py     # Phoenix HallucinationEvaluator + QACorrectness + Relevance
│
├── scenarios/
│   ├── sales_scenarios.py   # 3 pre-configured Sales Agent scenarios
│   └── support_scenarios.py # 3 pre-configured Support Agent scenarios
│
└── examples/
    ├── run_all_scenarios.py         # Run everything → populate both dashboards
    ├── example_01_sales_agent.py    # Sales Agent only
    ├── example_02_support_agent.py  # Support Agent only
    ├── example_03_observability_demo.py  # LangSmith + Phoenix setup walkthrough
    └── example_04_evaluation_demo.py     # Phoenix evals walkthrough
```

---

## Running modes

```bash
# Mock LLM — no API key needed, uses scripted responses
MOCK_LLM=true python examples/run_all_scenarios.py

# Real Anthropic (Claude Sonnet + Haiku)
LLM_PROVIDER=anthropic python examples/run_all_scenarios.py

# Real OpenAI (GPT-4o + GPT-4o-mini)
LLM_PROVIDER=openai python examples/run_all_scenarios.py

# Phoenix local UI (http://localhost:6006) — no cloud account needed
PHOENIX_MODE=local python examples/run_all_scenarios.py
```

Both Anthropic and OpenAI are fully supported:
- Same tool definitions (Anthropic format; auto-converted to function-calling for OpenAI)
- Same multi-turn history (provider-aware message building in `llm_client.py`)
- Same routing logic (Sonnet↔GPT-4o for complex; Haiku↔GPT-4o-mini for simple)
- Same LangSmith traces and Phoenix instrumentation

---

## Key concepts this code demonstrates

| Concept | Where in code | What to look for |
|---------|--------------|-----------------|
| Provider abstraction | `config/llm_client.py` | `call_llm()` dispatches to Anthropic or OpenAI |
| Tool format conversion | `config/llm_client.py` | `_anthropic_tools_to_openai()` |
| Multi-turn history | `config/llm_client.py` | `build_next_messages()` — provider-aware |
| Model routing | `agents/support_agent.py` | `route_ticket()` → `get_haiku_model()` / `get_capable_llm_model()` |
| LangSmith `@traceable` | `observability/langsmith_setup.py` | Wraps every tool and agent loop |
| Phoenix auto-instrumentation | `observability/phoenix_setup.py` | `AnthropicInstrumentor` + `OpenAIInstrumentor` |
| Hallucination detection | `evaluation/phoenix_evals.py` | `HallucinationEvaluator` scoring |
| Pricing by provider | `config/settings.py` | `COST_PER_M` — all four models |
