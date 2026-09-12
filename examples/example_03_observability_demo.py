#!/usr/bin/env python3
"""
examples/example_03_observability_demo.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 9 — Observability Stack Demo

Shows how LangSmith and Arize Phoenix capture a single agent run.
Uses the Sales Agent's FleetSync scenario.
Prints a detailed walkthrough of what appears in each dashboard.

Run:
  python examples/example_03_observability_demo.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from observability.langsmith_setup import print_langsmith_status, is_enabled as ls_enabled
from observability.phoenix_setup   import init_phoenix, print_phoenix_status, get_phoenix_url
from agents.sales_agent import SalesAgent
from scenarios.sales_scenarios import SCENARIO_FLEETSYNC


def explain_langsmith():
    print("""
  HOW LANGSMITH CAPTURES THIS TRACE
  ──────────────────────────────────
  The @agent_trace decorator on SalesAgent.run() creates the root span.
  Each tool function wrapped with wrap_tool() appears as a child span.

  In LangSmith you'll see:
    SalesAgent.run                     [root span, full duration]
    ├── search_web                     [tool span, 0.8s]
    │     input:  {"query": "logistics SaaS..."}
    │     output: {"results": [...], "count": 1}
    ├── read_crm_record                [tool span, 0.6s]
    ├── read_file                      [tool span, 0.4s]
    ├── draft_email                    [tool span, 0.3s]
    └── flag_for_deletion              [tool span, 0.2s]

  Token counts, latency, and estimated cost appear on each span.
  You can click any span to see the full input/output.
    """)


def explain_phoenix():
    print(f"""
  HOW PHOENIX CAPTURES THIS TRACE
  ────────────────────────────────
  AnthropicInstrumentor()/OpenAIInstrumentor() patches 
  anthropic.Anthropic/OpenAI at import time.
  Every client.messages.create() call becomes an OpenTelemetry span.
  No code changes needed in the agent itself.

  In Phoenix ({get_phoenix_url()}) you'll see:
    Traces tab:
      SalesAgent.run → all Anthropic/OpenAI calls as child spans
      Each span shows: model, input_tokens, output_tokens, latency

    Evals tab (after running evaluation):
      hallucination: not_hallucinated | hallucinated
      qa_correctness: correct | incorrect
      relevance: relevant | irrelevant
      Each result includes an explanation from the judge model.
    """)


def main():
    print("\n" + "═" * 62)
    print("  CHAPTER 9 — Observability Stack Walkthrough")
    print("═" * 62)

    # Setup
    phoenix_ok = init_phoenix(project_name="chapter-09-demo")
    print_langsmith_status()
    print_phoenix_status()

    # Run one scenario
    print(f"\n{'─'*62}")
    print("  Running FleetSync scenario to generate a trace...")
    print(f"{'─'*62}\n")

    agent = SalesAgent(scenario=SCENARIO_FLEETSYNC["scenario"])
    result = agent.run(task=SCENARIO_FLEETSYNC["task"])
    agent.print_trace()

    # Explain what just appeared in each tool
    print(f"\n{'─'*62}")
    print("  WHAT JUST HAPPENED IN YOUR DASHBOARDS")
    print(f"{'─'*62}")

    if ls_enabled():
        print("\n  ✓ LangSmith:")
        print("    https://smith.langchain.com → chapter-09-demo → Traces")
        print("    A new root trace 'SalesAgent.run' was just created.")
        print("    Click it to see the nested tool spans.")
    else:
        print("\n  · LangSmith not active (set LANGSMITH_TRACING=true)")
        print("    When enabled, traces appear at smith.langchain.com")

    if phoenix_ok:
        print(f"\n  ✓ Phoenix: {get_phoenix_url()}")
        print("    A new trace was just captured in the Traces tab.")
        print("    The Anthropic/OpenAI API call is visible as an LLM span.")
    else:
        print(f"\n  · Phoenix not active")
        print("    Install: pip install arize-phoenix openinference-instrumentation-anthropic")

    explain_langsmith()
    explain_phoenix()

    s = agent.get_run_summary()
    print(f"\n  Run summary: {s['input_tokens']} input / {s['output_tokens']} output tokens")
    print(f"  Estimated cost: ${s['cost_usd']:.4f} · Mode: {s['provider']}")
    print("\n" + "═" * 62)


if __name__ == "__main__":
    main()
