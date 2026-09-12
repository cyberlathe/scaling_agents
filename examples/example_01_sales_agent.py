#!/usr/bin/env python3
"""
examples/example_01_sales_agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 9 — Sales Intelligence Agent

Runs all 3 Sales Agent scenarios with full LangSmith tracing.
Each scenario produces a distinct trace in your LangSmith project.

What to look for in LangSmith after running:
  Scenario 1 (FleetSync normal): 6 tool spans, all green, ~15,000 tokens
  Scenario 2 (Bluecore mismatch): 3 spans only, output_tokens low, brief incomplete
  Scenario 3 (Hardened): draft_email called (not send_email), flag_for_deletion used

Run:
  python examples/example_01_sales_agent.py
  LLM_PROVIDER=anthropic python examples/example_01_sales_agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
from observability.langsmith_setup import print_langsmith_status
from observability.phoenix_setup   import init_phoenix, print_phoenix_status
from agents.sales_agent import SalesAgent
from scenarios.sales_scenarios import ALL_SCENARIOS


def main():
    print("\n" + "═" * 62)
    print("  CHAPTER 9 — Sales Intelligence Agent")
    print("  3 scenarios → LangSmith traces")
    print("═" * 62)

    # Initialise observability
    init_phoenix(project_name="chapter-09-sales")
    print_langsmith_status()
    print_phoenix_status()
    print()

    for i, sc in enumerate(ALL_SCENARIOS, 1):
        print(f"\n{'─'*62}")
        print(f"  Scenario {i}/3: {sc['name']}")
        print(f"{'─'*62}")
        print(f"  {sc['description']}\n")
        print(f"  Task: {sc['task']}\n")

        agent = SalesAgent(scenario=sc["scenario"])
        result = agent.run(task=sc["task"])
        # agent.print_trace()

        s = agent.get_run_summary()
        print(f"\n  ✓ Scenario complete")
        print(f"    Tools called: {s['tools_called']}")
        print(f"    Tokens: {s['input_tokens']} in / {s['output_tokens']} out")
        print(f"    Cost: ${s['cost_usd']:.4f}")
        print(f"    Mode: {s['provider']}")

        time.sleep(1)

    print("\n" + "═" * 62)
    print("  Done. Check your LangSmith project for 3 root traces.")
    print("  smith.langchain.com → chapter-09-sales → Traces")
    print("═" * 62)


if __name__ == "__main__":
    main()
