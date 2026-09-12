#!/usr/bin/env python3
"""
examples/example_02_support_agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 9 — Customer Support Agent

Runs all 3 Support Agent scenarios:
  1. Password reset → Haiku (~$0.036)
  2. Billing dispute → Sonnet (~$0.29)
  3. Data export → Hallucination trigger → Phoenix flags it

After all tickets, runs Phoenix evaluation and prints scores.

What to look for:
  LangSmith: 3 traces, routing decisions visible in each
  Phoenix Traces: Anthropic calls with token counts
  Phoenix Evals: Scenario 3 should show hallucination=hallucinated

Run:
  python examples/example_02_support_agent.py
  LLM_PROVIDER=anthropic python examples/example_02_support_agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import asyncio
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
from observability.langsmith_setup import print_langsmith_status
from observability.phoenix_setup   import init_phoenix, print_phoenix_status, get_phoenix_url
from agents.support_agent import SupportAgent
from evaluation.phoenix_evals import run_phoenix_evals, run_simple_evals
from scenarios.support_scenarios import ALL_SCENARIOS
from config.settings import ANTHROPIC_KEY, OPENAI_KEY


async def main():
    print("\n" + "═" * 62)
    print("  CHAPTER 9 — Customer Support Agent")
    print("  3 scenarios → routing + LangSmith + Phoenix evals")
    print("═" * 62)

    # Initialise observability
    init_phoenix(project_name="chapter-09-support")
    print_langsmith_status()
    print_phoenix_status()
    print()

    ticket_results = []

    for i, sc in enumerate(ALL_SCENARIOS, 1):
        print(f"\n{'─'*62}")
        print(f"  Scenario {i}/3: {sc['name']}")
        print(f"{'─'*62}")
        print(f"  {sc['description']}\n")
        print(f"  Ticket: \"{sc['ticket'][:100]}...\"")
        print(f"  Expected route: {sc['expected_route']}\n")

        agent = SupportAgent(scenario=sc["scenario"])
        result = agent.handle_ticket(sc["ticket"])
        result["scenario"] = sc["name"]

        agent.print_trace()

        print(f"\n  Response (first 200 chars):")
        print(f"  {result['response'][:200]}")

        if result.get("cited_urls"):
            print(f"\n  Cited URLs: {result['cited_urls']}")
            if sc.get("expect_hallucination"):
                print("  ⚠  This scenario contains broken URL citations — Phoenix will flag them")

        s = agent.get_run_summary()
        print(f"\n  ✓ Route: {s['tier']} | Model: {s['model'].split('-')[1]} | Cost: ${s['cost_usd']:.5f}")

        ticket_results.append(result)
        time.sleep(1)

    # ── Phoenix evaluation ────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  PHOENIX EVALUATION")
    print(f"{'─'*62}")

    if ANTHROPIC_KEY or OPENAI_KEY:
        await run_phoenix_evals(ticket_results, project_name="chapter-09-support")
    else:
        print("  No ANTHROPIC_API_KEY or OPENAI_API_KEY — running URL check only (no LLM judge)")
        run_simple_evals(ticket_results)

    print(f"\n  View evaluation results at: {get_phoenix_url()}")
    print("  → Evals tab → hallucination / qa_correctness / relevance")

    print("\n" + "═" * 62)
    print("  Done. Traces in LangSmith, evals in Phoenix.")
    print("═" * 62)


if __name__ == "__main__":
    asyncio.run(main())
