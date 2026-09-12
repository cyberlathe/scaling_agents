#!/usr/bin/env python3
"""
examples/example_04_evaluation_demo.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 9 — Phoenix Evaluation Pipeline Demo

Runs all 3 Support Agent scenarios, then runs the full Phoenix
evaluation pipeline:
  1. URL verification (no LLM needed)
  2. HallucinationEvaluator (via Haiku judge)
  3. QACorrectnessEvaluator
  4. RelevanceEvaluator

Prints evaluation scores to the terminal and logs them to Phoenix.
The hallucination in Scenario 3 should appear as a flagged result.

Run:
  python examples/example_04_evaluation_demo.py
  LLM_PROVIDER=anthropic python examples/example_04_evaluation_demo.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from observability.phoenix_setup   import init_phoenix, print_phoenix_status, get_phoenix_url
from observability.langsmith_setup import print_langsmith_status
from agents.support_agent import SupportAgent
from evaluation.phoenix_evals import run_phoenix_evals, run_simple_evals
from scenarios.support_scenarios import ALL_SCENARIOS
from config.settings import ANTHROPIC_KEY


def main():
    print("\n" + "═" * 62)
    print("  CHAPTER 9 — Phoenix Evaluation Pipeline Demo")
    print("  3 Support Agent tickets → hallucination + QA + relevance")
    print("═" * 62)

    phoenix_ok = init_phoenix(project_name="chapter-09-evals")
    print_langsmith_status()
    print_phoenix_status()

    print(f"\n{'─'*62}")
    print("  STEP 1 — Run Support Agent on all 3 ticket scenarios")
    print(f"{'─'*62}\n")

    ticket_results = []
    for i, sc in enumerate(ALL_SCENARIOS, 1):
        print(f"  Ticket {i}/3: {sc['name']}")
        print(f"  \"{sc['ticket'][:80]}...\"\n")

        agent  = SupportAgent(scenario=sc["scenario"])
        result = agent.handle_ticket(sc["ticket"])
        result["scenario"] = sc["name"]

        s = agent.get_run_summary()
        print(f"  → Route: {s['tier']} | ${s['cost_usd']:.5f} | {len(result.get('cited_urls',[]))} URLs cited\n")
        ticket_results.append(result)

    print(f"\n{'─'*62}")
    print("  STEP 2 — Phoenix Evaluation")
    print(f"{'─'*62}")

    if ANTHROPIC_KEY:
        print("""
  Phoenix runs three evaluators on each ticket response:

  HallucinationEvaluator:
    Compares each claim in the response against the retrieved help articles.
    Flags any claim not supported by the provided context.

  QACorrectnessEvaluator:
    Checks whether the answer is factually correct given the question
    and the retrieved help article content.

  RelevanceEvaluator:
    Checks whether the response actually addressed the customer's question.

  Judge model: claude-haiku-4-5-20251001 (cheap, consistent, sufficient for scoring)
    """)
        eval_df = run_phoenix_evals(ticket_results, project_name="chapter-09-evals")
    else:
        print("\n  No ANTHROPIC_API_KEY — running URL verification check only")
        print("  (Add ANTHROPIC_API_KEY to .env for full LLM-based evaluation)")
        eval_df = run_simple_evals(ticket_results)

    print(f"\n{'─'*62}")
    print("  STEP 3 — View Results in Phoenix")
    print(f"{'─'*62}")
    print(f"""
  Open Phoenix at: {get_phoenix_url()}

  Traces tab:
    See the 3 Anthropic API calls captured automatically via
    AnthropicInstrumentor. Token counts and latency are shown per call.

  Evals tab:
    Three columns: hallucination | qa_correctness | relevance
    Scenario 3 (hallucination trigger) should show:
      hallucination: hallucinated (broken URL detected)
    """)

    print("═" * 62)


if __name__ == "__main__":
    main()
