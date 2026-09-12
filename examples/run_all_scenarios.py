#!/usr/bin/env python3
"""
examples/run_all_scenarios.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 9 — From Prototype to Production

THE FULL DEMO: runs all 6 pre-configured scenarios across both agents
and populates your LangSmith and Arize Phoenix dashboards.

After running, open:
  LangSmith:  https://smith.langchain.com → your project → Traces
  Phoenix:    http://localhost:6006 (local) or app.phoenix.arize.com (cloud)

Run:
  python examples/run_all_scenarios.py
  MOCK_LLM=true python examples/run_all_scenarios.py
  LLM_PROVIDER=anthropic python examples/run_all_scenarios.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table
from rich         import print as rprint

from observability.langsmith_setup import print_langsmith_status
from observability.phoenix_setup   import init_phoenix, print_phoenix_status, get_phoenix_url
from agents.sales_agent   import SalesAgent
from agents.support_agent import SupportAgent
from evaluation.phoenix_evals import run_phoenix_evals, run_simple_evals
from scenarios.sales_scenarios   import ALL_SCENARIOS as SALES_SCENARIOS
from scenarios.support_scenarios import ALL_SCENARIOS as SUPPORT_SCENARIOS
from config.settings import MOCK_LLM, ANTHROPIC_KEY

console = Console()


def header():
    console.print(Panel.fit(
        "[bold green]Chapter 9 — From Prototype to Production[/]\n"
        "[dim]Running all scenarios → LangSmith + Arize Phoenix[/]",
        border_style="green"
    ))


def section(title: str):
    console.print(f"\n[bold cyan]{'─'*60}[/]")
    console.print(f"[bold cyan]  {title}[/]")
    console.print(f"[bold cyan]{'─'*60}[/]")


def run_sales_scenarios() -> list[dict]:
    """Run all 3 Sales Agent scenarios. Returns run summaries."""
    section("SALES INTELLIGENCE AGENT — 3 scenarios")
    summaries = []

    for i, sc in enumerate(SALES_SCENARIOS, 1):
        console.print(f"\n[bold]Scenario {i}/3:[/] {sc['name']}")
        console.print(f"[dim]{sc['description']}[/]\n")

        agent = SalesAgent(scenario=sc["scenario"])
        result = agent.run(task=sc["task"])
        agent.print_trace()

        summary = agent.get_run_summary()
        summary["scenario_name"] = sc["name"]
        summaries.append(summary)

        console.print(f"[green]✓[/] Done — ${summary['cost_usd']:.4f} · {summary['elapsed_s']}s")
        time.sleep(0.5)  # brief pause between scenarios

    return summaries


def run_support_scenarios() -> list[dict]:
    """Run all 3 Support Agent scenarios. Returns ticket result dicts for eval."""
    section("CUSTOMER SUPPORT AGENT — 3 scenarios")
    ticket_results = []

    for i, sc in enumerate(SUPPORT_SCENARIOS, 1):
        console.print(f"\n[bold]Scenario {i}/3:[/] {sc['name']}")
        console.print(f"[dim]{sc['description']}[/]\n")
        console.print(f"[bold]Ticket:[/] {sc['ticket']}\n")

        agent = SupportAgent(scenario=sc["scenario"])
        result = agent.handle_ticket(sc["ticket"])

        # Annotate result with scenario metadata for eval
        result["scenario"] = sc["name"]

        agent.print_trace()
        console.print(f"\n[bold]Response:[/] {result['response'][:200]}...")
        console.print(f"[bold]Routed to:[/] {result['routed_to']}")

        if result.get("cited_urls"):
            console.print(f"[bold]Cited URLs:[/] {result['cited_urls']}")

        summary = agent.get_run_summary()
        console.print(f"[green]✓[/] Done — ${summary['cost_usd']:.5f} · {summary['elapsed_s']}s")

        ticket_results.append(result)
        time.sleep(0.5)

    return ticket_results


def print_cost_summary(sales_summaries: list, support_results: list):
    """Print a cost summary table across all scenarios."""
    section("COST SUMMARY")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Agent / Scenario",    style="dim", width=32)
    table.add_column("Model",               width=18)
    table.add_column("Tokens In",           justify="right")
    table.add_column("Tokens Out",          justify="right")
    table.add_column("Cost (USD)",          justify="right")

    from config.settings import ANTHROPIC_MODEL_SONNET, ANTHROPIC_MODEL_HAIKU, estimate_cost_usd

    for s in sales_summaries:
        table.add_row(
            s["scenario_name"][:30],
            ANTHROPIC_MODEL_SONNET.split("-")[0] + "…",
            str(s["input_tokens"]),
            str(s["output_tokens"]),
            f"${s['cost_usd']:.4f}",
        )

    for r in support_results:
        model = r.get("model_used", ANTHROPIC_MODEL_SONNET)
        inp   = r.get("input_tokens", 0)
        out   = r.get("output_tokens", 0)
        cost  = estimate_cost_usd(model, inp, out)
        short_model = "haiku" if "haiku" in model.lower() else "sonnet"
        table.add_row(
            str(r.get("scenario",""))[:30],
            short_model,
            str(inp),
            str(out),
            f"${cost:.5f}",
        )

    console.print(table)


def print_dashboard_links():
    section("YOUR DASHBOARDS")
    console.print("\n  [bold]LangSmith traces:[/]")
    console.print("    https://smith.langchain.com → your project → Traces")
    console.print("    You should see 6 root traces (3 Sales, 3 Support)")
    console.print("    Each trace shows nested tool spans with token counts and latency\n")

    phoenix_url = get_phoenix_url()
    console.print("  [bold]Arize Phoenix:[/]")
    console.print(f"    {phoenix_url}")
    console.print("    Traces tab: all Anthropic calls as OpenTelemetry spans")
    console.print("    Evals tab: hallucination + QA correctness + relevance scores")
    console.print("    The hallucination scenario (Scenario 3) should show a flagged eval\n")


def main():
    header()

    # ── Initialise observability ──────────────────────────────────
    section("OBSERVABILITY SETUP")
    phoenix_ok = init_phoenix(project_name="chapter-09")
    print_langsmith_status()
    print_phoenix_status()

    mode = "mock LLM" if MOCK_LLM else f"real LLM (Anthropic)"
    console.print(f"\n  [bold]LLM mode:[/] {mode}")
    if MOCK_LLM:
        console.print("  [dim]Set LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY to use the real model[/]")

    console.print("\n  Starting in 2 seconds...")
    time.sleep(2)

    # ── Run Sales scenarios ───────────────────────────────────────
    sales_summaries = run_sales_scenarios()

    # ── Run Support scenarios ─────────────────────────────────────
    ticket_results = run_support_scenarios()

    # ── Run Phoenix evaluation ────────────────────────────────────
    section("ARIZE PHOENIX EVALUATION")
    console.print("  Running hallucination + QA + relevance evaluators on Support Agent results...")
    console.print("  [dim](This uses Haiku as judge — requires ANTHROPIC_API_KEY)[/]\n")

    if ANTHROPIC_KEY:
        eval_df = run_phoenix_evals(ticket_results, project_name="chapter-09")
    else:
        console.print("  No ANTHROPIC_API_KEY — running URL verification only")
        eval_df = run_simple_evals(ticket_results)

    # ── Cost summary ──────────────────────────────────────────────
    print_cost_summary(sales_summaries, ticket_results)

    # ── Dashboard links ───────────────────────────────────────────
    print_dashboard_links()

    console.print(Panel.fit(
        "[bold green]All 6 scenarios complete.[/]\n"
        "[dim]Traces are in LangSmith. Eval scores are in Phoenix.[/]",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
