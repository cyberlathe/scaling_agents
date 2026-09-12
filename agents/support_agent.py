"""
agents/support_agent.py
Customer Support Agent — Chapter 9.

Routes tickets to the fast/cheap model (Haiku / GPT-4o-mini) or the
capable model (Sonnet / GPT-4o) based on a lightweight classifier —
the same routing logic described in the chapter regardless of provider.

Works with Anthropic (Claude) or OpenAI (GPT-4o).
Set LLM_PROVIDER=anthropic or LLM_PROVIDER=openai in .env.
"""
import json
import re
import time

from config.settings import (
    MOCK_LLM, LLM_PROVIDER,
    get_capable_llm_model, get_fast_cheap_llm_model,
    estimate_cost_usd,
)
from config.llm_client import (
    call_llm,
    MockLLMClient, parse_anthropic_response,
)
from observability.langsmith_setup import agent_trace, wrap_tool
from tools.mock_helpdesk import search_help_articles, verify_urls

# ── Routing classifiers ───────────────────────────────────────────
_TIER0 = re.compile(
    r'\b(password|reset|login|sign.in|2fa|two.factor|locked.out|verify.email)\b',
    re.IGNORECASE,
)
_ESCALATE = re.compile(
    r'\b(legal|lawsuit|gdpr|data.breach|subpoena|enterprise.contract|sla.violation|chargeback)\b',
    re.IGNORECASE,
)


class TicketRoute:
    def __init__(self, model: str, tier: str, rationale: str, confidence: float):
        self.model      = model
        self.tier       = tier
        self.rationale  = rationale
        self.confidence = confidence

    def __repr__(self):
        return f"TicketRoute(tier={self.tier!r}, model={self.model!r})"


def route_ticket(ticket_text: str, rag_confidence: float = 0.0) -> TicketRoute:
    """
    Classify a ticket and return the routing decision.
    Provider-aware: returns the right model string for the configured provider.
    """
    if _ESCALATE.search(ticket_text):
        return TicketRoute(
            model="__escalate__", tier="escalate",
            rationale="Escalation trigger — routing to human agent",
            confidence=0.95,
        )
    if _TIER0.search(ticket_text):
        return TicketRoute(
            model=get_fast_cheap_llm_model(), tier="tier-0 (fast model)",
            rationale="Auth/password pattern — deterministic answer, fast model sufficient",
            confidence=0.90,
        )
    if rag_confidence >= 0.85:
        return TicketRoute(
            model=get_fast_cheap_llm_model(), tier="tier-1 (fast model)",
            rationale=f"High RAG confidence ({rag_confidence:.2f}) — answer in docs",
            confidence=rag_confidence,
        )
    return TicketRoute(
        model=get_capable_llm_model(), tier="tier-2 (capable model)",
        rationale="Mid-complexity — needs nuanced reasoning or synthesis",
        confidence=0.70,
    )


SYSTEM_PROMPT = """
You are a customer support agent for a B2B SaaS project management platform.

Answer the customer's question clearly and helpfully. Be concise.

When referencing help articles, ONLY cite articles that exist in the provided
context. Do not invent article URLs or article IDs. If you're unsure, say
you'll have the team follow up rather than guessing.

If the issue is outside your scope (legal, billing disputes, SLA violations),
explain that you're escalating to the appropriate team.
""".strip()

_search_help = wrap_tool(search_help_articles, run_type="retriever")


class SupportAgent:
    """
    Customer Support Agent — provider-agnostic.

    Usage:
        agent = SupportAgent(scenario="support_password")
        result = agent.handle_ticket("I can't log in — forgot my password")
        print(agent.get_run_summary())
    """

    def __init__(self, scenario: str = "support_password"):
        self.scenario    = scenario
        self._trace:     list[dict] = []
        self._run_start  = 0.0
        self._route:     TicketRoute | None = None
        self._in_tokens  = 0
        self._out_tokens = 0

        if MOCK_LLM:
            self._mock = MockLLMClient(scenario=scenario)
            self._mode = "mock"
        else:
            self._mock = None
            self._mode = LLM_PROVIDER

    @agent_trace(name="SupportAgent.handle_ticket", run_type="chain")
    def handle_ticket(self, ticket_text: str) -> dict:
        self._run_start  = time.time()
        self._trace      = []
        self._in_tokens  = 0
        self._out_tokens = 0

        # ── RAG retrieval ─────────────────────────────────────────
        rag = _search_help(query=ticket_text, top_k=3)
        rag_confidence = rag.get("top_confidence", 0.0)
        retrieved_docs = rag.get("results", [])
        self._record("retrieval", {
            "query":            ticket_text[:60],
            "top_confidence":   rag_confidence,
            "articles_retrieved": [d["id"] for d in retrieved_docs],
        })

        # ── Routing ───────────────────────────────────────────────
        route = route_ticket(ticket_text, rag_confidence)
        self._route = route
        self._record("routing", {
            "tier":      route.tier,
            "model":     route.model,
            "rationale": route.rationale,
        })

        if route.model == "__escalate__":
            return {
                "response":   "This requires specialist review. Escalating your ticket now — "
                              "our team will be in touch within 2 business hours.",
                "routed_to":  "human",
                "escalated":  True,
                "cited_urls": [],
                "rag_context": [],
                "ticket":     ticket_text,
            }

        # ── Build context ─────────────────────────────────────────
        context_parts = [
            f"Article: {d['title']}\nURL: {d['url']}\n{d['content']}"
            for d in retrieved_docs
        ]
        context_str = "\n\n---\n\n".join(context_parts) or "No relevant articles found."

        user_message = (
            f"Customer ticket:\n{ticket_text}\n\n"
            f"Relevant help articles:\n{context_str}"
        )
        messages = [{"role": "user", "content": user_message}]

        # ── LLM call ─────────────────────────────────────────────
        if self._mock:
            raw    = self._mock.create_message(
                model=route.model, messages=messages, system=SYSTEM_PROMPT
            )
            parsed = parse_anthropic_response(raw)
        else:
            parsed = call_llm(
                model=route.model, messages=messages,
                system=SYSTEM_PROMPT, max_tokens=512,
            )

        self._in_tokens  = parsed["input_tokens"]
        self._out_tokens = parsed["output_tokens"]
        answer = parsed["text"] or ""

        self._record("llm", {
            "model":            route.model,
            "input_tokens":     parsed["input_tokens"],
            "output_tokens":    parsed["output_tokens"],
            "response_preview": answer[:200],
        })

        cited_urls = re.findall(r'https?://[^\s\'"<>\)]+', answer)

        return {
            "response":      answer,
            "routed_to":     route.tier,
            "model_used":    route.model,
            "escalated":     False,
            "cited_urls":    cited_urls,
            "rag_context":   [d["content"] for d in retrieved_docs],
            "ticket":        ticket_text,
            "input_tokens":  parsed["input_tokens"],
            "output_tokens": parsed["output_tokens"],
        }

    def _record(self, kind: str, data: dict):
        self._trace.append({
            "kind": kind,
            "ts":   round(time.time() - self._run_start, 3),
            **data,
        })

    def get_run_summary(self) -> dict:
        elapsed = round(time.time() - self._run_start, 2)
        model   = self._route.model if self._route and self._route.model != "__escalate__" else get_capable_llm_model()
        cost    = estimate_cost_usd(model, self._in_tokens, self._out_tokens)
        return {
            "scenario":      self.scenario,
            "provider":      self._mode,
            "model":         model,
            "elapsed_s":     elapsed,
            "tier":          self._route.tier if self._route else "unknown",
            "input_tokens":  self._in_tokens,
            "output_tokens": self._out_tokens,
            "cost_usd":      round(cost, 5),
        }

    def print_trace(self):
        icons = {"retrieval": "📚", "routing": "🔀", "llm": "🧠"}
        print("\n" + "─" * 60)
        print(f"  SUPPORT AGENT TRACE  [{self._mode.upper()}]")
        print("─" * 60)
        for entry in self._trace:
            icon = icons.get(entry["kind"], "·")
            ts   = f"+{entry['ts']:.2f}s"
            if entry["kind"] == "retrieval":
                print(f"\n  {icon} [{ts}] RAG · confidence: {entry['top_confidence']:.2f}")
                print(f"     Articles: {entry['articles_retrieved']}")
            elif entry["kind"] == "routing":
                print(f"\n  {icon} [{ts}] Route → {entry['tier']}")
                print(f"     Model: {entry['model']}")
                print(f"     Reason: {entry['rationale']}")
            elif entry["kind"] == "llm":
                print(f"\n  {icon} [{ts}] LLM · {entry['model']}")
                print(f"     {entry['input_tokens']}→{entry['output_tokens']} tokens")
                print(f"     {entry.get('response_preview','')[:100]}")
        s = self.get_run_summary()
        print(f"\n  Cost: ${s['cost_usd']:.5f} · {s['elapsed_s']}s · {s['tier']}")
        print("─" * 60)
