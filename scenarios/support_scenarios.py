"""
scenarios/support_scenarios.py
Pre-configured Customer Support Agent scenarios for Chapter 9.

These cover the three routing tiers from the chapter and produce
meaningful Phoenix evaluation scores — including a deliberate
hallucination that Phoenix flags.

Scenario 1 — Password reset (Tier-0 → Haiku):
  Simple auth query → routed to Haiku → $0.036/ticket
  LangSmith trace shows: RAG retrieval → Haiku call → response
  Phoenix: high QA correctness + relevance, no hallucination

Scenario 2 — Billing dispute (Tier-2 → Sonnet):
  Complex billing question → routed to Sonnet → $0.29/ticket
  LangSmith trace shows: RAG retrieval → Sonnet call → response
  Phoenix: evaluators check whether billing plan details are accurate

Scenario 3 — Hallucination trigger:
  Query about data export → agent cites non-existent help article URLs
  LangSmith trace: RAG retrieval → Sonnet call → response with broken links
  Phoenix: HallucinationEvaluator flags broken URLs → appears in Evals tab
"""

SCENARIO_PASSWORD_RESET = {
    "name":     "Password Reset — Tier-0 (Fast Model)",
    "scenario": "support_password",
    "ticket": "Hi, I forgot my password and I can't log into my account. How do I reset it?",
    "description": (
        "Classic tier-0 ticket. Matched by the _TIER0 regex classifier → routed to Haiku. "
        "Costs ~$0.036/ticket vs $0.29 for Sonnet. "
        "Phoenix evaluators should score: not hallucinated, QA correct, relevant."
    ),
    "expected_route": "tier-0 (fast model)",
}

SCENARIO_BILLING = {
    "name":     "Billing Dispute — Tier-2 (Capable Model)",
    "scenario": "support_billing",
    "ticket": (
        "I just received my invoice and the amount doesn't match what I expected. "
        "I'm on the Pro plan and I thought it was $99/month but I was charged $299. "
        "Can you explain this charge and whether I'm entitled to a refund?"
    ),
    "description": (
        "Complex billing question requiring plan knowledge and nuanced explanation. "
        "No regex match → routed to Sonnet. Costs $0.29/ticket. "
        "Phoenix evaluators check: does the response accurately explain annual billing? "
        "QA correctness depends on whether the agent uses the billing article correctly."
    ),
    "expected_route": "tier-2 (capable model)",
}

SCENARIO_HALLUCINATION = {
    "name":     "Data Export — Hallucination Trigger",
    "scenario": "support_hallucination",
    "ticket": (
        "I need to export all my project data as a CSV file. "
        "Where do I find this option and what format will the data be in?"
    ),
    "description": (
        "The mock response for this scenario includes citations to help article URLs "
        "that do NOT exist in the help article database "
        "(articles/data-export-99999 and articles/csv-format-guide-88888). "
        "Phoenix's HallucinationEvaluator + URL verification will flag these. "
        "This is the scenario from Section 9.7 — the 14 hallucination alerts on the dashboard."
    ),
    "expected_route": "tier-2 (capable model)",
    "expect_hallucination": True,
}

ALL_SCENARIOS = [SCENARIO_PASSWORD_RESET, SCENARIO_BILLING, SCENARIO_HALLUCINATION]
