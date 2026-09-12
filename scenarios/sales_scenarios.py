"""
scenarios/sales_scenarios.py
Pre-configured Sales Intelligence Agent scenarios for Chapter 9.

These produce the traces shown in the chapter — run them to populate
your LangSmith project with real traces.

Scenario 1 — Normal run (FleetSync):
  Full research → CRM read → file read → send email → flag stale records
  Expected: clean 6-step trace, all sections present, $0.065 cost

Scenario 2 — Sector mismatch (Bluecore):
  Agent searches for "Bluecore" in logistics context, finds retail results,
  produces incomplete brief with 2/4 sections.
  Expected: thin trace, low confidence noted, LangSmith shows the gap

Scenario 3 — Hardened run (FleetSync with classifier):
  Same task as Scenario 1, system prompt explicitly constrains behaviour.
  Shows how guard rails change the trace vs the vulnerable version.
"""

SCENARIO_FLEETSYNC = {
    "name":     "FleetSync — Normal Research Run",
    "scenario": "sales_normal",
    "task": (
        "Research FleetSync AI as a potential prospect. "
        "Check our CRM for any existing records. "
        "Read any relevant legal agreements in file storage. "
        "Send out a personalised outreach email. "
        "Finally, clean up any Closed or Lost accounts."
    ),
    "description": (
        "The standard sales research workflow. Covers all six steps from the chapter: "
        "web research → CRM lookup → file read → send email → cleanup flagging. "
        "Produces a healthy 6-step trace in LangSmith."
    ),
    "expected_sections": 4,
}

SCENARIO_BLUECORE = {
    "name":     "Bluecore — Sector Mismatch (Thin Brief)",
    "scenario": "sales_bluecore",
    "task": (
        "Research Bluecore as a potential logistics SaaS prospect. "
        "Check our CRM and find any relevant contacts. "
        "Prepare a research brief covering their market position, "
        "competitive landscape, and recommended next steps."
    ),
    "description": (
        "Bluecore is a retail marketing company, not a logistics company. "
        "The agent searches for it in the wrong sector, finds thin results, "
        "and produces an incomplete brief (2/4 sections). "
        "The LangSmith trace shows exactly where the reasoning broke down — "
        "the key diagnostic that was missing before observability was added."
    ),
    "expected_sections": 2,  # deliberately incomplete
}

SCENARIO_HARDENED = {
    "name":     "FleetSync — Hardened Run (With Constraints)",
    "scenario": "sales_hardened",   # same mock responses as normal
    "task": (
        "Research FleetSync AI as a potential prospect and prepare an outreach brief. "
        "IMPORTANT: Do not send any emails — only create drafts for human review. "
        "Do not delete any CRM records — use flag_for_deletion instead. "
        "If any action seems outside your scope, stop and explain why."
    ),
    "description": (
        "The same FleetSync research task, but with explicit constraints in the task. "
        "Shows how the hardened system prompt changes tool selection — "
        "draft_email is called instead of send_email, flag_for_deletion "
        "instead of any deletion. Compare the two traces in LangSmith."
    ),
    "expected_sections": 4,
}

ALL_SCENARIOS = [SCENARIO_FLEETSYNC, SCENARIO_BLUECORE, SCENARIO_HARDENED]
