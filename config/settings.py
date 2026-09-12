"""
config/settings.py
All environment and runtime settings for Chapter 9.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Provider ─────────────────────────────────────────────────
# Options: anthropic | openai | mock
# Default: mock (scripted responses, no credentials needed)
LLM_PROVIDER  = os.getenv("LLM_PROVIDER", "mock").lower()
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_KEY    = os.getenv("OPENAI_API_KEY", "")

# Model names
ANTHROPIC_MODEL_SONNET = "claude-sonnet-4-6"
ANTHROPIC_MODEL_HAIKU  = "claude-haiku-4-5-20251001"
OPENAI_MODEL_GPT4O     = "gpt-4o"
OPENAI_MODEL_GPT4O_MINI = "gpt-4o-mini"

# Auto-detect mock mode: explicit flag, provider=mock, or no key for chosen provider
def _is_mock() -> bool:
    if os.getenv("MOCK_LLM", "").lower() in ("true", "1", "yes"):
        return True
    if LLM_PROVIDER == "mock":
        return True
    if LLM_PROVIDER == "anthropic" and not ANTHROPIC_KEY:
        return True
    if LLM_PROVIDER == "openai" and not OPENAI_KEY:
        return True
    return False

MOCK_LLM = _is_mock()

# ── Model selection by provider ───────────────────────────────────
# Agents always use these — provider-specific model names resolved here.
def get_capable_llm_model() -> str:
    """Return the 'capable' model for the configured provider."""
    if LLM_PROVIDER == "openai":
        return OPENAI_MODEL_GPT4O
    return ANTHROPIC_MODEL_SONNET

def get_fast_cheap_llm_model() -> str:
    """Return the 'fast/cheap' model for the configured provider."""
    if LLM_PROVIDER == "openai":
        return OPENAI_MODEL_GPT4O_MINI
    return ANTHROPIC_MODEL_HAIKU

# ── LangSmith ────────────────────────────────────────────────────
LANGSMITH_ENABLED  = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_API_KEY  = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT  = os.getenv("LANGSMITH_PROJECT", "chapter-09")

# ── Arize Phoenix ─────────────────────────────────────────────────
PHOENIX_MODE     = os.getenv("PHOENIX_MODE", "local")
PHOENIX_API_KEY  = os.getenv("PHOENIX_API_KEY", "")
PHOENIX_ENDPOINT = os.getenv( "PHOENIX_COLLECTOR_ENDPOINT","")

# ── Agent ────────────────────────────────────────────────────────
AUTO_APPROVE = os.getenv("AUTO_APPROVE", "true").lower() in ("true", "1", "yes")

# ── Pricing (USD per million tokens, 2026) ────────────────────────
COST_PER_M = {
    ANTHROPIC_MODEL_SONNET:  {"input": 3.00,  "output": 15.00},
    ANTHROPIC_MODEL_HAIKU:   {"input": 0.25,  "output": 1.25},
    OPENAI_MODEL_GPT4O:      {"input": 2.50,  "output": 10.00},
    OPENAI_MODEL_GPT4O_MINI: {"input": 0.15,  "output": 0.60},
}

def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = COST_PER_M.get(model, {"input": 3.0, "output": 15.0})
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
