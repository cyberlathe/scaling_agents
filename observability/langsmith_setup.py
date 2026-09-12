"""
observability/langsmith_setup.py
LangSmith initialisation and @traceable wrappers.

LangSmith captures every tool call, LLM call, and agent run as a
nested trace — visible at smith.langchain.com after setting:
  LANGSMITH_TRACING=true
  LANGSMITH_API_KEY=ls__...
  LANGSMITH_PROJECT=chapter-09

No code changes needed beyond importing this module and using the
wrapped tool functions instead of the raw ones.
"""
import os
import functools
from config.settings import LANGSMITH_ENABLED, LANGSMITH_PROJECT

# Set env vars that the LangSmith SDK reads at import time
if LANGSMITH_ENABLED:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", LANGSMITH_PROJECT)

# Guard import — langsmith is optional
try:
    from langsmith import traceable, Client as LSClient
    _LS_AVAILABLE = True
except ImportError:
    _LS_AVAILABLE = False
    def traceable(*args, **kwargs):
        """No-op decorator when langsmith is not installed."""
        def decorator(fn):
            return fn
        # Handle both @traceable and @traceable(run_type=...)
        if args and callable(args[0]):
            return args[0]
        return decorator


def is_enabled() -> bool:
    return LANGSMITH_ENABLED and _LS_AVAILABLE


def wrap_tool(fn, run_type: str = "tool"):
    """
    Wrap a tool function with @traceable so every call appears as a
    child span in the LangSmith trace, with inputs and outputs captured.
    """
    if not is_enabled():
        return fn
    return traceable(run_type=run_type, name=fn.__name__)(fn)


def agent_trace(name: str, run_type: str = "chain"):
    """
    Decorator for the agent's top-level run() method.
    Creates the root span in LangSmith that all tool/LLM spans nest under.
    """
    if not is_enabled():
        def passthrough(fn):
            return fn
        return passthrough
    return traceable(run_type=run_type, name=name)


def llm_trace(name: str):
    """Decorator for LLM call wrappers — appears as an 'llm' span in LangSmith."""
    if not is_enabled():
        def passthrough(fn):
            return fn
        return passthrough
    return traceable(run_type="llm", name=name)


def print_langsmith_status():
    if is_enabled():
        print(f"  ✓ LangSmith tracing active → project: '{LANGSMITH_PROJECT}'")
        print(f"    View traces at: https://smith.langchain.com")
        print(f"    Note: @traceable wraps tool calls. Anthropic/OpenAI spans appear as child nodes.")
    elif LANGSMITH_ENABLED and not _LS_AVAILABLE:
        print("  ⚠ LANGSMITH_TRACING=true but langsmith package not installed")
        print("    Run: pip install langsmith")
    else:
        print("  · LangSmith tracing disabled (set LANGSMITH_TRACING=true to enable)")
