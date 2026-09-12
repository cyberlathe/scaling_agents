"""
observability/phoenix_setup.py
Arize Phoenix initialisation and auto-instrumentation.

Phoenix captures every Anthropic SDK call as an OpenTelemetry span.
AnthropicInstrumentor patches anthropic.Anthropic at import time —
no other code changes needed.

Two modes:
  PHOENIX_MODE=local  → Phoenix UI at http://localhost:6006 (default)
  PHOENIX_MODE=cloud  → send to https://app.phoenix.arize.com
"""
import os
from config.settings import PHOENIX_MODE, PHOENIX_API_KEY, PHOENIX_ENDPOINT, LLM_PROVIDER

_phoenix_session = None
_phoenix_instrumented = False
_phoenix_provider = None
_PHOENIX_AVAILABLE = False
_OPENINFERENCE_AVAILABLE = False

# Try importing Phoenix
try:
    import phoenix as px
    _PHOENIX_AVAILABLE = True
except ImportError:
    pass


_ANTHROPIC_INSTRUMENTOR_AVAILABLE = False
_OPENAI_INSTRUMENTOR_AVAILABLE = False

if LLM_PROVIDER == "anthropic":
    # Try importing OpenInference Anthropic instrumentation
    try:
        from openinference.instrumentation.anthropic import AnthropicInstrumentor
        _ANTHROPIC_INSTRUMENTOR_AVAILABLE = True
        _OPENINFERENCE_AVAILABLE = True
    except ImportError:
        _ANTHROPIC_INSTRUMENTOR_AVAILABLE = False
elif LLM_PROVIDER == "openai":
    # Try importing OpenInference OpenAI instrumentation
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        _OPENAI_INSTRUMENTOR_AVAILABLE = True
        _OPENINFERENCE_AVAILABLE = True
    except ImportError:
        _OPENAI_INSTRUMENTOR_AVAILABLE = False


def init_phoenix(project_name: str = "chapter-09") -> bool:
    """
    Initialise Phoenix and instrument the Anthropic SDK.

    Returns True if Phoenix is active, False if unavailable/not configured.
    Call once at the start of your script — before creating any Anthropic clients.
    """
    global _phoenix_session, _phoenix_instrumented, _phoenix_provider

    if not _PHOENIX_AVAILABLE:
        return False

    if PHOENIX_MODE == "local":
        # Launch Phoenix local UI (http://localhost:6006)
        # Calling launch_app() when already running is safe — it's a no-op.
        try:
            _phoenix_session = px.launch_app()
        except Exception as e:
            print(f"  ⚠ Phoenix local launch failed: {e}")
            return False

        # Point OpenTelemetry exporter at the local Phoenix collector
        # os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"

    elif PHOENIX_MODE == "cloud":
        if not PHOENIX_API_KEY:
            print("  ⚠ PHOENIX_MODE=cloud but PHOENIX_API_KEY is not set")
            return False
        # os.environ["PHOENIX_API_KEY"]            = PHOENIX_API_KEY
        # os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = PHOENIX_ENDPOINT

    # Auto-instrument Anthropic and/or OpenAI SDKs — every client call
    # becomes an OpenTelemetry span forwarded to Phoenix automatically.
    if not _phoenix_instrumented and _OPENINFERENCE_AVAILABLE:
        try:
            from phoenix.otel import register

            collector_endpoint = PHOENIX_ENDPOINT.rstrip("/") or "http://localhost:6006"
            provider = register(
                project_name=project_name,
                auto_instrument=True,
                batch=True,
                endpoint=collector_endpoint + "/v1/traces",
                headers={"Authorization": f"Bearer {PHOENIX_API_KEY}"},
                protocol="http/protobuf",
                # verbose=False,
            )
            _phoenix_provider = provider

            # Instrument whichever SDKs are installed
            if _ANTHROPIC_INSTRUMENTOR_AVAILABLE:
                AnthropicInstrumentor().instrument(tracer_provider=provider)
            if _OPENAI_INSTRUMENTOR_AVAILABLE:
                OpenAIInstrumentor().instrument(tracer_provider=provider)

            _phoenix_instrumented = True
        except Exception as e:
            print(f"  ⚠ Phoenix instrumentation failed: {e}")
            return False

    return _phoenix_instrumented


def flush_phoenix() -> None:
    """Flush batched OpenTelemetry spans before querying Phoenix."""
    if _phoenix_provider is not None:
        _phoenix_provider.force_flush()


def get_phoenix_url() -> str:
    """Return the URL where Phoenix traces can be viewed."""
    if PHOENIX_MODE == "local":
        return "http://localhost:6006"
    return PHOENIX_ENDPOINT


def print_phoenix_status():
    if not _PHOENIX_AVAILABLE:
        print("  · Arize Phoenix not available (pip install arize-phoenix)")
        return
    if not _OPENINFERENCE_AVAILABLE:
        print("  · OpenInference not available")
        print("    pip install openinference-instrumentation-anthropic openinference-instrumentation-openai")
        return
    if _phoenix_instrumented:
        mode     = "local" if PHOENIX_MODE == "local" else "cloud"
        providers = []
        if _ANTHROPIC_INSTRUMENTOR_AVAILABLE:
            providers.append("Anthropic")
        if _OPENAI_INSTRUMENTOR_AVAILABLE:
            providers.append("OpenAI")
        print(f"  ✓ Arize Phoenix active ({mode}) → {get_phoenix_url()}")
        print(f"    Instrumenting: {', '.join(providers)}")
    else:
        print("  · Phoenix not instrumented — call init_phoenix() first")

