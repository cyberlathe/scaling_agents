"""
evaluation/phoenix_evals.py
Arize Phoenix evaluation pipeline for the Customer Support Agent.

Runs three built-in evaluators against a batch of ticket results:
  - HallucinationEvaluator: detects claims unsupported by retrieved context
  - QACorrectnessEvaluator: checks answers match the help article content
  - RelevanceEvaluator: did the response address the actual question?

Also runs a lightweight URL-verification check (no LLM needed) to flag
cited help article URLs that don't exist in the real article database.

Results are logged back to Phoenix and visible in the Evals tab.
"""
import pandas as pd
from typing import Optional


from tools.mock_helpdesk import verify_urls

# Guard imports — phoenix-evals is optional
try:
    from phoenix.evals.metrics.hallucination import HallucinationEvaluator
    from phoenix.evals.metrics.retrieval_relevance import RetrievalRelevanceEvaluator
    from phoenix.evals.metrics.correctness import CorrectnessEvaluator
    from phoenix.evals import LLM, evaluate_dataframe
    # from phoenix.evals import run_evals
    from phoenix.client import Client

    _PHOENIX_EVALS_AVAILABLE = True
except ImportError:
    _PHOENIX_EVALS_AVAILABLE = False

from config.settings import ANTHROPIC_KEY, ANTHROPIC_MODEL_HAIKU, OPENAI_KEY, OPENAI_MODEL_GPT4O_MINI, PHOENIX_API_KEY, PHOENIX_ENDPOINT


def build_eval_dataframe(ticket_results: list[dict]) -> pd.DataFrame:
    """
    Convert a list of ticket result dicts into the DataFrame shape
    that Phoenix evaluators expect.

    Each row = one ticket interaction.
    Columns required by Phoenix:
      - input:    the customer's question (or full ticket text)
      - output:   the agent's response
      - reference: the help article context that was retrieved (RAG context)
    """
    rows = []
    for r in ticket_results:
        rows.append({
            "input":     r.get("ticket", ""),
            "output":    r.get("response", ""),
            "reference": "\n\n".join(r.get("rag_context", [])) or "No context retrieved.",
            # Metadata carried through for the URL check
            "cited_urls":   r.get("cited_urls", []),
            "model_used":   r.get("model_used", "unknown"),
            "routed_to":    r.get("routed_to", "unknown"),
            "scenario":     r.get("scenario", "unknown"),
        })
    return pd.DataFrame(rows)


def run_url_verification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Non-LLM hallucination check: verify every URL cited in the response
    actually exists in the help article database.
    Adds columns: urls_checked, broken_urls, has_broken_url
    """
    results = []
    for _, row in df.iterrows():
        cited = row.get("cited_urls", [])
        if isinstance(cited, str):
            import ast
            try:
                cited = ast.literal_eval(cited)
            except Exception:
                cited = []
        verified = verify_urls(cited) if cited else {"broken": [], "all_valid": True, "checked": 0}
        results.append({
            "urls_checked":    verified["checked"],
            "broken_urls":     verified["broken"],
            "has_broken_url":  not verified["all_valid"],
        })
    return df.assign(**pd.DataFrame(results).to_dict("list"))


def run_phoenix_evals(
    ticket_results: list[dict],
    project_name: str = "chapter-09",
    concurrency: int = 4,
) -> Optional[pd.DataFrame]:
    """
    Run the full Phoenix evaluation pipeline on a batch of ticket results.

    Returns a DataFrame with all scores, or None if Phoenix is unavailable.
    Results are also logged back to Phoenix (visible in the Evals tab).

    Args:
        ticket_results: list of dicts returned by SupportAgent.handle_ticket()
        project_name:   Phoenix project to log results to
        concurrency:    parallel evaluator calls
    """
    if not _PHOENIX_EVALS_AVAILABLE:
        print("  ⚠ phoenix-evals not available. Run: pip install arize-phoenix-evals")
        return None

    if not (ANTHROPIC_KEY or OPENAI_KEY):
        print("  ⚠ ANTHROPIC_API_KEY or OPENAI_API_KEY required for Phoenix LLM evaluators.")
        print("    URL verification check will still run (no LLM needed).")

    df_results = build_eval_dataframe(ticket_results)
    print(f"\n  Running Phoenix evals on {len(df_results)} ticket(s)...")

    # ── Step 1: URL verification (no LLM needed) ──────────────────
    df = run_url_verification(df_results)
    broken_count = df["has_broken_url"].sum()
    if broken_count:
        print(f"  ⚠ URL check: {broken_count} ticket(s) contain broken help article links")
        for _, row in df[df["has_broken_url"]].iterrows():
            print(f"    Broken URLs: {row['broken_urls']}")
    else:
        print(f"  ✓ URL check: all cited URLs are valid")

    # ── Step 2: LLM-based evaluators (requires Anthropic key) ─────
    if not (ANTHROPIC_KEY or OPENAI_KEY) or not _PHOENIX_EVALS_AVAILABLE:
        print("  · Skipping LLM evaluators (no API key)")
        return df_results

    try:
        if ANTHROPIC_KEY:
            judge = LLM(provider="anthropic", model=ANTHROPIC_MODEL_HAIKU, api_key=ANTHROPIC_KEY)
        elif OPENAI_KEY:
            judge = LLM(provider="openai", model=OPENAI_MODEL_GPT4O_MINI, api_key=OPENAI_KEY)

        px_client = Client(
            base_url="https://app.phoenix.arize.com/s/cyberlathe",
            api_key=PHOENIX_API_KEY
            )
        df = px_client.spans.get_spans_dataframe(project_identifier=project_name)
        parent_spans = df[df["span_kind"] == "CHAIN"]

        # from phoenix.evals import async_evaluate_dataframe
        # from phoenix.trace import suppress_tracing

        # with suppress_tracing():
        #     results_df = evaluate_dataframe(
        #         dataframe=parent_spans,
        #         evaluators=[
        #                         HallucinationEvaluator(judge),   # flags unsupported factual claims
        #                         CorrectnessEvaluator(judge),   # checks answers match retrieved context
        #                         RetrievalRelevanceEvaluator(judge),       # did response address the actual question?
        #                     ]
        #     )

        results_df = evaluate_dataframe(
            dataframe=parent_spans,
            evaluators=[
                            HallucinationEvaluator(judge),   # flags unsupported factual claims
                            CorrectnessEvaluator(judge),   # checks answers match retrieved context
                            RetrievalRelevanceEvaluator(judge),       # did response address the actual question?
                        ]
        )

        from phoenix.evals.utils import to_annotation_dataframe

        evaluations = to_annotation_dataframe(
            dataframe=results_df
        )

        Client().spans.log_span_annotations_dataframe(
            dataframe=evaluations
        )

        # # Merge all eval results into one DataFrame
        # combined = df.copy()
        # for eval_df, name in [
        #     (hallucination_eval, "hallucination"),
        #     (qa_eval, "qa_correctness"),
        #     (relevance_eval, "relevance"),
        # ]:
        #     if eval_df is not None and not eval_df.empty:
        #         for col in ["label", "score", "explanation"]:
        #             if col in eval_df.columns:
        #                 combined[f"{name}_{col}"] = eval_df[col].values

        # Log back to Phoenix — results appear in Evals tab
        # try:
        #     px_client = Client()
        #     px_client.spans.log_span_annotations_dataframe(evaluations)

        print("  ✓ Eval scores logged to Phoenix Evals tab")
        # except Exception as e:
        #     print(f"  · Could not log to Phoenix (not critical): {e}")

        _print_eval_summary(evaluations)
        return evaluations

    except Exception as e:
        print(f"  ⚠ LLM evaluators failed: {e}")
        return df


def _print_eval_summary(df: pd.DataFrame):
    """Print a clean summary of eval results to the terminal."""
    print("\n" + "─" * 60)
    print("  PHOENIX EVALUATION RESULTS")
    print("─" * 60)

    for i, row in df.iterrows():
        print(f"\n  Ticket {i+1}: {str(row.get('input',''))[:50]}...")
        print(f"  Model: {row.get('model_used','?')} · Route: {row.get('routed_to','?')}")
        print(f"  Broken URLs: {row.get('has_broken_url', False)}")

        for metric in ["hallucination", "qa_correctness", "relevance"]:
            label = row.get(f"{metric}_label", "—")
            score = row.get(f"{metric}_score", "—")
            expl  = str(row.get(f"{metric}_explanation", ""))[:80]
            print(f"  {metric:20s}: {label} (score: {score})")
            if expl:
                print(f"    ↳ {expl}")

    print("\n" + "─" * 60)
    if "hallucination_label" in df.columns:
        hallucinated = (df["hallucination_label"] == "hallucinated").sum()
        print(f"  Hallucinations flagged: {hallucinated}/{len(df)}")
    if "qa_correctness_label" in df.columns:
        correct = (df["qa_correctness_label"] == "correct").sum()
        print(f"  QA Correct:             {correct}/{len(df)}")
    if "relevance_label" in df.columns:
        relevant = (df["relevance_label"] == "relevant").sum()
        print(f"  Relevant:               {relevant}/{len(df)}")
    print("─" * 60)


def run_simple_evals(ticket_results: list[dict]) -> pd.DataFrame:
    """
    Fallback evaluator that works without the Phoenix LLM evaluators.
    Runs only the URL verification check.
    Useful for testing the pipeline without an Anthropic key.
    """
    df = build_eval_dataframe(ticket_results)
    df = run_url_verification(df)

    print("\n" + "─" * 60)
    print("  SIMPLE EVALUATION (URL check only)")
    print("─" * 60)
    for i, row in df.iterrows():
        broken = row.get("has_broken_url", False)
        urls   = row.get("broken_urls", [])
        status = "⚠ BROKEN LINKS" if broken else "✓ URLs OK"
        print(f"  Ticket {i+1}: {status}")
        if broken:
            for u in urls:
                print(f"    ✗ {u}")
    print("─" * 60)
    return df
