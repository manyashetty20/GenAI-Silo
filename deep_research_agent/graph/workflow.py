"""
Plan → Execute → Verify loop for agentic deep research (no external graph lib).
"""
from typing import Any

from deep_research_agent.agents import (
    plan_research,
    search_agent,
    extract_nuggets,
    synthesize_report,
    verify_citations,
)
from deep_research_agent.config import get_config


def run_research(
    query: str,
    end_date: str | None = None,
) -> tuple[str, list[dict], dict[str, Any]]:
    """
    Run a simple Plan → Execute → Verify pipeline.

    Steps:
    1. Planner: decompose query into sub-questions and search queries.
    2. Execute: search → nuggets → initial synthesis.
    3. Verify: audit claim–citation pairs; optionally retry synthesis.
    """
    cfg = get_config().agent

    # 1. Plan
    plan = plan_research(query)

    # 2. Execute (search + nuggets + first report)
    papers = search_agent(
        plan.get("search_queries", [query]),
        main_query=query,
        end_date=end_date,
    )
    nuggets = extract_nuggets(papers)
    report = synthesize_report(query, plan, papers, nuggets)

    # 3. Verify loop
    verified = False
    verify_result: dict[str, Any] | None = None
    iteration = 0

    while iteration < cfg.max_verify_iterations:
        iteration += 1
        verify_result = verify_citations(report, papers)
        verified = bool(verify_result.get("valid", False))
        corrected_report = verify_result.get("corrected_report", report)
        report = corrected_report
        if verified:
            break
        # Optionally, you could inject verify_result['issues'] into a refined
        # synthesis prompt here for a smarter retry. For now, we trust the
        # verifier's corrected_report and just re-check it in the next loop.

    stats = {
        "verified": verified,
        "verify_result": verify_result,
        "num_papers": len(papers),
        "iterations": iteration,
    }
    return report, papers, stats
