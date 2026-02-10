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


def generate_reference_block(papers: list[dict]) -> str:
    """Helper to ensure references are consistently formatted and appended."""
    ref_lines = []
    for i, p in enumerate(papers, 1):
        authors_list = p.get("authors", [])
        authors = ", ".join(authors_list[:3])
        if len(authors_list) > 3:
            authors += " et al."
        title = p.get("title", "Unknown Title")
        url = p.get("url", f"arXiv:{p.get('arxiv_id', 'N/A')}")
        ref_lines.append(f"[{i}] {authors}. {title}. {url}")
    return "\n\n## References\n" + "\n".join(ref_lines)


def run_research(
    query: str,
    end_date: str | None = None,
) -> tuple[str, list[dict], dict[str, Any]]:
    """
    Run a Plan → Execute → Verify pipeline.
    Ensures metadata integrity for the DeepScholar Benchmark.
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
    
    if not papers:
        return "No relevant papers found for the given query.", [], {"verified": False, "num_papers": 0}

    nuggets = extract_nuggets(papers)
    
    # Generate initial report (Synthesizer appends references internally)
    report = synthesize_report(query, plan, papers, nuggets)

    # 3. Verify loop
    verified = False
    verify_result: dict[str, Any] | None = None
    iteration = 0

    while iteration < cfg.max_verify_iterations:
        iteration += 1
        
        # Verify the current version of the report
        verify_result = verify_citations(report, papers)
        verified = bool(verify_result.get("valid", False))
        
        # Update the report with the verifier's corrections
        report = verify_result.get("corrected_report", report)
        
        # CRITICAL FIX: Ensure ## References exists in the report after verification.
        # If the verifier's LLM omitted the references in its corrected_report,
        # we programmatically re-attach the guaranteed metadata.
        if "## References" not in report:
            report += generate_reference_block(papers)
            
        if verified:
            break

    stats = {
        "verified": verified,
        "verify_result": verify_result,
        "num_papers": len(papers),
        "iterations": iteration,
    }
    
    return report, papers, stats