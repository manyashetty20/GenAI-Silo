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
    Optimized for DeepScholar verifiability and nugget coverage.
    """
    cfg = get_config().agent

    # 1. Plan: Decompose query into sub-questions
    plan = plan_research(query)

    # 2. Execute: Recursive search and nugget extraction
    papers = search_agent(
        plan.get("search_queries", [query]),
        main_query=query,
        end_date=end_date,
    )
    
    if not papers:
        return "No relevant papers found.", [], {"verified": False, "num_papers": 0}

    nuggets = extract_nuggets(papers)
    
    # Generate initial synthesis (Synthesizer now focused on body text)
    report = synthesize_report(query, plan, papers, nuggets)

    # 3. Verify loop: Audit claim-citation pairs
    verified = False
    verify_result: dict[str, Any] | None = None
    iteration = 0

    while iteration < cfg.max_verify_iterations:
        iteration += 1
        
        # Verify current report against nuggets
        verify_result = verify_citations(report, papers)
        verified = bool(verify_result.get("valid", False))
        
        # Update report with corrections
        report = verify_result.get("corrected_report", report)
        
        # FORCE: Ensure references are attached even if the Verifier LLM omits them
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