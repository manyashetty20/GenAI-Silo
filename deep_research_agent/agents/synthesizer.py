"""Synthesizer / Writer Agent: produces a structured, cited scientific answer (DeepScholar-style)."""
from typing import Any
from deep_research_agent.llm import invoke

# Updated System Prompt to emphasize verifiability
SYSTEM = """You write a highly technical "Related Work" section in Markdown. 
Rules:
- Focus on specific metrics, SNR gains, and architectural comparisons.
- Every factual claim must be followed by an inline citation like [1] or [2].
- Use only information from the provided nuggets; do not speculate.
- DO NOT generate the References section; just write the body text. I will append the list."""

def synthesize_report(
    query: str,
    plan: dict[str, Any],
    papers: list[dict],
    nuggets_by_paper: list[dict],
    max_tokens: int = 8192,
) -> str:
    """
    Produce final Markdown report with accurate metadata and inline citations.
    """
    ref_map = {p.get("id"): p for p in papers}
    nugget_map = {n.get("id"): n.get("nuggets", []) for n in nuggets_by_paper if n.get("id")}

    # 1. Build the REAL reference block that will be appended at the end
    ref_lines = []
    for i, p in enumerate(papers, 1):
        pid = p.get("id", "")
        # Get actual metadata from the paper object
        authors_list = p.get("authors", [])
        authors = ", ".join(authors_list[:3])
        if len(authors_list) > 3:
            authors += " et al."
        
        title = p.get("title", "Unknown Title")
        url = p.get("url", f"arXiv:{p.get('arxiv_id', 'N/A')}")
        ref_lines.append(f"[{i}] {authors}. {title}. {url}")
    
    formatted_references = "\n\n## References\n" + "\n".join(ref_lines)

    # 2. Build the Nugget text for the LLM to reason over
    nugget_text = []
    for pid, nuggets in nugget_map.items():
        idx = next((i for i, x in enumerate(papers, 1) if x.get("id") == pid), None)
        tag = f"[{idx}]" if idx else ""
        for n in (nuggets or [])[:5]:
            nugget_text.append(f"- {n} {tag}")

    prompt = f"""Research query: {query}
Strategy: {plan.get('strategy', '')}

Nuggets (Each claim includes its citation index):
{chr(10).join(nugget_text)}

Write only the technical synthesis. Use the indices provided in the nuggets (e.g., [1]) for your citations."""

    # 3. Get the synthesis and append the GUARANTEED accurate references
    report_body = invoke(prompt, system=SYSTEM, temperature=0.2, max_tokens=max_tokens)
    
    return report_body + formatted_references