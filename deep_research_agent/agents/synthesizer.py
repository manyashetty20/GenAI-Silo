"""Synthesizer / Writer Agent: produces a structured, cited scientific answer."""
from typing import Any
from deep_research_agent.llm import invoke

# Updated System Prompt for Benchmark Parsing
SYSTEM = """You write a highly technical "Related Work" section in Markdown. 
Rules:
- You MUST start the response with the header "# Related Work".
- Use double newlines (\\n\\n) between every single paragraph.
- Every factual claim must be followed by an inline citation like [1] or [2].
- Focus on specific metrics, SNR gains, and architectural comparisons.
- DO NOT generate the References section; just write the body text."""

def synthesize_report(
    query: str,
    plan: dict[str, Any],
    papers: list[dict],
    nuggets_by_paper: list[dict],
    max_tokens: int = 8192,
) -> str:
    ref_map = {p.get("id"): p for p in papers}
    nugget_map = {n.get("id"): n.get("nuggets", []) for n in nuggets_by_paper if n.get("id")}

    # 1. Build Reference Block
    ref_lines = []
    for i, p in enumerate(papers, 1):
        authors_list = p.get("authors", [])
        authors = ", ".join(authors_list[:3]) + (" et al." if len(authors_list) > 3 else "")
        title = p.get("title", "Unknown Title")
        url = p.get("url", "N/A")
        ref_lines.append(f"[{i}] {authors}. {title}. {url}")
    
    formatted_references = "\n\n## References\n" + "\n".join(ref_lines)

    # 2. Build Nugget text for LLM
    nugget_text = []
    for pid, nuggets in nugget_map.items():
        idx = next((i for i, x in enumerate(papers, 1) if x.get("id") == pid), None)
        tag = f"[{idx}]" if idx else ""
        for n in (nuggets or [])[:5]:
            nugget_text.append(f"- {n} {tag}")

    prompt = f"""Research query: {query}
Strategy: {plan.get('strategy', '')}

Nuggets:
{chr(10).join(nugget_text)}

Write the technical synthesis. Use the indices provided (e.g., [1])."""

    # 3. Assemble with mandatory double newline
    report_body = invoke(prompt, system=SYSTEM, temperature=0.2, max_tokens=max_tokens)
    
    return report_body + "\n\n" + formatted_references