"""Synthesizer / Writer Agent: produces a structured, cited scientific answer (DeepScholar-style)."""
from typing import Any

from deep_research_agent.llm import invoke


SYSTEM = """You write a "Related Work" style section in Markdown. Rules:
- Structure with clear headings and paragraphs.
- Every factual claim must be followed by an inline citation in the format [AuthorYear] or [1], [2], etc., matching the reference list.
- Use only information from the provided nuggets and papers; do not add unsupported claims.
- End with a "## References" section listing each paper as: [id] Author(s). Title. URL (or "arXiv:id")."""


def synthesize_report(
    query: str,
    plan: dict[str, Any],
    papers: list[dict],
    nuggets_by_paper: list[dict],
    max_tokens: int = 8192,
) -> str:
    """
    Produce final Markdown report with inline citations and references.
    nuggets_by_paper: list of { "id": paper_id, "nuggets": [...] } from reader.
    """
    ref_map = {p.get("id"): p for p in papers}
    nugget_map = {n.get("id"): n.get("nuggets", []) for n in nuggets_by_paper if n.get("id")}

    ref_block = []
    for i, p in enumerate(papers, 1):
        pid = p.get("id", "")
        authors = ", ".join(p.get("authors", [])[:3])
        if len(p.get("authors", [])) > 3:
            authors += " et al."
        title = p.get("title", "")
        url = p.get("url", "")
        ref_block.append(f"[{i}] {authors}. {title}. {url}")

    nugget_text = []
    for pid, nuggets in nugget_map.items():
        p = ref_map.get(pid, {})
        idx = next((i for i, x in enumerate(papers, 1) if x.get("id") == pid), None)
        tag = f"[{idx}]" if idx else pid
        for n in (nuggets or [])[:5]:
            nugget_text.append(f"- {n} {tag}")

    prompt = f"""Research query: {query}

Strategy: {plan.get('strategy', '')}

Nuggets (claim [citation]):
{chr(10).join(nugget_text)}

References (use these indices for citations):
{chr(10).join(ref_block)}

Write the Related Work section in Markdown with inline citations. End with ## References and the same reference list."""

    return invoke(prompt, system=SYSTEM, temperature=0.4, max_tokens=max_tokens)
