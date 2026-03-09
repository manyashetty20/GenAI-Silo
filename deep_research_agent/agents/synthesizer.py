"""Synthesizer / Writer Agent: produces a structured, cited scientific answer."""

from typing import Any, Optional


SYSTEM = """You write a highly technical "Related Work" section in Markdown.

Rules:
- You MUST start the response with the header "# Related Work".
- Use double newlines (\\n\\n) between every paragraph.
- Every factual claim must be followed by an inline citation like [1] or [2].
- Focus on specific metrics, architectural comparisons, and benchmark results.
- DO NOT generate the References section; only write the body text.
"""


def synthesize_report(
    query: str,
    plan: dict[str, Any],
    papers: list[dict],
    nuggets: list[dict],
    llm: Optional[Any] = None,
    max_tokens: int = 4096,
) -> str:

    if llm is None:
        raise RuntimeError("LLM instance must be provided to synthesize_report().")

    # Map paper id → nuggets
    nugget_map = {n.get("id"): n.get("nuggets", []) for n in nuggets if n.get("id")}

    # 1. Build Reference Block
    ref_lines = []

    for i, p in enumerate(papers, 1):

        authors_list = p.get("authors", [])
        authors = ", ".join(authors_list[:3])

        if len(authors_list) > 3:
            authors += " et al."

        title = p.get("title", "Unknown Title")
        url = p.get("url") or f"https://arxiv.org/abs/{p.get('arxiv_id','')}"

        ref_lines.append(f"[{i}] {authors}. {title}. {url}")

    formatted_references = "\n\n## References\n" + "\n".join(ref_lines)

    # 2. Convert nuggets into prompt context
    nugget_lines = []

    for pid, nug_list in nugget_map.items():

        idx = next((i for i, x in enumerate(papers, 1) if x.get("id") == pid), None)

        citation = f"[{idx}]" if idx else ""

        for n in nug_list[:5]:
            nugget_lines.append(f"- {n} {citation}")

    nugget_text = "\n".join(nugget_lines)

    prompt = f"""
Research query:
{query}

Research strategy:
{plan.get('strategy','')}

Extracted research nuggets:
{nugget_text}

Write the technical Related Work synthesis.
Use the citation indices provided (e.g., [1], [2]).
"""

    # 3. Generate report
    report_body = llm.invoke(prompt, system=SYSTEM)

    return report_body.strip() + "\n\n" + formatted_references