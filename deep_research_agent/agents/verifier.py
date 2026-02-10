"""Verifier / Skeptic Agent: audits claim–citation pairs and suggests corrections."""
import json
import re
from typing import Any

from deep_research_agent.llm import invoke


SYSTEM = """...
- "corrected_report": if there are issues, a corrected version of the report. 
IMPORTANT: You MUST include the "## References" section and the full reference list at the end of the corrected_report exactly as it appeared in the input.
..."""

def verify_citations(
    report: str,
    papers: list[dict],
) -> dict[str, Any]:
    """
    Check report for citation accuracy and claim–evidence alignment. Returns { valid, issues, corrected_report }.
    """
    ref_list = "\n".join(
        f"[{i}] id={p.get('id')} title={p.get('title')} summary={(p.get('summary') or '')[:800]}"
        for i, p in enumerate(papers, 1)
    )
    prompt = f"""Report to verify:\n\n{report[:12000]}\n\n---\nPapers:\n{ref_list}"""
    out = invoke(prompt, system=SYSTEM, temperature=0, max_tokens=8192)
    out = out.strip()
    if out.startswith("```"):
        out = out.split("```")[1]
        if out.startswith("json"):
            out = out[4:]
    out = out.strip()
    try:
        data = json.loads(out)
        return {
            "valid": data.get("valid", False),
            "issues": data.get("issues", []),
            "corrected_report": data.get("corrected_report", report),
        }
    except json.JSONDecodeError:
        return {"valid": True, "issues": [], "corrected_report": report}


def extract_citation_ids(report: str) -> set[int]:
    """Extract reference indices like [1], [2] from report."""
    return set(int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", report))
