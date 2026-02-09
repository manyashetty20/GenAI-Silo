"""Verifier / Skeptic Agent: audits claim–citation pairs and suggests corrections."""
import json
import re
from typing import Any

from deep_research_agent.llm import invoke


SYSTEM = """You are a verifier for a research synthesis. Given a report and the list of papers (with id, title, summary), you check:
1. Does each inline citation [1], [2], etc. refer to a real reference in the References section?
2. Is each cited claim actually supported by the cited paper's content?
Output a JSON object:
- "valid": true if the report is fully correct, false otherwise
- "issues": list of strings describing each problem (e.g. "Claim X cites [3] but paper 3 does not support it")
- "corrected_report": if there are issues, a corrected version of the report (same structure, fix citations/claims); if valid, the same report as input.
Output only valid JSON."""


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
