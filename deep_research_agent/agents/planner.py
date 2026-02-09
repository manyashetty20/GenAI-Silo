"""Planner Agent: decomposes the benchmark query into sub-questions and a research strategy."""
import json
from typing import Any

from deep_research_agent.llm import invoke


SYSTEM = """You are a research planner for scientific literature synthesis. Given a main research query (e.g. a "related work" style question), output a JSON object with:
- "sub_questions": list of 3-6 focused sub-questions that cover the main query (each answerable with papers).
- "search_queries": list of 4-8 search query strings to use for arXiv and Semantic Scholar (concise, keyword-rich).
- "strategy": one short paragraph describing the research strategy.

Output only valid JSON, no markdown or extra text."""


def plan_research(query: str) -> dict[str, Any]:
    """Decompose query into sub_questions, search_queries, and strategy."""
    prompt = f"Research query:\n{query}"
    out = invoke(prompt, system=SYSTEM, temperature=0.3, max_tokens=1500)
    out = out.strip()
    if out.startswith("```"):
        out = out.split("```")[1]
        if out.startswith("json"):
            out = out[4:]
    out = out.strip()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {
            "sub_questions": [query],
            "search_queries": [query],
            "strategy": "Direct search for the query.",
        }
