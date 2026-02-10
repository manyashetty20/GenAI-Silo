"""Planner Agent: decomposes the benchmark query into sub-questions and a research strategy."""
import json
from typing import Any

from deep_research_agent.llm import invoke

SYSTEM = """...
Limit your research plan to a maximum of 3 highly targeted search queries to ensure efficiency.
..."""


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
