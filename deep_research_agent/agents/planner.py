"""
Planner Agent: decomposes the benchmark query into sub-questions and a research strategy.
"""
import json
from typing import Any, Optional


SYSTEM = """
You are an expert research planner.

Given a research question, generate a concise research plan.

Return JSON with the following fields:
- sub_questions: list of important research sub-questions
- search_queries: list of literature search queries (max 3)
- strategy: short description of the research strategy

Limit your research plan to a maximum of 3 highly targeted search queries to ensure efficiency.
"""


def plan_research(query: str, llm: Optional[Any] = None) -> dict[str, Any]:
    """
    Decompose query into sub_questions, search_queries, and strategy.
    """

    prompt = f"Research query:\n{query}"

    if llm is None:
        raise RuntimeError("LLM instance must be provided to planner.")

    out = llm.invoke(prompt, system=SYSTEM)

    out = out.strip()

    # Remove markdown code fences if present
    if out.startswith("```"):
        out = out.split("```")[1]
        if out.startswith("json"):
            out = out[4:]

    out = out.strip()

    try:
        return json.loads(out)

    except json.JSONDecodeError:

        # Safe fallback if LLM returns invalid JSON
        return {
            "sub_questions": [query],
            "search_queries": [query],
            "strategy": "Direct literature search using the original query."
        }