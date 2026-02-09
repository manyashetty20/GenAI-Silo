"""Reader / Nugget Agent: extracts specific facts, metrics, and mechanisms from papers."""
import json
from typing import Any

from deep_research_agent.llm import invoke


SYSTEM = """You are a scientific reader. For each paper, extract 2-5 short "nuggets": specific facts, numbers, methods, or claims that would support a related-work synthesis. Each nugget must be grounded in the paper. Output a JSON array of objects, one per paper, with keys:
- "id": same as the paper id provided
- "nuggets": list of strings (each one sentence, no citation yet)

Output only valid JSON array, no markdown."""


def extract_nuggets(papers: list[dict]) -> list[dict[str, Any]]:
    """For each paper, extract nuggets (facts/claims) for synthesis."""
    if not papers:
        return []
    # Batch into chunks to avoid token limits (e.g. 5 papers per call)
    batch_size = 5
    results = []
    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        parts = []
        for p in batch:
            parts.append(
                f"Paper id: {p.get('id', '')}\nTitle: {p.get('title', '')}\nAbstract/summary: {(p.get('summary') or '')[:1500]}\n"
            )
        prompt = "Extract nuggets from these papers:\n\n" + "\n---\n".join(parts)
        out = invoke(prompt, system=SYSTEM, temperature=0.2, max_tokens=2000)
        out = out.strip()
        if out.startswith("```"):
            out = out.split("```")[1]
            if out.startswith("json"):
                out = out[4:]
        out = out.strip()
        try:
            arr = json.loads(out)
            if isinstance(arr, list):
                results.extend(arr)
            else:
                results.append(arr)
        except json.JSONDecodeError:
            for p in batch:
                results.append({"id": p.get("id"), "nuggets": [p.get("title", "")]})
    return results
