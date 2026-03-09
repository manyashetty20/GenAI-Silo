from typing import List, Dict, Optional, Any
import concurrent.futures
from tavily import TavilyClient

from deep_research_agent.config import get_config
from deep_research_agent.retrieval import (
    search_arxiv,
    search_semantic_scholar,
    rerank
)

_SEARCH_CACHE = {}

# ------------------------------------------------
# KEYWORDS used for topic filtering
# ------------------------------------------------

RAG_KEYWORDS = [
    "retrieval",
    "retrieval augmented",
    "rag",
    "retriever",
    "knowledge retrieval",
    "language model",
    "llm",
    "transformer",
    "dense retrieval",
    "vector search",
]


# ------------------------------------------------
# QUERY EXPANSION
# ------------------------------------------------

def expand_queries(main_query: str, llm: Any) -> List[str]:

    prompt = f"""
You are an expert academic search assistant.

Generate 4 specific literature search queries for the topic:

{main_query}

Focus on:
- core methods
- benchmark papers
- evaluation techniques
- limitations

Return only the queries, one per line.
"""

    resp = llm.invoke(prompt)

    queries = [
        q.strip("- ").strip()
        for q in resp.split("\n")
        if len(q.strip()) > 5
    ]

    return queries[:4]


# ------------------------------------------------
# TAVILY SEARCH
# ------------------------------------------------

def tavily_search(query):

    api_key = get_config().retrieval.tavily_api_key
    client = TavilyClient(api_key=api_key)

    results = client.search(
        query=query,
        search_depth="advanced",
        max_results=5
    )

    papers = []

    for r in results["results"]:

        papers.append({
            "title": r["title"],
            "summary": r["content"],
            "url": r["url"]
        })

    return papers


# ------------------------------------------------
# TOPIC FILTER
# removes irrelevant scientific domains
# ------------------------------------------------

def is_relevant(paper: Dict) -> bool:

    text = (
        (paper.get("title") or "") +
        " " +
        (paper.get("summary") or "")
    ).lower()

    return any(keyword in text for keyword in RAG_KEYWORDS)


# ------------------------------------------------
# MAIN SEARCH AGENT
# ------------------------------------------------

def search_agent(
    search_queries: List[str],
    main_query: Optional[str] = None,
    end_date: Optional[str] = None,
    top_k: Optional[int] = None,
    llm=None
):

    cfg = get_config()

    expanded = list(search_queries)

    # Query expansion
    if len(search_queries) == 1 and llm:
        expanded += expand_queries(search_queries[0], llm)

    all_papers = []
    seen = set()

    # -----------------------------------------
    # Run search queries
    # -----------------------------------------

    def run_query(q):

        if q in _SEARCH_CACHE:
            return _SEARCH_CACHE[q]

        with concurrent.futures.ThreadPoolExecutor() as ex:

            f1 = ex.submit(search_arxiv, q, 20, end_date)
            f2 = ex.submit(search_semantic_scholar, q, 20)
            f3 = ex.submit(tavily_search, q)

            papers = f1.result() + f2.result() + f3.result()

        _SEARCH_CACHE[q] = papers

        return papers

    # -----------------------------------------
    # Collect results
    # -----------------------------------------

    for q in expanded[:5]:

        batch = run_query(q)

        for p in batch:

            pid = p.get("id") or p.get("title")

            if pid not in seen:

                seen.add(pid)
                all_papers.append(p)

    if not all_papers:
        return []

    # -----------------------------------------
    # FILTER IRRELEVANT PAPERS
    # -----------------------------------------

    filtered = [p for p in all_papers if is_relevant(p)]

    # fallback if filter removed everything
    if len(filtered) < 5:
        filtered = all_papers

    # -----------------------------------------
    # RERANK
    # -----------------------------------------

    rerank_query = main_query or expanded[0]

    ranked = rerank(
        rerank_query,
        filtered,
        text_key="summary",
        top_k=top_k or cfg.retrieval.top_k_after_rerank,
        model_name="BAAI/bge-reranker-base"
    )

    return ranked