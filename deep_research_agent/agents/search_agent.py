"""Search Agent: performs recursive, adaptive literature search using plan's queries."""
from typing import Optional

from deep_research_agent.config import get_config
from deep_research_agent.retrieval import search_arxiv, search_semantic_scholar, rerank


def search_agent(
    search_queries: list[str],
    main_query: Optional[str] = None,
    end_date: Optional[str] = None,
    top_k: Optional[int] = None,
) -> list[dict]:
    """
    Run search for each query on arXiv and Semantic Scholar, merge, dedupe by id, rerank by main_query, return top_k.
    """
    cfg = get_config()
    r_cfg = cfg.retrieval
    end_date = end_date or r_cfg.end_date
    top_k = top_k or r_cfg.top_k_after_rerank

    seen_ids: set[str] = set()
    all_papers: list[dict] = []

    for q in search_queries[:8]:
        arxiv_papers = search_arxiv(q, max_results=r_cfg.max_arxiv_results, end_date=end_date)
        ss_papers = search_semantic_scholar(q, max_results=r_cfg.max_semantic_scholar_results)
        for p in arxiv_papers + ss_papers:
            pid = p.get("id") or p.get("arxiv_id") or p.get("title", "")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                all_papers.append(p)

    if not all_papers:
        return []

    rerank_query = main_query or search_queries[0]
    return rerank(
        rerank_query,
        all_papers,
        text_key="summary",
        top_k=top_k,
        model_name=cfg.embedding.model_name,
        device=cfg.embedding.device,
    )
