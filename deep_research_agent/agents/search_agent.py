"""Search Agent: performs recursive, adaptive literature search using plan's queries."""
from typing import Optional, Any
import time

from deep_research_agent.config import get_config
from deep_research_agent.retrieval import search_arxiv, search_semantic_scholar, rerank

# Simple global cache to prevent redundant API calls during the same run
_SEARCH_CACHE: dict[str, list[dict]] = {}

def search_agent(
    search_queries: list[str],
    main_query: Optional[str] = None,
    end_date: Optional[str] = None,
    top_k: Optional[int] = None,
) -> list[dict]:
    """
    Run search for each query on arXiv and Semantic Scholar with caching and rate-limit safety.
    Optimized for the DeepScholar Benchmark by focusing on verifiability and nugget coverage.
    """
    cfg = get_config()
    r_cfg = cfg.retrieval
    
    # Use end_date from config if not provided in the call
    end_date = end_date or r_cfg.end_date
    top_k = top_k or r_cfg.top_k_after_rerank

    seen_ids: set[str] = set()
    all_papers: list[dict] = []

    # Limit to a maximum of 4 queries to prevent hitting API rate limits during recursive loops
    max_queries = 4 
    
    for q in search_queries[:max_queries]:
        # 1. Check Cache first to avoid the 10-second arXiv 429 penalty
        if q in _SEARCH_CACHE:
            batch = _SEARCH_CACHE[q]
        else:
            # 2. Fetch fresh results if not cached
            # search_arxiv now handles the internal 3-second delay and 429 backoff
            arxiv_papers = search_arxiv(q, max_results=r_cfg.max_arxiv_results, end_date=end_date)
            ss_papers = search_semantic_scholar(q, max_results=r_cfg.max_semantic_scholar_results)
            
            batch = arxiv_papers + ss_papers
            _SEARCH_CACHE[q] = batch

        # 3. Deduplicate by paper ID or title
        for p in batch:
            pid = p.get("id") or p.get("arxiv_id") or p.get("title", "")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                all_papers.append(p)

    if not all_papers:
        return []

    # 4. Rerank results to ensure the Reader and Writer agents receive the highest-quality evidence
    # This directly improves the verifiability metric for DeepScholar
    rerank_query = main_query or search_queries[0]
    return rerank(
        rerank_query,
        all_papers,
        text_key="summary",
        top_k=top_k,
        model_name=cfg.embedding.model_name,
        device=cfg.embedding.device,
    )