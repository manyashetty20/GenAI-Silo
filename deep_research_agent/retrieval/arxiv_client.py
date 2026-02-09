"""arXiv search client for scientific literature."""
from datetime import datetime
from typing import Optional
import os

import arxiv


def _disable_ssl_verification():
    """Disable SSL verification if env var is set (for testing/development only)."""
    verify_env = os.getenv("ARXIV_VERIFY_SSL", "1").lower()
    if verify_env in ("0", "false"):
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        # Monkey-patch requests Session to not verify SSL
        import requests
        requests.Session.verify = False


def search_arxiv(
    query: str,
    max_results: int = 30,
    end_date: Optional[str] = None,
) -> list[dict]:
    """
    Search arXiv and return list of paper dicts with id, title, summary, url, date, authors.
    end_date: YYYY-MM-DD to only include papers before this date (for benchmark reproducibility).
    """
    # Disable SSL verification if configured via ARXIV_VERIFY_SSL=0
    _disable_ssl_verification()
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    papers = []
    cutoff = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

    for result in client.results(search):
        if cutoff and result.published and result.published.replace(tzinfo=None) >= cutoff:
            continue
        papers.append({
            "id": result.entry_id.split("/")[-1],
            "arxiv_id": result.entry_id.split("/")[-1],
            "title": result.title,
            "summary": result.summary,
            "url": result.entry_id,
            "date": result.published.isoformat() if result.published else None,
            "authors": [a.name for a in result.authors],
            "source": "arxiv",
        })
    return papers
