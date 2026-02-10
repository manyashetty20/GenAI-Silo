"""arXiv search client for scientific literature."""
from datetime import datetime
from typing import Optional
import os
import time  # Add this import
import arxiv
import sys

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
    max_results: int = 20, # Reduced default from 30 to 20 to lower API load
    end_date: Optional[str] = None,
) -> list[dict]:
    """
    Search arXiv with rate-limiting to prevent HTTP 429.
    """
    _disable_ssl_verification()
    
    # Add a small delay before every search to respect rate limits
    time.sleep(3) 
    
    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=3, # Built-in rate limiting for the arxiv library
        num_retries=3
    )
    
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    
    papers = []
    cutoff = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

    try:
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
    except arxiv.HTTPError as e:
        if "429" in str(e):
            print("ArXiv Rate Limit hit. Waiting 10 seconds...", file=sys.stderr)
            time.sleep(10)
            return [] # Return empty list so the agent can try next query
        raise e
        
    return papers
