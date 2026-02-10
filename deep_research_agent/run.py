#!/usr/bin/env python3
"""
CLI entry point for the agentic deep research system.
"""
import argparse
import sys
import os
import pandas as pd
from pathlib import Path

# Ensure package root is on path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from deep_research_agent.graph import run_research

def main() -> None:
    p = argparse.ArgumentParser(description="Agentic Deep Research (DeepScholar-style)")
    p.add_argument("--query", "-q", required=True, help="Research query")
    p.add_argument("--end-date", default=None, help="YYYY-MM-DD: only papers before this date")
    p.add_argument("--output", "-o", default=None, help="Write report to this file")
    p.add_argument("--output-dir", default=None, help="Base directory for DeepScholar eval")
    p.add_argument("--query-id", default="0", help="Query index/ID (e.g. 0, 1, 2)")
    args = p.parse_args()

    # Run the research pipeline
    report, papers, stats = run_research(query=args.query, end_date=args.end_date)

    # Standard output saving
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Wrote report to {args.output}", file=sys.stderr)

    # DEEPSCHOLAR BENCHMARK FORMATTING
    if args.output_dir:
        # Create directory: base/query_id/ (e.g., results/deepscholar_base/0/)
        query_dir = Path(args.output_dir) / str(args.query_id)
        query_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save synthesis as 'intro.md'
        intro_file = query_dir / "intro.md"
        intro_file.write_text(report, encoding="utf-8")
        
        # 2. Save metadata as 'paper.csv' with required headers
        csv_data = []
        for i, p in enumerate(papers, 1):
            csv_data.append({
                "id": i,
                "title": p.get("title", "Unknown Title"),
                "snippet": p.get("abstract") or p.get("snippet") or "Abstract not available"
            })
        
        pd.DataFrame(csv_data).to_csv(query_dir / "paper.csv", index=False)
        print(f"Benchmark files generated in: {query_dir}", file=sys.stderr)

    if not args.output and not args.output_dir:
        print(report)

if __name__ == "__main__":
    main()