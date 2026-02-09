#!/usr/bin/env python3
"""
CLI entry point for the agentic deep research system.
Run from repo root (GenAI): python -m deep_research_agent.run --query "Your research question"
Or with end date for reproducibility: python -m deep_research_agent.run --query "..." --end-date 2025-01-01
"""
import argparse
import sys
from pathlib import Path

# Ensure package root is on path when run as script (e.g. python run.py from deep_research_agent)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from deep_research_agent.graph import run_research


def main() -> None:
    p = argparse.ArgumentParser(description="Agentic Deep Research (DeepScholar-style)")
    p.add_argument("--query", "-q", required=True, help="Research query (e.g. related-work question)")
    p.add_argument("--end-date", default=None, help="YYYY-MM-DD: only papers before this date")
    p.add_argument("--output", "-o", default=None, help="Write report to this file (default: stdout)")
    p.add_argument("--output-dir", default=None, help="DeepScholar eval: write report to dir/query_id.txt")
    p.add_argument("--query-id", default="default", help="Query id for --output-dir (e.g. for eval)")
    args = p.parse_args()

    report, papers, stats = run_research(query=args.query, end_date=args.end_date)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Wrote report to {args.output}", file=sys.stderr)
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{args.query_id}.md"
        out_file.write_text(report, encoding="utf-8")
        print(f"Wrote {out_file}", file=sys.stderr)

    if not args.output and not args.output_dir:
        print(report)
    else:
        print(f"Papers: {stats.get('num_papers', 0)}, Verified: {stats.get('verified', False)}", file=sys.stderr)


if __name__ == "__main__":
    main()
