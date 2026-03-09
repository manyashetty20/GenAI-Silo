#!/usr/bin/env python3
import argparse
import sys
import os
import pandas as pd
from pathlib import Path

from deep_research_agent.graph import run_research
from deep_research_agent.llm import get_llm

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--query","-q",required=True)
    parser.add_argument("--end-date",default=None)
    parser.add_argument("--output","-o",default=None)
    parser.add_argument("--output-dir",default=None)
    parser.add_argument("--query-id",default="0")

    args = parser.parse_args()

    # initialize LLM
    llm = get_llm()

    report, papers, stats = run_research(
        query=args.query,
        llm=llm,
        end_date=args.end_date
    )

    if args.output:
        Path(args.output).write_text(report)

    if args.output_dir:

        query_dir = Path(args.output_dir) / str(args.query_id)
        query_dir.mkdir(parents=True, exist_ok=True)

        (query_dir/"intro.md").write_text(report)

        rows = []

        for i,p in enumerate(papers,1):

            rows.append({
                "id":i,
                "title":p.get("title"),
                "snippet":p.get("abstract") or p.get("summary","")
            })

        pd.DataFrame(rows).to_csv(query_dir/"paper.csv",index=False)

    if not args.output and not args.output_dir:
        print(report)

if __name__ == "__main__":
    main()