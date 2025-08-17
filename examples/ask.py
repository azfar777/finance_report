import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from retrieval import Retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", "--query", dest="query", required=True)
    ap.add_argument("--section", default=None)
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    r = Retriever()
    hits = r.retrieve(args.query, section=args.section, k_final=args.k)
    print(json.dumps(hits, indent=2))


if __name__ == "__main__":
    main()
