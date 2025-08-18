from __future__ import annotations
import os, json
from typing import List, Dict

from finance.kpis import compute_kpis
from retrieval import Retriever
from models import get_generator
from prompts import DEFAULT_QUERIES, build_prompt

def collect_excerpts(r: Retriever, queries=DEFAULT_QUERIES, k_final=6) -> List[Dict]:
    out = []
    for q, section in queries:
        hits = r.retrieve(q, section=section, k_final=min(3, k_final))
        out.extend(hits)
    # de-duplicate by (file,page,text[:60])
    seen, uniq = set(), []
    for h in out:
        key = (h["meta"].get("file"), h["meta"].get("page"), h["text"][:60])
        if key not in seen:
            seen.add(key); uniq.append(h)
    return uniq[:k_final]

def generate_decision_brief(ticker: str, years: int = 5, k_final: int = 6) -> Dict:
    kpis = compute_kpis(ticker, period_years=years)
    try:
        r = Retriever()
        ctx = collect_excerpts(r, k_final=k_final)
    except FileNotFoundError:
        ctx = []

    prompt = build_prompt(ticker.upper(), kpis, ctx)
    pipe = get_generator()
    # keep it deterministic & concise
    out = pipe(prompt, max_new_tokens=450, do_sample=False)[0]["generated_text"]
    # take only the model's answer portion
    answer = out.split("Answer:", 1)[-1].strip()

    citemap = [{"id": f"R{i+1}", "file": c["meta"]["file"], "page": c["meta"]["page"], "section": c["meta"]["section"]} for i, c in enumerate(ctx)]
    return {
      "ticker": kpis.get("ticker", ticker.upper()),
      "as_of": kpis.get("as_of"),
      "kpis": kpis,
      "narrative": answer,
      "citations": citemap
    }

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()
    print(json.dumps(generate_decision_brief(args.ticker, args.years, args.k), indent=2))
