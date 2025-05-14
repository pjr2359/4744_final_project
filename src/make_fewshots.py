"""
make_fewshots.py
================
Scan pooled_text.txt + pooled_truth.txt and write a JSON list of
(rule, string, author_truth) triples you can paste into FEWSHOT_POOLS.

Usage
-----
    python -m src.make_fewshots --per-bucket 6 --seed 42

Outputs
-------
    fewshot_pool.json   # e.g. 24 high‑quality examples
"""

from __future__ import annotations
import argparse, json, re, random, pathlib
from collections import defaultdict, Counter

# ---------- quantifier buckets (same regexes as in error_analysis) ----------
QUANT_PATTERNS = {
    "some":      re.compile(r"\bsome\b",  re.I),
    "every":     re.compile(r"\bevery\b|\ball\b",  re.I),
    "no":        re.compile(r"\bno\b|\bnone\b",    re.I),
    "exactly":   re.compile(r"\bexactly\s+one\b|\bthe\b.+\bis\b", re.I),
    "other":     re.compile(r".*"),
}

def bucket(rule: str) -> str:
    for name, pat in QUANT_PATTERNS.items():
        if pat.search(rule):
            return name
    return "other"

# ----------------------------------------------------------------------------
def main(per_bucket: int, seed: int):
    random.seed(seed)

    # --- read rules ---------------------------------------------------------
    id2rule={}
    with open("data/pooled_text.txt") as fp:
        for line in fp:
            if not line.strip(): continue
            _id,*tok=line.strip().split()
            id2rule[_id]=" ".join(tok)

    # --- collect (rule,string,label) candidates ----------------------------
    buckets=defaultdict(list)
    with open("data/pooled_truth.txt") as fp:
        for line in fp:
            parts=line.strip().split()
            if not parts: continue
            _id = parts[0]
            rule=id2rule.get(_id)
            if not rule: continue
            b=bucket(rule)
            for s,l in zip(parts[1::2], parts[2::2]):
                if l.lower() in ("t","f"):
                    buckets[b].append((rule, s, l.lower()=="t"))

    # --- sample ------------------------------------------------------------
    chosen=[]
    for b,items in buckets.items():
        k=min(per_bucket, len(items))
        chosen.extend(random.sample(items, k))

    random.shuffle(chosen)
    out_path=pathlib.Path("fewshot_pool.json")
    json.dump(chosen, out_path.open("w"), indent=2)
    print(f"✨  wrote {len(chosen)} triples to {out_path}")
    print("Example:", chosen[0])

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--per-bucket", type=int, default=6,
                    help="number of examples per quantifier bucket")
    ap.add_argument("--seed", type=int, default=42)
    main(**vars(ap.parse_args()))
