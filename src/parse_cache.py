# src/parse_cache.py
import json, re, itertools, pathlib, nltk
from tqdm import tqdm

GRAMMAR = nltk.load_parser("data/ps6_grammar.fcfg", trace=0)

def best_parsing_transcript(id_prefix):
    """Return the first n‑best hypothesis that parses; else None."""
    pattern = re.compile(fr"^{id_prefix}-\d+\s+(.*)$")
    with open("data/1000.txt") as nb:
        for line in nb:
            m = pattern.match(line.rstrip())
            if not m:
                continue
            sent   = m.group(1)
            tokens = [tok.lower() for tok in sent.split()]   # ⬅︎ lower‑case!
            try:
                # Grammar.check_coverage() may raise *ValueError*.
                next(GRAMMAR.parse(tokens))
                return sent          # ✔ found a parsable transcript
            except (StopIteration, ValueError):
                continue
    return None     

def build_cache(ids):
    cache = {}
    for idx in tqdm(ids):
        sent = best_parsing_transcript(idx)
        cache[idx] = sent
    pathlib.Path("cache").mkdir(exist_ok=True)
    json.dump(cache, open("cache/parses.json", "w"), indent=2)

if __name__ == "__main__":
    # toy smoke‑test on first 50 IDs
    ids = [f"aiden-a-{i}" for i in range(1,51)]
    build_cache(ids)
