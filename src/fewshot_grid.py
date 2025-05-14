"""
fewshot_grid.py – clean‑tier‑1, non‑blocking, thread‑pooled runner
==================================================================
Creates K stochastic passes for each few‑shot size N and writes

    cache/llm_labels_<model>_n<N>_run<R>.json
"""
from __future__ import annotations
import os, time, random, argparse, itertools, json, queue, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any, Literal

import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from google.api_core.exceptions import InternalServerError, ResourceExhausted
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# ───────────────────── shared request bucket ──────────────────────
from src.query_llm import MODEL_CONFIGS, _claim_token, REQUEST_TIMESTAMPS

# ────────────────────── demo pool (unchanged) ─────────────────────
FEWSHOT_POOLS: List[Tuple[str, str, bool]] = [
    ("every leter that follows a consonant is vocalic", "sokka", False),
    ("every vowel is unique and final", "hel", False),
    ("no consonant is adjacent to a glide", "dawn", False),
    ("no consonant is unique", "aaaa", True),
    ("some glide that letter four follows is unvoiced and capitalized or is final", "www", False),
    ("the letter that immediately precedes letter four is letter three", "clock", True),
    ("at least one glide is repeated", "yyaaww", True),
    ("every vowel that letter one precedes precedes a unvoiced centered even glide", "bree", False),
    ("every consonant that letter two precedes is voiced or unvoiced and no k is consonantal and capital", "i", True),
    ("a glide is unique", "wwwhattt", False),
    ("some fricatives precede letter two or follow letter four and are adjacent to letter three", "woei", False),
    ("no capital letter follows a fricative or more than one vowel and at least one glide", "fayyf", True),
    ("only letter two is unique and is voiced", "fbss", True),
    ("there is exactly one letter that precedes a d", "dau", False),
    ("the letter that precedes letter five is a vowel", "attat", True),
    ("no vowel is voiced and is final", "piti", False),
    ("the letter that immediately precedes letter four is letter three", "papaya", True),
    ("every u that letter three is adjacent to is capitalized and final", "ottur", False),
    ("letter one is unique", "h", True),
    ("all vowels are not voiced and there is no u that follows some glides", "pay", False),
    ("the penultimate letter is not a vowel", "ieou", False),
    ("exactly one consonant is penultimate", "should", True),
    ("every vocalic even letter that precedes some vowel is a i and immediately follows a glide", "aaaa", False),
    ("no consonant is adjacent to a glide", "some", True),
    ("all mirrored letters are not capitalized", "mam", True),
    ("only letter two is unique", "tst", True),
    ("only vowels are capitalized", "a", True),
    ("there are some glides that are centered", "fawn", False),
    ("no consonant is a t", "incredible", True),
    ("some vowels precede a unique consonant or more than one vocalic even letter and are capitalized", "odd", False),
]

N_VALUES, K_RUNS, TEMP = [0, 8, 24], 3, 0.3

# ───────────────────── helper: build few‑shot block ───────────────
def build_fewshot(n: int) -> List[Tuple[str, str]]:
    shots = []
    for rule, s, ans in random.sample(FEWSHOT_POOLS, n):
        shots.append(("user",  f"Consider the string '{s}'. True or False: {rule}"))
        shots.append(("model", "True" if ans else "False"))
    return shots

# ───────────────────── helper: gold dataset ───────────────────────
def load_pairs() -> List[Tuple[str, str, str, bool]]:
    id2rule: Dict[str, str] = {}
    with open("data/pooled_text.txt") as f:
        for ln in f:
            _id, *tok = ln.strip().split()
            id2rule[_id] = " ".join(tok)

    pairs: List[Tuple[str, str, str, bool]] = []
    with open("data/pooled_truth.txt") as f:
        for ln in f:
            p = ln.strip().split()
            if not p: continue
            rule = id2rule.get(p[0])
            if not rule: continue
            for s, l in zip(p[1::2], p[2::2]):
                if l.lower() in ("t", "f"):
                    pairs.append((p[0], rule, s, l.lower() == "t"))
    return pairs

# ───────────────────── prompt + retryable call ────────────────────
def _messages(rule: str, s: str, shots):
    m = [{"role":"user","parts":[
        "You are an expert logician working with strings of letters. "
        "Respond 'True' or 'False' only."
    ]}]
    for r, txt in shots:
        m.append({"role": r, "parts": [txt]})
    m.append({"role": "user", "parts": [
        f"Consider the string '{s}'. True or False: {rule.lower()}?"
    ]})
    return m

@retry(
    retry=retry_if_exception_type((InternalServerError, ResourceExhausted)),
    wait=wait_exponential(min=2, max=20),
    stop=stop_after_attempt(6),
    reraise=True,
)
def call_gemini(model, rpm: int, rule: str, s: str, shots) -> bool|Literal["SAFETY"]:
    _claim_token(rpm)                     # only request bucket – tokens << 1 M/min
    
    # Add extra delay for flash-preview to avoid ResourceExhausted errors
    if "flash-preview" in model.model_name:
        time.sleep(random.uniform(0.5, 1.0))  # Add significant delay
    
    rsp = model.generate_content(_messages(rule, s, shots),
                                 generation_config=GEN_CFG,
                                 safety_settings={"harassment": "block_none"})
    fr = rsp.candidates[0].finish_reason
    if fr == "SAFETY":
        return "SAFETY"
    tok = rsp.text.strip().split()
    return bool(tok) and tok[0].lower().startswith("t")

# ───────────────────── worker wrapper (no blocking) ───────────────
def safe_call(model, rpm, rule, s, shots):
    try:
        return call_gemini(model, rpm, rule, s, shots)
    except Exception as e:
        # Special handling for ResourceExhausted with flash-preview
        if isinstance(e, ResourceExhausted) and "flash-preview" in model.model_name:
            print(f"ResourceExhausted for flash-preview, sleeping 5s...")
            time.sleep(5.0)  # Long sleep on resource exhausted
            try:
                # One more try with longer wait
                time.sleep(random.uniform(1.0, 2.0))
                return call_gemini(model, rpm, rule, s, shots)
            except Exception:
                pass
        return f"ERROR:{type(e).__name__}"

# ───────────────────── run one (model, N, run) cell ───────────────
def run_grid(model_key: str, n: int, run: int, test_mode: bool = False):
    cfg, rpm = MODEL_CONFIGS[model_key], MODEL_CONFIGS[model_key]["rpm"]
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGE_API_KEY"))
    model = genai.GenerativeModel(cfg["model_id"])

    shots = build_fewshot(n)
    data  = load_pairs()
    if test_mode:
        data = data[:5]  # Just 5 samples for quick testing
        print(f"TEST MODE: Using only {len(data)} samples")
    total = len(data)

    # Much lower thread count for flash-preview to avoid resource exhaustion
    if model_key.endswith("flash-preview"):
        pool = 4  # Very limited concurrency for preview model
    else:
        pool = min(int(rpm/60*0.8), 24)
    name = f"{model_key}-n{n}-r{run}"
    out: Dict[str, List[Dict[str, Any]]] = {}

    with ThreadPoolExecutor(max_workers=pool) as ex, tqdm(total=total, desc=name) as bar:
        fut2meta = {ex.submit(safe_call, model, rpm, rule, s, shots): (_id, s, gold)
                    for _id, rule, s, gold in data}

        for fut in as_completed(fut2meta):
            _id, s, gold = fut2meta[fut]
            pred         = fut.result()
            out.setdefault(_id, []).append(
                {"test_string": s, "llm_label": pred, "author_label": gold})
            bar.update(1)

    Path("cache").mkdir(exist_ok=True)
    fn = f"cache/llm_labels_{model_key}_n{n}_run{run}.json"
    json.dump(out, open(fn, "w"), indent=2)
    print("✔ saved", fn)

# ───────────────────── constants reused in call_gemini ────────────
GEN_CFG = GenerationConfig(temperature=TEMP, max_output_tokens=4)

# ───────────────────── CLI driver ─────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODEL_CONFIGS, required=True)
    ap.add_argument("--single-n", type=int, help="only run one N")
    ap.add_argument("--test", action="store_true", help="Run a small test with 5 samples")
    args = ap.parse_args()

    Ns   = [args.single_n] if args.single_n is not None else N_VALUES
    runs = 1 if args.model.endswith("preview") else K_RUNS

    for N, R in itertools.product(Ns, range(runs)):
        run_grid(args.model, N, R, test_mode=args.test)
        REQUEST_TIMESTAMPS.clear()   # clean slate for next cell
