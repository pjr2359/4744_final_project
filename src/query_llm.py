
"""
query_llm.py – simple bucketed, thread‑pooled Gemini labeller
-------------------------------------------------------------
Exports
    • MODEL_CONFIGS
    • _claim_token           ←  used by fewshot_grid
    • REQUEST_TIMESTAMPS     ←  ″
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
from threading import Lock

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from tqdm import tqdm

# ── model options ──────────────────────────────────────────────────
MODEL_CONFIGS = {
    "flash-lite":   {"model_id": "gemini-2.0-flash-lite",      "rpm": 4000},
    "flash":        {"model_id": "gemini-2.0-flash",           "rpm": 2000},
    "flash-preview":{"model_id": "gemini-2.5-flash-preview-04-17", "rpm": 1000},
    "pro":          {"model_id": "gemini-1.5-pro",             "rpm": 1000},
    "pro-preview":  {"model_id": "gemini-2.5-pro-preview",     "rpm": 150},
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODEL_CONFIGS, default="flash-lite")
    return p.parse_args()

# ── prompt builder (no few‑shots) ─────────────────────────────────
def build_prompt(rule: str, s: str):
    return [
        {"role": "user", "parts": [
            "You are an expert logician working with strings of letters. "
            "Respond with exactly one word: True or False."]},
        {"role": "user", "parts": [
            f"Consider the string '{s}'. True or False: {rule.lower()}?"]}
    ]

# ── request‑rate bucket (shared) ──────────────────────────────────
ONE_MIN = 60.0
REQUEST_TIMESTAMPS: List[float] = []
_BUCKET_LOCK = Lock()

def _claim_token(rpm: int):
    """Block until we are inside the per‑minute request quota."""
    while True:
        now = time.monotonic()
        with _BUCKET_LOCK:
            REQUEST_TIMESTAMPS[:] = [t for t in REQUEST_TIMESTAMPS if now - t < ONE_MIN]
            if len(REQUEST_TIMESTAMPS) < rpm:
                REQUEST_TIMESTAMPS.append(now)
                return
            sleep = (REQUEST_TIMESTAMPS[0] + ONE_MIN) - now
        time.sleep(max(0.0, sleep))

# ── retryable single call ─────────────────────────────────────────
@retry(
    retry=retry_if_exception_type(ResourceExhausted),
    wait=wait_exponential(min=2, max=30),
    stop=stop_after_attempt(6))
def ask(model, rpm, rule, s):
    _claim_token(rpm)
    rsp = model.generate_content(build_prompt(rule, s),
                                 safety_settings={"harassment": "block_none"})
    tok = rsp.text.strip().split()
    return bool(tok) and tok[0].lower().startswith("t")

# ── main labelling driver ─────────────────────────────────────────
def main(model_key: str):
    cfg, rpm = MODEL_CONFIGS[model_key], MODEL_CONFIGS[model_key]["rpm"]
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGE_API_KEY"))
    model = genai.GenerativeModel(cfg["model_id"])

    # load benchmark
    id2rule: Dict[str, str] = {}
    with open("data/pooled_text.txt") as f:
        for ln in f:
            _id, *tok = ln.strip().split()
            id2rule[_id] = " ".join(tok)

    jobs = []
    with open("data/pooled_truth.txt") as f:
        for ln in f:
            p = ln.strip().split()
            if not p or p[0] not in id2rule:
                continue
            for s, l in zip(p[1::2], p[2::2]):
                jobs.append((p[0], s, l.lower()))

    pool = min(int(rpm/60 * 8), 16)
    out: Dict[str, List[Dict[str, Any]]] = {}

    with ThreadPoolExecutor(pool) as ex, tqdm(total=len(jobs), desc=model_key) as bar:
        fut2meta = {ex.submit(ask, model, rpm, id2rule[_id], s): (_id, s, l)
                    for _id, s, l in jobs}
        for fut in as_completed(fut2meta):
            _id, s, l = fut2meta[fut]
            try:
                pred = fut.result()
            except Exception as e:
                pred = f"ERROR:{type(e).__name__}"
            out.setdefault(_id, []).append(
                {"test_string": s, "llm_label": pred, "author_label": l})
            bar.update(1)

    Path("cache").mkdir(exist_ok=True)
    fn = f"cache/llm_labels_{model_key}.json"
    json.dump(out, open(fn, "w"), indent=2)
    print("✔ saved", fn)

if __name__ == "__main__":
    a = parse_args()
    main(a.model)
