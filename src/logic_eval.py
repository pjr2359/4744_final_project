# src/logic_eval.py
from collections import Counter
def classify_letter(ch):
    VOWELS = "aeiouAEIOU"
    GLIDES = "yYwW"
    return dict(
        is_vowel = ch in VOWELS,
        is_consonant = ch.isalpha() and ch not in VOWELS+GLIDES,
        is_glide = ch in GLIDES,
        is_capital = ch.isupper(),
    )

def build_model(string):
    letters = list(string)
    features = {i+1: classify_letter(ch) for i, ch in enumerate(letters)}
    # helpers for “unique”, “letter two”, “penultimate”, etc.
    freq = Counter(letters)
    return dict(letters=letters, feats=features, freq=freq)

def eval_formula(model, formula):
    """
    Evaluate a *hand‑rolled* subset of the logical language:
    ‘SOME’, ‘EVERY’, ‘NO’, ‘AT LEAST ONE’, ‘THE letter that …’
    …

    Returns True / False / None (undefined).
    """
    # *** stub – fill with your own combinator‑based evaluator ***
    raise NotImplementedError


import json, pandas as pd, tqdm, src.logic_eval as L

cache = json.load(open("cache/parses.json"))
rules = {line.split()[0]: " ".join(line.split()[1:]) 
         for line in open("data/pooled_text.txt")}

rows=[]
for idx, truth_line in zip(rules, open("data/pooled_truth.txt")):
    # truth_line has many variants; we keep only the FIRST string for now
    gold_string = truth_line.split()[0]
    gold_label  = gold_string[-1].lower() == "t"   # quick hack
    string      = re.sub(r"[^A-Za-z]","", gold_string)  # strip weird chars
    model       = L.build_model(string)
    logic_pred  = L.eval_formula(model, rules[idx])
    rows.append((idx, gold_label, logic_pred))

df = pd.DataFrame(rows, columns=["id","author","logic"])
df.to_csv("cache/gold_vs_author.csv", index=False)
