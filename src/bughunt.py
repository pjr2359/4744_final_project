import pandas as pd
df = pd.read_csv("cache/full_metrics.csv")   # merge of all columns

candidates = df[(df.logic == df.llm) & (df.logic != df.author)].copy()
candidates.to_csv("bug_candidates.csv", index=False)
print(f"Flagged {len(candidates)} suspicious items.")


def is_presupposition(rule):
    return bool(re.search(r"\bTHE\b|\bEXACTLY ONE\b", rule))

df["presupposition"] = df["id"].map(lambda i: is_presupposition(rules[i]))


amb = df[df.presupposition]
plain= df[~df.presupposition]
print("LLM accuracy ambiguous :", (amb.llm==amb.author).mean())
print("LLM accuracy plain      :", (plain.llm==plain.author).mean())
