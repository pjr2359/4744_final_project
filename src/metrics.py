import pandas as pd, json
df = pd.read_csv("cache/gold_vs_author.csv")
llm = json.load(open("cache/llm_labels.json"))

df["llm"] = df["id"].map(llm)
acc_overall = (df["llm"] == df["author"]).mean()
acc_logic   = (df["logic"] == df["author"]).mean()

conf = (df["llm"] == df["logic"]).mean()

print(f"LLM vs Author accuracy   : {acc_overall:.3%}")
print(f"Logic vs Author accuracy : {acc_logic:.3%}")
print(f"LLM agrees with Logic    : {conf:.3%}")
