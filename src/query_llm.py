# src/query_llm.py
import openai, os, time, json, random
from tenacity import retry, stop_after_attempt, wait_exponential

openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM = """You are an expert logician. 
A "string" is a sequence of letters. 
Definitions:
- Vowel = A,E,I,O,U (capital or lowercase)
- Consonant = any other alphabetic letter that is not a glide
- Glide = Y or W (any case)
Answer only "True" or "False". """

FEW_SHOTS = [
    dict(role="user",
         content="Consider the string 'IMMEDIATELY'. True or False: some vowels that are capitalized precede letter two"),
    dict(role="assistant", content="True"),
    dict(role="user",
         content="Consider the string 'DeBATE'. True or False: every glide precedes a vowel"),
    dict(role="assistant", content="False"),
]

@retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(5))
def ask_gpt(rule, string):
    q = f"Consider the string '{string}'. True or False: {rule.lower()}?"
    msgs = [{"role":"system","content":SYSTEM}] + FEW_SHOTS + [{"role":"user","content":q}]
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini", temperature=0.0, messages=msgs
    )
    ans = resp.choices[0].message.content.strip().split()[0].lower()
    return ans.startswith("t")

def label_dataset(id2rule, id2string):
    out={}
    for _id in tqdm.tqdm(id2rule):
        out[_id] = ask_gpt(id2rule[_id], id2string[_id])
    json.dump(out, open("cache/llm_labels.json","w"), indent=2)
