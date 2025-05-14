"""
src/query_llm.py  –  Gemini edition
-----------------------------------
Labels every (rule, string) pair in the StringTruth benchmark by querying a
Google Gemini model via the google‑generativeai SDK.

Env‑var required:
    GOOGE_API_KEY   (typo kept for user request)
    or GOOGLE_API_KEY (official spelling)

Outputs:
    cache/llm_labels.json   {id: true/false}
"""

from __future__ import annotations

import os, re, json
from pathlib import Path
from typing import Dict, List, Any
import time # Added for rate limiting

import google.generativeai as genai
from tenacity import retry, wait_exponential, stop_after_attempt
from tqdm import tqdm

# ---------------------------------------------------------------------#
# 0.  Configure Gemini client
# ---------------------------------------------------------------------#

MODEL_CONFIGS = {
    "flash-lite": {
        "model_id": "gemini-2.0-flash-lite", 
        "rpm": 30, 
        "daily_limit": 1500 # For informational purposes
    },
    "flash": {
        "model_id": "gemini-2.0-flash", 
        "rpm": 15, 
        "daily_limit": 1500 # For informational purposes
    },
    "pro-preview": {
        "model_id": "models/gemini-1.5-pro-latest", # Actual ID for Gemini 1.5 Pro (was gemini-2.5-pro-preview-05-06)
        "rpm": 2, # Gemini 1.5 Pro has a 2 RPM limit for free tier
        "daily_limit": 500 # For informational purposes (example, check official docs)
    },
    "pro-1.5": { # Adding the one you manually set, assuming it's gemini-1.5-pro
        "model_id": "gemini-1.5-pro", # Or "models/gemini-1.5-pro-latest"
        "rpm": 2, # Free tier limit for 1.5 Pro
        "daily_limit": 1500 # Example, check official docs for current limits
    },
    "2.5-flash-preview": { # Adding the one you manually set, assuming it's gemini-1.5-pro
        "model_id": "gemini-2.5-flash-preview-04-17", # Or "models/gemini-1.5-pro-latest"
        "rpm": 10, # Free tier limit for 1.5 Pro
        "daily_limit": 500 # Example, check official docs for current limits
    }
}

# ---> SELECT YOUR MODEL HERE <----
SELECTED_MODEL_KEY = "2.5-flash-preview" # Options: "flash-lite", "flash", "pro-preview", "pro-1.5"

if SELECTED_MODEL_KEY not in MODEL_CONFIGS:
    raise ValueError(f"Invalid SELECTED_MODEL_KEY: '{SELECTED_MODEL_KEY}'. Choose from {list(MODEL_CONFIGS.keys())}")

CURRENT_MODEL_CONFIG = MODEL_CONFIGS[SELECTED_MODEL_KEY]
MODEL_NAME = CURRENT_MODEL_CONFIG["model_id"]
RATE_LIMIT_PER_MINUTE = CURRENT_MODEL_CONFIG["rpm"]

print(f"Using model: {MODEL_NAME} with RPM: {RATE_LIMIT_PER_MINUTE}")

api_key = os.getenv("GOOGE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError(
        "Set GOOGE_API_KEY (or GOOGLE_API_KEY) in your environment first."
    )

genai.configure(api_key=api_key)

# pick Flash for speed or Pro for accuracy
model = genai.GenerativeModel(MODEL_NAME)

# Rate limiting variables
REQUEST_TIMESTAMPS: List[float] = []
ONE_MINUTE_IN_SECONDS = 60

# Optional: Set to an integer to process only the first N lines of input files
# Set to None to process all lines.
MAX_LINES_TO_PROCESS = 3 # e.g., 5 to process only the first 5 lines


# ---------------------------------------------------------------------#
# 1.  Prompt pieces
# ---------------------------------------------------------------------#
SYSTEM_PROMPT = (
    "You are an expert logician working with strings of letters.\n"
    "Definitions:\n"
    "  • Vowel = A,E,I,O,U (any case)\n"
    "  • Consonant = any other alphabetic letter that is not Y or W\n"
    "  • Glide = Y or W (any case)\n"
    "Respond with exactly one word: 'True' or 'False'."
)

FEW_SHOTS = [
    (
        "user",
        "Consider the string 'IMMEDIATELY'. "
        "True or False: some vowels that are capitalized precede letter two",
    ),
    ("model", "True"),
    (
        "user",
        "Consider the string 'DeBATE'. "
        "True or False: every glide precedes a vowel",
    ),
    ("model", "False"),
]


def _gemini_messages(rule: str, string: str):
    """Build message list in Gemini format."""
    msgs = [{"role": "user", "parts": [SYSTEM_PROMPT]}]
    # add few‑shot pairs
    for role, text in FEW_SHOTS:
        msgs.append({"role": role, "parts": [text]})
    # actual query
    query = f"Consider the string '{string}'. True or False: {rule.lower()}?"
    msgs.append({"role": "user", "parts": [query]})
    return msgs


# ---------------------------------------------------------------------#
# 2.  Single query helper
# ---------------------------------------------------------------------#
@retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(5))
def ask_gemini(rule: str, string: str) -> bool:
    """
    Returns True / False according to Gemini's first word in its reply.
    Retries on transient errors (rate limit, 5xx).
    Implements a 30 RPM rate limiter.
    """
    global REQUEST_TIMESTAMPS

    current_time = time.monotonic()

    # Remove timestamps older than one minute
    REQUEST_TIMESTAMPS = [
        ts for ts in REQUEST_TIMESTAMPS if current_time - ts < ONE_MINUTE_IN_SECONDS
    ]

    if len(REQUEST_TIMESTAMPS) >= RATE_LIMIT_PER_MINUTE:
        # Calculate how long to sleep
        # Sleep until the oldest request in the window is more than a minute old
        time_to_wait = (REQUEST_TIMESTAMPS[0] + ONE_MINUTE_IN_SECONDS) - current_time
        if time_to_wait > 0:
            time.sleep(time_to_wait)
    
    # Update current_time after potential sleep
    current_time = time.monotonic()
    REQUEST_TIMESTAMPS.append(current_time)
    # Ensure the list doesn't grow indefinitely beyond what's needed for the window check
    # (though the filter above should handle it, this is an extra safeguard)
    if len(REQUEST_TIMESTAMPS) > RATE_LIMIT_PER_MINUTE:
        REQUEST_TIMESTAMPS = REQUEST_TIMESTAMPS[-RATE_LIMIT_PER_MINUTE:]


    msgs = _gemini_messages(rule, string)
    response = model.generate_content(msgs, safety_settings={"harassment": "block_none"})
    # take first non‑empty token
    first = response.text.strip().split()[0].lower()
    return first.startswith("t")


# ---------------------------------------------------------------------#
# 3.  Driver – executed with "python -m src.query_llm"
# ---------------------------------------------------------------------#
def main() -> None:
    # ------ 3a. Load rules from pooled_text.txt ----------
    id2rule: Dict[str, str] = {}
    rules_file_path = "data/pooled_text.txt"
    try:
        with open(rules_file_path) as f_txt:
            for i, rule_line in enumerate(f_txt):
                if MAX_LINES_TO_PROCESS is not None and i >= MAX_LINES_TO_PROCESS:
                    print(f"INFO: Processed only the first {MAX_LINES_TO_PROCESS} lines from {rules_file_path} due to MAX_LINES_TO_PROCESS setting.")
                    break
                _id, *rule_tokens = rule_line.strip().split()
                id2rule[_id] = " ".join(rule_tokens)
    except FileNotFoundError:
        print(f"ERROR: Rules file not found: {rules_file_path}")
        return
    except Exception as e:
        print(f"ERROR: Could not read rules file {rules_file_path}: {e}")
        return

    print(f"Loaded {len(id2rule):,} rules from {os.path.basename(rules_file_path)}.")
    if not id2rule:
        print("No rules loaded, exiting.")
        return

    # ------ 3b. Calculate total items to query for tqdm from pooled_truth.txt ------
    total_items_to_query = 0
    truth_file_path = "data/pooled_truth.txt"
    truth_lines_to_process = [] # Store lines to avoid reading file twice if MAX_LINES_TO_PROCESS is low

    try:
        with open(truth_file_path) as f_truth_temp:
            for i, line in enumerate(f_truth_temp):
                if MAX_LINES_TO_PROCESS is not None and i >= MAX_LINES_TO_PROCESS:
                    # This print is for the pre-scan, the main loop will also print
                    # print(f"INFO: Pre-scanning only the first {MAX_LINES_TO_PROCESS} lines from {truth_file_path} for counting.")
                    break
                truth_lines_to_process.append(line)
                parts = line.strip().split()
                if not parts:
                    continue
                _id_temp = parts[0]
                if _id_temp in id2rule: # Only count if rule exists for this ID
                    # Each pair is a (string, label), so (len(parts) - 1) gives total elements for pairs
                    # Divide by 2 for number of pairs
                    total_items_to_query += (len(parts) - 1) // 2
    except FileNotFoundError:
        print(f"ERROR: Truth file not found: {truth_file_path}")
        return
    except Exception as e:
        print(f"ERROR: Could not read truth file {truth_file_path} for counting: {e}")
        return

    if total_items_to_query == 0:
        print(f"No (test_string, author_label) pairs to process from {truth_file_path} (or no matching IDs in rules file).")
        # Still create an empty json if no items but rules were loaded.
        if id2rule:
            Path("cache").mkdir(exist_ok=True)
            json.dump({}, open("cache/llm_labels.json", "w"), indent=2)
            print("✅  wrote empty cache/llm_labels.json")
        return

    # ------ 3c. Query the model for each (rule, test_string) pair ----------
    # New structure for labels: {id: [{"test_string": str, "llm_label": bool, "author_label": str}, ...]}
    all_llm_labels: Dict[str, List[Dict[str, Any]]] = {}
    
    print(f"Starting LLM queries for approximately {total_items_to_query} items...")

    with tqdm(total=total_items_to_query, desc="Gemini-labelling pairs") as pbar:
        for i, truth_line in enumerate(truth_lines_to_process): # Use the stored lines
            # MAX_LINES_TO_PROCESS is already handled for truth_lines_to_process
            
            parts = truth_line.strip().split()
            if not parts:
                continue
            _id = parts[0]

            if _id not in id2rule:
                # This case might happen if MAX_LINES_TO_PROCESS for rules was less than for truth_file
                # or if an ID in truth file simply doesn't have a corresponding rule.
                # We also need to adjust pbar if we skip items that were counted.
                skipped_pairs_for_id = (len(parts) -1) // 2
                if skipped_pairs_for_id > 0:
                    pbar.update(skipped_pairs_for_id) # Account for skipped items in progress bar
                continue

            rule = id2rule[_id]
            
            if _id not in all_llm_labels:
                all_llm_labels[_id] = []

            # Iterate over (test_string, author_label) pairs
            # parts[0] is id, so pairs start from index 1
            for j in range(1, len(parts), 2):
                if j + 1 < len(parts): # Ensure there's a string and its label
                    test_string = parts[j]
                    author_label_char = parts[j+1].lower() # 't', 'f', or 'u'

                    try:
                        llm_result = ask_gemini(rule, test_string)
                        all_llm_labels[_id].append({
                            "test_string": test_string,
                            "llm_label": llm_result,
                            "author_label": author_label_char
                        })
                    except Exception as e:
                        print(f"ERROR: Failed to get LLM label for id {_id}, string '{test_string}': {e}")
                        # Optionally add a placeholder or skip
                        all_llm_labels[_id].append({
                            "test_string": test_string,
                            "llm_label": "ERROR", # Placeholder for error
                            "author_label": author_label_char
                        })
                    pbar.update(1)
                else:
                    # This means there's a dangling test_string without a label, or parts ended.
                    # If it was counted, we need to adjust pbar.
                    # However, the count (len(parts) - 1) // 2 already handles this.
                    print(f"WARNING: Skipping malformed pair for id {_id} at end of line: '{parts[j:]}'")


    # ------ 3d. Save ----------
    Path("cache").mkdir(exist_ok=True)
    # Sanitize model name for filename
    safe_model_name = MODEL_NAME.replace("/", "_") # Replace slashes if model ID has them e.g. models/gemini-1.5-pro-latest
    output_file = f"cache/llm_labels_{safe_model_name}.json"
    try:
        with open(output_file, "w") as f:
            json.dump(all_llm_labels, f, indent=2)
        print(f"✅  Wrote {len(all_llm_labels)} IDs with their LLM labels to {output_file}")
        processed_item_count = sum(len(v) for v in all_llm_labels.values())
        print(f"   Total (test_string, llm_label, author_label) entries: {processed_item_count}")

    except Exception as e:
        print(f"ERROR: Could not write output file {output_file}: {e}")


if __name__ == "__main__":
    main()
