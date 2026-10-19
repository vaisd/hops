"""
demo.py - run one question live, printing each step as it happens.

Wraps the model call and the search tool to print progress, without
changing agent.py, so the benchmark code stays exactly as it ran.

Usage:
    python demo.py            # runs corp-4
    python demo.py muni-2     # runs any question by id
"""

import os
import sys
import json
import time
from dotenv import load_dotenv
from openai import OpenAI

import agent

load_dotenv()

QUESTION_ID = sys.argv[1] if len(sys.argv) > 1 else "corp-4"
MODEL = "subconscious/deepseek-v4.1-flash-marathon"

with open("questions.jsonl") as f:
    rows = [json.loads(line) for line in f if line.strip()]
row = next(r for r in rows if r["id"] == QUESTION_ID)

client = OpenAI(
    api_key=os.environ["SUBCONSCIOUS_API_KEY"],
    base_url="https://api.subconscious.dev/v1",
)

# Print each model call as it returns: step number, prompt size, time.
_create = client.chat.completions.create
step = {"n": 0}

def create(*args, **kwargs):
    start = time.perf_counter()
    response = _create(*args, **kwargs)
    step["n"] += 1
    print(
        f"\nstep {step['n']}: {response.usage.prompt_tokens:>6,} prompt tokens"
        f"   ({time.perf_counter() - start:.2f}s)"
    )
    return response

client.chat.completions.create = create

# Print each search query as the model requests it.
_search = agent.TOOL_FUNCTIONS["web_search"]

def search(query):
    print(f"    search: {query}")
    return _search(query)

agent.TOOL_FUNCTIONS["web_search"] = search

print(f"Q: {row['question']}")

trace = agent.run_agent(row["question"], client=client, model=MODEL)

print("\n" + "-" * 60)
print(f"answer: {trace['answer'] or trace['error']}")
print(f"gold:   {row['gold']}")
print(
    f"{trace['n_steps']} steps, {trace['total_tokens']:,} total tokens, "
    f"{trace['model_seconds']:.2f}s model time"
)