"""
probe.py - test candidate questions before committing to the full set.

A question is only useful for this benchmark if it forces several rounds of
search. Run candidates through here first and keep the ones that produce
3 or more steps with a correct answer.

Usage:
    python probe.py
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from agent import run_agent

load_dotenv()

MODEL = "subconscious/deepseek-v4.1-flash-marathon"
BASE_URL = "https://api.subconscious.dev/v1"

# Candidates to test. Gold answers MUST be verified by you against a real
# source before a question goes into the final set.
CANDIDATES = [
    {
        "id": "agg-1",
        "question": "Among the four teams that played in the 2026 NFL conference championship games, which team's home stadium has the smallest seating capacity?",
        "gold": "New England Patriots"
    },
    {
        "id": "sb-2",
        "question": (
            "The offensive coordinator of the team that won Super Bowl LX was "
            "hired as head coach by which NFL team after that season?"
        ),
        "gold": "Las Vegas Raiders",
    },
    # Add your own below. Aim for 3+ hops and a one-string answer.
]


def normalize(s):
    """Loose match so trivial formatting differences don't count as wrong."""
    if s is None:
        return ""
    return " ".join(s.lower().replace(".", "").replace(",", "").split())


def judge(answer, gold):
    a, g = normalize(answer), normalize(gold)
    if not a:
        return "EMPTY"
    if a == g:
        return "PASS"
    if g in a or a in g:
        return "REVIEW"
    return "FAIL"


def main():
    client = OpenAI(
        api_key=os.environ["SUBCONSCIOUS_API_KEY"],
        base_url=BASE_URL,
    )

    print(f"{'id':8} {'steps':>5} {'tools':>5} {'tok':>7} "
          f"{'model_s':>8} {'tool_s':>7}  verdict  answer")
    print("-" * 95)

    traces = []
    for c in CANDIDATES:
        t = run_agent(c["question"], client=client, model=MODEL)
        t["id"] = c["id"]
        t["gold"] = c["gold"]
        t["verdict"] = judge(t["answer"], c["gold"])
        traces.append(t)

        print(
            f"{c['id']:8} {t['n_steps']:>5} {t['n_tool_calls']:>5} "
            f"{t['total_tokens']:>7} {t['model_seconds']:>8.2f} "
            f"{t['tool_seconds']:>7.2f}  {t['verdict']:7}  "
            f"{(t['answer'] or t['error'] or '')[:40]}"
        )

    # Save full traces so you can inspect what the model actually searched for.
    with open("probe_results.json", "w") as f:
        json.dump(traces, f, indent=2)

    # Summary: is this question set hard enough to be worth running?
    step_counts = [t["n_steps"] for t in traces]
    avg_steps = sum(step_counts) / len(step_counts) if step_counts else 0
    n_easy = sum(1 for s in step_counts if s <= 2)

    print("-" * 95)
    print(f"average steps: {avg_steps:.1f}")
    print(f"{n_easy}/{len(traces)} questions resolved in 2 steps or fewer")
    if avg_steps < 3:
        print(
            "\nThis set is too easy. Questions that finish in 2 steps cannot "
            "show a difference\nin context accumulation, because there is "
            "barely any context to accumulate.\nRewrite so the first hop's "
            "answer is required to form the second search."
        )


if __name__ == "__main__":
    main()