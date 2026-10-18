"""
run.py - the full benchmark: every question, both arms, N repeats.

Arms are interleaved per question, and the order alternates (A-then-B, then
B-then-A), so time-of-day load hits both arms equally and neither arm always
gets the warm search cache. Each trace is appended to a JSONL file as soon as
it finishes; rerunning with the same --out resumes and skips completed runs.

Usage:
    python run.py                      # all questions, both arms, 3 repeats
    python run.py --dry-run 1          # first question only, one repeat
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI

from agent import run_agent, config_fingerprint, MAX_STEPS
from probe import judge

load_dotenv()


def make_arms():
    """Both arms run the identical loop; only endpoint, model, routing differ."""
    provider = os.environ.get("BASELINE_PROVIDER")
    model = os.environ.get("BASELINE_MODEL")
    if not (provider and model and os.environ.get("OPENROUTER_API_KEY")):
        sys.exit("Set OPENROUTER_API_KEY, BASELINE_MODEL and BASELINE_PROVIDER in .env "
                 "(e.g. BASELINE_PROVIDER=deepinfra/fp8).")
    return {
        "subconscious": dict(
            client=OpenAI(api_key=os.environ["SUBCONSCIOUS_API_KEY"],
                          base_url="https://api.subconscious.dev/v1"),
            model="subconscious/deepseek-v4.1-flash-marathon",
            extra_body=None,
            pinned=None,
        ),
        "baseline": dict(
            client=OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                          base_url="https://openrouter.ai/api/v1"),
            model=model,
            # Pin to one provider and forbid fallthrough, so P95 isn't scattered
            # across whichever upstreams OpenRouter happens to route to.
            extra_body={"provider": {"order": [provider], "only": [provider],
                                     "allow_fallbacks": False}},
            pinned=provider,
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dry-run", type=int, metavar="N",
                    help="run only the first N questions, one repeat")
    ap.add_argument("--out", default=None, help="JSONL path (reuse to resume)")
    args = ap.parse_args()

    with open("questions.jsonl") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    repeats = args.repeats
    if args.dry_run:
        questions, repeats = questions[:args.dry_run], 1

    arms = make_arms()
    names = list(arms)
    fp = config_fingerprint()
    out = args.out or f"results/run_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    done = set()
    if os.path.exists(out):
        with open(out) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["id"], r["arm"], r["repeat"]))

    print(f"config {fp}  max_steps={MAX_STEPS}  baseline pin={arms['baseline']['pinned']}"
          f"  model={arms['baseline']['model']}  -> {out}")
    total = len(questions) * repeats * len(names)
    n = len(done)

    for rep in range(1, repeats + 1):
        for qi, q in enumerate(questions):
            order = names if (rep + qi) % 2 == 0 else names[::-1]
            for pos, arm in enumerate(order):
                if (q["id"], arm, rep) in done:
                    continue
                a = arms[arm]
                started = datetime.now(timezone.utc).isoformat()
                t = run_agent(q["question"], client=a["client"], model=a["model"],
                              extra_body=a["extra_body"])
                served = {s["provider"] for s in t["steps"] if s.get("provider")}
                rec = {
                    "id": q["id"], "family": q["family"], "arm": arm, "repeat": rep,
                    "order_position": pos, "started_at": started,
                    "config": fp, "model": a["model"], "pinned_provider": a["pinned"],
                    "served_by": sorted(served),
                    "verdict": judge(t["answer"], q["gold"], q.get("aliases", [])),
                    **t,
                }
                with open(out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                n += 1
                print(f"[{n:3}/{total}] rep{rep} {q['id']:8} {arm:12} steps={t['n_steps']:2} "
                      f"tok={t['total_tokens']:6} model_s={t['model_seconds']:6.2f} "
                      f"{rec['verdict']:6} served={','.join(served) or '-'}")
                if t["error"]:
                    print(f"         error: {t['error']}")


if __name__ == "__main__":
    main()
