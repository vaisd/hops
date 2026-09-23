"""
agent.py - a flat tool-calling loop, instrumented for benchmarking.

The same loop runs both arms of the benchmark. The only difference between
arms is base_url and model, which is what makes the comparison fair.

Timing is split three ways:
  model_seconds    - time spent waiting on the inference API (differs by arm)
  tool_seconds     - time spent running tools (held constant across arms)
  overhead_seconds - everything else (your own code; should be near zero)
"""

import os
import json
import time
import requests
import hashlib
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# ---------------------------------------------------------------------------
# The tool the model is allowed to call.
#
# Two parts are needed: a JSON description the model reads (TOOLS), and an
# actual Python function that does the work (web_search). The model never
# runs anything itself. It only ever says "please call web_search with these
# arguments" and hands control back to you.
# ---------------------------------------------------------------------------


CACHE_PATH = "search_cache.json"
try:
    with open(CACHE_PATH) as f:
        _CACHE = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    _CACHE = {}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information. Use this whenever you "
                "need a fact you are not certain about, especially recent ones."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def web_search(query: str) -> str:
    """Run a real web search and return a short text summary of the top hits.

    Both arms call this identical function, so search quality is held constant
    and any difference you measure comes from the model or runtime, not search.
    """
    key = hashlib.sha256(query.strip().lower().encode()).hexdigest()
    if key in _CACHE:
        return _CACHE[key]
    
    resp = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": os.environ["TAVILY_API_KEY"],
            "query": query,
            "max_results": 3,
            "search_depth": "basic",
        },
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])

    if not results:
        _CACHE[key] = "No results found."
        with open(CACHE_PATH, "w") as f:
            json.dump(_CACHE, f)
        return "No results found."

    lines = []
    for r in results:
        # Truncate hard. Long search results balloon your input tokens on
        # every subsequent round-trip, which is exactly the cost effect you
        # are trying to measure cleanly.
        content = r.get("content", "")[:500]
        lines.append(f"{r.get('title', '')}\n{content}")

    out = "\n\n".join(lines)
    _CACHE[key] = out
    with open(CACHE_PATH, "w") as f:
        json.dump(_CACHE, f)
    return out



# Maps tool names to the functions that implement them, so the loop can
# dispatch by name without a pile of if-statements.
TOOL_FUNCTIONS = {"web_search": web_search}


SYSTEM_PROMPT = (
    "You are a research assistant. Answer the user's question using the "
    "web_search tool as many times as needed. You may reason in your reply, "
    "but the LAST line of your final message must be exactly:\n"
    "ANSWER: <the answer>\n"
    "where <the answer> is a single name, number, or short phrase with no "
    "extra words."
)


# ---------------------------------------------------------------------------
# The loop.
# ---------------------------------------------------------------------------

def run_agent(question, client, model, max_steps=8, timeout=120):
    """Run one question to completion and return a full trace.

    A 'step' is one round-trip to the model. Each step, the model either
    asks for tools or gives a final answer. Every step resends the whole
    conversation so far, which is why input tokens grow each time.
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    # Per-step records, plus running totals.
    steps = []
    prompt_tokens = 0
    completion_tokens = 0
    tool_calls_made = 0
    tool_seconds = 0.0
    answer = None
    error = None

    run_start = time.perf_counter()

    for step_index in range(max_steps):
        step_start = time.perf_counter()

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                temperature=0,
                timeout=timeout,
            )
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            break

        step_seconds = time.perf_counter() - step_start

        usage = response.usage
        prompt_tokens += usage.prompt_tokens
        completion_tokens += usage.completion_tokens

        message = response.choices[0].message

        steps.append(
            {
                "step": step_index,
                "seconds": round(step_seconds, 3),
                "tool_seconds": 0.0,  # filled in below if tools run
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "finish_reason": response.choices[0].finish_reason,
                "n_tool_calls": len(message.tool_calls or []),
            }
        )

        # No tool calls means the model is done and this is the answer.
        if not message.tool_calls:
            raw = (message.content or "").strip()
            answer = raw
            for line in reversed(raw.splitlines()):
                if line.strip().upper().startswith("ANSWER:"):
                    answer = line.split(":", 1)[1].strip()
                    break
            break

        # The model asked for tools. Two appends are required, in this order:
        #
        #   1. The assistant's own message, INCLUDING its tool_calls. The model
        #      needs to see what it asked for.
        #   2. One message per result, each tagged with the matching
        #      tool_call_id so the model knows which answer goes with which
        #      request.
        #
        # Skipping step 1, or omitting tool_call_id, is the most common bug
        # in hand-written agent loops.

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
        )

        step_tool_seconds = 0.0

        for tool_call in message.tool_calls:
            name = tool_call.function.name

            # Time the tool call, including the failure path. A tool that
            # times out is still a latency event and belongs in the data.
            tool_start = time.perf_counter()
            try:
                args = json.loads(tool_call.function.arguments)
                result = TOOL_FUNCTIONS[name](**args)
            except Exception as e:
                # Feed the error back to the model rather than crashing.
                # A real agent has to survive a flaky tool.
                result = f"Tool error: {type(e).__name__}: {e}"
            elapsed = time.perf_counter() - tool_start

            step_tool_seconds += elapsed
            tool_seconds += elapsed
            tool_calls_made += 1

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

        # Attach this step's tool time to the record created above.
        steps[-1]["tool_seconds"] = round(step_tool_seconds, 3)

    else:
        # Loop finished without break, meaning max_steps was exhausted.
        error = f"hit max_steps ({max_steps}) without a final answer"

    total_seconds = time.perf_counter() - run_start
    model_seconds = sum(s["seconds"] for s in steps)

    return {
        "question": question,
        "answer": answer,
        "error": error,
        "total_seconds": round(total_seconds, 3),
        "model_seconds": round(model_seconds, 3),
        "tool_seconds": round(tool_seconds, 3),
        "overhead_seconds": round(total_seconds - model_seconds - tool_seconds, 3),
        "n_steps": len(steps),
        "n_tool_calls": tool_calls_made,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Smoke test: one question, one arm.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    client = OpenAI(
        api_key=os.environ["SUBCONSCIOUS_API_KEY"],
        base_url="https://api.subconscious.dev/v1",
    )

    question = (
        "Which team won the 2026 Super Bowl, and who was that team's "
        "head coach during the game?"
    )

    trace = run_agent(
        question,
        client=client,
        model="subconscious/deepseek-v4.1-flash-marathon",
    )

    print(json.dumps(trace, indent=2))