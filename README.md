# hops

A question set for comparing two inference backends on multi-hop search, plus
the tool-calling loop (`agent.py`) that runs both arms identically.

## questions.jsonl

One row per question: `id`, `family`, `hops`, `question`, `gold`, `aliases`,
`source` (list of URLs, pinned to a page revision where possible), `notes`.

Rules each row is held to:

- The top two candidates are not close; `notes` records the margin.
- Attributes don't drift (founding years, land area, not enrollment or capacity).
- Attributes are reliable across sources. This is separate from margin: early
  colonial incorporation dates and corporate founding dates (predecessors,
  mergers, renamings) disagree between sources, so even a wide gap on such an
  attribute rests on which source was picked. Woburn 1642 vs Reading 1644 was
  rejected on both counts.
- Every row cites the page its gold was checked against.
- Each family has one hard question built as a chain of sequential
  dependencies, where each search can't be written until the previous returns.

## Step cap

`MAX_STEPS = 12` in `agent.py`, identical for both arms. A step is one model
round-trip. Raised from 8 after muni-4 hit 8 without answering; at 12 it
finished correctly in 10, so it measures the question rather than the cap.
Its query log shows 6 of those 10 steps went to resolving the first link
(the old name "Wood End"), so expect high variance across repeats on it.

Each step record includes `tool_args`, the raw arguments of every tool call,
so dependent and independent queries can be told apart in the traces.

## Aliases are frozen

Alias lists are written when a row is created, before it is probed. They are
not edited after arm two runs. If a REVIEW verdict later justifies a new alias,
it is added, **both arms are rescored**, and the change is logged below.

Alias changes after freeze: none.

## Retired IDs

IDs are never reused once results exist.

| id | reason |
|---|---|
| sb-1 | Sports family dropped: sports coverage collapses chains into single articles. Probed at 2 steps. |
| sb-2 | Sports family dropped; also compared stadium capacities (drift) with a ~4% margin. |
| uni-4 | Probed at 3 steps / 12 parallel tool calls: wide, not hard. Replaced by uni-5. |

## Probing

    python probe.py <family>

Reads that family from `questions.jsonl` and writes `probe_results_<family>.json`.
Keep questions that take 3+ steps with a correct answer.
