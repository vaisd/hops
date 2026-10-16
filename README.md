# hops

A question set for comparing two inference backends on multi-hop search, plus
the tool-calling loop (`agent.py`) that runs both arms identically.

## questions.jsonl

One row per question: `id`, `family`, `hops`, `question`, `gold`, `aliases`,
`source` (list of URLs, pinned to a page revision where possible), `notes`.

Rules each row is held to:

- The top two candidates are not close; `notes` records the margin.
- Attributes don't drift (founding years, land area, not enrollment or capacity).
- Every row cites the page its gold was checked against.
- Each family has one hard question built as a chain of sequential
  dependencies, where each search can't be written until the previous returns.

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
