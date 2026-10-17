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

## Design rule: hops come from what the model doesn't know

The number of steps a question takes is set by what the model can't recall, not
by how many hops the question asks for. A chain whose first link is famous
collapses: the model recalls it and skips toward the end (infra-2 finished in 2
steps; the Anderson Bridge chain in infra-3 resolved its first two links from
memory). A chain whose first link can't be recalled forces the sequence:
"Wood End" (muni-4), a 2026 index change (corp-4), a 2026 rename (corp-1).

The same effect shows up in milder form. In 5 of the first 15 probes (uni-1,
uni-3, muni-3, infra-1, infra-2) the model's first query already named the
whole candidate set. Those aren't broken questions, but a "find the set, then
compare" question whose set is recalled is a 3-step question in a 4-step
costume: only the attribute lookup is left. The `hops` field records the
design, and probe step counts record what actually happened.

Unrecallable isn't enough on its own; consecutive links must also live on
different pages. infra-5 starts from a September 2026 metro opening the model
can't know, but one search result stated both the new line and its terminus
station, so two links collapsed into one search and it finished in 3 steps.
The chains that held (uni-5, muni-4, corp-4) each require a comparison across
several pages between links, which no single page summarizes.

Practical check before writing a question: can the model name the first link
without searching? If yes, the chain needs a less recallable starting point.

## Step cap

`MAX_STEPS = 12` in `agent.py`, identical for both arms. A step is one model
round-trip. Raised from 8 after muni-4 hit 8 without answering; at 12 it
finished correctly in 10, so it measures the question rather than the cap.
Its query log shows 6 of those 10 steps went to resolving the first link
(the old name "Wood End"), so expect high variance across repeats on it.

**Step counts are not deterministic at temperature 0.** Re-probing with an
empty cache moved several questions by one step (uni-5 7→6, muni-4 10→9,
muni-2 3→4), because Tavily returned slightly different results and the
model's next query depended on them. That is a second source of
nondeterminism alongside the model itself; repeats capture both.

Each step record includes `tool_args`, the raw arguments of every tool call,
so dependent and independent queries can be told apart in the traces.

## Search cache

`search_cache.json` maps each query to its Tavily result. Policy for the full run:

1. Start from whatever cache exists.
2. Run arm one; it fills the cache with any query it issues.
3. Run arm two against that warm cache.
4. Commit the resulting cache alongside the results as a frozen artifact. The
   whole benchmark is then replayable without a Tavily key.

Within the run, both arms see byte-identical results for any query they share,
so differences come from the runtime, not from Tavily's index shifting.

**Arm two sees a warmer cache than arm one.** Its `tool_seconds`, and so its
`total_seconds`, are lower for reasons that have nothing to do with the model.
That is why results report `model_seconds`, steps and tokens, never
`total_seconds`. This matters most on muni-4, where 6 of 10 probe steps went
to one hard lookup: if arm two's searches for that link are cached, the arms
differ in tool latency there, but not in steps or tokens.

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
| infra-2 | Probed at 2 steps: the model named all three Boston Harbor tunnels and their lengths in its first query. |

## Rejected candidates

Questions considered and thrown out before they got an ID, grouped by the rule
they broke. These are the evidence that the remaining set was filtered, not
just collected.

**Margin too small** (top two close enough that the gold is arguable)

| family | candidate | top two |
|---|---|---|
| universities | ACC charter members still in the league, most recently founded | Clemson 1889 vs NC State 1887 |
| municipalities | Lexington's neighbors, largest land area | Lincoln 14.4 vs Bedford 13.7 sq mi |
| municipalities | Lexington's neighbors, smallest land area | Belmont 4.7 vs Arlington 5.0 sq mi |
| municipalities | Wilmington's neighbors, earliest incorporation | Woburn 1642 vs Reading 1644 (also unreliable, below) |
| companies | Sept 22 2025 S&P additions, most recent IPO | Robinhood Jul 2021 vs AppLovin Apr 2021 |
| companies | July 2025 S&P additions, most recently founded | Datadog 2010 vs The Trade Desk / Block 2009 |
| infrastructure | Boston–Cambridge Charles River bridges, longest | Harvard Bridge 659.9 vs Longfellow 650.7 m |
| infrastructure | Red Line terminals including Mattapan, earliest | Ashmont 1928 vs Mattapan 1929 (infra-1 is worded to exclude Mattapan) |
| infrastructure | bridge that previously held Gordie Howe's "longest cable-stayed span in North America" title | Ravenel 471 vs Port Mann 470 m, and Mexico's Baluarte (520 m) makes the set ambiguous |
| papers | ICML 2026 honorable mentions, most / fewest authors | ties: 8 vs 8, 3 vs 3 |

**Attribute unreliable across sources** (gap may be wide, but depends on which date you accept)

| family | candidate | problem |
|---|---|---|
| universities | private ACC members, most recently founded | Miami 1925, but Duke can be dated 1838 or 1924 (renaming) |
| municipalities | Wilmington's neighbors, earliest incorporation | early colonial incorporation dates disagree between sources |
| companies | Mar 23 2026 S&P additions, earliest founded | Coherent's 1971 is II-VI's; EchoStar and Vertiv dates are spin-offs |
| companies | Mar 23 2026 S&P removals, earliest founded | Lamb Weston 1950 as a brand, 2016 as a spin-off |
| companies | Sept 22 2025 S&P additions, founding year | Emcor 1994 emerged from JWP's bankruptcy |
| companies | Mar 24 2025 S&P additions, founding year | TKO and Expand Energy are recent merger entities |
| companies | several 2025-26 batches, IPO year | SPAC listings (Vertiv), spin-off listings, Flex's 1994 re-IPO |
| infrastructure | 2026 airport openings | inauguration and first commercial flight are months apart (Halwara: Feb 1 vs May 15) |
| infrastructure | anything keyed to Nanjing Line 6's exact day | sources give Sep 29 and Sep 30; Nanjing South station Jun 26 and Jun 28, 2011 |
| papers | first-author affiliation | joint appointments, preprint vs camera-ready changes, name variants |
| papers | CVPR 2026 award papers, author count | SAM 3D lists 23 on arXiv as a team paper; proceedings count may differ |
| papers | ICML 2026 position papers, arXiv date | the API returned v1 dates that contradict the ID's month |

**Set membership ambiguous** (whether the winner belongs in the set is arguable)

| family | candidate | problem |
|---|---|---|
| municipalities | Ipswich's neighbors, most water area | Gloucester borders only across Essex Bay; without it, Rowley 2.1 vs Essex 1.9 |
| municipalities | Plymouth's neighbors, most water area | Duxbury touches only at Saquish Neck; without it, Bourne 12.2 vs Wareham 10.5 |
| municipalities | Hingham's neighbors, most water area | Hull borders across Hingham Bay; Hull 24.1 vs Cohasset 21.6 |
| companies | Dec 22 2025 S&P additions, HQ outside the US | CRH lists dual headquarters, Dublin and New York |
| papers | ICLR 2026 outstanding papers (any attribute) | secondary sources disagree on whether there were two or three |

**Hop can be skipped** (the question leaks a later hop, or one page answers it)

| family | candidate | problem |
|---|---|---|
| municipalities | chain starting from "Taunton North Purchase" (Easton) | the old name gives away Taunton, a later hop |
| municipalities | muni-4 ending on "most recently incorporated neighbor of Wilmington" | North Reading is also Reading's answer, so skipping a hop still scores |
| municipalities | Plymouth's neighbor in a different county | Plymouth's own article states the Bourne county line |
| sports (whole family) | Super Bowl chains | coverage summarizes whole seasons into single articles |
| infrastructure | Toronto Line 5 vs Line 6, longer | the model knows the planned lengths; only the opening dates are post-cutoff |
| infrastructure | chain through the Detroit–Windsor crossings | Ambassador Bridge and the tunnel are recalled; only the first link is new |

**Unreachable by the benchmark's search tool.** Recent facts can be true and
still fail here: before committing a question whose gold depends on a 2025-26
fact, `web_search` was called directly to confirm Tavily surfaces it. paper-4's
last link passed only with natural phrasings, noted in its row.

## Probing

    python probe.py <family>

Reads that family from `questions.jsonl` and writes `probe_results_<family>.json`.
Keep questions that take 3+ steps with a correct answer.
