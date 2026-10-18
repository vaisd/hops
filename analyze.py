"""
analyze.py - headline numbers and the per-step latency-vs-context analysis.

Usage:
    python analyze.py results/run_<timestamp>.jsonl

Prints the headline table, run-validity checks, run-to-run consistency, and
the slope analysis, and writes slope_chart.png next to the results file.

The slope question: if marathon's pruning works, its per-step time should grow
less as the conversation gets longer. Faster hardware lowers the whole line AND
its slope by roughly the same factor (prefill cost per token is a hardware
cost too), so raw slopes can't separate the two. The hardware-invariant
quantity is the relative slope: extra seconds per 1k prompt tokens, divided by
that arm's own per-step time at short context. Pruning would shrink it;
uniform speedup would not.
"""

import sys
import json
import statistics as st
from collections import defaultdict

import numpy as np

HARD = ["uni-5", "muni-4", "corp-4", "paper-4"]
ARMS = ["subconscious", "baseline"]


def p95(xs):
    return float(np.percentile(xs, 95))


def fit(steps):
    """OLS: seconds ~ a + b * prompt_k + c * completion_k. Returns (a, b, c)."""
    X = np.array([[1.0, s["prompt_tokens"] / 1000, s["completion_tokens"] / 1000] for s in steps])
    y = np.array([s["seconds"] for s in steps])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def relative_slope(steps, ref_prompt_k=1.0):
    """Extra seconds per 1k prompt tokens as a fraction of the arm's own per-step
    time at a short reference context, holding completion length at its median."""
    a, b, c = fit(steps)
    comp_k = st.median(s["completion_tokens"] for s in steps) / 1000
    base = a + b * ref_prompt_k + c * comp_k
    return b / base, (a, b, c, base)


def bootstrap(runs, stat, n=2000, seed=0):
    """Resample whole runs (steps within a run are not independent)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(runs), len(runs))
        vals.append(stat([s for i in pick for s in runs[i]["steps"]]))
    return np.percentile(vals, [2.5, 97.5])


def main(path):
    R = [json.loads(l) for l in open(path)]
    by = {a: [r for r in R if r["arm"] == a] for a in ARMS}

    # --- validity -----------------------------------------------------------
    print("configs:", sorted({r["config"] for r in R}))
    print("baseline served by:", sorted({p for r in by["baseline"] for p in r["served_by"]}))
    print("errors:", [(r["id"], r["arm"], r["repeat"], r["error"]) for r in R if r["error"]] or "none")

    # --- headline table -----------------------------------------------------
    def row(name, f):
        print(f"{name:28}" + "".join(f"{f(by[a]):>14}" for a in ARMS))

    print(f"\n{'':28}" + "".join(f"{a:>14}" for a in ARMS))
    row("Pass", lambda rs: f"{sum(r['verdict'] == 'PASS' for r in rs)}/{len(rs)}")
    row("Median steps", lambda rs: f"{st.median(r['n_steps'] for r in rs):g}")
    row("Median tokens", lambda rs: f"{st.median(r['total_tokens'] for r in rs):,.0f}")
    row("Median model seconds", lambda rs: f"{st.median(r['model_seconds'] for r in rs):.2f}")
    row("P95 model seconds", lambda rs: f"{p95([r['model_seconds'] for r in rs]):.2f}")
    row("Seconds per step (median)", lambda rs: f"{st.median(r['model_seconds'] / r['n_steps'] for r in rs):.2f}")
    row("Hard four, median tokens", lambda rs: f"{st.median(r['total_tokens'] for r in rs if r['id'] in HARD):,.0f}")

    # --- non-passes ---------------------------------------------------------
    print("\nnon-passes:")
    for r in R:
        if r["verdict"] != "PASS":
            print(f"  {r['id']:8} rep{r['repeat']} {r['arm']:12} {r['verdict']:6} steps={r['n_steps']:2} "
                  f"answer={(r['answer'] or r['error'] or '')[:60]!r}")

    # --- pairwise speed -----------------------------------------------------
    pair = defaultdict(dict)
    for r in R:
        pair[(r["id"], r["repeat"])][r["arm"]] = r["model_seconds"]
    ratios = [v["baseline"] / v["subconscious"] for v in pair.values()]
    print(f"\npairs where subconscious faster: {sum(x > 1 for x in ratios)}/{len(ratios)}  "
          f"(baseline/subconscious model_seconds: median {st.median(ratios):.1f}x, "
          f"min {min(ratios):.1f}x, max {max(ratios):.1f}x)")

    # --- run-to-run consistency ----------------------------------------------
    print("\ntoken spread across repeats (max-min), median over questions:")
    for a in ARMS:
        spread = defaultdict(list)
        for r in by[a]:
            spread[r["id"]].append(r["total_tokens"])
        d = [max(v) - min(v) for v in spread.values()]
        same = sum(x == 0 for x in d)
        print(f"  {a:12} median spread {st.median(d):>6,.0f} tokens; identical on all 3 repeats: {same}/20")

    # --- slope analysis -----------------------------------------------------
    print("\nper-step fit: seconds ~ a + b*prompt_k + c*completion_k (95% CI by run bootstrap)")
    fits = {}
    for a in ARMS:
        steps = [s for r in by[a] for s in r["steps"]]
        rel, (ia, b, c, base) = relative_slope(steps)
        b_ci = bootstrap(by[a], lambda ss: fit(ss)[1])
        rel_ci = bootstrap(by[a], lambda ss: relative_slope(ss)[0])
        fits[a] = (steps, ia, b, c)
        print(f"  {a:12} n={len(steps):3}  intercept {ia:5.2f}s  "
              f"slope {b:.3f}s/1k prompt [{b_ci[0]:.3f}, {b_ci[1]:.3f}]  "
              f"completion {c:.2f}s/1k out  "
              f"| per-step at 1k ctx {base:.2f}s, relative slope {rel*100:.1f}%/1k "
              f"[{rel_ci[0]*100:.1f}, {rel_ci[1]*100:.1f}]")
    print("  prompt-token range per step:",
          {a: (min(s["prompt_tokens"] for s in fits[a][0]), max(s["prompt_tokens"] for s in fits[a][0])) for a in ARMS})

    chart(fits, path.rsplit("/", 1)[0] + "/slope_chart.png")


def chart(fits, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color = {"subconscious": "#2a78d6", "baseline": "#eb6834"}
    ink, muted, grid, surface = "#1a1a19", "#6b6a64", "#e6e5e0", "#fcfcfb"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=surface)

    for ax, relative in zip(axes, [False, True]):
        ax.set_facecolor(surface)
        ys_all, label_y = [], {}
        for a in ARMS:
            steps, ia, b, c = fits[a]
            comp_k = st.median(s["completion_tokens"] for s in steps) / 1000
            base = ia + b * 1.0 + c * comp_k
            norm = base if relative else 1.0
            x = np.array([s["prompt_tokens"] / 1000 for s in steps])
            y = np.array([s["seconds"] for s in steps]) / norm
            ax.scatter(x, y, s=22, color=color[a], alpha=0.45, edgecolors=surface, linewidths=1, zorder=2)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, (ia + b * xs + c * comp_k) / norm, color=color[a], lw=2, zorder=3, label=a)
            ys_all.extend(y)
            label_y[a] = (xs[-1], (ia + b * xs[-1] + c * comp_k) / norm)
        # Direct labels at line ends, nudged apart when the lines converge.
        (xa, ya), (xb, yb) = label_y["subconscious"], label_y["baseline"]
        nudge = 8 if abs(ya - yb) < 0.08 * max(ys_all) else 0
        ax.annotate("subconscious", (xa, ya), xytext=(6, -nudge), textcoords="offset points",
                    color=ink, fontsize=9, va="center")
        ax.annotate("baseline", (xb, yb), xytext=(6, nudge), textcoords="offset points",
                    color=ink, fontsize=9, va="center")
        # Clip the y-axis at the 99th percentile so one slow step doesn't flatten the rest.
        top = float(np.percentile(ys_all, 99)) * 1.15
        off = sum(v > top for v in ys_all)
        ax.set_ylim(0, top)
        if off:
            ax.text(0.99, 0.98, f"{off} step{'s' * (off > 1)} above axis", transform=ax.transAxes,
                    ha="right", va="top", color=muted, fontsize=8)
        ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=ink)
        ax.set_xlabel("prompt tokens at this step (thousands)", color=muted, fontsize=9)
        ax.set_ylabel("× own per-step time at 1k context" if relative else "seconds for this step",
                      color=muted, fontsize=9)
        ax.set_title("Relative: hardware-invariant" if relative else "Absolute",
                     color=ink, fontsize=11, loc="left")
        ax.grid(color=grid, lw=0.8, zorder=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors=muted, labelsize=8, length=0)

    fig.suptitle("Per-step model time vs. context length, one dot per step", color=ink, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=surface)
    print(f"\nchart -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
