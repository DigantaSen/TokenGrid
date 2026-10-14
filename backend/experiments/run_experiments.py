"""
Step 2 of the TokenGrid evaluation: replay traces through the simulator.

    python -m experiments.run_experiments            # results/groq
    python -m experiments.run_experiments --mock     # results/mock (pipeline test only)

Experiments
  forecast  can an honest agent predict its own token cost? learned
            predictor vs. "every task costs the average" (cross-fitted)
  E1        policy comparison at the default swarm (TokenGrid vs. baseline
            schedulers, plus TokenGrid without the learned predictor)
  E2        liar robustness: share of dishonest agents 0 -> 50%
  E3        swarm scaling under a fixed rate limit: TokenGrid's core claim,
            cost per completed task stays flat as the swarm grows
  E4        the capacity fix. (a) selection: four round-budget rules compared
            on separate DEV seeds (100-119), which is where the rule was
            chosen; (b) confirmation: original vs. chosen rule on the
            evaluation seeds, paired, at 0 / 25 / 50% liars; (c) the same
            confirmation on tasks collected after the rule was chosen
  E5        ten forecasters (constant, per-benchmark average, length-only
            ridge, embedding ridge / LightGBM / LightGBM log-median, kNN,
            exact input + predicted output, ensemble, cost buckets) plus
            online per-agent correction: forecast quality, distance from the
            reputation cliff measured in play, and a near-cliff stress test.
            The primary forecaster used by E1-E4 is the one with the largest
            reputation change per honest call on the DEV seeds.
  E6        safe bids: honest agents bid the q-quantile of their likely cost
  E7        forecast noise: how much worse forecasts cost the system

Every configuration runs over the same seeds for every policy (same agents,
same liars, same task sequences), so policy differences are paired; 95%
CIs are t-intervals over seeds.

Writes results/<run>/report/{summary.md, metrics.json, e2_liars.png, e3_scaling.png}.
"""
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy import stats

from app.token_predictor import (crossfit_buckets, crossfit_constant, crossfit_group_mean, crossfit_knn,
                                 crossfit_predict, embed_texts, forecast_quality, handcrafted_features)
from experiments.collect_traces import load_traces, traces_path
from app import config
from app.tasks import DATASETS, load_tasks
from experiments.simulator import CAPACITY_RULES, POLICIES, SimConfig, TraceTask, simulate

BASE_N = 20            # the rate limit is sized against this swarm; E3 grows N past it
CAPACITY_RATIO = 0.25  # window token limit = 25% of what BASE_N agents could consume -> real contention
TG_CONST = "tokengrid (no learned predictor)"
TG_ORIGINAL = "tokengrid (original 50% rule)"
# Variant name -> (forecast pool, underlying policy, SimConfig overrides).
VARIANTS = {TG_CONST: ("constant", "tokengrid", {}), TG_ORIGINAL: ("learned", "tokengrid", {"capacity_rule": "original"})}
DEV_SEEDS = range(100, 120)  # used only to choose the capacity rule, never for reported results
E4_METRICS = [("utilization", "rate-limit used"), ("correct_per_window", "correct / window"),
              ("honest_correct_per_window", "honest correct / window"), ("violation_rate", "429 rate"),
              ("honest_token_share", "honest token share"), ("dropped_dishonest", "liars dropped"),
              ("dropped_honest", "honest dropped")]

E1_METRICS = [("correct_per_window", "correct / window", 1), ("honest_correct_per_window", "honest correct / window", 1),
              ("cost_per_correct_usd", "µ$ / correct", 1e6), ("utilization", "rate-limit used", 1),
              ("violation_rate", "429 rate", 1), ("wait_p95", "wait p95 (rounds)", 1),
              ("honest_token_share", "honest token share", 1), ("size_bias", "size bias", 1),
              ("dropped_honest", "honest dropped", 1), ("dropped_dishonest", "liars dropped", 1)]


def mean_ci(x) -> tuple[float, float, float]:
    x = np.asarray([v for v in x if not np.isnan(v)], dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(x.mean())
    if len(x) < 2 or x.std() == 0:
        return m, m, m
    h = float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
    return m, m - h, m + h


def fmt(m, lo, hi, scale=1.0) -> str:
    if np.isnan(m):
        return "—"
    m, lo, hi = m * scale, lo * scale, hi * scale
    digits = 3 if abs(m) < 10 else 1
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]" if hi != lo else f"{m:.{digits}f}"


# Forecasters compared in E5. Each turns the task context (see load_features)
# into one cross-fitted token forecast per task, which honest agents bid.
ENSEMBLE_PARTS = ("embedding (ridge)", "embedding (lgbm)", "kNN (10 nearest)")
FORECASTERS = {
    "constant": lambda c: crossfit_constant(c["tokens"]),
    "per-benchmark average": lambda c: crossfit_group_mean(c["datasets"], c["tokens"]),
    "length-only (ridge)": lambda c: crossfit_predict(c["text"], c["tokens"]),
    "embedding (ridge)": lambda c: crossfit_predict(c["full"], c["tokens"]),
    "embedding (lgbm)": lambda c: crossfit_predict(c["full"], c["tokens"], learner="lgbm"),
    "embedding (lgbm, log-median)": lambda c: crossfit_predict(c["full"], c["tokens"], learner="lgbm_median"),
    "kNN (10 nearest)": lambda c: crossfit_knn(c["emb"], c["tokens"]),
    # Input tokens can be counted before the call; only the output (answer + thinking) is unknown.
    "exact input + predicted output": lambda c: c["input"] + crossfit_predict(c["full"], c["output"], learner="lgbm"),
    # Geometric mean = averaging in log space, where the ledger's score lives.
    "ensemble (ridge + lgbm + kNN)": lambda c: np.exp(np.mean([np.log(forecast(p, c)) for p in ENSEMBLE_PARTS], axis=0)),
    "cost buckets (short/med/long)": lambda c: crossfit_buckets(c["full"], c["tokens"]),
}
# Forecasters with a learned error distribution, so E6 can turn them into percentile bids.
QUANTILE_CAPABLE = {"length-only (ridge)": ("text", "ridge"), "embedding (ridge)": ("full", "ridge"),
                    "embedding (lgbm)": ("full", "lgbm"), "embedding (lgbm, log-median)": ("full", "lgbm_median")}
BASELINES = ("constant", "per-benchmark average")
CANDIDATES = [k for k in FORECASTERS if k != "constant"]   # eligible to become the primary forecaster
DEFAULT_PRIMARY = "embedding (ridge)"   # used only when no report exists yet to say which one was selected
ONLINE = "{} + online per-agent correction"
QUANTILES = (0.5, 0.6, 0.7, 0.8, 0.9)
NOISE_SIGMAS = (0.0, 0.25, 0.5, 0.75, 1.0)
STRESS_SIGMA = 0.35   # extra forecast noise near the E7 cliff: how much margin does each forecaster leave?


def load_features(rows: list[dict], cache: Path) -> dict:
    """Everything a forecaster may use, plus the targets:
    text      length/number counts + benchmark one-hot
    emb       MiniLM sentence embedding (cached by task id)
    full      emb + text
    datasets  benchmark name per task
    input     prompt tokens (countable before the call)
    output    completion tokens (unknown until the call)
    tokens    input + output (the quantity agents bid)"""
    texts = [r["text"] for r in rows]
    ids = [r["task_id"] for r in rows]
    emb = None
    if cache.exists():
        z = np.load(cache)
        if z["task_ids"].tolist() == ids:
            emb = z["emb"]
    if emb is None:
        emb = embed_texts(texts)
        np.savez(cache, task_ids=np.array(ids), emb=emb)
    datasets = [r["dataset"] for r in rows]
    text = handcrafted_features(texts, datasets)
    return {"text": text, "emb": emb, "full": np.hstack([emb, text]), "datasets": datasets,
            "input": np.array([r["input_tokens"] for r in rows], dtype=float),
            "output": np.array([r["output_tokens"] for r in rows], dtype=float),
            "tokens": np.array([r["total_tokens"] for r in rows], dtype=float)}


def forecast(name: str, ctx: dict, quantile: float | None = None) -> np.ndarray:
    if quantile is None:
        return np.maximum(1.0, FORECASTERS[name](ctx))
    features, learner = QUANTILE_CAPABLE[name]
    return crossfit_predict(ctx[features], ctx["tokens"], learner=learner, quantile=quantile)


def make_pool(rows: list[dict], forecasts) -> list[TraceTask]:
    return [TraceTask(r["task_id"], int(r["total_tokens"]), float(r["cost_usd"]), bool(r["correct"]), float(f))
            for r, f in zip(rows, forecasts)]


def build_pools(rows: list[dict], cache: Path, primary: str = DEFAULT_PRIMARY) -> tuple[list[TraceTask], list[TraceTask], dict]:
    """(pool forecast by `primary`, pool forecast by the constant baseline, forecast quality)."""
    ctx = load_features(rows, cache)
    tokens = ctx["tokens"]
    learned, const = forecast(primary, ctx), forecast("constant", ctx)
    quality = {"learned": forecast_quality(learned, tokens), "constant": forecast_quality(const, tokens),
               "primary": primary,
               "tokens_mean": float(tokens.mean()), "tokens_cv": float(tokens.std() / tokens.mean()),
               "accuracy": float(np.mean([r["correct"] for r in rows]))}
    return make_pool(rows, learned), make_pool(rows, const), quality


def tokengrid_runs(pool, cfg: SimConfig, seeds) -> list[dict]:
    return [simulate(pool, "tokengrid", replace(cfg, seed=s)) for s in seeds]


def run_grid(pool, pool_const, cfg: SimConfig, policies: list[str], seeds) -> dict[str, list[dict]]:
    seeds = range(seeds) if isinstance(seeds, int) else seeds
    out = {}
    for p in policies:
        which, policy, overrides = VARIANTS.get(p, ("learned", p, {}))
        use = pool_const if which == "constant" else pool
        out[p] = [simulate(use, policy, replace(cfg, seed=s, **overrides)) for s in seeds]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=300)
    args = ap.parse_args()

    tpath = traces_path(args.mock)
    rows = [r for r in load_traces(tpath) if r["error"] is None]
    if not rows:
        raise SystemExit(f"no traces at {tpath} — run experiments.collect_traces first")
    rows = list({r["task_id"]: r for r in rows}.values())  # latest row per task
    run_dir = tpath.parent
    print(f"{len(rows)} traced tasks from {tpath}")

    ctx = load_features(rows, run_dir / "embeddings.npz")
    tokens = ctx["tokens"]
    base = SimConfig(n_agents=BASE_N, rounds=args.rounds,
                     tokens_per_window=int(CAPACITY_RATIO * BASE_N * SimConfig.window_rounds * float(tokens.mean())),
                     requests_per_window=BASE_N * SimConfig.window_rounds)

    print(f"E5 {len(FORECASTERS)} forecasters: quality, then selection of the primary forecaster on dev seeds...")
    fc = {name: forecast(name, ctx) for name in FORECASTERS}
    fc_quality = {name: forecast_quality(f, tokens) for name, f in fc.items()}
    fc_pools = {name: make_pool(rows, f) for name, f in fc.items()}
    e5_select = {name: tokengrid_runs(fc_pools[name], base, DEV_SEEDS) for name in CANDIDATES}
    # Selection rule (declared before the extended comparison was run): E5's first run showed
    # throughput is tied across learned forecasters, and E7 showed the margin from the
    # reputation cliff is what matters - so pick the largest average reputation change per
    # honest call on the dev seeds, with honest correct work as the tie-breaker.
    primary = max(CANDIDATES, key=lambda n: (round(np.mean([r["honest_rep_change_per_call"] for r in e5_select[n]]), 3),
                                             np.mean([r["honest_correct_per_window"] for r in e5_select[n]])))
    print(f"  primary forecaster: {primary}")
    pool, pool_const, quality = build_pools(rows, run_dir / "embeddings.npz", primary)

    print("E1 policy comparison...")
    e1_policies = POLICIES + [TG_CONST, TG_ORIGINAL]
    e1 = run_grid(pool, pool_const, base, e1_policies, args.seeds)

    print("E2 liar robustness...")
    liar_levels = [0.0, 0.1, 0.25, 0.5]
    e2_policies = ["fifo", "wsjf", "tokengrid"]
    e2 = {f: run_grid(pool, pool_const, replace(base, dishonest_frac=f), e2_policies, args.seeds) for f in liar_levels}

    print("E3 swarm scaling...")
    sizes = [5, 10, 20, 40, 80]
    e3_policies = ["fifo", "wsjf", "tokengrid", "oracle"]
    e3 = {n: run_grid(pool, pool_const, replace(base, n_agents=n), e3_policies, args.seeds) for n in sizes}

    print("E4 capacity fix: selection on dev seeds, confirmation on evaluation seeds...")
    e4_select = {f: {rule: run_grid(pool, pool_const, replace(base, dishonest_frac=f, capacity_rule=rule),
                                    ["tokengrid", "fifo"], DEV_SEEDS) for rule in CAPACITY_RULES} for f in (0.25, 0.5)}
    e4_confirm = {f: {rule: run_grid(pool, pool_const, replace(base, dishonest_frac=f, capacity_rule=rule),
                                     ["tokengrid"], args.seeds)["tokengrid"] for rule in ("original", "release_at_reset")}
                  for f in (0.0, 0.25, 0.5)}
    # The rule was chosen when only the first 100 tasks per benchmark existed;
    # every task collected after that is data the choice never saw.
    first = {t.task_id for d in DATASETS for t in load_tasks(d, 100)}
    pool_new = [t for t in pool if t.task_id not in first]
    e4_heldout = {"n_tasks": len(pool_new)}
    if len(pool_new) >= 100:
        e4_heldout.update({rule: tokengrid_runs(pool_new, replace(base, capacity_rule=rule), range(args.seeds))
                           for rule in ("original", "release_at_reset")})

    print("E5 forecasters on evaluation seeds (+ online correction, + near-cliff stress test); "
          "E6 safe (quantile) bids; E7 forecast noise...")
    seeds = range(args.seeds)
    z = np.random.default_rng(0).standard_normal(len(tokens))  # fixed noise draw: every sigma scales the same errors
    online = ONLINE.format(primary)
    e5 = {name: tokengrid_runs(fc_pools[name], base, seeds) for name in FORECASTERS}
    e5[online] = tokengrid_runs(pool, replace(base, online_correction=True), seeds)
    stressed = {name: make_pool(rows, f * np.exp(STRESS_SIGMA * z)) for name, f in fc.items()}
    e5_stress = {name: tokengrid_runs(stressed[name], base, seeds) for name in FORECASTERS}
    e5_stress[online] = tokengrid_runs(stressed[primary], replace(base, online_correction=True), seeds)

    e6_base = primary if primary in QUANTILE_CAPABLE else "embedding (lgbm)"
    e6, e6_quality = {}, {}
    for q in QUANTILES:
        f = forecast(e6_base, ctx, quantile=q)
        e6_quality[q] = forecast_quality(f, tokens)
        e6[q] = tokengrid_runs(make_pool(rows, f), base, seeds)
    e7, e7_quality = {}, {}
    for sigma in NOISE_SIGMAS:
        f = fc[primary] * np.exp(sigma * z)
        e7_quality[sigma] = forecast_quality(f, tokens)
        e7[sigma] = tokengrid_runs(make_pool(rows, f), base, range(args.seeds))

    out = run_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({
        "source": "mock" if args.mock else "groq", "n_tasks": len(rows), "seeds": args.seeds,
        "sim_config": asdict(base), "forecast": quality, "e1": e1,
        "e2": {str(k): v for k, v in e2.items()}, "e3": {str(k): v for k, v in e3.items()},
        "e4_select_dev_seeds": {str(k): v for k, v in e4_select.items()},
        "e4_confirm": {str(k): v for k, v in e4_confirm.items()},
        "e4_heldout_tasks": e4_heldout,
        "primary_forecaster": primary, "forecaster_quality": fc_quality,
        "e5_select_dev_seeds": e5_select, "e5": e5, "e5_stress": e5_stress, "e5_stress_sigma": STRESS_SIGMA,
        "e6_base_forecaster": e6_base,
        "e6_quality": {str(k): v for k, v in e6_quality.items()}, "e6": {str(k): v for k, v in e6.items()},
        "e7_quality": {str(k): v for k, v in e7_quality.items()}, "e7": {str(k): v for k, v in e7.items()},
    }, indent=1))
    write_plots(out, e2, liar_levels, e2_policies, e3, sizes, e3_policies)
    write_forecast_plot(out, e6, e7)
    write_summary(out, args, rows, base, quality, e1, e1_policies, e2, liar_levels, e2_policies, e3, sizes, e3_policies)
    with (out / "summary.md").open("a", encoding="utf-8") as f:
        f.write(e4_summary(e4_select, e4_confirm, e4_heldout))
        f.write(forecast_summary(primary, fc_quality, e5_select, e5, e5_stress, e6_base, e6, e6_quality, e7, e7_quality))
    print(f"report: {out / 'summary.md'}")


def write_plots(out, e2, liar_levels, e2_policies, e3, sizes, e3_policies):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for p in e2_policies:
        for ax, metric, label in [(axes[0], "honest_token_share", "honest agents' share of tokens"),
                                  (axes[1], "correct_per_window", "correct tasks per window")]:
            m = [mean_ci([r[metric] for r in e2[f][p]]) for f in liar_levels]
            x = [100 * f for f in liar_levels]
            ax.plot(x, [v[0] for v in m], marker="o", label=p)
            ax.fill_between(x, [v[1] for v in m], [v[2] for v in m], alpha=0.15)
            ax.set_xlabel("dishonest agents (%)")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("E2 — liar robustness (mean, 95% CI over seeds)")
    fig.tight_layout()
    fig.savefig(out / "e2_liars.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for p in e3_policies:
        for ax, metric, label, scale in [(axes[0], "cost_per_correct_usd", "µ$ per correct task", 1e6),
                                         (axes[1], "correct_per_window", "correct tasks per window", 1)]:
            m = [mean_ci([r[metric] for r in e3[n][p]]) for n in sizes]
            ax.plot(sizes, [v[0] * scale for v in m], marker="o", label=p)
            ax.fill_between(sizes, [v[1] * scale for v in m], [v[2] * scale for v in m], alpha=0.15)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("swarm size N (rate limit fixed)")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("E3 — swarm scaling (mean, 95% CI over seeds)")
    fig.tight_layout()
    fig.savefig(out / "e3_scaling.png", dpi=140)
    plt.close(fig)


def write_summary(out, args, rows, base, q, e1, e1_policies, e2, liar_levels, e2_policies, e3, sizes, e3_policies):
    ds_counts = {}
    for r in rows:
        ds_counts[r["dataset"]] = ds_counts.get(r["dataset"], 0) + 1
    L = []
    if args.mock:
        L += ["> **MOCK TRACES — pipeline test only. Nothing in this report is a finding.**", ""]
    L += ["# TokenGrid evaluation", "",
          f"{len(rows)} traced tasks ({', '.join(f'{k}: {v}' for k, v in sorted(ds_counts.items()))}) · "
          f"model accuracy {q['accuracy']:.1%} · mean {q['tokens_mean']:.0f} tokens/task (CV {q['tokens_cv']:.2f}) · "
          f"{args.seeds} seeds × {base.rounds} rounds · rate limit {base.tokens_per_window:,} tokens / "
          f"{base.window_rounds}-round window · {base.dishonest_frac:.0%} dishonest agents unless varied", "",
          "## Forecast: can an honest agent predict its own cost?", "",
          "Cross-fitted. 'Bonus' / 'penalty' = share of honest calls the reputation ledger would reward "
          f"(accuracy > {config.REPUTATION_ACCURACY_HIGH}) or punish (< {config.REPUTATION_ACCURACY_LOW}).", "",
          "| forecaster | Spearman | MAE (tokens) | mean ledger accuracy | bonus | penalty |", "|---|---|---|---|---|---|"]
    for name in ("learned", "constant"):
        f = q[name]
        L.append(f"| {name} | {f['spearman']:.2f} | {f['mae_tokens']:.0f} | {f['mean_tokengrid_accuracy']:.2f} | "
                 f"{f['frac_bonus']:.1%} | {f['frac_penalty']:.1%} |")

    L += ["", f"## E1 — policy comparison (N={base.n_agents})", "", "Mean [95% CI] over seeds.", "",
          "| policy | " + " | ".join(lbl for _, lbl, _ in E1_METRICS) + " |", "|---|" + "---|" * len(E1_METRICS)]
    for p in e1_policies:
        L.append(f"| {p} | " + " | ".join(fmt(*mean_ci([r[k] for r in e1[p]]), scale=s) for k, _, s in E1_METRICS) + " |")

    L += ["", "### Paired differences: tokengrid minus …", "",
          "Same seeds → same agents, liars and task sequences. CI excluding 0 = significant at 5%.", "",
          "| vs | Δ correct / window | Δ honest correct / window | Δ µ$ / correct |", "|---|---|---|---|"]
    for p in e1_policies:
        if p == "tokengrid":
            continue
        d_c = [a["correct_per_window"] - b["correct_per_window"] for a, b in zip(e1["tokengrid"], e1[p])]
        d_h = [a["honest_correct_per_window"] - b["honest_correct_per_window"] for a, b in zip(e1["tokengrid"], e1[p])]
        d_cost = [a["cost_per_correct_usd"] - b["cost_per_correct_usd"] for a, b in zip(e1["tokengrid"], e1[p])]
        L.append(f"| {p} | {fmt(*mean_ci(d_c))} | {fmt(*mean_ci(d_h))} | {fmt(*mean_ci(d_cost), scale=1e6)} |")

    L += ["", "## E2 — liar robustness", "", "Honest agents' share of tokens (higher = liars captured less capacity).", "",
          "| policy | " + " | ".join(f"{f:.0%} liars" for f in liar_levels) + " |", "|---|" + "---|" * len(liar_levels)]
    for p in e2_policies:
        L.append(f"| {p} | " + " | ".join(fmt(*mean_ci([r["honest_token_share"] for r in e2[f][p]])) for f in liar_levels) + " |")

    L += ["", "## E3 — swarm scaling (fixed rate limit)", "", "µ$ per correct task.", "",
          "| policy | " + " | ".join(f"N={n}" for n in sizes) + " |", "|---|" + "---|" * len(sizes)]
    for p in e3_policies:
        L.append(f"| {p} | " + " | ".join(fmt(*mean_ci([r["cost_per_correct_usd"] for r in e3[n][p]]), scale=1e6) for n in sizes) + " |")

    L += ["", "![E2](e2_liars.png)", "", "![E3](e3_scaling.png)", ""]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")


def write_forecast_plot(out, e6, e7):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    qs = list(e6)
    m = [mean_ci([r["honest_correct_per_window"] for r in e6[q]]) for q in qs]
    ax.plot([100 * q for q in qs], [v[0] for v in m], marker="o", color="#1f77b4", label="honest correct / window")
    ax.fill_between([100 * q for q in qs], [v[1] for v in m], [v[2] for v in m], alpha=0.15, color="#1f77b4")
    ax.set_xlabel("honest bid = this percentile of predicted cost")
    ax.set_ylabel("honest correct tasks per window")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot([100 * q for q in qs], [100 * np.mean([r["violation_rate"] for r in e6[q]]) for q in qs],
             marker="s", color="#d62728", label="429 rate (%)")
    ax2.set_ylabel("rejected calls (%)", color="#d62728")
    ax.set_title("E6 — safe bids", fontsize=10)

    ax = axes[1]
    sig = list(e7)
    for metric, label in [("honest_correct_per_window", "honest correct / window"), ("correct_per_window", "all correct / window")]:
        m = [mean_ci([r[metric] for r in e7[s]]) for s in sig]
        ax.plot(sig, [v[0] for v in m], marker="o", label=label)
        ax.fill_between(sig, [v[1] for v in m], [v[2] for v in m], alpha=0.15)
    ax.set_xlabel("added forecast noise σ (log scale; 0 = learned forecaster as is)")
    ax.set_ylabel("correct tasks per window")
    ax.set_title("E7 — worse forecasts", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "e6_e7_forecasts.png", dpi=140)
    plt.close(fig)


def forecast_summary(primary, fc_quality, e5_select, e5, e5_stress, e6_base, e6, e6_quality, e7, e7_quality) -> str:
    sys_metrics = [("honest_correct_per_window", "honest correct / window"), ("correct_per_window", "correct / window"),
                   ("violation_rate", "429 rate"), ("dropped_honest", "honest dropped"),
                   ("dropped_dishonest", "liars dropped"), ("utilization", "rate-limit used")]
    cliff = [("honest_rep_change_per_call", "rep change / honest call"), ("honest_penalty_rate", "honest calls penalized"),
             ("honest_ledger_accuracy", "ledger accuracy (in play)"), ("dropped_honest", "honest dropped"),
             ("honest_correct_per_window", "honest correct / window"), ("violation_rate", "429 rate")]
    L = ["", f"## E5 — {len(fc_quality)} forecasters", "",
         "Forecast quality (cross-fitted, all tasks). Ledger accuracy = the reputation ledger's own "
         "min(pred, actual)/max(pred, actual); rep change / call = bonus × bonus rate − penalty × penalty rate "
         "for an honest agent bidding this forecast (below 0 = drifting toward removal, the E7 cliff); "
         "coverage = share of tasks that cost no more than forecast.", "",
         "| forecaster | Spearman | MAE (tokens) | ledger accuracy | bonus | penalty | rep change / call | coverage |",
         "|---|---|---|---|---|---|---|---|"]
    for name, f in fc_quality.items():
        L.append(f"| {name} | {f['spearman']:.2f} | {f['mae_tokens']:.0f} | {f['mean_tokengrid_accuracy']:.3f} | "
                 f"{f['frac_bonus']:.1%} | {f['frac_penalty']:.1%} | {f['rep_change_per_call']:+.2f} | {f['coverage']:.1%} |")
    sel = {n: np.mean([r["honest_rep_change_per_call"] for r in runs]) for n, runs in e5_select.items()}
    L += ["", f"Selection on dev seeds {DEV_SEEDS.start}–{DEV_SEEDS.stop - 1} by the largest reputation change per "
              "honest call (in play): " + ", ".join(f"{n} {v:+.2f}" for n, v in sorted(sel.items(), key=lambda kv: -kv[1])) +
              f" → **primary: {primary}** (used for E1–E4).", "",
          "### Distance from the reputation cliff (evaluation seeds, 25% liars, measured in play)", "",
          "| forecaster | " + " | ".join(lbl for _, lbl in cliff) + " |", "|---|" + "---|" * len(cliff)]
    for name, runs in e5.items():
        L.append(f"| {name} | " + " | ".join(fmt(*mean_ci([r[k] for r in runs])) for k, _ in cliff) + " |")
    stress_sigma = STRESS_SIGMA
    L += ["", f"### Near-cliff stress test: every forecast given extra noise σ = {stress_sigma} (same noise for all)", "",
          "More margin from the cliff = fewer honest agents removed when forecasts get worse.", "",
          "| forecaster | rep change / honest call | honest dropped | honest correct / window |", "|---|---|---|---|"]
    for name, runs in e5_stress.items():
        L.append(f"| {name} | " + " | ".join(fmt(*mean_ci([r[k] for r in runs]))
                                             for k in ("honest_rep_change_per_call", "dropped_honest", "honest_correct_per_window")) + " |")
    L += ["", "### System results (evaluation seeds, 25% liars). Mean [95% CI].", "",
          "| forecaster | " + " | ".join(lbl for _, lbl in sys_metrics) + " |", "|---|" + "---|" * len(sys_metrics)]
    for name, runs in e5.items():
        L.append(f"| {name} | " + " | ".join(fmt(*mean_ci([r[k] for r in runs])) for k, _ in sys_metrics) + " |")
    for ref in BASELINES:
        L += ["", f"Paired differences vs. **{ref}** (CI excluding 0 = significant at 5%):", "",
              "| forecaster | Δ rep change / honest call | Δ honest correct / window | Δ honest dropped (stress test) |",
              "|---|---|---|---|"]
        for name, runs in e5.items():
            if name in BASELINES:
                continue
            cells = [fmt(*mean_ci([a[k] - b[k] for a, b in zip(runs, e5[ref])]))
                     for k in ("honest_rep_change_per_call", "honest_correct_per_window")]
            cells.append(fmt(*mean_ci([a["dropped_honest"] - b["dropped_honest"]
                                       for a, b in zip(e5_stress[name], e5_stress[ref])])))
            L.append(f"| {name} | " + " | ".join(cells) + " |")

    L += ["", f"## E6 — safe bids ({e6_base})", "",
          "Honest agents bid the q-th percentile of their predicted cost instead of the average estimate.", "",
          "| bid | coverage | ledger accuracy | " + " | ".join(lbl for _, lbl in sys_metrics) + " |",
          "|---|---|---|" + "---|" * len(sys_metrics)]
    L.append(f"| average (point) | {fc_quality[e6_base]['coverage']:.1%} | {fc_quality[e6_base]['mean_tokengrid_accuracy']:.3f} | " +
             " | ".join(fmt(*mean_ci([r[k] for r in e5[e6_base]])) for k, _ in sys_metrics) + " |")
    for q, runs in e6.items():
        f = e6_quality[q]
        L.append(f"| {q:.0%} percentile | {f['coverage']:.1%} | {f['mean_tokengrid_accuracy']:.3f} | " +
                 " | ".join(fmt(*mean_ci([r[k] for r in runs])) for k, _ in sys_metrics) + " |")

    L += ["", f"## E7 — forecast noise ({primary})", "",
          "Forecasts multiplied by exp(σ·z), z ~ N(0,1) fixed per task: σ = 0 is the learned forecaster as is.", "",
          "| σ | Spearman | ledger accuracy | penalty | " + " | ".join(lbl for _, lbl in sys_metrics) + " |",
          "|---|---|---|---|" + "---|" * len(sys_metrics)]
    for s, runs in e7.items():
        f = e7_quality[s]
        L.append(f"| {s:g} | {f['spearman']:.2f} | {f['mean_tokengrid_accuracy']:.3f} | {f['frac_penalty']:.1%} | " +
                 " | ".join(fmt(*mean_ci([r[k] for r in runs])) for k, _ in sys_metrics) + " |")
    L += ["", "![E6 and E7](e6_e7_forecasts.png)", ""]
    return "\n".join(L) + "\n"


def e4_summary(e4_select: dict, e4_confirm: dict, e4_heldout: dict | None = None) -> str:
    L = ["", "## E4 — the capacity fix", "",
         "Each round may spend 50% of the remaining window headroom (original rule). Headroom left at the "
         "window reset is lost, so the chosen rule `release_at_reset` lets the last round before the reset "
         "spend all of it.", "",
         f"### (a) Selection — dev seeds {DEV_SEEDS.start}–{DEV_SEEDS.stop - 1} (where the rule was chosen; means)", "",
         "| liars | rule | policy | " + " | ".join(lbl for _, lbl in E4_METRICS) + " |",
         "|---|---|---|" + "---|" * len(E4_METRICS)]
    for f, by_rule in e4_select.items():
        for rule, by_pol in by_rule.items():
            for pol, runs in by_pol.items():
                L.append(f"| {f:.0%} | {rule} | {pol} | " +
                         " | ".join(f"{np.mean([r[k] for r in runs]):.3f}" for k, _ in E4_METRICS) + " |")
    L += ["", "### (b) Confirmation — evaluation seeds, tokengrid, release_at_reset minus original", "",
          "Paired by seed. CI excluding 0 = significant at 5%.", "",
          "| liars | " + " | ".join(f"Δ {lbl}" for _, lbl in E4_METRICS) + " |", "|---|" + "---|" * len(E4_METRICS)]
    for f, by_rule in e4_confirm.items():
        cells = []
        for k, _ in E4_METRICS:
            d = [a[k] - b[k] for a, b in zip(by_rule["release_at_reset"], by_rule["original"])]
            cells.append(fmt(*mean_ci(d)))
        L.append(f"| {f:.0%} | " + " | ".join(cells) + " |")
    if e4_heldout and "original" in e4_heldout:
        L += ["", f"### (c) Held-out tasks — only the {e4_heldout['n_tasks']} tasks collected after the rule was chosen", "",
              "Evaluation seeds, 25% liars, tokengrid, release_at_reset minus original, paired.", "",
              "| " + " | ".join(f"Δ {lbl}" for _, lbl in E4_METRICS) + " |", "|" + "---|" * len(E4_METRICS)]
        L.append("| " + " | ".join(
            fmt(*mean_ci([a[k] - b[k] for a, b in zip(e4_heldout["release_at_reset"], e4_heldout["original"])]))
            for k, _ in E4_METRICS) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
