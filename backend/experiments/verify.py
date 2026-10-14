"""
Independent checks on the real traces and the experiment report.

Unit tests (tests/) check the code on tiny hand-made cases. This script
checks the *actual data and results*: are the traces internally consistent,
could the predictor be cheating, does the simulator respect the rate limit
on real data, does a re-run reproduce the report exactly, and is the
explanation for TokenGrid's lower rate-limit usage actually right?

    python -m experiments.verify            # results/groq
    python -m experiments.verify --mock

Prints PASS / FAIL per check; exits non-zero if anything fails.
"""
import argparse
import json
import sys
from dataclasses import replace

import numpy as np
from scipy.stats import spearmanr

from app.accounting import cost_of
from app.tasks import Task, grade
from app.token_predictor import crossfit_predict, handcrafted_features
from experiments.collect_traces import load_traces, traces_path
from experiments.run_experiments import BASE_N, CAPACITY_RATIO, DEFAULT_PRIMARY, build_pools
from experiments.simulator import POLICIES, SimConfig, simulate

results = []


def check(name: str, ok: bool, detail: str):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    tpath = traces_path(args.mock)
    all_rows = load_traces(tpath)
    rows = list({r["task_id"]: r for r in all_rows if r["error"] is None}.values())
    run_dir = tpath.parent
    print(f"{len(rows)} traced tasks in {tpath}\n")

    # 1. Trace integrity -------------------------------------------------
    keys = [(r["task_id"], r["model"]) for r in all_rows if r["error"] is None]
    check("no duplicate successful traces", len(keys) == len(set(keys)),
          f"{len(keys)} rows, {len(set(keys))} unique")
    bad_sum = [r for r in rows if r["total_tokens"] != r["input_tokens"] + r["output_tokens"]]
    check("total = input + output tokens", not bad_sum, f"{len(bad_sum)} mismatches")
    bad_cost = [r for r in rows if abs(r["cost_usd"] - cost_of(r["model"], r["input_tokens"], r["output_tokens"])) > 1e-12]
    check("cost matches token counts x price", not bad_cost, f"{len(bad_cost)} mismatches")

    def regrade(r):
        if r["pred"] is None:
            return False
        if r["dataset"] == "gsm8k":
            try:
                return abs(float(r["pred"]) - float(r["gold"])) < 1e-6
            except ValueError:
                return r["pred"] == r["gold"]
        return r["pred"] == r["gold"]
    bad_grade = [r for r in rows if regrade(r) != r["correct"]]
    check("'correct' agrees with extracted answer vs gold", not bad_grade, f"{len(bad_grade)} disagreements")
    truncated = sum(r["output_tokens"] >= r["max_tokens"] for r in rows)
    check("answers not cut off by the token cap", truncated / max(1, len(rows)) < 0.02,
          f"{truncated} of {len(rows)} hit the cap")
    # Audit grading from the saved full replies (stored for tasks collected after
    # replies were added): grading the reply again must give the same verdict.
    with_reply = [r for r in rows if r.get("response")]
    if with_reply:
        kinds = {"gsm8k": "numeric", "mmlu": "mc", "arc_challenge": "mc"}
        regraded = [grade(Task(r["task_id"], r["dataset"], r["text"], r["gold"], kinds[r["dataset"]]), r["response"])
                    for r in with_reply]
        bad = sum((p, c) != (r["pred"], r["correct"]) for (p, c), r in zip(regraded, with_reply))
        check("saved replies re-grade to the same verdict", bad == 0, f"{bad} of {len(with_reply)} replies disagree")
    else:
        print("[SKIP] reply audit: no saved replies in these traces")

    # 2. Predictor leakage -------------------------------------------------
    # If the pipeline accidentally let a task's own cost into its forecast,
    # it would still "predict" well after the costs are randomly shuffled.
    # Rebuild forecasts with the forecaster the saved report selected, so the
    # reproducibility check below compares like with like.
    saved_path = run_dir / "report" / "metrics.json"
    primary = (json.loads(saved_path.read_text()).get("primary_forecaster") if saved_path.exists() else None) or DEFAULT_PRIMARY
    pool, _, quality = build_pools(rows, run_dir / "embeddings.npz", primary)
    texts, datasets = [r["text"] for r in rows], [r["dataset"] for r in rows]
    z = np.load(run_dir / "embeddings.npz")
    X = np.hstack([z["emb"], handcrafted_features(texts, datasets)])
    tokens = np.array([r["total_tokens"] for r in rows], dtype=float)
    shuffled = np.random.default_rng(0).permutation(tokens)
    rho_shuf = spearmanr(crossfit_predict(X, shuffled), shuffled).statistic
    rho_real = quality["learned"]["spearman"]
    check("no leakage: shuffled costs are unpredictable", abs(rho_shuf) < 0.15,
          f"Spearman on shuffled costs {rho_shuf:+.2f} vs real {rho_real:.2f}")

    # 3. Simulator invariants on real traces ------------------------------
    base = SimConfig(n_agents=BASE_N, rounds=300,
                     tokens_per_window=int(CAPACITY_RATIO * BASE_N * SimConfig.window_rounds * quality["tokens_mean"]),
                     requests_per_window=BASE_N * SimConfig.window_rounds)
    for p in POLICIES:
        a, b = simulate(pool, p, base), simulate(pool, p, base)
        check(f"{p}: rate limit respected and run deterministic", a["utilization"] <= 1 + 1e-9 and a == b,
              f"utilization {a['utilization']:.3f}")

    # 4. Reproducibility of the saved report ------------------------------
    report = run_dir / "report" / "metrics.json"
    if report.exists():
        saved = json.loads(report.read_text())
        mismatches = 0
        for p in POLICIES:
            for s in range(min(3, saved["seeds"])):
                fresh = simulate(pool, p, replace(base, seed=s))
                old = saved["e1"][p][s]
                mismatches += any(
                    not (np.isnan(fresh[k]) and np.isnan(old[k])) and abs(fresh[k] - old[k]) > 1e-9
                    for k in fresh if isinstance(fresh[k], float))
        check("re-running reproduces the saved report exactly", mismatches == 0,
              f"{mismatches} of {len(POLICIES) * min(3, saved['seeds'])} runs differ")
    else:
        print("[SKIP] reproducibility: no report yet")

    # 5. Claim check: why did TokenGrid use less of the rate limit? -------
    # Hypothesis: the ORIGINAL capacity rule (spend at most 50% of *remaining*
    # headroom per round, max 20% of that per agent) leaves end-of-window
    # crumbs that only liars' tiny under-bids can fill. If so: with no
    # liars, every policy should waste about the same; and letting a round
    # spend all remaining headroom should recover most of the waste.
    original = replace(base, capacity_rule="original")
    util = {p: simulate(pool, p, replace(original, dishonest_frac=0.0))["utilization"] for p in ("fifo", "wsjf", "tokengrid")}
    spread = max(util.values()) - min(util.values())
    check("original rule, no liars: all policies waste equally (waste is not caused by reputation)",
          spread < 0.02, ", ".join(f"{p} {u:.3f}" for p, u in util.items()))
    orig_util = simulate(pool, "tokengrid", original)["utilization"]
    full_util = simulate(pool, "tokengrid", replace(base, capacity_rule="spend_all"))["utilization"]
    check("the 50%-per-round rule causes the unused capacity", full_util > orig_util + 0.03,
          f"tokengrid utilization {orig_util:.3f} with the 50% rule -> {full_util:.3f} if a round may spend all headroom")

    # 6. The fix (release_at_reset, now TokenGrid's rule) -----------------
    # Across the evaluation seeds it must recover the capacity without
    # weakening the protection against liars.
    pairs = [(simulate(pool, "tokengrid", replace(base, seed=s)), simulate(pool, "tokengrid", replace(original, seed=s)))
             for s in range(10)]
    d_util = np.mean([a["utilization"] - b["utilization"] for a, b in pairs])
    d_honest = np.mean([a["honest_correct_per_window"] - b["honest_correct_per_window"] for a, b in pairs])
    check("fix recovers unused capacity", d_util > 0.05, f"utilization +{d_util:.3f} on average over 10 seeds")
    check("fix increases honest agents' correct work", d_honest > 0, f"{d_honest:+.2f} honest correct tasks per window")
    share_new = min(a["honest_token_share"] for a, _ in pairs)
    liars_new, liars_old = sum(a["dropped_dishonest"] for a, _ in pairs), sum(b["dropped_dishonest"] for _, b in pairs)
    # Honest agents can be wrongly dropped when forecasts are poor — that's a
    # predictor-quality effect present under either rule, so the fix is held
    # to "no worse than before", not to an absolute zero.
    honest_new, honest_old = sum(a["dropped_honest"] for a, _ in pairs), sum(b["dropped_honest"] for _, b in pairs)
    check("fix keeps liar protection",
          share_new > 0.97 and liars_new == liars_old and honest_new <= honest_old,
          f"honest token share >= {share_new:.3f}; liars dropped {liars_new} (before {liars_old}); "
          f"honest dropped {honest_new} (before {honest_old})")
    viol = np.mean([a["violation_rate"] for a, _ in pairs])
    check("fix keeps 429 rejections rare", viol < 0.02, f"429 rate {viol:.3%}")

    # 7. Forecasters --------------------------------------------------------
    # A "safe bid" at the q-th percentile should cover about q of tasks, and a
    # learned forecaster should score better on the ledger than a constant guess.
    text_feats = handcrafted_features(texts, datasets)
    cover = {q: float((tokens <= crossfit_predict(text_feats, tokens, quantile=q)).mean()) for q in (0.5, 0.8, 0.9)}
    check("percentile forecasts are calibrated (within 5 points)", all(abs(c - q) <= 0.05 for q, c in cover.items()),
          ", ".join(f"q{int(q * 100)} covers {c:.1%}" for q, c in cover.items()))
    check("learned forecaster beats a constant guess on the ledger's accuracy score",
          quality["learned"]["mean_tokengrid_accuracy"] > quality["constant"]["mean_tokengrid_accuracy"],
          f"{primary} {quality['learned']['mean_tokengrid_accuracy']:.3f} vs constant {quality['constant']['mean_tokengrid_accuracy']:.3f}")

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
