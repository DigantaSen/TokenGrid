"""
Quick terminal check of the auction + reputation ledger, without needing
uvicorn running.

    python run_demo.py            # uses DRY_RUN if no GROQ_API_KEY is set
    GROQ_API_KEY=... python run_demo.py
"""
from app import config
from app.accounting import ledger
from app.agent_simulator import run_auction_round, spawn_agents
from app.event_log import event_log
from app.llm_provider import get_provider
from app.reputation import reputation_ledger
from app.swarm_sizer import swarm_sizer


def main():
    provider = get_provider()
    print(f"mode: {'DRY_RUN' if config.DRY_RUN else 'LIVE (Groq)'}")
    print(f"models: simple={config.MODEL_SIMPLE} complex={config.MODEL_COMPLEX}\n")

    agents = spawn_agents(12, provider)
    print(f"strategies: {', '.join(f'{a.agent_id}={a.strategy}' for a in agents)}\n")

    n_rounds = 15
    for round_num in range(1, n_rounds + 1):
        summary = run_auction_round(agents)
        cap = summary["capacity"]
        swarm = summary["swarm"]
        print(
            f"--- round {round_num} --- K={cap['max_requests']} token_budget={cap['max_tokens']:.0f} "
            f"per_agent_cap={cap['per_agent_cap_tokens']:.0f} source={cap['source']} "
            f"| won={summary['won']} queued={summary['queued']} rejected={summary['rejected']} "
            f"| N={swarm['current_n']} N*={swarm['optimal_n']}"
        )
        for e in summary["entries"]:
            if e["status"] in ("WON", "FLAGGED", "DROPPED"):
                cache_tag = " [CACHE HIT]" if e["cache_hit"] else ""
                accuracy_str = f"{e['accuracy']:.0%}" if e["accuracy"] is not None else "n/a"
                print(
                    f"  {e['agent_id']:<10} {e['strategy']:<10} status={e['status']:<8} "
                    f"score={e['bid_score']:.3f} predicted={e['predicted_tokens']:<5} "
                    f"actual={e['actual_tokens']:<5} accuracy={accuracy_str} "
                    f"rep {e['reputation_before']:.0f}->{e['reputation_after']:.0f} "
                    f"compression={e['compression_ratio']:.1f}x{cache_tag}"
                )
        print(f"  cache: {summary['cache_stats']}")
        print(f"  savings: {summary['savings']}")
        if summary["resize"]["added"] or summary["resize"]["removed"]:
            print(f"  resize: +{len(summary['resize']['added'])} -{len(summary['resize']['removed'])} -> N={summary['resize']['new_n']}")
        print()

    print("=== reputation snapshot ===")
    for a in reputation_ledger.snapshot():
        print(f"  {a['agent_id']:<10} rep={a['reputation']:>5.1f} avg_acc={a['avg_accuracy']:.0%} "
              f"calls={a['calls']} liar={a['flagged_liar']} dropped={a['dropped']}")

    print("\n=== ledger totals ===")
    for k, v in ledger.totals().items():
        print(f"  {k}: {v}")

    print(f"\n=== swarm sizer (N={len(agents)}) ===")
    status = swarm_sizer.status(len(agents))
    print(f"  optimal_n={status['optimal_n']} tick_count={status['tick_count']}")
    print(f"  cost at current N: {status['cost_at_current_n']}")
    print(f"  cost at optimal N*: {status['cost_at_optimal_n']}")

    print("\n=== last 15 events ===")
    for e in reversed(event_log.recent(15)):
        print(f"  [{e.category}] {e.message}")


if __name__ == "__main__":
    main()
