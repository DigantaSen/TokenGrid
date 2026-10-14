import random
from dataclasses import replace

from experiments.simulator import POLICIES, SimConfig, TraceTask, _make_agents, simulate


def pool(n=300, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        tokens = int(rng.lognormvariate(5.5, 0.5))
        out.append(TraceTask(f"t{i}", tokens, tokens * 5e-7, rng.random() < 0.7, tokens * rng.uniform(0.8, 1.25)))
    return out


CFG = SimConfig(n_agents=20, rounds=120, tokens_per_window=12_000, requests_per_window=200)


def test_rate_limit_never_exceeded_and_deterministic():
    p = pool()
    for policy in POLICIES:
        r = simulate(p, policy, CFG)
        assert 0 < r["utilization"] <= 1.0 + 1e-9, policy
        assert r == simulate(p, policy, CFG), f"{policy} not deterministic"


def test_agents_and_liars_identical_across_policies():
    a, b = _make_agents(pool(), CFG), _make_agents(pool(), CFG)
    assert [(x.agent_id, x.strategy, [t.task_id for t in x.tasks[:5]]) for x in a] == \
           [(x.agent_id, x.strategy, [t.task_id for t in x.tasks[:5]]) for x in b]
    assert sum(x.strategy != "honest" for x in a) == round(CFG.dishonest_frac * CFG.n_agents)


def test_oracle_never_triggers_429():
    assert simulate(pool(), "oracle", CFG)["violations"] == 0


def test_tokengrid_reputation_protects_honest_agents_from_liars():
    cfg = SimConfig(n_agents=20, rounds=200, tokens_per_window=12_000, requests_per_window=200, dishonest_frac=0.5)
    tg, fifo = simulate(pool(), "tokengrid", cfg), simulate(pool(), "fifo", cfg)
    assert tg["dropped_dishonest"] > 0
    assert tg["honest_token_share"] > fifo["honest_token_share"]


def test_online_correction_learns_a_systematic_bias():
    # Every honest forecast is 2x too low; with online correction honest bids
    # should move toward the truth, so the ledger penalizes them less.
    biased = [TraceTask(t.task_id, t.total_tokens, t.cost_usd, t.correct, t.total_tokens / 2.6) for t in pool()]
    cfg = SimConfig(n_agents=20, rounds=200, tokens_per_window=12_000, requests_per_window=200, dishonest_frac=0.0)
    plain = simulate(biased, "tokengrid", cfg)
    learned = simulate(biased, "tokengrid", replace(cfg, online_correction=True))
    assert learned["honest_ledger_accuracy"] > plain["honest_ledger_accuracy"]
    assert learned["honest_penalty_rate"] < plain["honest_penalty_rate"]
