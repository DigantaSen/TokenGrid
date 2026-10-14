"""
Trace-driven simulation of a TokenGrid swarm under a shared rate limit.

What is real and what is simulated:
  real       the mechanism — TokenGrid's own compute_capacity (K and round
             budget from remaining headroom), run_auction (ranking, per-agent
             cap, budget packing) and ReputationLedger (accuracy scoring,
             flagging, dropping) are imported and run unchanged
  real       each task's token cost and correctness — replayed from traces
             of real model calls (experiments/collect_traces.py)
  simulated  the clock (rounds), the rate-limit window (fixed windows of
             `window_rounds` rounds), and which agents lie

A call whose *actual* tokens exceed the remaining window headroom is
rejected (a 429): nothing is spent and the task stays pending. That is the
real cost of a bad forecast, beyond the reputation hit.

Policies — all clear through identical K / budget / per-agent-cap rules;
only the ranking key differs:
    fifo       longest-waiting agent first
    random     uniform random order
    urgency    highest urgency first
    wsjf       urgency / predicted_tokens — Smith's weighted-shortest-job-first,
               i.e. TokenGrid's bid score *without* reputation
    tokengrid  reputation * urgency / predicted_tokens, reputation updated
               from real usage, agents dropped below threshold (the system)
    oracle     urgency / TRUE tokens, packing by true tokens (upper bound:
               perfect forecasts, no liars possible)

Capacity rules — how much of the remaining window headroom a round may spend:
    release_at_reset  TokenGrid's rule (default): 50% per round, but 100% in
                      the last round before the window resets, since headroom
                      left at the reset is lost anyway
    original          50% every round — the rule before the fix
    spend_all         100% every round        } alternatives considered when
    even_pacing       remaining / rounds left } designing the fix; simulation only
"""
import math
import random
from dataclasses import dataclass, field, replace

import numpy as np

from app import config
from app.auction import Bid, compute_capacity, run_auction, score
from app.event_log import event_log
from app.reputation import ReputationLedger

POLICIES = ["fifo", "random", "urgency", "wsjf", "tokengrid", "oracle"]
CAPACITY_RULES = ["original", "release_at_reset", "spend_all", "even_pacing"]
ONLINE_ALPHA = 0.2  # how fast an honest agent's online correction follows its latest errors


@dataclass
class TraceTask:
    task_id: str
    total_tokens: int
    cost_usd: float
    correct: bool
    forecast: float        # an honest agent's prediction for this task (learned or constant)


@dataclass
class SimConfig:
    n_agents: int = 20
    rounds: int = 300
    window_rounds: int = 10            # rate-limit window length, in auction rounds
    tokens_per_window: int = 60_000
    requests_per_window: int = 400
    dishonest_frac: float = 0.25       # split evenly between inflators and freeloaders
    seed: int = 0
    capacity_rule: str = "release_at_reset"
    # Honest agents learn from their own track record: each keeps a running
    # average of log(actual / forecast) over its completed calls and scales
    # its next bid by it. Liars don't (correcting would undo their lie).
    online_correction: bool = False


@dataclass
class _Agent:
    agent_id: str
    strategy: str                      # "honest" | "inflator" | "freeloader"
    base_urgency: float
    tasks: list                        # this agent's own task sequence (identical across policies)
    distort: list                      # per-task freeloader under-report factor
    k: int = 0                         # index of current task
    rounds_queued: int = 0
    assigned_round: int = 0
    dropped: bool = False
    completions: int = 0
    tokens_used: int = 0
    online: bool = False
    bias: float = 0.0                  # running average log(actual / forecast), used when online

    @property
    def task(self) -> TraceTask:
        return self.tasks[self.k % len(self.tasks)]

    def reported(self) -> int:
        honest = self.task.forecast
        if self.strategy == "inflator":
            return max(1, int(honest * config.INFLATOR_MULTIPLIER))
        if self.strategy == "freeloader":
            return max(1, int(honest * self.distort[self.k % len(self.distort)]))
        return max(1, int(honest * math.exp(self.bias))) if self.online else max(1, int(honest))

    def learn(self, actual_tokens: int):
        if self.online and self.strategy == "honest":
            err = math.log(max(1, actual_tokens) / max(1.0, self.task.forecast))
            self.bias = (1 - ONLINE_ALPHA) * self.bias + ONLINE_ALPHA * err

    def urgency(self) -> float:
        return min(1.0, self.base_urgency + 0.05 * self.rounds_queued)  # same aging rule as SimAgent


def _make_agents(pool: list[TraceTask], cfg: SimConfig) -> list[_Agent]:
    """Depends only on cfg.seed, so every policy faces the same agents, the
    same liars and the same per-agent task sequences — comparisons are paired."""
    rng = random.Random(cfg.seed)
    n_bad = round(cfg.dishonest_frac * cfg.n_agents)
    bad = set(rng.sample(range(cfg.n_agents), n_bad))
    agents, bad_seen = [], 0
    for i in range(cfg.n_agents):
        strategy = "honest"
        if i in bad:
            strategy = "inflator" if bad_seen % 2 == 0 else "freeloader"
            bad_seen += 1
        arng = random.Random(f"{cfg.seed}-{i}")
        seq = arng.sample(pool, len(pool))
        distort = [arng.uniform(*config.FREELOADER_UNDERBID_RANGE) for _ in range(64)]
        agents.append(_Agent(f"agent-{i:03d}", strategy, arng.uniform(0.4, 0.8), seq, distort,
                             online=cfg.online_correction and strategy == "honest"))
    return agents


def _capacity(cfg: SimConfig, rnd: int, remaining_requests: int, remaining_tokens: int):
    rounds_left = cfg.window_rounds - (rnd % cfg.window_rounds)
    if cfg.capacity_rule == "original":
        return compute_capacity(remaining_requests, remaining_tokens)
    if cfg.capacity_rule == "release_at_reset":
        return compute_capacity(remaining_requests, remaining_tokens, rounds_until_reset=rounds_left)
    base = compute_capacity(remaining_requests, remaining_tokens)
    fraction = 1.0 if cfg.capacity_rule == "spend_all" else 1.0 / rounds_left
    max_tokens = max(1, math.floor(remaining_tokens * fraction))
    return replace(base, max_tokens=max_tokens, per_agent_cap_tokens=max_tokens * config.PER_AGENT_CAP_FRACTION)


def simulate(pool: list[TraceTask], policy: str, cfg: SimConfig) -> dict:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy}")
    if cfg.capacity_rule not in CAPACITY_RULES:
        raise ValueError(f"unknown capacity rule {cfg.capacity_rule}")
    agents = _make_agents(pool, cfg)
    by_id = {a.agent_id: a for a in agents}
    ledger = ReputationLedger()
    prng = random.Random(f"policy-{cfg.seed}")

    remaining_tokens = remaining_requests = 0
    completed = correct = honest_correct = violations = 0
    honest_calls = honest_bonus = honest_penalty = 0  # the ledger's verdicts on honest agents' calls
    honest_acc_sum = 0.0
    cost = 0.0
    waits, done_tokens = [], []

    with event_log.muted():
        for rnd in range(cfg.rounds):
            if rnd % cfg.window_rounds == 0:
                remaining_tokens, remaining_requests = cfg.tokens_per_window, cfg.requests_per_window

            active = [a for a in agents if not a.dropped]
            if not active:
                break
            capacity = _capacity(cfg, rnd, remaining_requests, remaining_tokens)

            if policy == "oracle":
                bids = [Bid(a.agent_id, a.task.total_tokens, a.urgency(), config.REPUTATION_START) for a in active]
            else:
                bids = [Bid(a.agent_id, a.reported(), a.urgency(),
                            ledger.get(a.agent_id) if policy == "tokengrid" else config.REPUTATION_START)
                        for a in active]

            if policy in ("tokengrid",):
                key = score
            elif policy in ("wsjf", "oracle"):
                key = lambda b: b.urgency / b.predicted_tokens  # noqa: E731
            elif policy == "urgency":
                key = lambda b: b.urgency  # noqa: E731
            else:
                draw = {b.agent_id: prng.random() for b in bids}
                if policy == "fifo":
                    key = lambda b: by_id[b.agent_id].rounds_queued + 1e-3 * draw[b.agent_id]  # noqa: E731
                else:
                    key = lambda b: draw[b.agent_id]  # noqa: E731

            result = run_auction(bids, capacity, key=key)

            for outcome in result.winners:
                a = by_id[outcome.agent_id]
                t = a.task
                if t.total_tokens > remaining_tokens or remaining_requests <= 0:
                    violations += 1          # 429: forecast under-shot the headroom
                    a.rounds_queued += 1
                    continue
                remaining_tokens -= t.total_tokens
                remaining_requests -= 1
                completed += 1
                correct += int(t.correct)
                honest_correct += int(t.correct and a.strategy == "honest")
                cost += t.cost_usd
                waits.append(rnd - a.assigned_round)
                done_tokens.append(t.total_tokens)
                a.completions += 1
                a.tokens_used += t.total_tokens
                if a.strategy == "honest":
                    p, act = max(1, outcome.predicted_tokens), max(1, t.total_tokens)
                    acc = min(p, act) / max(p, act)
                    honest_calls += 1
                    honest_acc_sum += acc
                    honest_bonus += acc > config.REPUTATION_ACCURACY_HIGH
                    honest_penalty += acc < config.REPUTATION_ACCURACY_LOW
                a.learn(t.total_tokens)
                if policy == "tokengrid":
                    ev = ledger.record_prediction_outcome(a.agent_id, outcome.predicted_tokens, t.total_tokens)
                    a.dropped = ev.dropped
                a.k += 1
                a.rounds_queued = 0
                a.assigned_round = rnd + 1

            for outcome in result.queued:
                by_id[outcome.agent_id].rounds_queued += 1

    honest = [a for a in agents if a.strategy == "honest"]
    total_tokens = sum(a.tokens_used for a in agents)
    hc = np.array([a.completions for a in honest], dtype=float)
    pool_mean = float(np.mean([t.total_tokens for t in pool]))
    n_windows = max(1, cfg.rounds // cfg.window_rounds)
    return {
        "policy": policy,
        "completed": completed,
        "correct": correct,
        "correct_per_window": correct / n_windows,
        # Liars' tasks are real work too, so total throughput alone can make
        # dropping a liar look like a loss; this is the work honest agents got done.
        "honest_correct_per_window": honest_correct / n_windows,
        # Share of the rate limit actually spent. TokenGrid clears at most
        # ROUND_TOKEN_FRACTION of *remaining* headroom per round, so headroom
        # it holds back for later rounds can expire unused at the window boundary.
        "utilization": total_tokens / (cfg.tokens_per_window * n_windows),
        "cost_usd": cost,
        "cost_per_correct_usd": cost / correct if correct else float("nan"),
        "violations": violations,
        "violation_rate": violations / max(1, completed + violations),
        "wait_p50": float(np.percentile(waits, 50)) if waits else float("nan"),
        "wait_p95": float(np.percentile(waits, 95)) if waits else float("nan"),
        "honest_token_share": sum(a.tokens_used for a in honest) / total_tokens if total_tokens else float("nan"),
        "honest_starved_frac": float((hc == 0).mean()) if len(hc) else float("nan"),
        "jain_fairness_honest": float(hc.sum() ** 2 / (len(hc) * (hc ** 2).sum())) if hc.sum() else float("nan"),
        # < 1: the policy completes cheaper-than-average tasks, i.e. part of any
        # cost-per-task gain is from skipping expensive work, not doing it cheaper.
        "size_bias": float(np.mean(done_tokens)) / pool_mean if done_tokens else float("nan"),
        "dropped_dishonest": sum(a.dropped for a in agents if a.strategy != "honest"),
        "dropped_honest": sum(a.dropped for a in honest),
        # How the reputation ledger judges honest agents' bids, measured in play:
        # rep change per call = bonus * bonus rate - penalty * penalty rate.
        # Below zero, honest agents drift toward removal (the E7 cliff).
        "honest_ledger_accuracy": honest_acc_sum / honest_calls if honest_calls else float("nan"),
        "honest_penalty_rate": honest_penalty / honest_calls if honest_calls else float("nan"),
        "honest_bonus_rate": honest_bonus / honest_calls if honest_calls else float("nan"),
        "honest_rep_change_per_call": (config.REPUTATION_BONUS * honest_bonus - config.REPUTATION_PENALTY * honest_penalty)
                                      / honest_calls if honest_calls else float("nan"),
    }
