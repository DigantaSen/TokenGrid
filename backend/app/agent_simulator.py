"""
Agent simulator, now wired through the auction and reputation ledger.

Each agent has a strategy that shapes how it *predicts* its own token cost —
this is where the auction's incentive properties actually get exercised:

    honest      predicts close to a true estimate
    inflator    predicts 3x true cost (lowers its own bid_score on purpose,
                to demonstrate why nobody rational does this)
    freeloader  predicts far below true cost to game bid_score upward —
                the reputation ledger is what catches this over time
    efficient   predicts a bit low AND actually asks the model for brevity,
                so its prediction stays honest relative to its own (lower)
                real usage
"""
import itertools
import math
import random
import time
from dataclasses import dataclass

from . import config
from .accounting import CallRecord, cost_of, ledger
from .auction import Bid, compute_capacity, run_auction
from .context_compressor import context_compressor, estimate_tokens
from .event_log import event_log
from .llm_provider import CompletionResult, LLMProvider
from .metrics_history import metrics_history
from .model_router import classify_complexity
from .reputation import reputation_ledger
from .savings import savings
from .semantic_cache import semantic_cache
from .swarm_sizer import swarm_sizer

SAMPLE_TASKS = [
    ("Classify the sentiment of this review: 'Shipping was fast, product broke in a week.'", "simple"),
    ("Summarize in one sentence: quarterly revenue rose 8% but churn also rose 3%.", "simple"),
    ("Is this a question or a statement: 'The server restarted itself twice today.'", "simple"),
    ("Write a short design doc outlining a rate limiter for a multi-tenant API.", "complex"),
    ("Explain the tradeoffs between optimistic and pessimistic locking in a database.", "complex"),
    ("Draft a plan to migrate a monolith to microservices with minimal downtime.", "complex"),
    ("Debug why a Python asyncio task silently swallows exceptions and propose a fix.", "complex"),
    ("Extract the named entities from: 'Satya met with the board in Redmond on Tuesday.'", "simple"),
]

MODEL_FOR_COMPLEXITY = {
    "simple": config.MODEL_SIMPLE,
    "complex": config.MODEL_COMPLEX,
}

_task_counter = itertools.count(0)


def base_token_estimate(task: str, complexity: str) -> int:
    """A rough, strategy-neutral estimate of what a call should cost —
    the number an *honest* agent would predict. Stands in for whatever a
    real agent's own cost model would compute."""
    prompt_tokens = max(10, len(task) // 4) + 20  # +system prompt overhead
    # Roughly the midpoint of what the complexity tier's max_tokens cap allows.
    output_tokens = config.MAX_OUTPUT_TOKENS[complexity] // 2
    return prompt_tokens + output_tokens


@dataclass
class AgentRunResult:
    agent_id: str
    task: str
    complexity: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_s: float
    error: str | None
    cache_hit: bool = False
    cache_similarity: float = 0.0
    compression_ratio: float = 1.0
    history_summarized: bool = False


@dataclass
class BidRoundEntry:
    agent_id: str
    strategy: str
    predicted_tokens: int
    urgency: float
    reputation_before: float
    bid_score: float
    status: str
    reason: str
    actual_tokens: int = None
    accuracy: float = None
    reputation_after: float = None
    cost_usd: float = None
    model: str = None
    cache_hit: bool = False
    compression_ratio: float = 1.0


class SimAgent:
    def __init__(self, agent_id: str, strategy: str, provider: LLMProvider):
        self.agent_id = agent_id
        self.strategy = strategy
        self.provider = provider

        self.pending_task: tuple[str, str] | None = None
        self.current_prediction: int | None = None
        self.base_urgency = random.uniform(0.4, 0.8)
        self.rounds_queued = 0
        self.status = "IDLE"
        self.tokens_used_total = 0
        self.history: list[dict] = []  # accumulated (user, assistant) turns, across rounds
        self.force_predicted_tokens: int | None = None  # one-shot override, used by /stress/flash-drain

    def predict_tokens(self, task: str, complexity: str) -> int:
        base = base_token_estimate(task, complexity)
        if self.strategy == "inflator":
            return max(1, int(base * config.INFLATOR_MULTIPLIER))
        if self.strategy == "freeloader":
            return max(1, int(base * random.uniform(*config.FREELOADER_UNDERBID_RANGE)))
        if self.strategy == "efficient":
            return max(1, int(base * random.uniform(0.6, 0.8)))
        return max(1, int(base * random.uniform(0.9, 1.15)))  # honest

    def ensure_task(self):
        if self.pending_task is None:
            idx = next(_task_counter)
            task, complexity = SAMPLE_TASKS[idx % len(SAMPLE_TASKS)]
            self.pending_task = (task, complexity)
            self.current_prediction = self.predict_tokens(task, complexity)
            self.rounds_queued = 0

    def current_urgency(self) -> float:
        # Queued agents get more urgent over time — prevents starvation
        # without letting them jump the queue purely via reputation.
        return min(1.0, self.base_urgency + 0.05 * self.rounds_queued)

    def execute_pending(self) -> AgentRunResult:
        task, complexity = self.pending_task
        model = MODEL_FOR_COMPLEXITY[complexity]

        # Semantic cache is checked on the canonical task text — strategy
        # wording differences shouldn't stop a repeat task from hitting.
        cache_result = semantic_cache.lookup(task)
        if cache_result.hit:
            result = CompletionResult(
                text=cache_result.entry.response_text,
                input_tokens=0, output_tokens=0,
                model=cache_result.entry.model, latency_s=0.0,
            )
            rec: CallRecord = ledger.record(self.agent_id, result, call_type="cache")
            self.pending_task = None
            return AgentRunResult(
                agent_id=self.agent_id, task=task, complexity=complexity, model=rec.model,
                input_tokens=0, output_tokens=0, total_tokens=0, cost_usd=0.0, latency_s=0.0, error=None,
                cache_hit=True, cache_similarity=cache_result.similarity,
            )

        # Model router: a cheap real call decides simple-vs-complex at
        # runtime rather than trusting the task's pre-assigned label —
        # that's what makes this routing rather than a lookup table.
        routed_complexity, classify_result = classify_complexity(self.provider, task)
        ledger.record(self.agent_id, classify_result, call_type="router")
        routed_model = MODEL_FOR_COMPLEXITY[routed_complexity]
        if routed_complexity != complexity:
            event_log.add(
                "router",
                f"agent {self.agent_id} task routed to '{routed_complexity}' ({routed_model}), "
                f"differs from dataset label '{complexity}'",
            )

        task_text = f"{task} Answer in one short sentence." if self.strategy == "efficient" else task
        messages, compressed_history, report = context_compressor.build_messages(self.agent_id, self.history, task_text)

        result = self.provider.complete(routed_model, messages, max_tokens=config.MAX_OUTPUT_TOKENS[routed_complexity])
        rec: CallRecord = ledger.record(self.agent_id, result, call_type="work")
        self.tokens_used_total += rec.total_tokens

        # Persist the (already-compressed) history plus this new turn for next time.
        self.history = compressed_history + [
            {"role": "user", "content": task_text},
            {"role": "assistant", "content": result.text},
        ]

        if rec.error is None:
            semantic_cache.store(task, result.text, routed_model, rec.input_tokens, rec.output_tokens, rec.cost_usd)
            tokens_saved_by_compression = max(0, report.estimated_uncompressed_tokens - report.actual_sent_tokens)
            # Compression targets the prompt side, so price the saving at
            # this model's input rate.
            cost_saved_by_compression = tokens_saved_by_compression * config.PRICING_USD_PER_TOKEN[routed_model]["input"]
            savings.add_compression_saving(tokens_saved_by_compression, cost_saved_by_compression)
            if routed_model == config.MODEL_SIMPLE:
                # Counterfactual: what this exact call would have cost on the
                # big model, same real token counts, priced at its rate.
                naive_cost = cost_of(config.MODEL_COMPLEX, rec.input_tokens, rec.output_tokens)
                savings.add_routing_cost_saved(max(0.0, naive_cost - rec.cost_usd))

        self.pending_task = None
        return AgentRunResult(
            agent_id=self.agent_id, task=task, complexity=routed_complexity, model=routed_model,
            input_tokens=rec.input_tokens, output_tokens=rec.output_tokens, total_tokens=rec.total_tokens,
            cost_usd=rec.cost_usd, latency_s=rec.latency_s, error=rec.error,
            cache_hit=False, cache_similarity=cache_result.similarity,
            compression_ratio=round(report.ratio, 2), history_summarized=report.history_summarized,
        )


_agent_index_counter = itertools.count(0)  # reset by spawn_agents(); resize_swarm() continues it


def spawn_agents(n: int, provider: LLMProvider) -> list[SimAgent]:
    global _agent_index_counter
    n = max(1, min(n, config.MAX_NUM_AGENTS))
    strategies = config.AGENT_STRATEGIES  # honest, inflator, freeloader, efficient
    agents = [SimAgent(f"agent-{i:03d}", strategies[i % len(strategies)], provider) for i in range(n)]
    _agent_index_counter = itertools.count(n)
    event_log.add(
        "system",
        f"spawned {n} agents "
        f"({', '.join(f'{s}={sum(1 for a in agents if a.strategy == s)}' for s in strategies)})",
    )
    return agents


def resize_swarm(agents: list[SimAgent], target_n: int) -> dict:
    """Grows or shrinks the swarm toward target_n. Shrinking removes the
    lowest-reputation agents first — a defensible rule on its own, and it
    doubles as garbage collection for agents reputation already dropped."""
    global _agent_index_counter
    current_n = len(agents)
    added, removed = [], []

    if target_n > current_n and agents:
        provider = agents[0].provider
        strategies = config.AGENT_STRATEGIES
        for _ in range(target_n - current_n):
            idx = next(_agent_index_counter)
            new_agent = SimAgent(f"agent-{idx:03d}", strategies[idx % len(strategies)], provider)
            agents.append(new_agent)
            added.append(new_agent.agent_id)
    elif target_n < current_n:
        removable = sorted(agents, key=lambda a: reputation_ledger.get(a.agent_id))
        for a in removable[: current_n - target_n]:
            agents.remove(a)
            removed.append(a.agent_id)

    if added or removed:
        detail = []
        if added:
            detail.append(f"spawned {len(added)}")
        if removed:
            detail.append(f"killed {len(removed)} (lowest reputation first)")
        event_log.add(
            "swarm",
            f"resized swarm {current_n} -> {len(agents)} agents toward N*={target_n}: {', '.join(detail)}",
        )
    return {"added": added, "removed": removed, "new_n": len(agents)}


def run_round(agents: list[SimAgent]) -> list[AgentRunResult]:
    """Step-1-style plain round (no auction): every agent runs once. Kept
    for quick sanity checks of raw token accounting."""
    results = []
    for i, agent in enumerate(agents):
        task, complexity = SAMPLE_TASKS[i % len(SAMPLE_TASKS)]
        agent.pending_task = (task, complexity)
        results.append(agent.execute_pending())
    return results


def rounds_until_token_reset(totals: dict, now: float | None = None) -> int | None:
    """How many more auction rounds start before the provider's token limit
    is fully restored; None when no reset time has been seen (dry run, cold
    start). A round takes at least TICK_MIN_INTERVAL_SECONDS — longer when
    real calls run — so this over-counts the rounds left: it can miss the
    chance to release expiring headroom, but never releases it too early."""
    reset_at = totals.get("rate_limit_tokens_reset_at")
    if reset_at is None:
        return None
    seconds_left = max(0.0, reset_at - (time.time() if now is None else now))
    return max(1, math.ceil(seconds_left / config.TICK_MIN_INTERVAL_SECONDS))


def run_auction_round(agents: list[SimAgent]) -> dict:
    """The real flow: agents bid, the auction clears K of them off live
    rate-limit headroom, winners actually call the LLM, reputation updates
    off real vs. predicted tokens, losers requeue with rising urgency."""
    active_agents = [a for a in agents if reputation_ledger.is_active(a.agent_id)]
    for a in active_agents:
        a.ensure_task()
        if a.force_predicted_tokens is not None:
            # One-shot override, e.g. from /stress/flash-drain — applied
            # after ensure_task so it always wins even if a task is already
            # pending, then consumed immediately.
            a.current_prediction = a.force_predicted_tokens
            a.force_predicted_tokens = None

    bids = [
        Bid(
            agent_id=a.agent_id,
            predicted_tokens=a.current_prediction,
            urgency=a.current_urgency(),
            reputation=reputation_ledger.get(a.agent_id),
        )
        for a in active_agents
    ]

    totals = ledger.totals()
    capacity = compute_capacity(totals["rate_limit_remaining_requests"], totals["rate_limit_remaining_tokens"],
                                rounds_until_token_reset(totals))
    auction_result = run_auction(bids, capacity)

    by_id = {a.agent_id: a for a in active_agents}
    entries: list[BidRoundEntry] = []

    for outcome in auction_result.winners:
        agent = by_id[outcome.agent_id]
        run_result = agent.execute_pending()

        if run_result.cache_hit:
            # A cache hit never touches reputation: the prediction was never
            # actually tested against real execution (0 tokens were spent
            # because the answer was already known, not because the agent
            # mispredicted), so scoring it as a bad prediction would punish
            # agents for the cache doing its job.
            agent.status = "WON"
            agent.rounds_queued = 0
            entries.append(BidRoundEntry(
                agent_id=agent.agent_id, strategy=agent.strategy,
                predicted_tokens=outcome.predicted_tokens, urgency=outcome.urgency,
                reputation_before=outcome.reputation, bid_score=outcome.bid_score,
                status="WON", reason=outcome.reason + " (served from cache)",
                actual_tokens=0, accuracy=None,
                reputation_after=outcome.reputation, cost_usd=0.0,
                model=run_result.model, cache_hit=True, compression_ratio=1.0,
            ))
            continue

        rep_event = reputation_ledger.record_prediction_outcome(
            agent.agent_id, outcome.predicted_tokens, run_result.total_tokens
        )
        if rep_event.newly_dropped:
            agent.status = "DROPPED"
        elif rep_event.newly_flagged:
            agent.status = "FLAGGED"
        else:
            agent.status = "WON"
        agent.rounds_queued = 0
        entries.append(BidRoundEntry(
            agent_id=agent.agent_id, strategy=agent.strategy,
            predicted_tokens=outcome.predicted_tokens, urgency=outcome.urgency,
            reputation_before=outcome.reputation, bid_score=outcome.bid_score,
            status=agent.status, reason=outcome.reason,
            actual_tokens=run_result.total_tokens, accuracy=round(rep_event.accuracy, 3),
            reputation_after=rep_event.new_reputation, cost_usd=run_result.cost_usd,
            model=run_result.model, cache_hit=run_result.cache_hit,
            compression_ratio=run_result.compression_ratio,
        ))

    for outcome in auction_result.queued:
        agent = by_id[outcome.agent_id]
        agent.rounds_queued += 1
        agent.status = "QUEUED"
        entries.append(BidRoundEntry(
            agent_id=agent.agent_id, strategy=agent.strategy,
            predicted_tokens=outcome.predicted_tokens, urgency=outcome.urgency,
            reputation_before=outcome.reputation, bid_score=outcome.bid_score,
            status="QUEUED", reason=outcome.reason,
        ))

    for outcome in auction_result.rejected:
        agent = by_id[outcome.agent_id]
        agent.status = "REJECTED"
        entries.append(BidRoundEntry(
            agent_id=agent.agent_id, strategy=agent.strategy,
            predicted_tokens=outcome.predicted_tokens, urgency=outcome.urgency,
            reputation_before=outcome.reputation, bid_score=outcome.bid_score,
            status="REJECTED", reason=outcome.reason,
        ))

    dropped_ids = [a.agent_id for a in agents if reputation_ledger.is_active(a.agent_id) is False]
    for a in agents:
        if a.agent_id in dropped_ids:
            a.status = "DROPPED"

    # Swarm sizer: one tick per auction round. N* is re-derived only every
    # SWARM_RECOMPUTE_INTERVAL_TICKS ticks; current_n takes one step toward
    # it every round regardless, so the swarm visibly converges over time.
    swarm_sizer.maybe_recompute(ledger.avg_cost_per_real_call())
    target_n = swarm_sizer.step_target(len(agents))
    resize_info = resize_swarm(agents, target_n)

    metrics_point = metrics_history.record_tick(len(auction_result.winners), len(agents))

    return {
        "capacity": {
            "max_requests": capacity.max_requests,
            "max_tokens": capacity.max_tokens,
            "per_agent_cap_tokens": capacity.per_agent_cap_tokens,
            "remaining_requests_seen": capacity.remaining_requests_seen,
            "remaining_tokens_seen": capacity.remaining_tokens_seen,
            "source": capacity.source,
        },
        "entries": [e.__dict__ for e in entries],
        "won": len(auction_result.winners),
        "queued": len(auction_result.queued),
        "rejected": len(auction_result.rejected),
        "dropped_agents": dropped_ids,
        "ledger_totals": ledger.totals(),
        "cache_stats": semantic_cache.stats(),
        "savings": savings.totals(),
        "swarm": swarm_sizer.status(len(agents)),
        "resize": resize_info,
        "metrics_point": metrics_point,
    }
