"""
Token auction — the novel mechanism.

Agents bid for the right to make an LLM call this round. bid_score rewards
high reputation and high urgency but *penalizes* claiming a large token
budget:

    bid_score = (reputation * urgency) / predicted_tokens

That division is load-bearing: inflating predicted_tokens can only ever
shrink your own score, so truthfully reporting your expected cost is the
dominant strategy. Under-reporting looks tempting (it inflates your score)
but the reputation ledger punishes the resulting low prediction accuracy —
see reputation.py. Don't "fix" the formula to remove either incentive.

K — how many bids clear this round — is derived from live rate-limit
headroom (Groq's x-ratelimit-remaining-* headers), never hardcoded. When the
swarm saturates the limit, this is the mechanism deciding who proceeds.
"""
import math
from dataclasses import dataclass, field

from . import config
from .event_log import event_log


@dataclass
class Bid:
    agent_id: str
    predicted_tokens: int
    urgency: float       # 0..1, how much this agent needs to run now
    reputation: float    # 0..100, from the reputation ledger


@dataclass
class BidOutcome:
    agent_id: str
    predicted_tokens: int
    urgency: float
    reputation: float
    bid_score: float
    status: str   # "WON" | "QUEUED" | "REJECTED"
    reason: str


@dataclass
class RoundCapacity:
    max_requests: int          # K: max number of bids that can clear this round
    max_tokens: int            # total predicted-token budget for the round
    per_agent_cap_tokens: float
    remaining_requests_seen: int
    remaining_tokens_seen: int
    source: str  # "live" | "fallback" — was this derived from a real header?


@dataclass
class AuctionRoundResult:
    capacity: RoundCapacity
    winners: list[BidOutcome] = field(default_factory=list)
    queued: list[BidOutcome] = field(default_factory=list)
    rejected: list[BidOutcome] = field(default_factory=list)


def compute_capacity(remaining_requests: int | None, remaining_tokens: int | None,
                     rounds_until_reset: int | None = None) -> RoundCapacity:
    """K comes from headroom. If we've never seen a real rate-limit header
    yet (cold start), fall back to a configured baseline rather than an
    arbitrary hardcoded K — the first real call replaces this immediately.

    Each round normally spends at most ROUND_TOKEN_FRACTION of the remaining
    token headroom, holding the rest back as a safety margin for later
    rounds. But headroom still unspent when the limit resets is simply lost,
    so in the last round before the reset (rounds_until_reset == 1) holding
    anything back buys nothing: that round may spend all of it. Measured
    effect: rate-limit utilization 88% -> ~98% with liar protection unchanged
    (docs/EVALUATION_GUIDE.md). The per-agent cap still applies, so no single
    bid can take more than PER_AGENT_CAP_FRACTION of it. rounds_until_reset
    of None means the reset time is unknown -> the plain fractional rule."""
    source = "live"
    if remaining_requests is None:
        remaining_requests = config.FALLBACK_REMAINING_REQUESTS
        source = "fallback"
    if remaining_tokens is None:
        remaining_tokens = config.FALLBACK_REMAINING_TOKENS
        source = "fallback"

    expiring = rounds_until_reset is not None and rounds_until_reset <= 1
    token_fraction = 1.0 if expiring else config.ROUND_TOKEN_FRACTION

    max_requests = max(1, math.floor(remaining_requests * config.ROUND_REQUEST_FRACTION))
    max_tokens = max(1, math.floor(remaining_tokens * token_fraction))
    per_agent_cap = max_tokens * config.PER_AGENT_CAP_FRACTION

    event_log.add(
        "auction",
        f"round capacity derived from {source} headroom: "
        f"remaining_requests={remaining_requests} -> K={max_requests} bids "
        f"({config.ROUND_REQUEST_FRACTION:.0%} of headroom), "
        f"remaining_tokens={remaining_tokens} -> token budget={max_tokens} "
        f"({token_fraction:.0%} of headroom"
        f"{' — last round before the limit resets, so nothing is held back' if expiring else ''}), "
        f"per-agent cap={per_agent_cap:.0f} tokens ({config.PER_AGENT_CAP_FRACTION:.0%} of round)",
        source=source,
    )

    return RoundCapacity(
        max_requests=max_requests,
        max_tokens=max_tokens,
        per_agent_cap_tokens=per_agent_cap,
        remaining_requests_seen=remaining_requests,
        remaining_tokens_seen=remaining_tokens,
        source=source,
    )


def score(bid: Bid) -> float:
    if bid.predicted_tokens <= 0:
        return -math.inf
    return (bid.reputation * bid.urgency) / bid.predicted_tokens


def run_auction(bids: list[Bid], capacity: RoundCapacity, key=score) -> AuctionRoundResult:
    """`key` ranks bids (highest first). The live system always uses `score`;
    experiments pass alternative keys (FIFO, random, ...) so baseline
    schedulers clear through the exact same K / budget / per-agent-cap rules
    and only the ranking differs."""
    result = AuctionRoundResult(capacity=capacity)
    if not bids:
        return result

    scored = sorted(bids, key=key, reverse=True)
    used_requests = 0
    used_tokens = 0

    for bid in scored:
        s = key(bid)

        if bid.predicted_tokens <= 0:
            outcome = BidOutcome(bid.agent_id, bid.predicted_tokens, bid.urgency, bid.reputation, s,
                                  "REJECTED", "invalid bid: predicted_tokens must be > 0")
            result.rejected.append(outcome)
            event_log.add("auction", f"agent {bid.agent_id} bid REJECTED: predicted_tokens={bid.predicted_tokens} invalid")
            continue

        if used_requests >= capacity.max_requests:
            outcome = BidOutcome(bid.agent_id, bid.predicted_tokens, bid.urgency, bid.reputation, s,
                                  "QUEUED", f"round full: K={capacity.max_requests} slots already taken")
            result.queued.append(outcome)
            continue

        if bid.predicted_tokens > capacity.per_agent_cap_tokens:
            outcome = BidOutcome(bid.agent_id, bid.predicted_tokens, bid.urgency, bid.reputation, s,
                                  "QUEUED",
                                  f"exceeds per-agent cap: predicted {bid.predicted_tokens} > "
                                  f"{capacity.per_agent_cap_tokens:.0f} (20% of round capacity)")
            result.queued.append(outcome)
            event_log.add(
                "auction",
                f"agent {bid.agent_id} QUEUED: predicted {bid.predicted_tokens} tokens exceeds "
                f"20% per-agent cap ({capacity.per_agent_cap_tokens:.0f}) — score={s:.3f}",
            )
            continue

        if used_tokens + bid.predicted_tokens > capacity.max_tokens:
            outcome = BidOutcome(bid.agent_id, bid.predicted_tokens, bid.urgency, bid.reputation, s,
                                  "QUEUED",
                                  f"token budget exhausted: {used_tokens}+{bid.predicted_tokens} > {capacity.max_tokens}")
            result.queued.append(outcome)
            continue

        # Accepted.
        used_requests += 1
        used_tokens += bid.predicted_tokens
        outcome = BidOutcome(bid.agent_id, bid.predicted_tokens, bid.urgency, bid.reputation, s,
                              "WON", f"cleared rank {used_requests}/{capacity.max_requests}, score={s:.3f}")
        result.winners.append(outcome)

    if result.queued or result.rejected:
        event_log.add(
            "auction",
            f"round cleared {len(result.winners)}/{len(bids)} bids "
            f"(K={capacity.max_requests}, token budget={capacity.max_tokens}); "
            f"{len(result.queued)} queued for next round, {len(result.rejected)} rejected",
        )
    else:
        event_log.add("auction", f"round cleared all {len(result.winners)} bids — no contention this round")

    return result
