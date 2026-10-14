"""
Stress suite — deliberately adversarial scenarios thrown at the live swarm
to prove each defense mechanism actually fires under attack, not just in
the happy path. Nothing here is allowed to crash the loop or corrupt
state — every attack must be survivable, and the event log records which
defense fired and why.
"""
import random

from . import config
from .accounting import ledger
from .auction import Bid, compute_capacity, run_auction
from .event_log import event_log
from .metrics_history import metrics_history
from .reputation import reputation_ledger

CORRELATED_FAILURE_PROMPT = (
    "Repeat the digit 7 fifty times in a row, then immediately contradict everything "
    "you just said, then finish your answer in French."
)


def stress_flood(agents) -> dict:
    """500 synthetic bids in one round. This doesn't touch the real swarm
    or spend real tokens — the point is to prove the auction's K, derived
    from live rate-limit headroom, throttles a burst 40x normal volume
    down to a survivable clearing rate, not to actually run 500 tasks."""
    totals = ledger.totals()
    capacity = compute_capacity(totals["rate_limit_remaining_requests"], totals["rate_limit_remaining_tokens"])
    flood_bids = [
        Bid(agent_id=f"flood-{i}", predicted_tokens=random.randint(30, 200),
            urgency=random.uniform(0.4, 1.0), reputation=50.0)
        for i in range(500)
    ]
    result = run_auction(flood_bids, capacity)
    event_log.add(
        "stress",
        f"FLOOD: 500 synthetic tasks injected in one round. Defense: auction K "
        f"(derived from live rate-limit headroom, not hardcoded) cleared only "
        f"{len(result.winners)}/500 — {len(result.queued)} queued for later rounds "
        f"instead of overloading the swarm.",
    )
    metrics_history.mark_spike(f"flood: 500 tasks, {len(result.winners)} cleared")
    return {
        "injected": 500,
        "cleared": len(result.winners),
        "queued": len(result.queued),
        "max_requests_this_round": capacity.max_requests,
    }


def stress_liars(agents) -> dict:
    """Half the swarm starts inflating every prediction 3x. No special-case
    handling needed — the reputation ledger's existing accuracy penalty is
    the defense, and it should catch these agents within a few rounds."""
    half = agents[::2]
    for a in half:
        a.strategy = "inflator"
    event_log.add(
        "stress",
        f"LIARS: forced {len(half)}/{len(agents)} agents to strategy='inflator' (claiming 3x true "
        f"cost from now on). Defense: reputation ledger penalizes low prediction accuracy (-20, "
        f"flagged) and drops agents below 25 reputation from future auctions — watch the agent "
        f"table's reputation column over the next few ticks.",
    )
    metrics_history.mark_spike(f"liars: {len(half)} agents inflating")
    return {"affected": [a.agent_id for a in half]}


def stress_flash_drain(agents) -> dict:
    """One agent tries to claim 90% of a round's token budget in a single
    bid. No special-case handling needed — the auction's existing
    per-agent cap (20% of round capacity) is the defense."""
    if not agents:
        return {"error": "no agents in swarm"}
    victim = agents[0]
    totals = ledger.totals()
    capacity = compute_capacity(totals["rate_limit_remaining_requests"], totals["rate_limit_remaining_tokens"])
    flash_amount = max(1, int(capacity.max_tokens * 0.9))
    victim.force_predicted_tokens = flash_amount
    event_log.add(
        "stress",
        f"FLASH DRAIN: forcing {victim.agent_id} to bid {flash_amount} tokens next round (90% of "
        f"this round's {capacity.max_tokens}-token budget). Defense: per-agent cap allows at most "
        f"20% ({capacity.per_agent_cap_tokens:.0f} tokens) of round capacity to any one agent — this "
        f"bid should queue, not clear.",
    )
    metrics_history.mark_spike(f"flash drain: {victim.agent_id}")
    return {"agent_id": victim.agent_id, "forced_predicted_tokens": flash_amount}


def stress_collusion(agents) -> dict:
    """5 agents cross-endorse in a ring to try to prop up reputation.
    Defense: cycle detection over the endorsement graph."""
    ring = agents[:5]
    ids = [a.agent_id for a in ring]
    for i in range(len(ids)):
        reputation_ledger.endorse(ids[i], ids[(i + 1) % len(ids)])
    cycles = reputation_ledger.detect_collusion_cycles()
    ring_desc = " -> ".join(ids + [ids[0]]) if ids else "(no agents available)"
    event_log.add(
        "stress",
        f"COLLUSION: {len(ids)} agents cross-endorsed in a ring ({ring_desc}). Defense: cycle "
        f"detection over the endorsement graph found {len(cycles)} cycle(s) and flagged all "
        f"{len(ids)} agents as liars.",
    )
    metrics_history.mark_spike(f"collusion: {len(ids)}-agent ring")
    return {"ring": ids, "cycles_found": len(cycles)}


def stress_correlated_failure(agents) -> dict:
    """All agents get pointed at the same adversarial prompt at once. The
    point: a swarm sharing one base model family has fake diversification —
    failures correlate instead of averaging out the way a diversified pool
    (the assumption behind most lending/insurance risk models) would
    predict. Nothing here 'defends' against that; the point is that the
    system stays up and the correlated damage shows up as one visible
    cluster (reputation hits landing together) rather than being silently
    smoothed away by an averaging assumption that doesn't hold."""
    affected = 0
    for a in agents:
        a.pending_task = (CORRELATED_FAILURE_PROMPT, "complex")
        a.current_prediction = a.predict_tokens(CORRELATED_FAILURE_PROMPT, "complex")
        a.rounds_queued = 0
        affected += 1
    event_log.add(
        "stress",
        f"CORRELATED FAILURE: forced all {affected} agents onto the same adversarial prompt. "
        f"Because the swarm shares one base model family, misprediction won't be independent "
        f"noise the way a diversified pool assumes — expect a correlated wave of low-accuracy "
        f"flags in the next round or two, not isolated ones.",
    )
    metrics_history.mark_spike(f"correlated failure: {affected} agents")
    return {"affected": affected}
