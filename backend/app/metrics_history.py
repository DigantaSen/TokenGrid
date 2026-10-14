"""
Time series backing the dashboard's hero graph: cost per completed task,
with TokenGrid vs. a without-TokenGrid counterfactual.

"With TokenGrid" is real cumulative $ spent (from the ledger — actual Groq
usage, includes every mechanism's own overhead, e.g. the router's
classification calls).

"Without TokenGrid" is that same real spend, minus the router's own
overhead (a naive system wouldn't run a classifier at all), plus what cache
hits / compression / routing each measurably saved — i.e. what a naive
"always call the big model with the full uncompressed prompt, no cache"
system would have spent to produce the same completed work.
"""
import time
from collections import deque
from dataclasses import dataclass, field

from .accounting import ledger
from .event_log import event_log
from .savings import savings
from .semantic_cache import semantic_cache
from .swarm_sizer import swarm_sizer

MAX_HISTORY_POINTS = 300


@dataclass
class MetricsHistory:
    points: deque = field(default_factory=lambda: deque(maxlen=MAX_HISTORY_POINTS))
    spikes: list = field(default_factory=list)
    completed_tasks_cumulative: int = 0
    communication_tokens_cumulative: float = 0.0

    def mark_spike(self, reason: str):
        spike = {"tick": len(self.points), "reason": reason, "timestamp": time.time()}
        self.spikes.append(spike)
        event_log.add("stress", f"spike marker: {reason}")

    def record_tick(self, completed_this_tick: int, current_n: int) -> dict:
        self.completed_tasks_cumulative += max(0, completed_this_tick)
        # Communication tax accrues every tick as a function of swarm size,
        # whether or not any agent won a bid this round — coordination
        # overhead doesn't wait for work to happen.
        self.communication_tokens_cumulative += swarm_sizer.coordination_tokens_for_n(current_n)

        totals = ledger.totals()
        by_type = totals["by_call_type"]
        real_cost = totals["total_cost_usd"]
        router_cost = by_type["router"]["total_cost_usd"]
        work_tokens_cumulative = by_type["work"]["total_tokens"] + by_type["router"]["total_tokens"]

        sav = savings.totals()
        cache_stats = semantic_cache.stats()

        # What a naive system (no router, no cache, no compression) would
        # have spent: real spend, minus the router overhead that wouldn't
        # exist without TokenGrid, plus every mechanism's measured saving.
        counterfactual_cost = (
            (real_cost - router_cost)
            + cache_stats["cost_saved_usd"]
            + sav["compression_cost_saved_usd_est"]
            + sav["routing_cost_saved_usd"]
        )

        completed = max(1, self.completed_tasks_cumulative)
        point = {
            "tick": len(self.points) + 1,
            "timestamp": time.time(),
            "completed_tasks_cumulative": self.completed_tasks_cumulative,
            "real_cost_cumulative": round(real_cost, 6),
            "counterfactual_cost_cumulative": round(counterfactual_cost, 6),
            "cost_per_task_with": round(real_cost / completed, 8),
            "cost_per_task_without": round(counterfactual_cost / completed, 8),
            "cache_hit_rate": cache_stats["hit_rate"],
            "rate_limit_remaining_requests": totals["rate_limit_remaining_requests"],
            "rate_limit_remaining_tokens": totals["rate_limit_remaining_tokens"],
            "work_tokens_cumulative": work_tokens_cumulative,
            "communication_tokens_cumulative": round(self.communication_tokens_cumulative),
        }
        self.points.append(point)
        return point

    def reset(self):
        self.points.clear()
        self.spikes = []
        self.completed_tasks_cumulative = 0
        self.communication_tokens_cumulative = 0.0

    def series(self) -> dict:
        return {"points": list(self.points), "spikes": self.spikes}


metrics_history = MetricsHistory()
