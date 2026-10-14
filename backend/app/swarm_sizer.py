"""
Swarm sizer: finds N*, the swarm size minimizing total cost.

    total_cost(N) = production_cost(N) + coordination_cost(N)

production_cost(N) models specialization gains on a fixed amount of
exogenous task demand — more agents means narrower, more efficient work per
agent, so cost per task falls with N (a decreasing power law anchored to
the swarm's live measured average cost per task, not a made-up constant).

coordination_cost(N) is the communication tax: every pair of agents costs
something to keep synchronized, so it grows O(N^2) — modeled directly as
coordination tokens per pair, priced like any other tokens.

N* is found by numeric search over the curve rather than a closed-form
derivative — simpler, robust to whatever the live cost inputs are, and it
hands back the full curve for free, which is exactly what the dashboard's
U-shaped chart needs.
"""
from dataclasses import dataclass, field

from . import config
from .event_log import event_log


@dataclass
class SwarmSizer:
    tick_count: int = 0
    optimal_n: int = config.DEFAULT_NUM_AGENTS
    cost_curve: list = field(default_factory=list)  # [{n, production_cost, coordination_cost, total_cost}, ...]

    def _comm_rate_per_token(self) -> float:
        rates = config.PRICING_USD_PER_TOKEN[config.MODEL_SIMPLE]
        return (rates["input"] + rates["output"]) / 2

    def coordination_tokens_for_n(self, n: int) -> float:
        """Pure token count (no pricing) — the communication tax in tokens,
        for the dashboard's work-vs-communication split."""
        pairs = n * (n - 1) / 2
        return pairs * config.SWARM_COMM_TOKENS_PER_PAIR

    def _compute_curve(self, avg_cost_per_task: float) -> list[dict]:
        comm_rate = self._comm_rate_per_token()
        curve = []
        for n in range(1, config.MAX_NUM_AGENTS + 1):
            production = (
                config.SWARM_TOTAL_TASKS_PER_TICK
                * avg_cost_per_task
                * (config.SWARM_BASELINE_N / n) ** config.SWARM_SPECIALIZATION_ALPHA
            )
            pairs = n * (n - 1) / 2
            coordination = pairs * config.SWARM_COMM_TOKENS_PER_PAIR * comm_rate
            curve.append({
                "n": n,
                "production_cost": production,
                "coordination_cost": coordination,
                "total_cost": production + coordination,
            })
        return curve

    def maybe_recompute(self, avg_cost_per_task: float) -> bool:
        """Advances the tick clock; re-derives N* only every
        SWARM_RECOMPUTE_INTERVAL_TICKS ticks (or on the very first tick)."""
        self.tick_count += 1
        due = self.tick_count == 1 or self.tick_count % config.SWARM_RECOMPUTE_INTERVAL_TICKS == 0
        if not due:
            return False

        curve = self._compute_curve(avg_cost_per_task)
        best = min(curve, key=lambda c: c["total_cost"])
        old_optimal = self.optimal_n
        self.optimal_n = best["n"]
        self.cost_curve = curve

        if old_optimal != self.optimal_n:
            event_log.add(
                "swarm",
                f"N* recomputed at tick {self.tick_count}: optimal swarm size {old_optimal} -> "
                f"{self.optimal_n} (avg cost/task=${avg_cost_per_task:.6f}, "
                f"min total cost=${best['total_cost']:.6f}/tick)",
            )
        else:
            event_log.add(
                "swarm",
                f"N* recomputed at tick {self.tick_count}: still {self.optimal_n} "
                f"(avg cost/task=${avg_cost_per_task:.6f})",
            )
        return True

    def step_target(self, current_n: int) -> int:
        """One incremental step toward N* — the swarm converges over
        several ticks rather than jumping straight there."""
        if current_n == self.optimal_n:
            return current_n
        step = config.SWARM_STEP_SIZE
        if current_n < self.optimal_n:
            return min(self.optimal_n, current_n + step)
        return max(self.optimal_n, current_n - step)

    def status(self, current_n: int) -> dict:
        at_current = next((c for c in self.cost_curve if c["n"] == current_n), None)
        at_optimal = next((c for c in self.cost_curve if c["n"] == self.optimal_n), None)
        return {
            "current_n": current_n,
            "optimal_n": self.optimal_n,
            "tick_count": self.tick_count,
            "cost_at_current_n": at_current,
            "cost_at_optimal_n": at_optimal,
            "curve": self.cost_curve,
        }

    def reset(self):
        self.tick_count = 0
        self.optimal_n = config.DEFAULT_NUM_AGENTS
        self.cost_curve = []


swarm_sizer = SwarmSizer()
