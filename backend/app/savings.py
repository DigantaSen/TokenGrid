"""
Aggregates savings by source, kept separate per the spec: cache and routing
savings must be attributable independently, not folded into one number.

Compression savings are token-denominated: compression doesn't change price,
it changes how many tokens get sent, so an estimated token count (char/4
heuristic on the uncompressed side, since there's no real second call to
diff against) is the right unit.

Routing savings are dollar-denominated: routing doesn't reduce token count
(the same call happens either way), it changes *price* by picking the
cheaper model when the task allows it — so the saving is measured in real
$ against a naive "always use the big model" baseline, using the actual
token counts the routed call really cost.
"""
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class SavingsTracker:
    _lock: Lock = field(default_factory=Lock)
    compression_tokens_saved_est: int = 0
    compression_cost_saved_usd_est: float = 0.0
    routing_cost_saved_usd: float = 0.0

    def add_compression_saving(self, tokens_est: int, cost_usd_est: float = 0.0):
        with self._lock:
            self.compression_tokens_saved_est += max(0, tokens_est)
            self.compression_cost_saved_usd_est += max(0.0, cost_usd_est)

    def add_routing_cost_saved(self, usd: float):
        with self._lock:
            self.routing_cost_saved_usd += max(0.0, usd)

    def reset(self):
        with self._lock:
            self.compression_tokens_saved_est = 0
            self.compression_cost_saved_usd_est = 0.0
            self.routing_cost_saved_usd = 0.0

    def totals(self) -> dict:
        with self._lock:
            return {
                "compression_tokens_saved_est": self.compression_tokens_saved_est,
                "compression_cost_saved_usd_est": round(self.compression_cost_saved_usd_est, 6),
                "routing_cost_saved_usd": round(self.routing_cost_saved_usd, 6),
            }


savings = SavingsTracker()
