"""
Token accounting layer.

Every LLM call in the system — real or dry-run — gets recorded here exactly
once, using the token counts the provider reported (never estimated). This is
the single source of truth the auction, reputation ledger, and dashboard will
all read from in later steps.
"""
import itertools
import time
from dataclasses import dataclass, field
from threading import Lock

from . import config
from .llm_provider import CompletionResult

_call_id_counter = itertools.count(1)


@dataclass
class CallRecord:
    id: int
    agent_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    timestamp: float
    remaining_requests: int | None
    remaining_tokens: int | None
    error: str | None = None
    call_type: str = "work"  # "work" | "router" | "cache" — for per-mechanism accounting
    tokens_reset_at: float | None = None  # wall-clock time the token limit is fully restored

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def cost_of(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = config.PRICING_USD_PER_TOKEN.get(model)
    if not rates:
        return 0.0
    return input_tokens * rates["input"] + output_tokens * rates["output"]


class TokenLedger:
    """In-memory, thread-safe ledger of every call made through TokenGrid."""

    def __init__(self):
        self._lock = Lock()
        self.records: list[CallRecord] = []

    def record(self, agent_id: str, result: CompletionResult, call_type: str = "work") -> CallRecord:
        cost = cost_of(result.model, result.input_tokens, result.output_tokens)
        now = time.time()
        rec = CallRecord(
            id=next(_call_id_counter),
            agent_id=agent_id,
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=cost,
            latency_s=result.latency_s,
            timestamp=now,
            remaining_requests=result.remaining_requests,
            remaining_tokens=result.remaining_tokens,
            error=result.error,
            call_type=call_type,
            tokens_reset_at=now + result.reset_tokens_s if result.reset_tokens_s is not None else None,
        )
        with self._lock:
            self.records.append(rec)
        return rec

    def avg_cost_per_real_call(self) -> float:
        """Average real $ cost of a non-cached, non-error work call — the
        swarm sizer's production-cost input. Falls back to a configured
        default before any real work call has happened yet."""
        with self._lock:
            real = [r for r in self.records if r.error is None and r.call_type == "work" and r.cost_usd > 0]
        if not real:
            return config.SWARM_DEFAULT_AVG_COST_PER_TASK
        return sum(r.cost_usd for r in real) / len(real)

    def totals(self) -> dict:
        with self._lock:
            records = list(self.records)
        ok = [r for r in records if r.error is None]
        total_input = sum(r.input_tokens for r in ok)
        total_output = sum(r.output_tokens for r in ok)
        total_cost = sum(r.cost_usd for r in ok)
        latest_remaining_requests = next(
            (r.remaining_requests for r in reversed(ok) if r.remaining_requests is not None),
            None,
        )
        latest_remaining_tokens = next(
            (r.remaining_tokens for r in reversed(ok) if r.remaining_tokens is not None),
            None,
        )
        latest_tokens_reset_at = next(
            (r.tokens_reset_at for r in reversed(ok) if r.tokens_reset_at is not None),
            None,
        )
        by_type = {}
        for call_type in ("work", "router", "cache"):
            subset = [r for r in ok if r.call_type == call_type]
            by_type[call_type] = {
                "calls": len(subset),
                "total_tokens": sum(r.total_tokens for r in subset),
                "total_cost_usd": round(sum(r.cost_usd for r in subset), 6),
            }

        return {
            "calls": len(records),
            "calls_ok": len(ok),
            "calls_errored": len(records) - len(ok),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cost_usd": round(total_cost, 6),
            "rate_limit_remaining_requests": latest_remaining_requests,
            "rate_limit_remaining_tokens": latest_remaining_tokens,
            "rate_limit_tokens_reset_at": latest_tokens_reset_at,
            "by_call_type": by_type,
        }

    def per_agent(self, agent_id: str) -> dict:
        with self._lock:
            records = [r for r in self.records if r.agent_id == agent_id]
        ok = [r for r in records if r.error is None]
        return {
            "agent_id": agent_id,
            "calls": len(records),
            "total_tokens": sum(r.total_tokens for r in ok),
            "total_cost_usd": round(sum(r.cost_usd for r in ok), 6),
        }

    def recent(self, n: int = 20) -> list[CallRecord]:
        with self._lock:
            return list(self.records[-n:])


# One process-wide ledger for now (in-memory only, per the constraints).
ledger = TokenLedger()
