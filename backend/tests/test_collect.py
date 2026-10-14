"""Trace collection on the free tier: wait out per-minute limits, stop cleanly at the daily one."""
import threading

from app.llm_provider import CompletionResult
from app.tasks import Task
from experiments.collect_traces import collect, is_daily_limit, load_traces, run_groq

TPM = ("429 rate-limited: Error code: 429 - Rate limit reached for model `openai/gpt-oss-120b` "
       "on tokens per minute (TPM): Limit 8000, Used 7900, Requested 500. Please try again in 3s.")
TPD = ("429 rate-limited: Error code: 429 - Rate limit reached for model `openai/gpt-oss-120b` "
       "on tokens per day (TPD): Limit 200000, Used 199800, Requested 500. Please try again in 6m.")


class FakeProvider:
    """Answers correctly; returns scripted errors on chosen call numbers."""

    def __init__(self, errors: dict[int, str]):
        self.errors, self.calls, self.lock = errors, 0, threading.Lock()

    def complete(self, model, messages, max_tokens=None):
        with self.lock:
            self.calls += 1
            n = self.calls
        if n in self.errors:
            return CompletionResult("", 0, 0, model, 0.0, error=self.errors[n])
        return CompletionResult("thinking...\nAnswer: 4", 50, 100, model, 0.1)


def tasks(n):
    return [Task(f"gsm8k-{i}", "gsm8k", "What is 2+2?", "4", "numeric") for i in range(n)]


def test_daily_vs_minute_limit_detection():
    assert is_daily_limit(TPD) and not is_daily_limit(TPM)
    assert not is_daily_limit(None) and not is_daily_limit("connection reset")


def test_per_minute_limit_is_retried_and_reply_is_stored():
    row = run_groq(FakeProvider({1: TPM, 2: TPM}), tasks(1)[0], "openai/gpt-oss-120b",
                   threading.Event(), base_delay=0.01)
    assert row["error"] is None and row["correct"]
    assert row["response"] == "thinking...\nAnswer: 4"


def test_daily_limit_stops_the_run_without_error_rows(tmp_path):
    out = tmp_path / "traces.jsonl"
    provider, stop = FakeProvider({6: TPD}), threading.Event()
    fn = lambda t: run_groq(provider, t, "openai/gpt-oss-120b", stop, base_delay=0.01)  # noqa: E731
    stats = collect(tasks(40), fn, out, workers=1, stop=stop)
    rows = load_traces(out)
    assert stats["daily_limit"] and stats["errors"] == 0
    assert len(rows) == stats["ok"] == 5            # the 5 calls before the limit were kept
    assert all(r["error"] is None for r in rows)    # nothing recorded as a failure
    assert provider.calls == 6                      # no calls made after the limit

