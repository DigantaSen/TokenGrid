"""
Step 1 of the TokenGrid evaluation: run every task through the real model
once and log what it actually cost and whether it was right.

Every experiment after this is a trace-driven simulation that replays these
logs through TokenGrid's auction — so policies, swarm sizes and attacks can
be compared thousands of times without another paid call, and every
comparison sees identical task costs.

    python -m experiments.collect_traces --n 500     # real Groq calls (needs GROQ_API_KEY)
    python -m experiments.collect_traces --mock      # synthetic traces, pipeline testing only

Resumable: tasks already logged without error are skipped. On Groq's free
tier the daily token limit (200K for gpt-oss-120b, ~500 tasks) is what runs
out; when it does, the run stops cleanly and the next run continues.
"""
import argparse
import hashlib
import json
import math
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from pathlib import Path

from app import config
from app.accounting import cost_of
from app.llm_provider import GroqProvider
from app.tasks import DATASETS, Task, grade, load_tasks

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
# Large enough that a correct answer is rarely cut off; truncation would
# otherwise show up as "wrong" and as artificially uniform token counts.
TRACE_MAX_TOKENS = 2048


class DailyLimitReached(Exception):
    """The provider's per-day quota is used up - retrying today is pointless."""


def is_daily_limit(error: str | None) -> bool:
    e = (error or "").lower()
    return "429" in e and ("per day" in e or "(tpd)" in e or "(rpd)" in e)


def traces_path(mock: bool) -> Path:
    return RESULTS_DIR / ("mock" if mock else "groq") / "traces.jsonl"


def load_traces(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _row(task: Task, model: str, source: str, text: str, input_tokens: int, output_tokens: int,
         latency_s: float, error: str | None) -> dict:
    pred, correct = grade(task, text) if error is None else (None, False)
    return {
        "task_id": task.task_id, "dataset": task.dataset, "text": task.text, "gold": task.gold,
        "model": model, "source": source, "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens, "cost_usd": cost_of(model, input_tokens, output_tokens),
        "response": text,  # full model output, so grading can be re-audited later
        "pred": pred, "correct": bool(correct), "latency_s": latency_s, "error": error,
        "max_tokens": TRACE_MAX_TOKENS, "timestamp": time.time(),
    }


def run_groq(provider, task: Task, model: str, stop: threading.Event, max_retries: int = 6,
             base_delay: float = 2.0) -> dict:
    delay = base_delay
    for attempt in range(max_retries + 1):
        if stop.is_set():
            raise DailyLimitReached("stopped: daily limit already reached")
        r = provider.complete(model, task.messages(), max_tokens=TRACE_MAX_TOKENS)
        if is_daily_limit(r.error):
            stop.set()  # here, not in the main thread: other workers must stop calling immediately
            raise DailyLimitReached(r.error)
        if r.error and "429" in r.error and attempt < max_retries:
            stop.wait(delay)  # per-minute limit: back off and retry (wakes early if the run is stopping)
            delay *= 2
            continue
        return _row(task, model, "groq", r.text, r.input_tokens, r.output_tokens, r.latency_s, r.error)


def run_mock(task: Task, model: str) -> dict:
    """Synthetic but structured: a latent per-task difficulty, partly visible
    in the text, drives both token usage (heavy-tailed, like reasoning
    models) and correctness — so a token predictor has real signal to find."""
    def u(salt: str) -> float:
        return int.from_bytes(hashlib.sha256(f"{task.task_id}|{model}|{salt}".encode()).digest()[:8], "big") / 2**64

    d = min(1.0, 0.5 * min(1.0, len(task.text) / 700) + 0.5 * u("d") + (0.15 if task.dataset == "gsm8k" else 0.0))
    noise = math.sqrt(-2 * math.log(max(u("n1"), 1e-12))) * math.cos(2 * math.pi * u("n2"))  # Box-Muller
    output_tokens = min(TRACE_MAX_TOKENS, int(math.exp(5.0 + 1.6 * d + 0.45 * noise)))
    correct = u("c") < 1 / (1 + math.exp(-(3.0 - 4.5 * d)))
    answer = task.gold if correct else ("B" if task.gold == "A" else "A") if task.kind == "mc" else f"{task.gold}1"
    return _row(task, model, "mock", f"[mock]\nAnswer: {answer}", len(task.text) // 4 + 40, output_tokens, 0.0, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=500, help="tasks per dataset")
    ap.add_argument("--model", default=config.MODEL_COMPLEX)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    out = traces_path(args.mock)
    out.parent.mkdir(parents=True, exist_ok=True)
    tasks = [t for name in DATASETS for t in load_tasks(name, args.n)]
    done = {(r["task_id"], r["model"]) for r in load_traces(out) if r["error"] is None}
    todo = [t for t in tasks if (t.task_id, args.model) not in done]
    print(f"{len(tasks)} tasks, {len(done)} already traced, {len(todo)} to run -> {out}")

    stop = threading.Event()  # set when the daily limit is hit; every worker checks it
    if args.mock:
        fn = lambda t: run_mock(t, args.model)  # noqa: E731
    else:
        if not config.GROQ_API_KEY:
            raise SystemExit("GROQ_API_KEY not set (backend/.env) - or pass --mock")
        provider = GroqProvider()
        fn = lambda t: run_groq(provider, t, args.model, stop)  # noqa: E731

    stats = collect(todo, fn, out, args.workers, stop)
    left = len(todo) - stats["ok"]
    if stats["daily_limit"]:
        print(f"Daily limit reached after {stats['ok']} tasks. Progress is saved - run the same command "
              f"again after the limit resets to continue ({left} tasks left).")
    elif stats["errors"]:
        print(f"{stats['errors']} tasks errored - rerun the same command to retry only those.")
    else:
        print(f"Done: {stats['ok']} tasks traced, nothing left to run.")


def collect(todo: list[Task], fn, out: Path, workers: int, stop: threading.Event) -> dict:
    """Runs fn over todo in parallel, appending each row as it completes. A
    DailyLimitReached from any worker stops the whole run: queued tasks are
    cancelled and in-flight ones finish without writing an error row."""
    stats = {"ok": 0, "errors": 0, "daily_limit": False}
    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, t) for t in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                row = fut.result()
            except (DailyLimitReached, CancelledError):
                if not stats["daily_limit"]:
                    stats["daily_limit"] = True
                    stop.set()
                    for other in futures:
                        other.cancel()
                continue
            f.write(json.dumps(row) + "\n")
            f.flush()
            if row["error"]:
                stats["errors"] += 1
                print(f"  error {row['task_id']}: {row['error'][:120]}")
            else:
                stats["ok"] += 1
            if i % 50 == 0:
                print(f"  {i}/{len(futures)} ({stats['ok']} ok, {stats['errors']} errors)")
    return stats


if __name__ == "__main__":
    main()
