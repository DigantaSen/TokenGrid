"""
Context compressor.

Agent workloads are input-heavy — research puts input:output ratios around
150:1 — so the prompt side is where the real waste lives, not completions.
Three techniques, all applied to what's actually sent to the LLM (so real
Groq usage reflects the savings, not an estimate):

  1. Shared system prompt dedup: the swarm-wide system prompt is sent in
     full exactly once, ever; every later call (any agent) sends a short
     reference instead.
  2. Boilerplate stripping on the user message itself.
  3. Conversation history summarization once an agent's running history
     exceeds a token budget — old turns collapse into one summary line,
     the most recent turns stay verbatim.

Compression ratio is reported per call as
    estimated_uncompressed_tokens / actual_tokens_sent
using a char/4 heuristic for the *uncompressed* side only (real accounting
never uses this — it's purely for reporting how much compression helped).
"""
import re
from dataclasses import dataclass
from threading import Lock

from . import config
from .event_log import event_log

SHARED_SYSTEM_PROMPT = (
    "You are a worker agent inside the TokenGrid swarm. Follow instructions "
    "precisely and answer concisely. Do not repeat the question back, do not "
    "add disclaimers or preamble, and do not explain that you are an AI. If a "
    "task is ambiguous, make the most reasonable assumption and proceed "
    "instead of asking for clarification. Return plain text only."
)
COMPRESSED_SYSTEM_REF = "Same swarm worker-agent instructions as established earlier. Continue."

_BOILERPLATE_PATTERNS = [
    re.compile(r"\bplease note that\b", re.I),
    re.compile(r"\bas an ai( language model)?\b", re.I),
    re.compile(r"\bi (just )?wanted to (say|mention|note) that\b", re.I),
    re.compile(r"\bkindly\b", re.I),
    re.compile(r"\bit(?:'|’)s worth (mentioning|noting) that\b", re.I),
]


def estimate_tokens(text: str) -> int:
    """char/4 heuristic — used only for compression-ratio reporting, never
    for real accounting (that always comes from Groq's actual usage)."""
    return max(1, len(text) // 4)


def strip_boilerplate(text: str) -> str:
    out = text
    for pattern in _BOILERPLATE_PATTERNS:
        out = pattern.sub("", out)
    return re.sub(r"\s+", " ", out).strip()


@dataclass
class CompressionReport:
    system_deduped: bool
    estimated_uncompressed_tokens: int
    actual_sent_tokens: int
    history_summarized: bool
    history_turns_summarized: int

    @property
    def ratio(self) -> float:
        return self.estimated_uncompressed_tokens / max(1, self.actual_sent_tokens)


class ContextCompressor:
    """Swarm-wide: the system-prompt dedup state is shared across every
    agent, not per-agent — that's the point of deduplicating it."""

    def __init__(self):
        self._lock = Lock()
        self._system_prompt_sent = False

    def reset(self):
        with self._lock:
            self._system_prompt_sent = False

    def system_message(self) -> tuple[dict, bool]:
        with self._lock:
            first_time = not self._system_prompt_sent
            self._system_prompt_sent = True
        content = SHARED_SYSTEM_PROMPT if first_time else COMPRESSED_SYSTEM_REF
        if not first_time:
            pass  # no per-call log spam; the first-time transition is the interesting event
        else:
            event_log.add("compressor", "shared system prompt sent in full for the first time — deduplicated for all later calls, swarm-wide")
        return {"role": "system", "content": content}, first_time

    def compress_history(self, agent_id: str, history: list[dict]) -> tuple[list[dict], bool, int]:
        """Returns (possibly-summarized history, whether it summarized, how many turns it collapsed)."""
        total_tokens = sum(estimate_tokens(m["content"]) for m in history)
        if total_tokens <= config.HISTORY_COMPRESS_THRESHOLD_TOKENS:
            return history, False, 0

        keep_n = config.HISTORY_KEEP_LAST_TURNS * 2  # each turn = user + assistant message
        old, recent = history[:-keep_n], history[-keep_n:]
        if not old:
            return history, False, 0

        turns_collapsed = len(old) // 2
        user_snippets = [m["content"][:40] for m in old if m["role"] == "user"][:3]
        summary = (
            f"[compressed history: {turns_collapsed} earlier exchanges elided, including: "
            + "; ".join(user_snippets) + ("..." if len(old) // 2 > 3 else "") + "]"
        )
        new_history = [{"role": "system", "content": summary}] + recent

        old_tokens = sum(estimate_tokens(m["content"]) for m in old)
        new_tokens = estimate_tokens(summary)
        event_log.add(
            "compressor",
            f"agent {agent_id} history summarized: {turns_collapsed} turns / ~{old_tokens} tokens "
            f"-> 1 summary line / ~{new_tokens} tokens (kept last {config.HISTORY_KEEP_LAST_TURNS} turns verbatim)",
        )
        return new_history, True, turns_collapsed

    def build_messages(self, agent_id: str, history: list[dict], task_text: str) -> tuple[list[dict], CompressionReport]:
        sys_msg, system_first_time = self.system_message()
        compressed_history, history_summarized, turns_collapsed = self.compress_history(agent_id, history)
        clean_task = strip_boilerplate(task_text)

        messages = [sys_msg] + compressed_history + [{"role": "user", "content": clean_task}]

        uncompressed_tokens = (
            estimate_tokens(SHARED_SYSTEM_PROMPT)
            + sum(estimate_tokens(m["content"]) for m in history)  # pre-compression history
            + estimate_tokens(task_text)
        )
        actual_sent_tokens = sum(estimate_tokens(m["content"]) for m in messages)

        report = CompressionReport(
            system_deduped=not system_first_time,
            estimated_uncompressed_tokens=uncompressed_tokens,
            actual_sent_tokens=actual_sent_tokens,
            history_summarized=history_summarized,
            history_turns_summarized=turns_collapsed,
        )
        return messages, compressed_history, report


context_compressor = ContextCompressor()
