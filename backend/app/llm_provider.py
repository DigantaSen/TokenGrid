"""
LLM provider adapter.

The rest of TokenGrid only ever talks to `LLMProvider.complete(model, messages)`.
Swapping Groq for Anthropic/OpenAI later means writing one new class here —
nothing upstream (auction, ledger, agents) should need to change.
"""
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from . import config


@dataclass
class CompletionResult:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    latency_s: float
    # Rate-limit headroom, when the provider exposes it. None in dry-run.
    remaining_requests: Optional[int] = None
    remaining_tokens: Optional[int] = None
    # Seconds until the token limit is fully restored (x-ratelimit-reset-tokens).
    reset_tokens_s: Optional[float] = None
    error: Optional[str] = None
    raw_headers: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, model: str, messages: list[dict], max_tokens: Optional[int] = None) -> CompletionResult:
        ...


class GroqProvider(LLMProvider):
    """Real calls to Groq via the OpenAI-compatible SDK. Token counts and
    rate-limit headroom come straight from the response — never estimated."""

    def __init__(self, api_key: str = None):
        self.client = OpenAI(
            api_key=api_key or config.GROQ_API_KEY,
            base_url=config.GROQ_BASE_URL,
        )

    def complete(self, model: str, messages: list[dict], max_tokens: Optional[int] = None) -> CompletionResult:
        start = time.monotonic()
        try:
            # with_raw_response so we can read the rate-limit headers Groq
            # returns alongside the parsed completion body. max_tokens caps
            # worst-case cost per call — some Groq models are reasoning
            # models that will happily emit thousands of tokens for an
            # open-ended prompt otherwise, which is exactly the runaway-cost
            # scenario the auction exists to prevent.
            kwargs = {"model": model, "messages": messages}
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            raw = self.client.chat.completions.with_raw_response.create(**kwargs)
            latency = time.monotonic() - start
            completion = raw.parse()
            headers = raw.headers
            usage = completion.usage

            return CompletionResult(
                text=completion.choices[0].message.content or "",
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                model=model,
                latency_s=latency,
                remaining_requests=_safe_int(headers.get("x-ratelimit-remaining-requests")),
                remaining_tokens=_safe_int(headers.get("x-ratelimit-remaining-tokens")),
                reset_tokens_s=parse_duration_s(headers.get("x-ratelimit-reset-tokens")),
                raw_headers={
                    k: v for k, v in headers.items() if k.lower().startswith("x-ratelimit")
                },
            )
        except Exception as e:
            latency = time.monotonic() - start
            status = getattr(getattr(e, "response", None), "status_code", None)
            return CompletionResult(
                text="",
                input_tokens=0,
                output_tokens=0,
                model=model,
                latency_s=latency,
                # Keep Groq's message on 429s: it names the limit that was hit
                # ("tokens per minute (TPM)" vs "tokens per day (TPD)"), and
                # only a per-minute limit is worth waiting out.
                error=f"429 rate-limited: {e}" if status == 429 else str(e),
            )


class DryRunProvider(LLMProvider):
    """No network calls at all. Fakes plausible token counts from message
    length so the whole pipeline (accounting, auction, dashboard) can run
    offline, deterministically-ish, on a laptop with no key and no wifi."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def complete(self, model: str, messages: list[dict], max_tokens: Optional[int] = None) -> CompletionResult:
        # Rough chars/4 ~ tokens heuristic, same rule of thumb most tokenizers
        # land near for English text.
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        input_tokens = max(1, prompt_chars // 4)

        is_complex = model == config.MODEL_COMPLEX
        base_output = self._rng.randint(80, 220) if is_complex else self._rng.randint(20, 60)
        output_tokens = min(base_output, max_tokens) if max_tokens else base_output

        text = f"[DRY_RUN {model}] simulated {output_tokens}-token response."

        # Fake a slowly-draining rate limit window so the dashboard has
        # something believable to show even offline.
        remaining_requests = self._rng.randint(20, 30)
        remaining_tokens = self._rng.randint(4000, 6000)

        time.sleep(0.05)  # keep the simulated latency non-zero but fast
        return CompletionResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            latency_s=0.05,
            remaining_requests=remaining_requests,
            remaining_tokens=remaining_tokens,
        )


def _safe_int(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


_DURATION_PART_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")


def parse_duration_s(v) -> Optional[float]:
    """Groq reset headers look like "7.66s", "2m59.56s" or "120ms".
    Returns seconds, or None if absent/unparseable — callers then fall back
    to the behavior that doesn't need a reset time."""
    if not isinstance(v, str) or not v.strip():
        return None
    v = v.strip()
    parts = _DURATION_PART_RE.findall(v)
    if not parts or "".join(n + u for n, u in parts) != v:
        return None
    scale = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
    return sum(float(n) * scale[u] for n, u in parts)


def get_provider() -> LLMProvider:
    if config.DRY_RUN:
        return DryRunProvider()
    return GroqProvider()
