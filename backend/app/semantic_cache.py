"""
Semantic cache: embed incoming task prompts, cosine-match against everything
seen before, and skip the LLM call entirely above a similarity threshold.

With N agents pulling from a shared, finite task pool (or just naturally
overlapping real-world requests), most of the swarm's "work" is redundant —
this is where a lot of the token savings story comes from, cheaply.

Real accounting rule still holds: a cache hit records 0 real tokens spent
(nothing was sent to the LLM). Tokens *saved* is tracked separately as the
real token count the original (uncached) call actually cost.
"""
import re
from dataclasses import dataclass, field
from threading import Lock

import numpy as np

from . import config
from .event_log import event_log


class Embedder:
    """Wraps sentence-transformers when available. Falls back to a cheap
    hashed bag-of-words vector if the model can't be loaded (no internet,
    no cached weights) — the semantic cache should degrade, not disappear,
    on bad venue wifi."""

    def __init__(self):
        self.mode = "pending"
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
            self.mode = "minilm"
            event_log.add("cache", f"embedder ready: {config.EMBEDDING_MODEL_NAME}")
        except Exception as e:
            self.mode = "fallback-hash"
            event_log.add("cache", f"embedding model unavailable ({e.__class__.__name__}); using fallback hashed embedder")

    def embed(self, text: str) -> np.ndarray:
        if self.mode == "minilm":
            vec = self._model.encode(text, normalize_embeddings=True)
            return np.asarray(vec, dtype=np.float32)
        return self._fallback_embed(text)

    @staticmethod
    def _fallback_embed(text: str, dim: int = 256) -> np.ndarray:
        vec = np.zeros(dim, dtype=np.float32)
        for tok in re.findall(r"\w+", text.lower()):
            vec[hash(tok) % dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


@dataclass
class CacheEntry:
    task_text: str
    response_text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    embedding: np.ndarray

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class CacheLookupResult:
    hit: bool
    entry: CacheEntry | None
    similarity: float


class SemanticCache:
    def __init__(self, threshold: float = config.CACHE_SIMILARITY_THRESHOLD):
        self.threshold = threshold
        self.embedder = Embedder()
        self._entries: list[CacheEntry] = []
        self._lock = Lock()
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0
        self.cost_saved_usd = 0.0

    def lookup(self, task_text: str) -> CacheLookupResult:
        query = self.embedder.embed(task_text)
        with self._lock:
            entries = list(self._entries)

        best_entry, best_sim = None, -1.0
        for entry in entries:
            sim = float(np.dot(query, entry.embedding))
            if sim > best_sim:
                best_entry, best_sim = entry, sim

        hit = best_entry is not None and best_sim >= self.threshold
        with self._lock:
            if hit:
                self.hits += 1
                self.tokens_saved += best_entry.total_tokens
                self.cost_saved_usd += best_entry.cost_usd
            else:
                self.misses += 1

        if hit:
            event_log.add(
                "cache",
                f"cache HIT (similarity={best_sim:.3f}) for '{task_text[:60]}...' — "
                f"served cached response, saved {best_entry.total_tokens} tokens (${best_entry.cost_usd:.6f})",
            )
        return CacheLookupResult(hit=hit, entry=best_entry, similarity=max(best_sim, 0.0))

    def store(self, task_text: str, response_text: str, model: str, input_tokens: int, output_tokens: int, cost_usd: float):
        entry = CacheEntry(task_text=task_text, response_text=response_text, model=model,
                            input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost_usd,
                            embedding=self.embedder.embed(task_text))
        with self._lock:
            self._entries.append(entry)

    def stats(self) -> dict:
        with self._lock:
            hits, misses, saved, cost_saved, size = self.hits, self.misses, self.tokens_saved, self.cost_saved_usd, len(self._entries)
        total = hits + misses
        return {
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hits / total, 3) if total else 0.0,
            "tokens_saved": saved,
            "cost_saved_usd": round(cost_saved, 6),
            "cache_size": size,
            "embedder_mode": self.embedder.mode,
            "threshold": self.threshold,
        }

    def reset(self):
        with self._lock:
            self._entries = []
            self.hits = 0
            self.misses = 0
            self.tokens_saved = 0
            self.cost_saved_usd = 0.0


semantic_cache = SemanticCache()
