"""
Model router: a cheap real call classifies each task's complexity before
it's executed, and routing follows that classification rather than trusting
any pre-assigned label — this is what makes it a router and not a lookup
table. Savings are measured against a naive "always use the big model"
baseline, using the real token counts the actual call cost, and tracked
separately from cache and compression savings.
"""
from . import config
from .llm_provider import CompletionResult, LLMProvider

CLASSIFY_SYSTEM_PROMPT = (
    "Classify the complexity of the task below as exactly one word: 'simple' or "
    "'complex'. Simple: short factual, classification, or extraction questions "
    "answerable in one sentence. Complex: anything requiring reasoning, writing, "
    "planning, or debugging. Reply with only that one word."
)


def classify_complexity(provider: LLMProvider, task_text: str) -> tuple[str, CompletionResult]:
    messages = [
        {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
        {"role": "user", "content": task_text},
    ]
    result = provider.complete(config.MODEL_SIMPLE, messages, max_tokens=config.ROUTER_CLASSIFY_MAX_TOKENS)
    label = (result.text or "").strip().lower()
    complexity = "complex" if "complex" in label else "simple"
    return complexity, result
