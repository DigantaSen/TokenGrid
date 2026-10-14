"""
Real tasks with ground truth, for evaluating TokenGrid offline.

The live demo's 8 SAMPLE_TASKS repeat, so after one round every call is a
cache hit and nothing downstream (reputation, auction contention) is ever
exercised. Experiments instead draw distinct tasks from three public
benchmarks, and every answer is graded — "completed" means *correct*, not
just "won the auction":

    gsm8k          grade-school math, numeric answer
    mmlu           57-subject knowledge, multiple choice
    arc_challenge  science reasoning, multiple choice
"""
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

PARQUET_URLS = {
    "gsm8k": "https://huggingface.co/api/datasets/openai/gsm8k/parquet/main/test/0.parquet",
    "mmlu": "https://huggingface.co/api/datasets/cais/mmlu/parquet/all/test/0.parquet",
    "arc_challenge": "https://huggingface.co/api/datasets/allenai/ai2_arc/parquet/ARC-Challenge/test/0.parquet",
}
DATASETS = list(PARQUET_URLS)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TASK_SYSTEM_PROMPT = (
    "Solve the problem. Think as needed, then end your response with a final "
    "line of exactly the form 'Answer: <answer>'."
)
LETTERS = "ABCDEFGHIJ"


@dataclass
class Task:
    task_id: str
    dataset: str
    text: str     # the user message actually sent
    gold: str
    kind: str     # "numeric" | "mc"
    meta: dict = field(default_factory=dict)

    def messages(self) -> list[dict]:
        return [{"role": "system", "content": TASK_SYSTEM_PROMPT}, {"role": "user", "content": self.text}]


def _read_parquet(name: str):
    import pandas as pd

    dest = DATA_DIR / f"{name}.parquet"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(PARQUET_URLS[name], tmp)
        tmp.replace(dest)
    return pd.read_parquet(dest)


def _format_mc(question: str, choices: list[str]) -> str:
    lines = [question.strip(), ""] + [f"{LETTERS[i]}. {c}" for i, c in enumerate(choices)]
    return "\n".join(lines + ["", "Answer with the letter of the correct option."])


def _build(name: str, df) -> list[Task]:
    tasks = []
    for idx, row in df.iterrows():
        if name == "gsm8k":
            tasks.append(Task(f"gsm8k-{idx}", name, row["question"].strip(),
                              normalize_number(row["answer"].split("####")[-1]), "numeric"))
        elif name == "mmlu":
            tasks.append(Task(f"mmlu-{idx}", name, _format_mc(row["question"], list(row["choices"])),
                              LETTERS[int(row["answer"])], "mc", {"subject": row["subject"]}))
        else:
            texts, labels = list(row["choices"]["text"]), list(row["choices"]["label"])
            # ARC labels are sometimes "1".."4" instead of "A".."D"; re-letter by position.
            tasks.append(Task(f"arc_challenge-{row['id']}", name, _format_mc(row["question"], texts),
                              LETTERS[labels.index(row["answerKey"])], "mc"))
    return tasks


def load_tasks(name: str, n: int | None = 500, seed: int = 0) -> list[Task]:
    """Seeded sample, so every experiment and model sees the same tasks."""
    df = _read_parquet(name)
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=seed)
    return _build(name, df)


# --- grading ---

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_ANSWER_LINE_RE = re.compile(r"answer\s*[:：]\s*(.+)", re.I)
_MC_RE = re.compile(r"answer\s*[:：]\s*\**\s*\(?([A-J])\b", re.I)


def normalize_number(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(s)
    except ValueError:
        return s
    return str(int(f)) if f == int(f) else str(f)


def extract_answer(text: str, kind: str) -> str | None:
    if not text:
        return None
    if kind == "mc":
        m = _MC_RE.findall(text)
        return m[-1].upper() if m else None
    lines = _ANSWER_LINE_RE.findall(text)
    nums = _NUM_RE.findall(lines[-1] if lines else text)
    return normalize_number(nums[-1]) if nums else None


def grade(task: Task, text: str) -> tuple[str | None, bool]:
    """(extracted answer, correct). Unparseable output counts as wrong."""
    pred = extract_answer(text, task.kind)
    if pred is None:
        return None, False
    if task.kind == "numeric":
        try:
            return pred, abs(float(pred) - float(task.gold)) < 1e-6
        except ValueError:
            return pred, pred == task.gold
    return pred, pred == task.gold
