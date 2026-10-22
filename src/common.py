"""Shared helpers: prompt format, simple metrics, run logging.

Everything in here is plain, small, and tested (tests/test_common.py).
No metric in this project is typed in by hand; numbers come from
evaluate.py reading model generations.
"""

import json
import platform
import re
import resource
import time
from pathlib import Path

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"
PROMPT_TEMPLATE_WITH_INPUT = (
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)


def build_prompt(instruction: str, input_text: str = "") -> str:
    """Format one example the Alpaca way. Used for both training and eval."""
    if input_text and input_text.strip():
        return PROMPT_TEMPLATE_WITH_INPUT.format(
            instruction=instruction.strip(), input=input_text.strip()
        )
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace (for exact match)."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize(prediction) == normalize(reference) else 0.0


def _lcs_length(a: list, b: list) -> int:
    """Longest common subsequence length, classic dynamic programming."""
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[len(b)]


def rouge_l_f1(prediction: str, reference: str) -> float:
    """ROUGE-L F1 on whitespace tokens."""
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def clean_generation(text: str) -> str:
    """Cut a raw generation at the start of a new section, if one appears."""
    for marker in ("\n### Instruction", "\n### Input", "\n### Response"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def peak_memory_mb() -> float:
    """Peak resident memory of this process so far (Linux ru_maxrss)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class RunLog:
    """Tiny run logger: collects facts, writes them as JSON at the end."""

    def __init__(self, name: str):
        self.data = {
            "run": name,
            "started_unix": time.time(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        }
        self._t0 = time.time()

    def add(self, **kwargs):
        self.data.update(kwargs)
        return self

    def finish(self, path: Path):
        self.data["wall_seconds"] = round(time.time() - self._t0, 2)
        self.data["peak_memory_mb"] = round(peak_memory_mb(), 1)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.data, indent=2))
        return self.data
