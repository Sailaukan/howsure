"""Notice when a model's answers stop looking like the ones its policy was fitted on.

A policy's promise only holds on data like the data it was fitted on. Labels arrive too late to
check that while serving, but a shift in which answers a model gives, or how confident it is,
shows up straight away. While that lasts, answers go to review instead of being auto-handled.
"""

from __future__ import annotations

import logging
import math
import threading
from bisect import bisect_right
from collections import Counter, deque

WINDOW = 200  # recent answers compared with the reference, so the first ones can't be judged
CHECK_EVERY = 20
ALPHA = 1e-4  # strict, because the check keeps running

logger = logging.getLogger("howsure")


def reference(answers: list[str], confidences: list[float]) -> dict:
    """What a model's answers looked like when its policy was fitted."""
    ordered = sorted(confidences)
    edges = sorted({ordered[len(ordered) * i // 10] for i in range(1, 10)})
    confidence_counts = [0] * (len(edges) + 1)
    for c in confidences:
        confidence_counts[bisect_right(edges, c)] += 1
    return {"answers": dict(Counter(answers)), "edges": edges, "confidence": confidence_counts}


def _log_choose(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _smaller_tail(ref_count: int, ref_total: int, count: int, total: int) -> float:
    """Fisher's exact test: how surprising this count is, given noise in both samples."""
    hits, everything = ref_count + count, ref_total + total
    denominator = _log_choose(everything, total)
    low, high = max(0, total - (everything - hits)), min(total, hits)

    def chance(x: int) -> float:
        return math.exp(_log_choose(hits, x) + _log_choose(everything - hits, total - x) - denominator)

    return min(sum(chance(x) for x in range(low, count + 1)), sum(chance(x) for x in range(count, high + 1)))


def _shifted(ref: dict, window: deque) -> bool:
    answers = Counter(name for name, _ in window)
    confidence = [0] * (len(ref["edges"]) + 1)
    for _, c in window:
        confidence[bisect_right(ref["edges"], c)] += 1
    names = set(ref["answers"]) | set(answers)
    pairs = [(ref["answers"].get(name, 0), answers[name]) for name in names] + list(zip(ref["confidence"], confidence))
    ref_total = sum(ref["confidence"])
    alpha = ALPHA / len(pairs)  # one test per bin
    return any(_smaller_tail(a, ref_total, b, len(window)) <= alpha for a, b in pairs)


class DriftMonitor:
    """Tracks one model's recent answers, per question. Safe to share between threads."""

    def __init__(self, policy):
        self.model = policy.model
        self.references = {qid: q["reference"] for qid, q in policy.questions.items() if q.get("reference")}
        self.windows = {qid: deque(maxlen=WINDOW) for qid in self.references}
        self.seen = dict.fromkeys(self.references, 0)
        self.drifted: set[str] = set()
        self.lock = threading.Lock()

    def observe(self, qid: str, answer: str, confidence: float) -> bool:
        """Record one answer. Returns whether this question has drifted."""
        ref = self.references.get(qid)
        if ref is None:
            return False
        with self.lock:
            window = self.windows[qid]
            window.append((answer, confidence))
            self.seen[qid] += 1
            if len(window) == WINDOW and self.seen[qid] % CHECK_EVERY == 0:
                shifted = _shifted(ref, window)
                if shifted != (qid in self.drifted):
                    state = "have shifted since calibration, so they go to review" if shifted else "are back to normal"
                    logger.warning("%s: answers to %r %s", self.model, qid, state)
                (self.drifted.add if shifted else self.drifted.discard)(qid)
            return qid in self.drifted
