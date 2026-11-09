"""Calibration, metrics and cutoff selection. Pure Python, no dependencies.

Rows are ``(probs, label)`` pairs: a probability per option and the index of the
correct one. Everything here works on the top answer ("top-label") the way a
production rule would: take the most likely option, act on it if it is confident.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

EPS = 1e-6


# ---------------------------------------------------------------- basics


def argmax(xs: Sequence[float]) -> int:
    best = 0
    for i, x in enumerate(xs):
        if x > xs[best]:
            best = i
    return best


def _softplus(z: float) -> float:
    return z + math.log1p(math.exp(-z)) if z > 0 else math.log1p(math.exp(z))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _logit(p: float) -> float:
    p = min(1 - EPS, max(EPS, p))
    return math.log(p / (1 - p))


# ---------------------------------------------------------------- calibration


def scale_temperature(probs: Sequence[float], temperature: float) -> list[float]:
    beta = 1.0 / temperature
    logs = [beta * math.log(max(p, EPS)) for p in probs]
    m = max(logs)
    exps = [math.exp(z - m) for z in logs]
    s = sum(exps)
    return [e / s for e in exps]


def fit_temperature(rows: Sequence) -> float:
    """Temperature T minimizing log loss of softmax(log p / T). T > 1 softens overconfident models.

    The loss is convex in 1/T, so a golden-section search over log(1/T) finds the optimum.
    """
    data = [([math.log(max(p, EPS)) for p in probs], y) for probs, y in rows if len(probs) > 1]
    if not data:
        return 1.0

    def loss(u: float) -> float:
        beta = math.exp(u)
        total = 0.0
        for logs, y in data:
            m = beta * max(logs)
            total += m + math.log(sum(math.exp(beta * z - m) for z in logs)) - beta * logs[y]
        return total

    lo, hi = math.log(0.05), math.log(20.0)
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = loss(a), loss(b)
    for _ in range(80):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = loss(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = loss(b)
    return 1.0 / math.exp((lo + hi) / 2)


def platt(p_yes: float, a: float, b: float) -> float:
    return _sigmoid(a * _logit(p_yes) + b)


def fit_platt(rows: Sequence) -> tuple[float, float]:
    """Platt scaling for yes/no: p' = sigmoid(a * logit(p) + b).

    ``a`` fixes over/underconfidence, ``b`` fixes a lean towards yes or no. Uses Platt's
    smoothed targets so a small or one-sided dataset can't push probabilities to 0 or 1.
    """
    xs = [_logit(probs[1]) for probs, _ in rows]
    ys = [y for _, y in rows]
    npos = sum(ys)
    nneg = len(ys) - npos
    tpos, tneg = (npos + 1) / (npos + 2), 1 / (nneg + 2)
    ts = [tpos if y else tneg for y in ys]

    def loss(a: float, b: float) -> float:
        return sum(_softplus(a * x + b) - t * (a * x + b) for x, t in zip(xs, ts))

    a, b = 1.0, 0.0
    cur = loss(a, b)
    for _ in range(100):
        ga = gb = haa = hab = hbb = 0.0
        for x, t in zip(xs, ts):
            q = _sigmoid(a * x + b)
            r, w = q - t, q * (1 - q)
            ga += r * x
            gb += r
            haa += w * x * x
            hab += w * x
            hbb += w
        haa += 1e-9
        hbb += 1e-9
        det = haa * hbb - hab * hab
        if det <= 1e-12:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        step = 1.0
        while step > 1e-8:
            na, nb = a - step * da, b - step * db
            new = loss(na, nb)
            if new <= cur:
                break
            step /= 2
        else:
            break
        moved = abs(na - a) + abs(nb - b)
        a, b, cur = na, nb, new
        if moved < 1e-10:
            break
    return a, b


# ---------------------------------------------------------------- metrics


def _bins(rows: Sequence) -> list[list[tuple[float, int]]]:
    """Equal-count bins of (confidence, correct), sorted by confidence."""
    pts = sorted((max(p), int(argmax(p) == y)) for p, y in rows)
    n = len(pts)
    k = max(1, min(10, n // 25))
    out, start = [], 0
    for i in range(k):
        end = round((i + 1) * n / k)
        if end > start:
            out.append(pts[start:end])
        start = end
    return out


def reliability(rows: Sequence) -> list[dict]:
    """Per bin: mean stated confidence, actual accuracy, count. For reliability diagrams."""
    return [
        {
            "confidence": sum(c for c, _ in b) / len(b),
            "accuracy": sum(ok for _, ok in b) / len(b),
            "count": len(b),
        }
        for b in _bins(rows)
    ]


def summarize(rows: Sequence) -> dict:
    """Accuracy, calibration error (ECE), Brier score, log loss and confident-error rate."""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    correct = brier = nll = confident_wrong = 0.0
    for probs, y in rows:
        top = argmax(probs)
        correct += top == y
        brier += sum((p - (1.0 if k == y else 0.0)) ** 2 for k, p in enumerate(probs))
        nll -= math.log(max(probs[y], EPS))
        confident_wrong += top != y and probs[top] >= 0.9
    ece = sum(len(b) * abs(sum(c for c, _ in b) / len(b) - sum(ok for _, ok in b) / len(b)) for b in _bins(rows)) / n
    return {
        "n": n,
        "accuracy": correct / n,
        "ece": ece,
        "brier": brier / n,
        "log_loss": nll / n,
        "confident_errors": confident_wrong / n,
    }


# ---------------------------------------------------------------- cutoffs


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n or p <= 0:
        return 1.0
    if p >= 1:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    lg = math.lgamma
    base = lg(n + 1)
    total = 0.0
    for i in range(k + 1):
        total += math.exp(base - lg(i + 1) - lg(n - i + 1) + i * lp + (n - i) * lq)
    return min(1.0, total)


def risk_curve(rows: Sequence) -> list[tuple[float, int, int]]:
    """(threshold, examples at or above it, errors among them), from the most confident down."""
    items = sorted(((max(p), int(argmax(p) != y)) for p, y in rows), key=lambda t: -t[0])
    curve, errors = [], 0
    for i, (conf, wrong) in enumerate(items):
        errors += wrong
        if i + 1 < len(items) and items[i + 1][0] == conf:
            continue  # ties: a threshold includes all of them
        curve.append((conf, i + 1, errors))
    return curve


def min_examples(max_error: float, delta: float) -> int:
    """The fewest examples that can show an error rate is within max_error, even with zero errors."""
    return math.ceil(math.log(delta) / math.log(1 - max_error))


def expected_cutoff(curve: Sequence, max_error: float, delta: float):
    """Lowest threshold whose error rate, plus one standard error, is within the target (max coverage).

    The margin keeps a lucky small sample from passing, and fades as examples grow. Thresholds
    covering too few examples to show the target at all are skipped, so a target the model can't
    reach gets no cutoff rather than a small slice that happened to look good.
    """
    n_min = min_examples(max_error, delta)
    best = None
    for t, n, k in curve:
        if n < n_min:
            continue
        smoothed = (k + 1) / (n + 2)  # so zero errors still has a margin
        if k / n + math.sqrt(smoothed * (1 - smoothed) / n) <= max_error:
            best = (t, n, k)
    return best


def safe_cutoff(curve: Sequence, max_error: float, delta: float):
    """Lowest threshold whose error rate is <= max_error with probability >= 1 - delta.

    Fixed-sequence testing (Learn-then-Test): walk thresholds from most to least confident,
    certify each with an exact binomial test at level delta, stop at the first failure.
    The order never looks at labels, so no multiple-testing correction is needed.
    Thresholds covering fewer than n_min examples can't pass even with zero errors and are
    skipped (that also depends only on confidences).
    """
    n_min = min_examples(max_error, delta)
    best = None
    for t, n, k in curve:
        if n < n_min:
            continue
        if binom_cdf(k, n, max_error) <= delta:
            best = (t, n, k)
        else:
            break
    return best


def answer_floor(rows: Sequence, truths: Sequence, max_error: float, delta: float) -> float:
    """The confidence a cutoff must stay above, so no true answer is clearly auto-handled wrong too often.

    The overall error rate lets a model miss almost every rare case: 0.5% fraud is only 0.5% errors
    even if all of it is called legitimate. So each true answer gets its own binomial test. A few
    examples can only show a large breach, which keeps noise from blocking a cutoff.
    """
    wrong: dict = {}
    counts: dict = {}
    for (probs, y), truth in zip(rows, truths):
        counts[truth] = counts.get(truth, 0) + 1
        if argmax(probs) != y:
            wrong.setdefault(truth, []).append(max(probs))
    delta /= max(1, len(counts))  # one test per answer
    floor = -1.0
    for truth, confidences in wrong.items():
        confidences.sort(reverse=True)
        allowed = 0
        while allowed < len(confidences) and 1 - binom_cdf(allowed, counts[truth], max_error) > delta:
            allowed += 1
        if allowed < len(confidences):
            floor = max(floor, confidences[allowed])
    return floor


def safe_coverage_with_more(curve: Sequence, max_error: float, delta: float, factor: int) -> float:
    """Rough share of examples a safe cutoff would cover with ``factor`` times the labels,
    assuming new labels look like the current ones. A planning estimate, not a promise."""
    if not curve:
        return 0.0
    total = curve[-1][1]
    scaled = [(t, n * factor, round(k * factor)) for t, n, k in curve]
    best = safe_cutoff(scaled, max_error, delta)
    return best[1] / (total * factor) if best else 0.0
