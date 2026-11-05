import math
import random

from howsure import stats


def softmax(zs):
    m = max(zs)
    e = [math.exp(z - m) for z in zs]
    return [x / sum(e) for x in e]


def sample(rng, probs):
    r, acc = rng.random(), 0.0
    for i, p in enumerate(probs):
        acc += p
        if r < acc:
            return i
    return len(probs) - 1


def test_temperature_recovers_overconfidence():
    rng = random.Random(1)
    rows = []
    for _ in range(3000):
        z = [rng.gauss(0, 1.5) for _ in range(4)]
        y = sample(rng, softmax(z))  # labels follow the honest probabilities
        rows.append((softmax([2 * x for x in z]), y))  # the model reports them twice as sharp
    t = stats.fit_temperature(rows)
    assert 1.75 < t < 2.3


def test_temperature_keeps_the_top_answer():
    probs = [0.7, 0.2, 0.1]
    for t in (0.5, 1.0, 3.0):
        out = stats.scale_temperature(probs, t)
        assert stats.argmax(out) == 0
        assert abs(sum(out) - 1) < 1e-9


def test_platt_recovers_scale_and_shift():
    rng = random.Random(2)
    rows = []
    for _ in range(5000):
        x = rng.gauss(0, 2)
        y = int(rng.random() < 1 / (1 + math.exp(-x)))
        reported = 1 / (1 + math.exp(-(2 * x + 1)))  # overconfident and leaning yes
        rows.append(([1 - reported, reported], y))
    a, b = stats.fit_platt(rows)
    assert abs(a - 0.5) < 0.06
    assert abs(b + 0.5) < 0.12


def test_binomial_cdf_exact_values():
    assert abs(stats.binom_cdf(0, 10, 0.5) - 1 / 1024) < 1e-12
    assert abs(stats.binom_cdf(5, 10, 0.5) - 638 / 1024) < 1e-12
    assert stats.binom_cdf(10, 10, 0.3) == 1.0
    assert stats.binom_cdf(-1, 10, 0.3) == 0.0


def test_upper_bound_zero_errors():
    # With 0 errors in n, the 95% bound solves (1 - p)^n = 0.05.
    expected = 1 - 0.05 ** (1 / 149)
    assert abs(stats.upper_bound(0, 149, 0.05) - expected) < 1e-6


def test_ece_small_for_honest_model():
    rng = random.Random(3)
    rows = []
    for _ in range(4000):
        p = softmax([rng.gauss(0, 2) for _ in range(3)])
        rows.append((p, sample(rng, p)))
    assert stats.summarize(rows)["ece"] < 0.03


def test_expected_cutoff_picks_most_coverage_within_target():
    # Confidences high to low, the third and fifth are wrong.
    rows = [([c, 1 - c], 0 if ok else 1) for c, ok in
            [(0.99, True), (0.98, True), (0.97, False), (0.96, True), (0.95, False), (0.9, True)]]
    curve = stats.risk_curve(rows)
    t, n, k = stats.expected_cutoff(curve, 0.34)
    assert (t, n, k) == (0.9, 6, 2)
    t, n, k = stats.expected_cutoff(curve, 0.25)
    assert (t, n, k) == (0.96, 4, 1)


def test_safe_cutoff_keeps_its_promise():
    """Over many simulated datasets, the certified error rate is exceeded at most ~delta of the time."""
    rng = random.Random(4)
    alpha, delta, violations, runs = 0.1, 0.1, 0, 150

    def draw(n):
        out = []
        for _ in range(n):
            c = 1 / 2 + rng.betavariate(5, 1) / 2
            wrong = rng.random() < (1 - c) * 1.3
            out.append(([c, 1 - c], 1 if wrong else 0))
        return out

    # True error rate above a threshold, from a large sample.
    big = sorted(draw(100_000), key=lambda r: -r[0][0])

    def true_error(t):
        sel = [y for p, y in big if p[0] >= t]
        return sum(sel) / len(sel)

    certified = 0
    for _ in range(runs):
        best = stats.safe_cutoff(stats.risk_curve(draw(300)), alpha, delta)
        if best:
            certified += 1
            violations += true_error(best[0]) > alpha
    assert certified > runs // 2  # it finds cutoffs, not just refuses
    assert violations / runs <= delta + 0.05


def test_safe_cutoff_refuses_with_tiny_data():
    rows = [([0.99, 0.01], 0)] * 20
    assert stats.safe_cutoff(stats.risk_curve(rows), 0.02, 0.05) is None  # needs 149+ examples
