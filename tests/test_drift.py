"""Noticing that answers no longer look like the ones a policy was fitted on."""

import logging
import random
from collections import deque

from howsure import demo, drift
from howsure.drift import WINDOW, DriftMonitor
from howsure.policy import Policy, fit


def draw(rng, n, yes=0.5, sharp=1.0):
    """Answers and confidences from a pretend model."""
    answers, confidences = [], []
    for _ in range(n):
        answers.append("yes" if rng.random() < yes else "no")
        confidences.append(min(0.999, 0.5 + rng.betavariate(2, 2) * 0.5 * sharp))
    return answers, confidences


def monitor(rng=None, n=2000):
    rng = rng or random.Random(1)
    policy = Policy(
        {"howsure": 1, "model": "m", "questions": {"q": {"type": "noul", "reference": drift.reference(*draw(rng, n))}}}
    )
    return DriftMonitor(policy)


def feed(m, answers, confidences):
    return [m.observe("q", a, c) for a, c in zip(answers, confidences)]


def test_answers_like_the_reference_are_not_flagged():
    m, rng = monitor(), random.Random(2)
    assert not any(feed(m, *draw(rng, 5000)))


def test_false_alarms_are_rare_across_many_runs():
    flagged = total = 0
    for seed in range(30):
        m, rng = monitor(random.Random(seed)), random.Random(seed + 1000)
        results = feed(m, *draw(rng, 2000))
        flagged += sum(results)
        total += len(results)
    assert flagged / total < 0.01


def test_nothing_is_judged_before_the_window_fills():
    m, rng = monitor(), random.Random(3)
    assert not any(feed(m, *draw(rng, WINDOW - 1, yes=0.99)))


def test_a_shift_in_which_answers_come_back_is_flagged():
    m, rng = monitor(), random.Random(4)
    assert any(feed(m, *draw(rng, WINDOW + 40, yes=0.95)))


def test_a_shift_in_confidence_is_flagged():
    m, rng = monitor(), random.Random(5)
    assert any(feed(m, *draw(rng, WINDOW + 40, sharp=0.2)))


def test_an_answer_never_seen_before_is_flagged_when_it_keeps_coming():
    m, rng = monitor(), random.Random(6)
    answers, confidences = draw(rng, WINDOW + 40)
    assert any(feed(m, ["brand_new"] * len(answers), confidences))


def test_it_recovers_when_answers_go_back_to_normal():
    m, rng = monitor(), random.Random(7)
    assert feed(m, *draw(rng, WINDOW + 40, yes=0.95))[-1]
    assert not feed(m, *draw(rng, WINDOW * 3))[-1]


def test_changes_are_logged_once_each_way(caplog):
    m, rng = monitor(), random.Random(8)
    with caplog.at_level(logging.WARNING, logger="howsure"):
        feed(m, *draw(rng, WINDOW * 2, yes=0.95))
        feed(m, *draw(rng, WINDOW * 3))
    messages = [r.getMessage() for r in caplog.records]
    assert sum("have shifted" in x for x in messages) == 1
    assert sum("back to normal" in x for x in messages) == 1


def test_questions_without_a_reference_are_ignored():
    m = monitor()
    assert not m.observe("other", "yes", 0.9)
    old = Policy({"howsure": 1, "model": "m", "questions": {"q": {"type": "noul"}}})
    assert not DriftMonitor(old).observe("q", "yes", 0.9)


def test_a_single_confidence_value_is_handled():
    reference = drift.reference(["a"] * 100, [0.999] * 100)
    assert len(reference["edges"]) == 1
    policy = Policy({"howsure": 1, "model": "m", "questions": {"q": {"type": "noul", "reference": reference}}})
    assert not any(feed(DriftMonitor(policy), ["a"] * 400, [0.999] * 400))


def test_a_fitted_policy_flags_drift_and_sends_answers_to_review():
    policy = fit(demo.make_records())[0]
    assert policy.questions["team"]["reference"]
    m = DriftMonitor(policy)
    sure = {"answers": {"team": {"type": "choice", "probabilities": {"billing": 0.0, "technical": 1.0, "other": 0.0}}}}
    before = policy.decide(sure, monitor=m)["team"]
    assert before.auto
    for _ in range(WINDOW + 40):
        latest = policy.decide(sure, monitor=m)["team"]
    assert not latest.auto
    assert "shifted" in latest.reason
    assert policy.decide(sure)["team"].auto  # the policy itself didn't change


def test_the_test_notices_a_share_going_up_or_down():
    assert drift._smaller_tail(50, 100, 95, 100) < 1e-6
    assert drift._smaller_tail(50, 100, 5, 100) < 1e-6
    assert drift._smaller_tail(50, 100, 50, 100) > 0.5


def test_testing_every_bin_raises_the_evidence_needed():
    ref = {"answers": {"yes": 1000, "no": 1000}, "edges": [i / 10 for i in range(1, 10)], "confidence": [200] * 10}
    confidences = [(i % 10 + 0.5) / 10 for i in range(WINDOW)]

    def window(yes):
        return deque(zip(["yes"] * yes + ["no"] * (WINDOW - yes), confidences))

    assert drift.ALPHA / 12 < drift._smaller_tail(1000, 2000, 130, WINDOW) < drift.ALPHA  # enough for one test, not 12
    assert not drift._shifted(ref, window(130))
    assert drift._shifted(ref, window(135))


def test_a_small_reference_needs_stronger_evidence():
    """A reference from few examples is itself noisy, which mustn't look like drift."""
    flagged = total = 0
    for seed in range(30):
        m, rng = monitor(random.Random(seed), n=60), random.Random(seed + 1000)
        results = feed(m, *draw(rng, 1000))
        flagged += sum(results)
        total += len(results)
    assert flagged / total < 0.01


def test_a_new_policy_format_without_enough_data_has_no_reference():
    policy = fit(demo.make_records(n=50))[0]
    assert policy.questions["team"]["reference"] is None
