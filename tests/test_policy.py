import copy

import pytest

from howsure import demo, load
from howsure.data import DataError
from howsure.policy import Policy, fit

RESPONSE = {
    "answers": {
        "team": {"type": "choice", "choice": "billing", "confidence": 0.9,
                 "probabilities": {"billing": 0.97, "technical": 0.02, "other": 0.01}},
        "refund": {"type": "noul", "noul": 0.6},
        "urgency": {"type": "score", "score": 1.4, "legend": ["Routine", "Soon", "Urgent"],
                    "probabilities": {"0": 0.2, "1": 0.2, "2": 0.6}},
    }
}


@pytest.fixture(scope="module")
def fitted():
    return fit(demo.make_records())


def test_fit_finds_the_overconfidence(fitted):
    policy, charts, notes = fitted
    team = policy.questions["team"]
    assert team["calibration"]["temperature"] > 1.5  # the demo model is overconfident
    assert team["stats"]["ece_after"] < team["stats"]["ece_before"]
    assert team["cutoff"] is not None and team["stats"]["coverage"] > 0.5
    assert team["stats"]["error"] <= 0.05
    assert set(charts) == {"team", "refund", "urgency"}


def test_safe_cutoff_is_stricter(fitted):
    policy, _, _ = fitted
    for q in policy.questions.values():
        if q["safe_cutoff"] is not None:
            assert q["safe_cutoff"] >= q["cutoff"]
            assert q["stats"]["safe_coverage"] <= q["stats"]["coverage"]


def test_split_is_the_same_for_every_model():
    a = fit(demo.make_records(model="a"))[0]
    b = fit(demo.make_records(model="b", sharpness=1.0))[0]
    assert a.data["data"]["hash"] == b.data["data"]["hash"]
    assert a.model == "a" and b.model == "b"


def test_decide(fitted):
    policy = fitted[0]
    d = policy.decide(RESPONSE)
    assert d["team"].answer == "billing"
    assert abs(sum(d["team"].probabilities.values()) - 1) < 1e-9
    assert d["team"].probability < 0.97  # calibration softened it
    assert isinstance(d["refund"].answer, bool)
    assert d["urgency"].answer == 2
    assert d["refund"].action == "review"  # 0.6 is nowhere near confident


def test_changed_question_goes_to_review(fitted):
    policy = fitted[0]
    request = {"state": "x", "questions": copy.deepcopy(demo.QUESTIONS)}
    assert policy.decide(RESPONSE, request)["team"].reason is None
    request["questions"]["team"]["instructions"] = "Which department?"
    d = policy.decide(RESPONSE, request)["team"]
    assert not d.auto and d.reason == "question changed since calibration"


def test_apply_adds_blocks_and_keeps_original(fitted):
    policy = fitted[0]
    out = policy.apply(RESPONSE)
    assert "howsure" not in RESPONSE["answers"]["team"]  # input untouched
    block = out["answers"]["urgency"]["howsure"]
    assert block["answer"] == 2 and block["level"] == "Urgent"
    assert block["action"] in ("auto", "review")
    assert out["answers"]["team"]["probabilities"] == RESPONSE["answers"]["team"]["probabilities"]


def test_apply_batch(fitted):
    policy = fitted[0]
    out = policy.apply({"results": [RESPONSE, RESPONSE]})
    assert all("howsure" in r["answers"]["team"] for r in out["results"])


def test_safe_mode_without_safe_cutoff_reviews(fitted):
    policy = Policy(copy.deepcopy(fitted[0].data))
    policy.questions["team"]["safe_cutoff"] = None
    d = policy.decide(RESPONSE, safe=True)["team"]
    assert not d.auto and "safe cutoff" in d.reason


def test_save_and_load(fitted, tmp_path):
    path = tmp_path / "p.json"
    fitted[0].save(str(path))
    again = load(str(path))
    assert again.decide(RESPONSE)["team"].probability == fitted[0].decide(RESPONSE)["team"].probability


def test_fit_needs_answers():
    with pytest.raises(DataError, match="No model answers"):
        fit([{"state": "s", "questions": {"q": {"type": "noul", "label": True}}}])


def test_fit_skips_tiny_questions():
    recs = demo.make_records(n=15)
    with pytest.raises(DataError, match="enough labeled"):
        fit(recs)
