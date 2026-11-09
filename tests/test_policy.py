import copy
import random

import pytest

from howsure import demo, load
from howsure import policy as policy_module
from howsure.data import DataError, Obs
from howsure.policy import Policy, fit

RESPONSE = {
    "answers": {
        "team": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.9,
            "probabilities": {"billing": 0.97, "technical": 0.02, "other": 0.01},
        },
        "refund": {"type": "noul", "noul": 0.6},
        "urgency": {
            "type": "score",
            "score": 1.4,
            "legend": ["Routine", "Soon", "Urgent"],
            "probabilities": {"0": 0.2, "1": 0.2, "2": 0.6},
        },
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


@pytest.mark.parametrize(
    "answer, reason",
    [
        ({"type": "choice", "probabilities": {"hallucinated_team": 0.99}}, "don't match"),
        ({"type": "choice", "probabilities": {"billing": 0.99}}, "don't match"),
        ({"type": "choice", "probabilities": {"billing": 1.5, "technical": -0.25, "other": -0.25}}, "unreadable"),
        ({"type": "choice", "probabilities": {"billing": 0.4, "technical": 0.1, "other": 0.0}}, "add up"),
    ],
)
def test_garbage_is_never_auto_even_without_the_request(fitted, answer, reason):
    d = fitted[0].decide({"answers": {"team": answer}})["team"]
    assert not d.auto and d.answer is None and reason in d.reason


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 5, -1, True, "0.9", None])
def test_garbage_yes_no_answers_are_never_auto(fitted, value):
    d = fitted[0].decide({"answers": {"refund": {"type": "noul", "noul": value}}})["refund"]
    assert not d.auto and d.answer is None


def test_the_request_decides_which_options_are_valid(fitted):
    answer = {"type": "choice", "probabilities": {"sales": 0.99, "legal": 0.01}}
    question = {"type": "choice", "instructions": "x", "criteria": {"sales": "a", "legal": "b"}}
    assert "don't match" in fitted[0].decide({"answers": {"team": answer}})["team"].reason
    d = fitted[0].decide({"answers": {"team": answer}}, {"questions": {"team": question}})["team"]
    assert d.answer == "sales" and d.reason == "question changed since calibration"


def test_a_new_model_version_goes_to_review():
    records = demo.make_records()
    for r in records:
        r["served_model"] = "vendor/m-20260930"
    policy = fit(records)[0]
    sure = {"type": "noul", "noul": 0.9999}
    assert policy.decide({"model": "vendor/m-20260930", "answers": {"refund": sure}})["refund"].auto
    d = policy.decide({"model": "vendor/m-20261101", "answers": {"refund": sure}})["refund"]
    assert not d.auto and d.reason == "model version changed since calibration"
    assert policy.decide({"answers": {"refund": sure}})["refund"].auto  # nothing to compare


def test_preds_without_a_version_turn_the_version_check_off(fitted):
    assert (
        fitted[0].decide({"model": "anything", "answers": {"refund": {"type": "noul", "noul": 0.9999}}})["refund"].auto
    )


def test_too_few_labels_auto_handle_nothing():
    policy = fit(demo.make_records(n=79))[0]
    for q in policy.questions.values():
        assert q["cutoff"] is None and q["safe_cutoff"] is None
        assert any("at least 80" in w for w in q["stats"]["warnings"])
    assert fit(demo.make_records(n=80))[0].questions["team"]["stats"]["held_out"]


def test_calibration_is_dropped_when_it_does_not_help_on_unseen_examples(monkeypatch):
    monkeypatch.setattr(
        policy_module, "fit_calibration", lambda qtype, rows: {"method": "temperature", "temperature": 20.0}
    )
    q = fit(demo.make_records())[0].questions["team"]
    assert q["calibration"] == {"method": "temperature", "temperature": 1.0}


def test_a_mostly_missed_rare_answer_blocks_the_cutoff():
    """5% of cases are rare, and the model is confidently wrong on them. Error rate alone would allow it."""
    rng = random.Random(2)
    records = []
    for i in range(3000):
        rare = rng.random() < 0.04
        p = rng.uniform(0.96, 0.999) if rare else rng.uniform(0.97, 0.9999)  # says yes either way
        records.append(
            {
                "state": {"i": i},
                "questions": {"q": {"type": "noul", "instructions": "?", "label": not rare}},
                "answers": {"q": {"type": "noul", "noul": p}},
            }
        )
    s = fit(records)[0].questions["q"]["stats"]
    other = s["uncommon"]
    assert other["wrong"] <= 0.15 * other["n"]  # the overall error rate alone would let nearly all of them through


def test_both_cutoffs_answer_to_the_rare_answer_check(monkeypatch):
    # 3% of cases are rare and always wrong, at a lower confidence. Overall that's within a 5% target.
    monkeypatch.setattr(policy_module, "fit_calibration", lambda qtype, rows: policy_module.no_calibration(qtype))
    rows = [
        Obs(f"{i:05d}", [0.05, 0.95], 0, ["false", "true"])
        if i % 33 == 0
        else Obs(f"{i:05d}", [0.01, 0.99], 1, ["false", "true"])
        for i in range(1000)
    ]
    fitted = policy_module._fit_question("noul", rows, 0.05, 0.05, 0.25)
    right = sum(r.label == 1 for r in fitted.check_rows)
    assert fitted.expected[1] == right and fitted.safe[1] == right


def test_exact_duplicates_only_count_once():
    records = demo.make_records(n=300)
    q, _, notes = fit(records + records + records)
    alone = fit(records)[0]
    assert q.questions["team"]["stats"]["n"] == 300
    assert q.questions["team"]["cutoff"] == alone.questions["team"]["cutoff"]
    assert any("team: skipped 600" in n and "duplicate" in n for n in notes)


def test_the_same_input_with_different_labels_is_not_a_duplicate():
    def record(state, label):
        question = {"type": "noul", "instructions": "?", "label": label}
        return {"state": state, "questions": {"q": question}, "answers": {"q": {"type": "noul", "noul": 0.7}}}

    records = [record("s", True), record("s", False), *(record(str(i), True) for i in range(20))]
    assert fit(records)[0].questions["q"]["stats"]["n"] == 22


def test_too_few_labels_get_one_clear_warning():
    q = fit(demo.make_records(n=30))[0].questions["team"]
    assert len(q["stats"]["warnings"]) == 1 and "at least 80" in q["stats"]["warnings"][0]


def test_fit_reports_garbage_it_skipped():
    records = demo.make_records()
    for r in records[:7]:
        r["answers"]["team"] = {"type": "choice", "probabilities": {"hallucinated_team": 1.0}}
    q, _, notes = fit(records)
    assert q.questions["team"]["stats"]["n"] == len(records) - 7
    assert any("team: skipped 7" in n and "don't match" in n for n in notes)


def test_few_uncommon_answers_get_a_warning():
    rng = random.Random(3)
    records = [
        {
            "state": {"i": i},
            "questions": {"q": {"type": "noul", "instructions": "?", "label": (t := rng.random() < 0.98)}},
            "answers": {"q": {"type": "noul", "noul": 0.9 if t else 0.2}},
        }
        for i in range(1000)
    ]
    assert any("label more of them" in w for w in fit(records)[0].questions["q"]["stats"]["warnings"])


def test_can_auto_agrees_with_decide(fitted):
    policy = Policy(copy.deepcopy(fitted[0].data))
    sure = {"answers": {"team": {"type": "choice", "probabilities": {"billing": 1.0, "technical": 0.0, "other": 0.0}}}}
    team = copy.deepcopy(demo.QUESTIONS["team"])
    assert policy.can_auto("team", team)
    assert policy.decide(sure, {"questions": {"team": team}})["team"].auto

    assert not policy.can_auto("unknown", team)

    reworded = {**team, "instructions": "Which department?"}
    assert not policy.can_auto("team", reworded)
    assert not policy.decide(sure, {"questions": {"team": reworded}})["team"].auto

    retyped = {**team, "type": "score"}
    assert not policy.can_auto("team", retyped)

    policy.questions["team"]["safe_cutoff"] = None
    assert policy.can_auto("team", team)
    assert not policy.can_auto("team", team, safe=True)
    assert not policy.decide(sure, {"questions": {"team": team}}, safe=True)["team"].auto

    policy.questions["team"]["cutoff"] = None
    assert not policy.can_auto("team", team)
    assert not policy.decide(sure, {"questions": {"team": team}})["team"].auto


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
