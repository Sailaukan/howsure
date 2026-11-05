import json

import pytest

from howsure.data import DataError, answer_probs, example_id, load_examples, parse_label, validate_labels

CHOICE = {"type": "choice", "instructions": "Team?", "criteria": {"billing": "x", "technical": "y", "other": "z"}}
NOUL = {"type": "noul", "instructions": "Refund?"}
SCORE = {"type": "score", "instructions": "Urgency?", "criteria": ["Routine", "Soon", "Urgent"]}


def test_choice_labels():
    assert parse_label(CHOICE, "technical") == 1
    assert parse_label(CHOICE, " Billing ") == 0
    with pytest.raises(DataError, match="not one of the options"):
        parse_label(CHOICE, "sales")


def test_noul_labels():
    for v in (True, "true", "Yes", 1, "1"):
        assert parse_label(NOUL, v) == 1
    for v in (False, "false", "no", 0):
        assert parse_label(NOUL, v) == 0
    with pytest.raises(DataError):
        parse_label(NOUL, "maybe")


def test_score_labels_by_index_or_name():
    assert parse_label(SCORE, 2) == 2
    assert parse_label(SCORE, "1") == 1
    assert parse_label(SCORE, "urgent") == 2
    with pytest.raises(DataError, match="out of range"):
        parse_label(SCORE, 3)


def test_answer_probs_normalizes_and_orders_levels():
    keys, probs = answer_probs("score", {"probabilities": {"2": 0.5, "0": 0.25, "1": 0.25}})
    assert keys == ["0", "1", "2"] and probs == [0.25, 0.25, 0.5]
    keys, probs = answer_probs("noul", {"noul": 0.8})
    assert keys == ["false", "true"] and abs(probs[1] - 0.8) < 1e-12
    keys, probs = answer_probs("choice", {"probabilities": {"a": 2, "b": 2}})
    assert probs == [0.5, 0.5]


def test_example_id_ignores_labels():
    a = {"state": "s", "questions": {"q": dict(NOUL, label=True)}}
    b = {"state": "s", "questions": {"q": dict(NOUL, label=False)}}
    assert example_id(a) == example_id(b)


def test_csv_with_shared_questions(tmp_path):
    q = tmp_path / "questions.json"
    q.write_text(json.dumps({"team": CHOICE, "refund": NOUL}))
    csv = tmp_path / "tickets.csv"
    csv.write_text("subject,body,team,refund\nCharged twice,Please refund,billing,yes\nLogin,Broken,technical,\n")
    recs = load_examples(str(csv), str(q))
    assert recs[0]["state"] == {"subject": "Charged twice", "body": "Please refund"}
    assert recs[0]["labels"] == {"team": "billing", "refund": "yes"}
    assert recs[1]["labels"] == {"team": "technical"}
    assert validate_labels(recs) == {"team": 2, "refund": 1}


def test_validate_labels_reports_every_bad_label():
    recs = [{"state": "s", "questions": {"team": dict(CHOICE, label="sales")}},
            {"state": "t", "questions": {"team": dict(CHOICE, label="nope")}}]
    with pytest.raises(DataError) as e:
        validate_labels(recs)
    assert "example 1" in str(e.value) and "example 2" in str(e.value)
