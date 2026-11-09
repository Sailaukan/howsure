"""The checks `fit` adds: baseline, less common answers, and results per document type."""

import json
import math
import random

import pytest

from howsure import demo
from howsure.cli import main
from howsure.data import DataError, Obs, field_of
from howsure.policy import _answer_checks, _answer_warnings, _clearly_above, _clearly_worse, _group_checks, fit


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def noul_records(n, yes_rate, yes_mean, no_mean, spread=1.5, seed=1, extra=None, start=0):
    """Yes/no answers from a model whose evidence is centered on yes_mean or no_mean."""
    rng = random.Random(seed)
    question = {"type": "noul", "instructions": "Same?"}
    records = []
    for i in range(n):
        truth = rng.random() < yes_rate
        p = sigmoid(rng.gauss(yes_mean if truth else no_mean, spread))
        records.append(
            {
                "model": "m",
                "state": {"i": start + i},
                "questions": {"same": {**question, "label": truth}},
                "answers": {"same": {"type": "noul", "noul": p}},
                **(extra or {}),
            }
        )
    return records


def honest_records(n, yes_rate, seed=1, extra=None, start=0):
    """A model that reports the real probability of yes, so calibration has nothing to fix."""
    rng = random.Random(seed)
    prior = math.log(yes_rate / (1 - yes_rate))
    records = []
    for i in range(n):
        p = sigmoid(prior + rng.gauss(0, 2.5))
        truth = rng.random() < p
        records.append(
            {
                "model": "m",
                "state": {"i": start + i},
                "questions": {"same": {"type": "noul", "instructions": "Same?", "label": truth}},
                "answers": {"same": {"type": "noul", "noul": p}},
                **(extra or {}),
            }
        )
    return records


def stats_of(records, **kwargs):
    return fit(records, **kwargs)[0].questions["same"]["stats"]


# ---------------------------------------------------------------- the common answer and the rest


def test_collapsing_to_the_common_answer_gets_warnings():
    s = stats_of(noul_records(2000, 0.95, 0.8, -1.0))
    assert s["common_answer"] == "true"
    assert 0.93 < s["common_accuracy"] < 0.97
    other = s["uncommon"]
    assert other["right"] + other["wrong"] + other["review"] == other["n"] > 0
    assert other["lost_to_calibration"] > other["gained_from_calibration"]
    assert any("at the expense of the others" in w for w in s["warnings"])


def test_doing_no_better_than_always_guessing_gets_a_warning():
    s = stats_of(noul_records(2000, 0.95, 0.5, -0.5))
    assert s["accuracy_after"] <= s["common_accuracy"]
    assert any("always answering 'true'" in w for w in s["warnings"])


def test_a_low_error_rate_does_not_hide_wrong_uncommon_answers():
    s = stats_of(noul_records(2000, 0.95, 0.8, -1.0))
    assert s["error"] <= 0.06  # looks fine overall
    assert s["uncommon"]["wrong"] > s["uncommon"]["right"]
    assert any("auto-handles" in w for w in s["warnings"])


@pytest.mark.parametrize("yes_rate", [0.5, 0.6, 0.7, 0.8])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_healthy_model_gets_no_warning(yes_rate, seed):
    assert stats_of(honest_records(1500, yes_rate, seed=seed))["warnings"] == []


def test_the_demo_gets_no_warning():
    policy = fit(demo.make_records())[0]
    assert {q: s["stats"]["warnings"] for q, s in policy.questions.items()} == {"team": [], "refund": [], "urgency": []}


def test_every_example_having_one_answer_is_called_out():
    records = noul_records(200, 1.0, 2.0, -2.0)
    s = stats_of(records)
    assert s["common_accuracy"] == 1 and s["uncommon"]["n"] == 0
    assert any("say nothing about any other answer" in w for w in s["warnings"])


def test_answer_checks_count_by_hand():
    def row(label, keys=("no", "yes")):
        return Obs("e", [0.5, 0.5], label, list(keys))

    rows = [row(1), row(1), row(1), row(0), row(0), row(0), row(1)]
    #            common: yes (4 of 7). The rest: right, wrong and review under a 0.8 cutoff, right and low.
    raw = [
        ([0.1, 0.9], 1),
        ([0.1, 0.9], 1),
        ([0.1, 0.9], 1),
        ([0.9, 0.1], 0),
        ([0.9, 0.1], 0),
        ([0.1, 0.9], 0),
        ([0.1, 0.9], 1),
    ]
    cal = [
        ([0.1, 0.9], 1),
        ([0.1, 0.9], 1),
        ([0.1, 0.9], 1),
        ([0.9, 0.1], 0),
        ([0.1, 0.9], 0),
        ([0.6, 0.4], 0),
        ([0.1, 0.9], 1),
    ]
    out = _answer_checks(rows, raw, cal, 0.8)
    assert out["common_answer"] == "yes" and out["common_accuracy"] == 4 / 7
    assert out["uncommon"] == {
        "n": 3,
        "right": 1,
        "wrong": 1,
        "review": 1,
        "lost_to_calibration": 1,
        "gained_from_calibration": 1,
    }
    assert out["common"] == {"n": 4, "lost_to_calibration": 0, "gained_from_calibration": 0}


@pytest.mark.parametrize(
    "common_n, common_gained, other_n, other_lost, warns",
    [
        (1000, 100, 14, 14, True),  # gave up every uncommon answer to gain 10% on the common one
        (345, 40, 255, 18, False),  # correcting a lean: both groups moved, and the common one gained more
        (1175, 13, 925, 13, False),  # noise around a tiny shift
        (245, 5, 14, 2, False),  # too few uncommon answers to blame anything
        (500, 0, 50, 0, False),  # nothing moved
        (10000, 200, 5000, 200, False),  # a real fall, but smaller than the error budget
    ],
)
def test_calibration_is_only_blamed_when_chance_cannot_explain_it(common_n, common_gained, other_n, other_lost, warns):
    s = {
        "n": 1000,
        "held_out": True,
        "common_answer": "a",
        "common_accuracy": 0.5,
        "accuracy_after": 0.9,
        "common": {"n": common_n, "lost_to_calibration": 0, "gained_from_calibration": common_gained},
        "uncommon": {
            "n": other_n,
            "right": 0,
            "wrong": 0,
            "review": other_n,
            "lost_to_calibration": other_lost,
            "gained_from_calibration": 0,
        },
    }
    assert any("at the expense of the others" in w for w in _answer_warnings(s, 0.05, 0.05)) == warns


def test_clearly_above():
    assert not _clearly_above(0, 10, 0.05, 0.05)
    assert not _clearly_above(1, 10, 0.05, 0.05)
    assert _clearly_above(5, 10, 0.05, 0.05)
    assert not _clearly_above(5, 100, 0.05, 0.05)  # exactly at the rate
    assert _clearly_above(12, 100, 0.05, 0.05)


def test_clearly_worse():
    assert _clearly_worse(164, 1000, 50, 1000, 0.05)  # 16.4% against 5%
    assert not _clearly_worse(34, 500, 22, 450, 0.05)  # 6.8% against 4.9%: both could be 5.8%
    assert not _clearly_worse(50, 1000, 164, 1000, 0.05)  # better is never worse
    assert not _clearly_worse(0, 0, 5, 100, 0.05) and not _clearly_worse(5, 100, 0, 0, 0.05)
    assert not _clearly_worse(0, 100, 0, 100, 0.05)


# ---------------------------------------------------------------- per document type


def two_types(second_mean=(-4.0, 4.0), n=1500):
    """Type a: a sharp model. Type b: the same scores mean something else, so a policy from a misleads on b."""
    a = noul_records(n, 0.5, 2.5, -2.5, spread=1.0, seed=1, extra={"meta": {"doc": "a"}})
    b = noul_records(n, 0.5, *second_mean, spread=1.0, seed=2, extra={"meta": {"doc": "b"}}, start=n)
    return a + b


def test_a_type_that_does_not_carry_over_gets_a_warning():
    s = stats_of(two_types(), by="meta.doc")
    by = {g["group"]: g for g in s["groups"]}
    assert by["a"]["n"] == by["b"]["n"] == 1500
    assert by["b"]["unseen_error"] > 0.5  # a policy from a is mostly wrong on b
    assert any("fitted without 'b'" in w and "carry over" in w for w in s["warnings"])


def test_a_policy_with_a_bad_type_is_flagged_even_when_it_was_fitted_on_it():
    s = stats_of(two_types(), by="meta.doc")
    assert any(w.startswith("this policy makes") for w in s["warnings"])


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_two_types_from_one_source_get_no_warning(seed):
    records = honest_records(1200, 0.6, seed=seed, extra={"meta": {"doc": "a"}}) + honest_records(
        1200, 0.6, seed=seed + 100, extra={"meta": {"doc": "b"}}, start=1200
    )
    s = stats_of(records, by="meta.doc")
    assert len(s["groups"]) == 2 and s["warnings"] == []


PLAIN = {"method": "platt", "a": 1.0, "b": 0.0}


def rows_of(group, n, wrong, start=0):
    """Rows that are all equally confident, so a cutoff takes all or none. Half of each answer, wrong ones last."""
    rows = []
    for i in range(n):
        truth = i % 2
        said = 1 - truth if i >= n - wrong else truth
        rows.append(Obs(f"{start + i:06d}", [0.1, 0.9] if said else [0.9, 0.1], truth, ["false", "true"], group=group))
    return rows


def group_warnings(a, b, check=None):
    rows = a + b
    return _group_checks("noul", rows, check if check is not None else rows, PLAIN, 0.5, 0.05, 0.05, 0.25)


def test_more_groups_need_stronger_evidence():
    # 9% against 4.5% on 200 each: z is 1.79, enough for one test at 5%, not for two.
    _, warnings = group_warnings(rows_of("a", 200, 18), rows_of("b", 200, 9, start=1000))
    assert not any(w.startswith("this policy makes") for w in warnings)
    _, warnings = group_warnings(rows_of("a", 200, 18), rows_of("b", 200, 5, start=1000))
    assert any(w.startswith("this policy makes") for w in warnings)


def test_worse_than_the_other_type_but_within_target_is_fine():
    groups, warnings = group_warnings(rows_of("a", 400, 14), rows_of("b", 400, 2, start=1000))
    assert groups[0]["error"] == 0.035
    assert warnings == []


def test_unseen_results_cover_the_whole_type_not_only_its_check_rows():
    a, b = rows_of("a", 200, 2), rows_of("b", 10, 5, start=1000)
    groups, _ = group_warnings(a, b, check=a + b[:5])
    assert groups[1]["n_check"] == 5 and groups[1]["error"] == 0.0
    assert groups[1]["unseen_error"] == 0.5


def test_unseen_warning_compares_with_what_was_promised():
    # 8% on a big type, but the promise rests on 60 check rows with one error: too noisy to call that clearly worse.
    groups, warnings = group_warnings(rows_of("a", 90, 1), rows_of("b", 1000, 80, start=1000))
    assert groups[1]["unseen_error"] == 0.08
    assert not any("fitted without 'b'" in w for w in warnings)
    # The same 8% against a promise resting on thousands of rows at about 1% is clearly worse.
    _, warnings = group_warnings(rows_of("a", 4000, 40), rows_of("b", 1000, 80, start=10000))
    assert any("fitted without 'b'" in w for w in warnings)


def test_unseen_warning_needs_the_target_missed():
    groups, warnings = group_warnings(rows_of("a", 4000, 8), rows_of("b", 1000, 40, start=10000))
    assert groups[1]["unseen_error"] == 0.04
    assert not any("fitted without 'b'" in w for w in warnings)


def test_small_data_is_fitted_and_checked_on_the_same_rows():
    small, big = stats_of(honest_records(79, 0.6)), stats_of(honest_records(80, 0.6))
    assert not small["held_out"] and small["n_fit"] == small["n_check"] == 79
    assert big["held_out"] and big["n_fit"] == 30 and big["n_check"] == 50


def test_by_changes_what_is_reported_and_nothing_else():
    records = two_types()
    plain, grouped = fit(records)[0].questions["same"], fit(records, by="meta.doc")[0].questions["same"]
    for key in ("calibration", "cutoff", "safe_cutoff", "fingerprint", "type"):
        assert plain[key] == grouped[key]
    assert "groups" not in plain["stats"] and "groups" in grouped["stats"]
    response = {"answers": {"same": {"type": "noul", "noul": 0.97}}}
    assert fit(records)[0].decide(response)["same"].auto == fit(records, by="meta.doc")[0].decide(response)["same"].auto


def test_one_group_reports_nothing():
    records = noul_records(300, 0.5, 2.0, -2.0, extra={"meta": {"doc": "a"}})
    assert "groups" not in stats_of(records, by="meta.doc")


def test_a_group_too_small_to_leave_out_still_gets_its_own_row():
    rows = [Obs(str(i), [0.1, 0.9], 1, ["false", "true"], group="big" if i else "only-one") for i in range(10)]
    groups, _ = _group_checks("noul", rows, rows, {"method": "platt", "a": 1.0, "b": 0.0}, 0.5, 0.05, 0.05, 0.25)
    assert {g["group"] for g in groups} == {"big", "only-one"}
    assert all("unseen_error" not in g for g in groups)  # neither side has enough to fit on


def test_missing_group_field():
    records = noul_records(300, 0.5, 2.0, -2.0)
    with pytest.raises(DataError, match="doc_type"):
        fit(records, by="doc_type")
    for r in records[:150]:
        r["doc_type"] = "a"
    for r in records[150:250]:
        r["doc_type"] = "b"
    _, _, notes = fit(records, by="doc_type")
    assert any("50 example(s) have no 'doc_type'" in n for n in notes)


def test_field_of():
    rec = {"a": {"b": 3}, "c": "x", "d": None}
    assert field_of(rec, "a.b") == "3" and field_of(rec, "c") == "x"
    assert field_of(rec, "a.z") is None and field_of(rec, "c.z") is None and field_of(rec, "d") is None


# ---------------------------------------------------------------- what people see


def test_cli_prints_baseline_groups_and_warnings(tmp_path, capsys):
    preds = tmp_path / "p.preds.jsonl"
    preds.write_text("\n".join(json.dumps(r) for r in two_types()))
    assert main(["fit", str(preds), "--by", "meta.doc"]) == 0
    out = capsys.readouterr().out
    assert "would be right" in out
    assert "fitted without it:" in out
    assert "warning: a policy fitted without 'b'" in out


def test_cli_prints_the_baseline_when_nothing_is_wrong(tmp_path, capsys):
    preds = tmp_path / "p.preds.jsonl"
    preds.write_text("\n".join(json.dumps(r) for r in honest_records(600, 0.6)))
    assert main(["fit", str(preds)]) == 0
    out = capsys.readouterr().out
    assert "warning:" not in out
    assert "always answering 'true' would be right" in out and "held-out cases:" in out


def test_report_escapes_answer_names(tmp_path):
    rng = random.Random(1)
    records = []
    for i in range(300):
        records.append(
            {
                "model": "m",
                "state": {"i": i},
                "questions": {
                    "pick": {
                        "type": "choice",
                        "instructions": "?",
                        "criteria": {"<i>x</i>": "a", "y": "b"},
                        "label": "<i>x</i>",
                    }
                },
                "answers": {
                    "pick": {
                        "type": "choice",
                        "probabilities": {"<i>x</i>": 0.9 + rng.random() / 20, "y": 0.1 - rng.random() / 20},
                    }
                },
            }
        )
    preds = tmp_path / "p.preds.jsonl"
    preds.write_text("\n".join(json.dumps(r) for r in records))
    assert main(["fit", str(preds)]) == 0
    html = (tmp_path / "p.report.html").read_text()
    assert "<i>x</i>" not in html and "&lt;i&gt;x&lt;/i&gt;" in html


def test_report_shows_warnings_and_groups_and_escapes_them(tmp_path):
    preds = tmp_path / "p.preds.jsonl"
    records = noul_records(600, 0.5, 2.5, -2.5, spread=1.0, seed=1, extra={"doc": "<b>a</b>"})
    records += noul_records(600, 0.5, -4.0, 4.0, spread=1.0, seed=2, extra={"doc": "<b>c</b>"}, start=600)
    preds.write_text("\n".join(json.dumps(r) for r in records))
    assert main(["fit", str(preds), "--by", "doc"]) == 0
    html = (tmp_path / "p.report.html").read_text()
    assert "By document type" in html
    assert "Fitted without this type" in html
    assert 'class="note"' in html and "carry over" in html
    assert "<b>a</b>" not in html and "&lt;b&gt;a&lt;/b&gt;" in html
    assert "Always answering" in html


def test_cli_explains_a_missing_group_field(tmp_path, capsys):
    preds = tmp_path / "p.preds.jsonl"
    preds.write_text("\n".join(json.dumps(r) for r in noul_records(100, 0.5, 2.0, -2.0)))
    assert main(["fit", str(preds), "--by", "doc_type"]) == 2
    assert "doc_type" in capsys.readouterr().err


def test_saved_policy_keeps_the_checks(tmp_path):
    policy = fit(noul_records(300, 0.8, 2.0, -2.0))[0]
    path = tmp_path / "p.json"
    policy.save(str(path))
    assert "uncommon" in json.loads(path.read_text())["questions"]["same"]["stats"]
