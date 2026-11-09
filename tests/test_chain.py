"""Fitting a chain: each model only on what the ones before it pass on."""

import json
import math
import random

import pytest

from howsure.cli import main
from howsure.data import DataError
from howsure.gateway import load_config
from howsure.policy import fit, fit_chain

QUESTION = {"type": "noul", "instructions": "Refund?"}


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def two_models(n, seed=1, start=0):
    """A cheap and a strong model, honest about their own evidence, on the same examples."""
    rng = random.Random(seed)
    cheap, strong = [], []
    for i in range(start, start + n):
        truth = rng.random() < 0.5
        for records, signal, name in ((cheap, 1.5, "cheap"), (strong, 2.5, "strong")):
            evidence = rng.gauss(signal if truth else 0, 1)
            p = sigmoid(signal * evidence - signal * signal / 2)
            records.append(
                {
                    "model": name,
                    "state": {"i": i},
                    "questions": {"refund": {**QUESTION, "label": truth}},
                    "answers": {"refund": {"type": "noul", "noul": p}},
                }
            )
    return cheap, strong


def test_a_later_model_is_only_fitted_on_what_was_passed_on():
    cheap, strong = two_models(3000)
    (first, _, _), (second, _, _) = fit_chain([cheap, strong])
    passed_on = sum(not first.decide({"answers": r["answers"]}, r)["refund"].auto for r in cheap)
    assert 0 < passed_on < 3000
    assert second.questions["refund"]["stats"]["n"] == passed_on
    assert first.questions["refund"]["stats"]["n"] == 3000


def test_the_first_model_is_the_same_as_fitting_it_alone_apart_from_the_certainty():
    cheap, strong = two_models(1000)
    alone = fit(cheap)[0].questions["refund"]
    first = fit_chain([cheap, strong])[0][0].questions["refund"]
    assert first["calibration"] == alone["calibration"] and first["cutoff"] == alone["cutoff"]


def test_a_chain_shares_out_its_certainty():
    cheap, strong = two_models(500)
    fitted = fit_chain([cheap, strong], sure=0.9)
    assert [p.data["target"]["sure"] for p, _, _ in fitted] == pytest.approx([0.95, 0.95])


def test_each_policy_records_what_it_comes_after():
    cheap, strong = two_models(500)
    (first, _, _), (second, _, _) = fit_chain([cheap, strong])
    assert first.data["after"] == [] and second.data["after"] == ["cheap"]


def test_chain_fit_keeps_later_models_nearer_the_target_than_fitting_them_alone():
    """The whole point: fitted alone, a later model's cutoff doesn't hold on what reaches it."""
    cheap, strong = two_models(4000)
    fresh_cheap, fresh_strong = two_models(20000, seed=2, start=4000)
    (first, _, _), (second, _, _) = fit_chain([cheap, strong])
    alone = fit(strong)[0]

    def errors(policy):
        auto = wrong = 0
        for c, s in zip(fresh_cheap, fresh_strong):
            if first.decide({"answers": c["answers"]}, c)["refund"].auto:
                continue
            d = policy.decide({"answers": s["answers"]}, s)["refund"]
            if d.auto:
                auto += 1
                wrong += d.answer != s["questions"]["refund"]["label"]
        return wrong / auto

    assert errors(second) < errors(alone)
    assert errors(second) < 0.07


def test_a_model_that_too_little_reaches_gets_an_empty_policy():
    cheap, strong = two_models(200)
    for rec in cheap:  # a first model that's right and sure every time leaves nothing
        rec["answers"]["refund"]["noul"] = 0.9999 if rec["questions"]["refund"]["label"] else 0.0001
    (first, _, _), (second, _, notes) = fit_chain([cheap, strong])
    assert first.questions["refund"]["cutoff"] is not None
    assert second.questions == {}


def test_bad_data_in_a_later_file_still_raises():
    cheap, strong = two_models(100)
    for rec in strong:
        del rec["answers"]
    with pytest.raises(DataError, match="No model answers"):
        fit_chain([cheap, strong])


def write_preds(tmp_path, name, records):
    path = tmp_path / f"{name}.preds.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return str(path)


def test_cli_fits_a_chain_from_several_files(tmp_path, capsys):
    cheap, strong = two_models(2000)
    a, b = write_preds(tmp_path, "cheap", cheap), write_preds(tmp_path, "strong", strong)
    assert main(["fit", a, b]) == 0
    out = capsys.readouterr().out
    assert "only what cheap passed on" in out
    assert "The whole chain (estimates):" in out and "refund:" in out
    assert json.loads((tmp_path / "strong.policy.json").read_text())["after"] == ["cheap"]
    assert (tmp_path / "strong.report.html").exists()


def test_cli_chain_refuses_single_file_options(tmp_path, capsys):
    cheap, strong = two_models(100)
    a, b = write_preds(tmp_path, "cheap", cheap), write_preds(tmp_path, "strong", strong)
    assert main(["fit", a, b, "--out", "x.json"]) == 2
    assert "only work with one file" in capsys.readouterr().err


def test_cli_saves_an_empty_policy_without_a_report(tmp_path, capsys):
    cheap, strong = two_models(200)
    for rec in cheap:
        rec["answers"]["refund"]["noul"] = 0.9999 if rec["questions"]["refund"]["label"] else 0.0001
    a, b = write_preds(tmp_path, "cheap", cheap), write_preds(tmp_path, "strong", strong)
    assert main(["fit", a, b]) == 0
    assert "too little reached this model" in capsys.readouterr().out
    assert (tmp_path / "strong.policy.json").exists() and not (tmp_path / "strong.report.html").exists()


def test_a_safe_chain_passes_on_what_safe_cutoffs_do_not_cover():
    cheap, strong = two_models(3000)
    normal = fit_chain([cheap, strong])
    safe = fit_chain([cheap, strong], safe=True)
    assert [p.data["safe_chain"] for p, _, _ in normal] == [False, False]
    assert [p.data["safe_chain"] for p, _, _ in safe] == [True, True]
    assert safe[1][0].questions["refund"]["stats"]["n"] > normal[1][0].questions["refund"]["stats"]["n"]


def test_cli_fits_a_safe_chain(tmp_path, capsys):
    cheap, strong = two_models(1000)
    a, b = write_preds(tmp_path, "cheap", cheap), write_preds(tmp_path, "strong", strong)
    assert main(["fit", a, b, "--safe"]) == 0
    assert "at most 5.0% errors" in capsys.readouterr().out  # what safe cutoffs certify, not a tighter guess
    assert json.loads((tmp_path / "strong.policy.json").read_text())["safe_chain"] is True


def test_cli_explains_that_safe_is_for_chains(tmp_path, capsys):
    cheap, _ = two_models(100)
    assert main(["fit", write_preds(tmp_path, "cheap", cheap), "--safe"]) == 2
    assert "for fitting a chain" in capsys.readouterr().err


# ---------------------------------------------------------------- routes must match how policies were fitted


@pytest.fixture
def chain_config(tmp_path):
    cheap, strong = two_models(1000)
    (first, _, _), (second, _, _) = fit_chain([cheap, strong])
    first.save(str(tmp_path / "cheap.json"))
    second.save(str(tmp_path / "strong.json"))
    fit(strong)[0].save(str(tmp_path / "alone.json"))

    def write(route, models=("cheap", "strong", "alone")):
        specs = {m: {"url": "http://x", "policy": f"{m}.json"} for m in models}
        path = tmp_path / "gateway.json"
        path.write_text(json.dumps({"models": specs, "routes": {"r": route}}))
        return str(path)

    return write


def test_a_route_in_the_fitted_order_is_fine(chain_config):
    assert load_config(chain_config(["cheap", "strong"])).warnings == []


def test_a_route_in_another_order_is_refused(chain_config):
    with pytest.raises(DataError, match="fitted after"):
        load_config(chain_config(["strong", "cheap"]))


def test_a_later_model_fitted_alone_gets_a_warning(chain_config):
    warnings = load_config(chain_config(["cheap", "alone"])).warnings
    assert len(warnings) == 1 and "wasn't fitted for this chain" in warnings[0]


def test_a_single_model_route_needs_no_chain_fit(chain_config):
    assert load_config(chain_config(["alone"])).warnings == []


def test_a_route_must_run_on_the_cutoffs_its_chain_was_fitted_for(tmp_path):
    cheap, strong = two_models(1000)
    (first, _, _), (second, _, _) = fit_chain([cheap, strong])
    first.save(str(tmp_path / "cheap.json"))
    second.save(str(tmp_path / "strong.json"))
    specs = {m: {"url": "http://x", "policy": f"{m}.json", "safe": True} for m in ("cheap", "strong")}
    path = tmp_path / "gateway.json"
    path.write_text(json.dumps({"models": specs, "routes": {"r": ["cheap", "strong"]}}))
    with pytest.raises(DataError, match="without --safe"):
        load_config(str(path))

    (first, _, _), (second, _, _) = fit_chain([cheap, strong], safe=True)
    first.save(str(tmp_path / "cheap.json"))
    second.save(str(tmp_path / "strong.json"))
    assert load_config(str(path)).warnings == []
    for spec in specs.values():
        spec["safe"] = False
    path.write_text(json.dumps({"models": specs, "routes": {"r": ["cheap", "strong"]}}))
    with pytest.raises(DataError, match="with --safe"):
        load_config(str(path))
