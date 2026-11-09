"""Fit a policy from labeled answers, and apply it to live System One responses.

A policy is a small JSON file: per question, a calibration (temperature or Platt
scaling) and two cutoffs on the calibrated probability of the top answer:

* ``cutoff``: the most automation whose error on held-out examples meets the target,
  with a margin of one standard error.
* ``safe_cutoff``: the most automation whose error meets the target with the chosen
  certainty (default 95%), or null when there aren't enough examples to promise it.

Neither may pass a cutoff at which any true answer is clearly auto-handled wrong too often.
"""

from __future__ import annotations

import copy
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import NormalDist
from typing import Any

from . import drift, stats
from .data import (
    BadAnswer,
    DataError,
    Obs,
    answer_probs,
    example_id,
    fingerprint,
    observations,
    option_names,
    stable_hash,
)

FORMAT_VERSION = 1
MIN_LABELS = 20
HOLD_OUT_FROM = 80  # with fewer labeled examples nothing can be checked on unseen data, so nothing is auto-handled
MIN_FIT, MAX_FIT = (
    30,
    300,
)  # calibration only has one or two parameters, so the other examples are better spent checking


# ---------------------------------------------------------------- calibration helpers


def fit_calibration(qtype: str, rows: list[Obs]) -> dict:
    pairs = [(r.probs, r.label) for r in rows]
    if qtype == "noul":
        a, b = stats.fit_platt(pairs)
        return {"method": "platt", "a": a, "b": b}
    return {"method": "temperature", "temperature": stats.fit_temperature(pairs)}


def no_calibration(qtype: str) -> dict:
    if qtype == "noul":
        return {"method": "platt", "a": 1.0, "b": 0.0}
    return {"method": "temperature", "temperature": 1.0}


def calibrate(calibration: dict, probs: list[float]) -> list[float]:
    if calibration["method"] == "platt":
        q = stats.platt(probs[1], calibration["a"], calibration["b"])
        return [1.0 - q, q]
    return stats.scale_temperature(probs, calibration["temperature"])


def describe(calibration: dict) -> str:
    """Plain-language reading of a calibration."""
    if calibration["method"] == "platt":
        a, b = calibration["a"], calibration["b"]
        parts = []
        if a < 0:
            return "its yes/no probabilities point the wrong way on this data"
        if a < 0.9:
            parts.append("overconfident")
        elif a > 1.1:
            parts.append("underconfident")
        if b < -0.2:
            parts.append("too quick to say yes")
        elif b > 0.2:
            parts.append("too quick to say no")
        return "model was " + " and ".join(parts) if parts else "model was already about right"
    t = calibration["temperature"]
    if t > 1.1:
        return f"model was overconfident; temperature {t:.2f}"
    if t < 0.9:
        return f"model was underconfident; temperature {t:.2f}"
    return f"model was already about right; temperature {t:.2f}"


# ---------------------------------------------------------------- fitting one question


def split_rows(rows: list[Obs], fit_share: float) -> tuple[list[Obs], list[Obs], bool]:
    """Fit rows, check rows, whether they differ. By example id: the same for every model on the same data."""
    rows = sorted(rows, key=lambda r: r.eid)
    if len(rows) < HOLD_OUT_FROM:
        return rows, rows, False
    n_fit = min(MAX_FIT, max(MIN_FIT, round(fit_share * len(rows))))
    return rows[:n_fit], rows[n_fit:], True


@dataclass
class _Fitted:
    calibration: dict
    fit_rows: list
    check_rows: list
    held_out: bool
    raw: list  # (probabilities, label) on the check rows, before calibration
    cal: list  # and after
    usable: list  # the part of the risk curve a cutoff may come from
    expected: tuple | None
    safe: tuple | None


def _fit_question(qtype: str, rows: list[Obs], max_error: float, delta: float, fit_share: float) -> _Fitted:
    fit_rows, check_rows, held_out = split_rows(rows, fit_share)
    calibration = fit_calibration(qtype, fit_rows)
    raw = [(r.probs, r.label) for r in check_rows]
    cal = [(calibrate(calibration, r.probs), r.label) for r in check_rows]
    if held_out and stats.summarize(cal)["log_loss"] >= stats.summarize(raw)["log_loss"]:
        calibration = no_calibration(qtype)  # it didn't help on examples it hadn't seen
        cal = [(calibrate(calibration, r.probs), r.label) for r in check_rows]
    usable: list = []
    if held_out:
        floor = stats.answer_floor(cal, [r.keys[r.label] for r in check_rows], max_error, delta)
        usable = [c for c in stats.risk_curve(cal) if c[0] > floor]
    return _Fitted(
        calibration,
        fit_rows,
        check_rows,
        held_out,
        raw,
        cal,
        usable,
        stats.expected_cutoff(usable, max_error, delta),
        stats.safe_cutoff(usable, max_error, delta),
    )


def _reference(fitted: _Fitted) -> dict | None:
    if not fitted.held_out:
        return None
    rows = fitted.check_rows
    return drift.reference([r.keys[stats.argmax(r.probs)] for r in rows], [max(r.probs) for r in rows])


def _chart(fitted: _Fitted) -> dict:
    n_check, expected, safe = len(fitted.check_rows), fitted.expected, fitted.safe
    return {
        "reliability_before": stats.reliability(fitted.raw),
        "reliability_after": stats.reliability(fitted.cal),
        "curve": [(cov / n_check, err / cov, t) for t, cov, err in stats.risk_curve(fitted.cal)],
        "cutoff_point": (expected[1] / n_check, expected[2] / expected[1]) if expected else None,
        "safe_point": (safe[1] / n_check, safe[2] / safe[1]) if safe else None,
    }


# ---------------------------------------------------------------- checks on a fitted question


def _clearly_above(k: int, n: int, rate: float, delta: float) -> bool:
    """Whether k out of n is more than chance at this rate would explain, at level delta."""
    return k > 0 and 1 - stats.binom_cdf(k - 1, n, rate) <= delta


def _clearly_worse(wrong: int, n: int, base_wrong: int, base_n: int, delta: float) -> bool:
    """Whether an error rate is clearly above another, allowing for noise in both (two-proportion z-test)."""
    if not n or not base_n:
        return False
    pooled = (wrong + base_wrong) / (n + base_n)
    if pooled in (0, 1):
        return False
    z = (wrong / n - base_wrong / base_n) / math.sqrt(pooled * (1 - pooled) * (1 / n + 1 / base_n))
    return z >= NormalDist().inv_cdf(1 - delta)


def _uncommon_paid(common: dict, other: dict, max_error: float, delta: float) -> bool:
    """Whether calibration clearly gained on the common answer at the expense of the others.

    Correcting a lean always flips some answers of both kinds, and calibrating moves decisions a
    little towards the common answer, so a small fall in balanced accuracy (the mean of the two
    groups' accuracy) is normal. This asks for a fall bigger than the error budget, and not noise.
    """
    if not common["n"] or not other["n"]:
        return False
    gains = (common["gained_from_calibration"], common["lost_to_calibration"])
    losses = (other["gained_from_calibration"], other["lost_to_calibration"])
    change = (gains[0] - gains[1]) / common["n"] + (losses[0] - losses[1]) / other["n"]
    variance = sum(gains) / common["n"] ** 2 + sum(losses) / other["n"] ** 2
    return change < -max_error and change / math.sqrt(variance) <= -NormalDist().inv_cdf(1 - delta)


def _answer_checks(check_rows: list[Obs], raw: list, cal: list, cutoff: float | None) -> dict:
    """How always giving the most common answer would score, and how the model does on the other answers."""
    truths = [r.keys[r.label] for r in check_rows]
    common, count = Counter(truths).most_common(1)[0]
    right = wrong = review = 0
    flips = {True: Counter(), False: Counter()}  # is this example's truth the common answer?
    for r, (raw_probs, y), (cal_probs, _) in zip(check_rows, raw, cal):
        is_common = r.keys[r.label] == common
        raw_ok, cal_ok = stats.argmax(raw_probs) == y, stats.argmax(cal_probs) == y
        flips[is_common]["lost"] += raw_ok and not cal_ok
        flips[is_common]["gained"] += cal_ok and not raw_ok
        if is_common:
            continue
        if cutoff is None or max(cal_probs) < cutoff:
            review += 1
        elif cal_ok:
            right += 1
        else:
            wrong += 1
    return {
        "common_answer": common,
        "common_accuracy": count / len(truths),
        "common": {
            "n": count,
            "lost_to_calibration": flips[True]["lost"],
            "gained_from_calibration": flips[True]["gained"],
        },
        "uncommon": {
            "n": len(truths) - count,
            "right": right,
            "wrong": wrong,
            "review": review,
            "lost_to_calibration": flips[False]["lost"],
            "gained_from_calibration": flips[False]["gained"],
        },
    }


def _answer_warnings(s: dict, max_error: float, delta: float) -> list[str]:
    """A low error rate can hide a model that only ever gives the common answer. Say so when it does."""
    if not s["held_out"]:  # the other checks would be judging a model on the examples it was fitted on
        return [
            f"only {s['n']} labeled examples, and at least {HOLD_OUT_FROM} are needed before anything is auto-handled"
        ]
    common, other = s["common_answer"], s["uncommon"]
    out = []
    needed = stats.min_examples(max_error, delta)
    if not other["n"]:
        out.append(
            f"every held-out example has the answer {common!r}, so these numbers say nothing about any other answer"
        )
    elif other["n"] < needed:
        out.append(
            f"only {other['n']} held-out examples have an answer other than {common!r}, and {needed} are needed "
            "to show they aren't being missed, so label more of them"
        )
    if other["n"] and s["accuracy_after"] <= s["common_accuracy"]:
        out.append(
            f"always answering {common!r} would be right {100 * s['common_accuracy']:.1f}% of the time, "
            f"and the model is right {100 * s['accuracy_after']:.1f}%"
        )
    if _uncommon_paid(s["common"], other, max_error, delta):
        out.append(
            f"calibration helps the answer {common!r} at the expense of the others: it turned "
            f"{other['lost_to_calibration']} of the {other['n']} less common answers from right to wrong"
        )
    if _clearly_above(other["wrong"], other["wrong"] + other["right"], 0.5, delta):
        out.append(
            f"of the less common answers it auto-handles, {other['wrong']} are wrong and {other['right']} right, "
            "which the overall error rate hides"
        )
    return out


def _auto_counts(rows: list[Obs], calibration: dict, cutoff: float | None) -> tuple[int, int]:
    """(auto-handled, wrong among them) under one calibration and cutoff."""
    auto = wrong = 0
    for r in rows:
        probs = calibrate(calibration, r.probs)
        if cutoff is not None and max(probs) >= cutoff:
            auto += 1
            wrong += stats.argmax(probs) != r.label
    return auto, wrong


def _group_checks(
    qtype: str,
    rows: list[Obs],
    check_rows: list[Obs],
    calibration: dict,
    cutoff: float | None,
    max_error: float,
    delta: float,
    fit_share: float,
) -> tuple[list[dict], list[str]]:
    """Per group: this policy's results, and those of a policy fitted without that group.

    The second number is the one to trust for documents the policy has never seen.
    """
    groups = sorted({r.group for r in rows if r.group is not None})
    if len(groups) < 2:
        return [], []
    alpha = delta / (2 * len(groups))  # two tests per group, so more groups mustn't mean more false alarms
    out, warnings = [], []
    rate = lambda k, n: round(k / n, 6) if n else None  # noqa: E731
    for group in groups:
        here = [r for r in rows if r.group == group]
        rest = [r for r in rows if r.group != group]
        checked = [r for r in check_rows if r.group == group]
        auto, wrong = _auto_counts(checked, calibration, cutoff)
        others = _auto_counts([r for r in check_rows if r.group != group], calibration, cutoff)
        entry = {
            "group": group,
            "n": len(here),
            "n_check": len(checked),
            "coverage": rate(auto, len(checked)),
            "error": rate(wrong, auto),
        }
        if auto and wrong / auto > max_error and _clearly_worse(wrong, auto, others[1], others[0], alpha):
            warnings.append(
                f"this policy makes {100 * wrong / auto:.1f}% errors on {group!r}, "
                f"over the {100 * max_error:.1f}% target "
                "and clearly worse than on the other types"
            )
        if len(rest) >= MIN_LABELS and max(len(r.probs) for r in rest) >= 2:
            without = _fit_question(qtype, rest, max_error, delta, fit_share)
            best = without.expected
            auto, wrong = _auto_counts(here, without.calibration, best[0] if best else None)
            entry["unseen_coverage"], entry["unseen_error"] = rate(auto, len(here)), rate(wrong, auto)
            if best and auto and wrong / auto > max_error and _clearly_worse(wrong, auto, best[2], best[1], alpha):
                warnings.append(
                    f"a policy fitted without {group!r} makes {100 * wrong / auto:.1f}% errors on it, over the "
                    f"{100 * max_error:.1f}% target and clearly worse than it promised, so a policy may not "
                    "carry over to new document types"
                )
        out.append(entry)
    return out, warnings


def _question_stats(
    qtype: str, rows: list[Obs], fitted: _Fitted, max_error: float, delta: float, fit_share: float
) -> dict:
    """What `fit` reports about one question."""
    before, after = stats.summarize(fitted.raw), stats.summarize(fitted.cal)
    expected, safe, n_check = fitted.expected, fitted.safe, len(fitted.check_rows)
    s: dict[str, Any] = {
        "n": len(rows),
        "n_fit": len(fitted.fit_rows),
        "n_check": n_check,
        "held_out": fitted.held_out,
        "accuracy_before": before["accuracy"],
        "accuracy_after": after["accuracy"],
        "ece_before": before["ece"],
        "ece_after": after["ece"],
        "brier_before": before["brier"],
        "brier_after": after["brier"],
        "confident_errors_before": before["confident_errors"],
        "confident_errors_after": after["confident_errors"],
        "coverage": expected[1] / n_check if expected else 0.0,
        "error": expected[2] / expected[1] if expected else None,
        "safe_coverage": safe[1] / n_check if safe else 0.0,
    }
    if expected and s["coverage"] - s["safe_coverage"] > 0.02:
        # What more labels would likely buy, assuming they look like these.
        more = {str(f): round(stats.safe_coverage_with_more(fitted.usable, max_error, delta, f), 4) for f in (2, 4)}
        if more["4"] >= s["safe_coverage"] + 0.05:  # only worth saying if it would help
            s["safe_coverage_with_more"] = more
    if qtype == "score":
        pairs = [
            (r.raw_score, int(r.keys[stats.argmax(r.probs)]))
            for r in fitted.check_rows
            if isinstance(r.raw_score, (int, float))
        ]
        if pairs:
            s["score_rounding_mismatch"] = sum(math.floor(x + 0.5) != top for x, top in pairs) / len(pairs)
    cutoff = expected[0] if expected else None
    s.update(_answer_checks(fitted.check_rows, fitted.raw, fitted.cal, cutoff))
    warnings = _answer_warnings(s, max_error, delta)
    groups, group_warnings = _group_checks(
        qtype, rows, fitted.check_rows, fitted.calibration, cutoff, max_error, delta, fit_share
    )
    if groups:
        s["groups"] = groups
    s["warnings"] = warnings + group_warnings
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}


# ---------------------------------------------------------------- fitting


def fit(
    records: list[dict],
    max_error: float = 0.05,
    sure: float = 0.95,
    name: str | None = None,
    fit_share: float = 0.25,
    by: str | None = None,
    allow_empty: bool = False,
) -> tuple[Policy, dict, list[str]]:
    """Fit a policy. Returns (policy, chart data for the report, notes).

    ``by`` names a field holding each example's document type, for per-group results and warnings.
    It only adds to what's reported: calibrations and cutoffs are the same with or without it.
    ``allow_empty`` returns a policy with no questions rather than raise, for models late in a chain.
    """
    if not 0 < max_error < 1:
        raise DataError("--max-error must be between 0 and 1, e.g. 0.05 for 5%")
    if not 0 < sure < 1:
        raise DataError("--sure must be between 0 and 1, e.g. 0.95")
    delta = 1 - sure
    obs, info, notes = observations(records, by=by) if records or not allow_empty else ({}, {}, [])
    if not obs and not allow_empty:
        raise DataError("No labeled questions with model answers found.")

    questions: dict[str, dict] = {}
    charts: dict[str, dict] = {}
    for qid, rows in obs.items():
        qtype = info[qid]["type"]
        if len(rows) < MIN_LABELS:
            notes.append(f"{qid}: only {len(rows)} labeled example(s), need at least {MIN_LABELS}; skipped")
            continue
        if max(len(r.probs) for r in rows) < 2:
            notes.append(f"{qid}: has a single option, nothing to decide; skipped")
            continue
        fitted = _fit_question(qtype, rows, max_error, delta, fit_share)
        option_sets = {tuple(sorted(r.keys)) for r in rows}
        fingerprints = info[qid]["fingerprints"]
        questions[qid] = {
            "type": qtype,
            "fingerprint": next(iter(fingerprints)) if len(fingerprints) == 1 else None,
            "options": list(next(iter(option_sets))) if qtype != "noul" and len(option_sets) == 1 else None,
            "calibration": fitted.calibration,
            "cutoff": fitted.expected[0] if fitted.expected else None,
            "safe_cutoff": fitted.safe[0] if fitted.safe else None,
            "reference": _reference(fitted),
            "stats": _question_stats(qtype, rows, fitted, max_error, delta, fit_share),
        }
        charts[qid] = _chart(fitted)

    if not questions and not allow_empty:
        raise DataError("No question had enough labeled examples to fit.\n  " + "\n  ".join(notes))

    models = Counter(r.get("model") for r in records if r.get("answers") and r.get("model"))
    data_hash = stable_hash(sorted(f"{qid}:{o.eid}:{o.label}" for qid, rows in obs.items() for o in rows))
    policy = Policy(
        {
            "howsure": FORMAT_VERSION,
            "model": name or (models.most_common(1)[0][0] if models else "unknown"),
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "target": {"max_error": max_error, "sure": sure},
            "data": {"examples": sum(1 for r in records if r.get("answers")), "hash": data_hash},
            "served_models": sorted({r["served_model"] for r in records if r.get("served_model")}),
            "questions": questions,
        }
    )
    return policy, charts, notes


def fit_chain(
    chain: list[list[dict]], max_error: float = 0.05, sure: float = 0.95, by: str | None = None, safe: bool = False
) -> list[tuple]:
    """Fit models in chain order, each only on the questions the models before it pass on.

    Takes each model's labeled answers on the same examples. A later model only ever sees the
    harder cases, so a cutoff fitted on all of them wouldn't hold. Fitting on what actually
    reaches it does, and if too little reaches it, it simply auto-handles nothing.
    Set ``safe`` if the chain will run on safe cutoffs, since those pass more on to the next model.
    Returns (policy, chart data, notes) per model.
    """
    certainty = 1 - (1 - sure) / len(chain)  # shared out, so the whole chain keeps the promise
    decided: set[tuple[str, str]] = set()
    earlier: list[str] = []
    fitted = []
    for records in chain:
        still_open = []
        for rec in records:
            eid = example_id(rec)
            questions = {qid: q for qid, q in rec["questions"].items() if (eid, qid) not in decided}
            if questions:
                still_open.append((eid, {**rec, "questions": questions}))
        policy, charts, notes = fit([rec for _, rec in still_open], max_error, certainty, by=by, allow_empty=True)
        policy.data["after"] = list(earlier)
        policy.data["safe_chain"] = safe
        earlier.append(policy.model)
        fitted.append((policy, charts, notes))
        for eid, rec in still_open:
            for qid, decision in policy.decide({"answers": rec.get("answers") or {}}, rec, safe=safe).items():
                if decision.auto:
                    decided.add((eid, qid))
    return fitted


# ---------------------------------------------------------------- applying


@dataclass
class Decision:
    """What to do with one answer."""

    answer: Any  # choice: option name; noul: True/False; score: level index
    probability: float  # calibrated probability of `answer`, or raw if `reason` says calibration doesn't apply
    auto: bool  # True: act on it; False: send to review
    reason: str | None = None  # why it went to review, if not just low probability
    probabilities: dict = field(default_factory=dict)

    @property
    def action(self) -> str:
        return "auto" if self.auto else "review"


class Policy:
    def __init__(self, data: dict):
        if data.get("howsure") != FORMAT_VERSION or not isinstance(data.get("questions"), dict):
            raise DataError("This doesn't look like a howsure policy file.")
        self.data = data
        self.questions: dict[str, dict] = data["questions"]

    @classmethod
    def load(cls, path: str) -> Policy:
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    @property
    def model(self) -> str:
        return self.data.get("model", "unknown")

    def can_auto(self, qid: str, qdef: dict, safe: bool = False) -> bool:
        """Whether ``decide`` could ever mark this question auto, whatever the model answers."""
        q = self.questions.get(qid)
        if q is None or q.get("safe_cutoff" if safe else "cutoff") is None:
            return False
        if qdef.get("type", q["type"]) != q["type"]:
            return False
        return not (q.get("fingerprint") and fingerprint(qdef) != q["fingerprint"])

    def decide(
        self, response: dict, request: dict | None = None, safe: bool = False, monitor=None
    ) -> dict[str, Decision]:
        """Decisions for every calibrated question in one System One response.

        Goes to review instead of trusting a stale calibration: an answer that can't be read,
        a question whose wording or options changed (pass the request too), a new model
        version, or answers that have drifted (pass a DriftMonitor).
        """
        out: dict[str, Decision] = {}
        answers = response.get("answers") or {}
        asked = (request or {}).get("questions") or {}
        served_models, served = self.data.get("served_models"), response.get("model")
        for qid, ans in answers.items():
            q = self.questions.get(qid)
            if q is None or not isinstance(ans, dict):
                continue
            reason = None
            if ans.get("type", q["type"]) != q["type"]:
                reason = "question type changed since calibration"
            elif qid in asked and q.get("fingerprint") and fingerprint(asked[qid]) != q["fingerprint"]:
                reason = "question changed since calibration"
            elif served_models and served and served not in served_models:
                reason = "model version changed since calibration"
            try:
                options = option_names(asked[qid]) if qid in asked else q.get("options")
                keys, probs = answer_probs(q["type"], ans, options)
            except BadAnswer as e:
                out[qid] = Decision(None, 0.0, False, str(e))
                continue
            top = stats.argmax(probs)
            if monitor is not None and monitor.observe(qid, keys[top], probs[top]) and reason is None:
                reason = "answers have shifted since calibration, so refit this policy"
            cal = calibrate(q["calibration"], probs) if reason is None else probs
            top = stats.argmax(cal)
            cutoff = q.get("safe_cutoff") if safe else q.get("cutoff")
            if reason is None and cutoff is None:
                reason = "not enough evidence for a safe cutoff" if safe else "no cutoff meets the target"
            if q["type"] == "noul":
                answer: Any = top == 1
            elif q["type"] == "score":
                answer = int(keys[top])
            else:
                answer = keys[top]
            auto = reason is None and cal[top] >= cutoff
            out[qid] = Decision(answer, cal[top], auto, reason, dict(zip(keys, cal)))
        return out

    def apply(self, response: dict, request: dict | None = None, safe: bool = False, monitor=None) -> dict:
        """A copy of the response with a ``howsure`` block added to each calibrated answer.

        Handles single responses and batch responses (``{"results": [...]}``).
        """
        out = copy.deepcopy(response)
        if isinstance(out.get("results"), list):
            reqs = (request or {}).get("requests") or []
            for i, res in enumerate(out["results"]):
                if isinstance(res, dict):
                    self._annotate(res, reqs[i] if i < len(reqs) else None, safe, monitor)
        else:
            self._annotate(out, request, safe, monitor)
        return out

    def _annotate(self, response: dict, request: dict | None, safe: bool, monitor=None) -> None:
        for qid, d in self.decide(response, request, safe, monitor).items():
            block: dict[str, Any] = {
                "answer": d.answer,
                "probability": round(d.probability, 4),
                "action": d.action,
                "probabilities": {k: round(v, 4) for k, v in d.probabilities.items()},
            }
            legend = response["answers"][qid].get("legend")
            if not isinstance(legend, list) and request:
                legend = (((request.get("questions") or {}).get(qid)) or {}).get("criteria")
            if self.questions[qid]["type"] == "score" and isinstance(legend, list) and isinstance(d.answer, int):
                if 0 <= d.answer < len(legend):
                    block["level"] = legend[d.answer]
            if d.reason:
                block["reason"] = d.reason
            response["answers"][qid]["howsure"] = block


def load(path: str) -> Policy:
    """Load a policy file written by ``howsure fit``."""
    return Policy.load(path)
