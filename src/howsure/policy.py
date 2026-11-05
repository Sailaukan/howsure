"""Fit a policy from labeled answers, and apply it to live System One responses.

A policy is a small JSON file: per question, a calibration (temperature or Platt
scaling) and two cutoffs on the calibrated probability of the top answer:

* ``cutoff``: the most automation whose error on held-out examples meets the target.
* ``safe_cutoff``: the most automation whose error meets the target with the chosen
  certainty (default 95%), or null when there aren't enough examples to promise it.
"""

from __future__ import annotations

import copy
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from . import stats
from .data import DataError, Obs, _hash, answer_probs, fingerprint, observations

FORMAT_VERSION = 1
MIN_LABELS = 20
HOLD_OUT_FROM = 80  # with fewer labeled examples, fit and check on the same data


# ---------------------------------------------------------------- calibration helpers


def fit_calibration(qtype: str, rows: list[Obs]) -> dict:
    pairs = [(r.probs, r.label) for r in rows]
    if qtype == "noul":
        a, b = stats.fit_platt(pairs)
        return {"method": "platt", "a": a, "b": b}
    return {"method": "temperature", "temperature": stats.fit_temperature(pairs)}


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
            parts.append("says yes too easily")
        elif b > 0.2:
            parts.append("says no too easily")
        return "model was " + " and ".join(parts) if parts else "model was already about right"
    t = calibration["temperature"]
    if t > 1.1:
        return f"model was overconfident; temperature {t:.2f}"
    if t < 0.9:
        return f"model was underconfident; temperature {t:.2f}"
    return f"model was already about right; temperature {t:.2f}"


# ---------------------------------------------------------------- fitting


def fit(
    records: list[dict],
    max_error: float = 0.05,
    sure: float = 0.95,
    name: str | None = None,
    fit_share: float = 0.25,
) -> tuple["Policy", dict, list[str]]:
    """Fit a policy. Returns (policy, chart data for the report, notes)."""
    if not 0 < max_error < 1:
        raise DataError("--max-error must be between 0 and 1, e.g. 0.05 for 5%")
    if not 0 < sure < 1:
        raise DataError("--sure must be between 0 and 1, e.g. 0.95")
    delta = 1 - sure
    obs, info, notes = observations(records)
    if not obs:
        raise DataError("No labeled questions with model answers found.")

    questions: dict[str, dict] = {}
    charts: dict[str, dict] = {}
    for qid, rows in obs.items():
        qtype = info[qid]["type"]
        if len(rows) < MIN_LABELS:
            notes.append(f"{qid}: only {len(rows)} labeled examples, need at least {MIN_LABELS}; skipped")
            continue
        if max(len(r.probs) for r in rows) < 2:
            notes.append(f"{qid}: has a single option, nothing to decide; skipped")
            continue

        # Split by example id: deterministic and the same for every model on the same data.
        rows = sorted(rows, key=lambda r: r.eid)
        n = len(rows)
        if n >= HOLD_OUT_FROM:
            n_fit = min(300, max(30, round(fit_share * n)))
            fit_rows, check_rows, held_out = rows[:n_fit], rows[n_fit:], True
        else:
            fit_rows = check_rows = rows
            held_out = False

        calibration = fit_calibration(qtype, fit_rows)
        raw = [(r.probs, r.label) for r in check_rows]
        cal = [(calibrate(calibration, r.probs), r.label) for r in check_rows]
        before, after = stats.summarize(raw), stats.summarize(cal)
        curve = stats.risk_curve(cal)
        n_check = len(check_rows)

        expected = stats.expected_cutoff(curve, max_error)
        safe = stats.safe_cutoff(curve, max_error, delta)
        s: dict[str, Any] = {
            "n": n,
            "n_fit": len(fit_rows),
            "n_check": n_check,
            "held_out": held_out,
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
            "safe_error_bound": stats.upper_bound(safe[2], safe[1], delta) if safe else None,
        }
        if expected and s["coverage"] - s["safe_coverage"] > 0.02:
            # What more labels would likely buy, assuming they look like these.
            more = {str(f): round(stats.safe_coverage_with_more(curve, max_error, delta, f), 4) for f in (2, 4)}
            if more["4"] >= s["safe_coverage"] + 0.05:  # only worth saying if it would help
                s["safe_coverage_with_more"] = more
        if qtype == "score":
            pairs = [
                (r.raw_score, int(r.keys[stats.argmax(r.probs)]))
                for r in check_rows
                if isinstance(r.raw_score, (int, float))
            ]
            if pairs:
                s["score_rounding_mismatch"] = sum(math.floor(x + 0.5) != top for x, top in pairs) / len(pairs)

        fps = info[qid]["fingerprints"]
        questions[qid] = {
            "type": qtype,
            "fingerprint": next(iter(fps)) if len(fps) == 1 else None,
            "calibration": calibration,
            "cutoff": expected[0] if expected else None,
            "safe_cutoff": safe[0] if safe else None,
            "stats": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()},
        }
        charts[qid] = {
            "reliability_before": stats.reliability(raw),
            "reliability_after": stats.reliability(cal),
            "curve": [(cov / n_check, err / cov, t) for t, cov, err in curve],
            "cutoff_point": (expected[1] / n_check, expected[2] / expected[1]) if expected else None,
            "safe_point": (safe[1] / n_check, safe[2] / safe[1]) if safe else None,
        }

    if not questions:
        raise DataError("No question had enough labeled examples to fit.\n  " + "\n  ".join(notes))

    models = Counter(r.get("model") for r in records if r.get("answers") and r.get("model"))
    data_hash = _hash(sorted(f"{qid}:{o.eid}:{o.label}" for qid, rows in obs.items() for o in rows))
    policy = Policy(
        {
            "howsure": FORMAT_VERSION,
            "model": name or (models.most_common(1)[0][0] if models else "unknown"),
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "target": {"max_error": max_error, "sure": sure},
            "data": {"examples": sum(1 for r in records if r.get("answers")), "hash": data_hash},
            "questions": questions,
        }
    )
    return policy, charts, notes


# ---------------------------------------------------------------- applying


@dataclass
class Decision:
    """What to do with one answer."""

    answer: Any  # choice: option name; noul: True/False; score: level index
    probability: float  # calibrated probability of `answer`
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
    def load(cls, path: str) -> "Policy":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    @property
    def model(self) -> str:
        return self.data.get("model", "unknown")

    def decide(self, response: dict, request: dict | None = None, safe: bool = False) -> dict[str, Decision]:
        """Decisions for every calibrated question in one System One response.

        Pass the request too, and a question whose wording or options changed since
        calibration goes to review instead of trusting a stale calibration.
        """
        out: dict[str, Decision] = {}
        answers = response.get("answers") or {}
        asked = (request or {}).get("questions") or {}
        for qid, ans in answers.items():
            q = self.questions.get(qid)
            if q is None or not isinstance(ans, dict):
                continue
            reason = None
            if ans.get("type", q["type"]) != q["type"]:
                reason = "question type changed since calibration"
            elif qid in asked and q.get("fingerprint") and fingerprint(asked[qid]) != q["fingerprint"]:
                reason = "question changed since calibration"
            try:
                keys, probs = answer_probs(q["type"], ans)
            except (KeyError, TypeError, ValueError):
                out[qid] = Decision(None, 0.0, False, "unreadable answer")
                continue
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

    def apply(self, response: dict, request: dict | None = None, safe: bool = False) -> dict:
        """A copy of the response with a ``howsure`` block added to each calibrated answer.

        Handles single responses and batch responses (``{"results": [...]}``).
        """
        out = copy.deepcopy(response)
        if isinstance(out.get("results"), list):
            reqs = (request or {}).get("requests") or []
            for i, res in enumerate(out["results"]):
                if isinstance(res, dict):
                    self._annotate(res, reqs[i] if i < len(reqs) else None, safe)
        else:
            self._annotate(out, request, safe)
        return out

    def _annotate(self, response: dict, request: dict | None, safe: bool) -> None:
        for qid, d in self.decide(response, request, safe).items():
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
