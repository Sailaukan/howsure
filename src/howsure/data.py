"""Labeled examples and model answers, in the System One request/response format.

A labeled example is a System One request plus a label per question, either inside
the question (the Kev training format) or in a top-level ``labels`` object:

    {"state": "...", "questions": {"team": {"type": "choice", ..., "label": "billing"}}}
    {"state": "...", "labels": {"team": "billing"}}          # with --questions file
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

SUPPORTED_TYPES = ("choice", "noul", "score")
_TRUE = {"true", "yes", "y", "1"}
_FALSE = {"false", "no", "n", "0"}


class DataError(ValueError):
    """A problem with the user's data, with a message meant for people."""


def _hash(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- reading files


def read_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise DataError(
                    f"{path}:{lineno}: not valid JSON ({e.msg}). Each line must be one JSON object."
                ) from None
            if not isinstance(rec, dict):
                raise DataError(f"{path}:{lineno}: expected a JSON object")
            records.append(rec)
    return records


def read_questions(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("questions"), dict):
        data = data["questions"]  # a whole request was given; take its questions
    if not isinstance(data, dict) or not data:
        raise DataError(f"{path}: expected an object of named questions")
    return data


def read_csv(path: str, questions: dict) -> list[dict]:
    """Columns named after a question hold its label; all other columns become the state."""
    records = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            state = {k: v for k, v in row.items() if k is not None and k not in questions}
            labels = {k: row[k] for k in questions if row.get(k) not in (None, "")}
            records.append({"state": state, "labels": labels})
    return records


def load_examples(path: str, questions_path: str | None = None) -> list[dict]:
    shared = read_questions(questions_path) if questions_path else None
    if path.lower().endswith(".csv"):
        if shared is None:
            raise DataError("A CSV file needs --questions questions.json (the questions your app sends).")
        records = read_csv(path, shared)
    else:
        records = read_jsonl(path)
    for i, rec in enumerate(records, 1):
        if "questions" not in rec:
            if shared is None:
                raise DataError(
                    f"{path}: example {i} has no 'questions'. "
                    "Put them on every line or pass --questions questions.json."
                )
            rec["questions"] = copy.deepcopy(shared)
        if "state" not in rec:
            raise DataError(f"{path}: example {i} has no 'state'.")
        if not isinstance(rec["questions"], dict):
            raise DataError(f"{path}: example {i}: 'questions' must be an object")
    return records


# ---------------------------------------------------------------- requests and ids


def strip_label(qdef: dict) -> dict:
    return {k: v for k, v in qdef.items() if k != "label"}


def request_of(rec: dict, model: str | None = None) -> dict:
    """The System One request for an example: its state and questions, without labels."""
    req = {"state": rec["state"], "questions": {qid: strip_label(q) for qid, q in rec["questions"].items()}}
    if model:
        req["model"] = model
    return req


def example_id(rec: dict) -> str:
    """Model-independent id of an example (state + questions, labels excluded)."""
    return _hash(request_of(rec))


def fingerprint(qdef: dict) -> str:
    """Identifies a question's wording and options, so a policy notices when they change."""
    return _hash({k: qdef.get(k) for k in ("type", "instructions", "criteria")})


# ---------------------------------------------------------------- labels and answers


def label_of(rec: dict, qid: str) -> Any:
    q = (rec.get("questions") or {}).get(qid) or {}
    if q.get("label") is not None:
        return q["label"]
    return (rec.get("labels") or {}).get(qid)


def option_names(qdef: dict) -> list[str]:
    """Choice options (criteria keys) or score levels (criteria list), in order."""
    crit = qdef.get("criteria")
    if isinstance(crit, dict):
        return list(crit.keys())
    if isinstance(crit, list):
        return [str(c) for c in crit]
    return []


def parse_label(qdef: dict, value: Any, options: list[str] | None = None) -> int:
    """Index of the correct option. noul: 0 = no, 1 = yes. score: level position from 0."""
    qtype = qdef.get("type")
    if qtype == "noul":
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)) and value in (0, 1):
            return int(value)
        s = str(value).strip().lower()
        if s in _TRUE:
            return 1
        if s in _FALSE:
            return 0
        raise DataError(f"label {value!r} is not yes/no (use true or false)")

    opts = options if options is not None else option_names(qdef)
    if qtype == "score" and not isinstance(value, bool):
        if isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit()):
            idx = int(value)
            if opts and not 0 <= idx < len(opts):
                raise DataError(f"level {idx} is out of range (0 to {len(opts) - 1})")
            return idx
    s = str(value)
    if s in opts:
        return opts.index(s)
    lowered = [o.lower() for o in opts]
    if s.strip().lower() in lowered:
        return lowered.index(s.strip().lower())
    shown = ", ".join(opts[:12]) + (", ..." if len(opts) > 12 else "")
    raise DataError(f"label {value!r} is not one of the options ({shown})")


def answer_probs(qtype: str, answer: dict, levels: list[str] | None = None) -> tuple[list[str], list[float]]:
    """Option keys and normalized probabilities from one System One answer.

    choice: keys are option names; score: level indices as strings ("0", "1", ...);
    noul: ["false", "true"].
    """
    if qtype == "noul":
        p = min(1.0, max(0.0, float(answer["noul"])))
        return ["false", "true"], [1.0 - p, p]
    probs = answer.get("probabilities")
    if not isinstance(probs, dict) or not probs:
        raise ValueError("answer has no probabilities")
    if qtype == "score":
        legend = list(answer.get("legend") or levels or [])

        def level(k: str) -> int:
            if str(k).isdigit():
                return int(k)
            if k in legend:
                return legend.index(k)
            raise ValueError(f"unknown level {k!r}")

        items = sorted(((level(k), float(v)) for k, v in probs.items()), key=lambda t: t[0])
        keys, vals = [str(i) for i, _ in items], [v for _, v in items]
    else:
        keys, vals = list(probs.keys()), [float(v) for v in probs.values()]
    total = sum(vals)
    if total <= 0:
        raise ValueError("probabilities sum to zero")
    return keys, [v / total for v in vals]


@dataclass
class Obs:
    """One labeled answer: what the model said and what was right."""

    eid: str
    probs: list[float]
    label: int  # index into probs
    keys: list[str]
    raw_score: float | None = None


def validate_labels(records: list[dict]) -> Counter:
    """Check every label before spending model calls. Returns labeled counts per question."""
    counts: Counter = Counter()
    errors = []
    for i, rec in enumerate(records, 1):
        for qid, qdef in rec["questions"].items():
            value = label_of(rec, qid)
            if value is None or qdef.get("type") not in SUPPORTED_TYPES:
                continue
            try:
                parse_label(qdef, value)
                counts[qid] += 1
            except DataError as e:
                errors.append(f"example {i}, question {qid!r}: {e}")
    if errors:
        more = f"\n  ... and {len(errors) - 10} more" if len(errors) > 10 else ""
        raise DataError("Some labels don't match their questions:\n  " + "\n  ".join(errors[:10]) + more)
    if not counts:
        raise DataError(
            "No labels found. Add a 'label' to each question (or a 'labels' object, or label columns in a CSV)."
        )
    return counts


def observations(records: list[dict]) -> tuple[dict[str, list[Obs]], dict[str, dict], list[str]]:
    """Group labeled answers by question. Returns (observations, question info, notes)."""
    obs: dict[str, list[Obs]] = {}
    info: dict[str, dict] = {}
    skipped: Counter = Counter()
    any_answers = False
    for i, rec in enumerate(records, 1):
        answers = rec.get("answers")
        if not isinstance(answers, dict):
            continue
        any_answers = True
        eid = example_id(rec)
        for qid, qdef in rec["questions"].items():
            value = label_of(rec, qid)
            if value is None:
                continue
            qtype = qdef.get("type")
            if qtype not in SUPPORTED_TYPES:
                skipped[(qid, f"type {qtype!r} is not supported")] += 1
                continue
            ans = answers.get(qid)
            if not isinstance(ans, dict):
                skipped[(qid, "no answer from the model")] += 1
                continue
            try:
                keys, probs = answer_probs(qtype, ans, option_names(qdef))
            except (KeyError, TypeError, ValueError):
                skipped[(qid, "unreadable answer")] += 1
                continue
            try:
                if qtype == "choice":
                    y = parse_label(qdef, value, keys)
                else:
                    y = parse_label(qdef, value)
            except DataError as e:
                raise DataError(f"example {i}, question {qid!r}: {e}") from None
            if qtype == "score":
                if str(y) not in keys:
                    skipped[(qid, "label level missing from the answer")] += 1
                    continue
                y = keys.index(str(y))
            entry = info.setdefault(qid, {"type": qtype, "fingerprints": set()})
            if entry["type"] != qtype:
                raise DataError(f"question {qid!r} is {entry['type']} in some examples and {qtype} in others")
            entry["fingerprints"].add(fingerprint(qdef))
            raw_score = ans.get("score") if qtype == "score" else None
            obs.setdefault(qid, []).append(Obs(eid, probs, y, keys, raw_score))
    if not any_answers:
        raise DataError(
            "No model answers found. Run `howsure run` first, or add an 'answers' object to each line."
        )
    notes = [f"{qid}: skipped {n} example(s): {why}" for (qid, why), n in skipped.items()]
    return obs, info, notes
