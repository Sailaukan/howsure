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
SUM_TOLERANCE = 0.02  # models round their probabilities, so they don't add up to exactly 1
_TRUE = {"true", "yes", "y", "1"}
_FALSE = {"false", "no", "n", "0"}


class DataError(ValueError):
    """A problem with the user's data, with a message meant for people."""


class BadAnswer(ValueError):
    """A model's answer that can't be trusted. The message says why."""


def stable_hash(obj: Any) -> str:
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
    return stable_hash(request_of(rec))


def fingerprint(qdef: dict) -> str:
    """Identifies a question's wording and options, so a policy notices when they change."""
    return stable_hash({k: qdef.get(k) for k in ("type", "instructions", "criteria")})


def field_of(rec: dict, path: str) -> str | None:
    """A record's value at a dotted path like ``meta.doc_type``, as text."""
    value: Any = rec
    for part in path.split("."):
        if not isinstance(value, dict) or value.get(part) is None:
            return None
        value = value[part]
    return str(value)


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


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def answer_probs(qtype: str, answer: dict, options: list[str] | None = None) -> tuple[list[str], list[float]]:
    """Option keys and normalized probabilities from one System One answer.

    choice: keys are option names; score: level indices as strings ("0", "1", ...);
    noul: ["false", "true"]. Pass the question's options to check the answer covers them.

    Raises BadAnswer rather than guess: rescaling a distribution over options nobody asked
    about, or over only some of them, would turn it into a confident answer.
    """
    if qtype == "noul":
        p = answer.get("noul")
        if not _is_number(p) or not 0 <= p <= 1:
            raise BadAnswer("unreadable answer")
        return ["false", "true"], [1.0 - p, p]
    probs = answer.get("probabilities")
    if not isinstance(probs, dict) or not probs or not all(_is_number(v) and v >= 0 for v in probs.values()):
        raise BadAnswer("unreadable answer")
    if qtype == "score":
        legend = answer.get("legend") if isinstance(answer.get("legend"), list) else (options or [])

        def level(k: str) -> int:
            if str(k).isdecimal():
                return int(k)
            if k in legend:
                return legend.index(k)
            raise BadAnswer("answer's options don't match the question")

        items = sorted(((level(k), v) for k, v in probs.items()), key=lambda t: t[0])
        keys, vals = [str(i) for i, _ in items], [v for _, v in items]
        matches = not options or [i for i, _ in items] == list(range(len(options)))
    else:
        keys, vals = list(probs.keys()), list(probs.values())
        matches = not options or set(keys) == set(options)
    if not matches:
        raise BadAnswer("answer's options don't match the question")
    total = sum(vals)
    if abs(total - 1) > SUM_TOLERANCE:
        raise BadAnswer("probabilities don't add up to 1")
    return keys, [v / total for v in vals]


@dataclass
class Obs:
    """One labeled answer: what the model said and what was right."""

    eid: str
    probs: list[float]
    label: int  # index into probs
    keys: list[str]
    raw_score: float | None = None
    group: str | None = None  # document type or similar, from ``fit --by``


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


def observations(records: list[dict], by: str | None = None) -> tuple[dict[str, list[Obs]], dict[str, dict], list[str]]:
    """Group labeled answers by question. Returns (observations, question info, notes)."""
    obs: dict[str, list[Obs]] = {}
    info: dict[str, dict] = {}
    skipped: Counter = Counter()
    any_answers = False
    grouped = ungrouped = 0
    seen: set[tuple] = set()
    for i, rec in enumerate(records, 1):
        answers = rec.get("answers")
        if not isinstance(answers, dict):
            continue
        any_answers = True
        eid = example_id(rec)
        group = field_of(rec, by) if by else None
        if by:
            grouped += group is not None
            ungrouped += group is None
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
            except BadAnswer as e:
                skipped[(qid, str(e))] += 1
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
            if (qid, eid, y) in seen:  # an exact repeat isn't new evidence, and would make every promise overconfident
                skipped[(qid, "exact duplicate of another example")] += 1
                continue
            seen.add((qid, eid, y))
            raw_score = ans.get("score") if qtype == "score" else None
            obs.setdefault(qid, []).append(Obs(eid, probs, y, keys, raw_score, group))
    if not any_answers:
        raise DataError("No model answers found. Run `howsure run` first, or add an 'answers' object to each line.")
    if by and not grouped:
        raise DataError(f"No example has a {by!r} field to group by. It can be a dotted path like meta.doc_type.")
    notes = [f"{qid}: skipped {n} example(s): {why}" for (qid, why), n in skipped.items()]
    if ungrouped:
        notes.append(f"{ungrouped} example(s) have no {by!r} field, so they're left out of the per-group results")
    return obs, info, notes
