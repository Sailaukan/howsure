"""Command line: howsure demo | run | fit | gateway | apply | compare."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import webbrowser

from . import __version__, client, demo, report
from .data import DataError, load_examples, read_jsonl
from .policy import Policy, describe, fit, fit_chain


def _pct(x, digits=0) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f}%"


def _stem(path: str) -> str:
    base = path
    for ext in (".preds.jsonl", ".jsonl", ".json", ".csv"):
        if base.lower().endswith(ext):
            return base[: -len(ext)]
    return base


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()


# ---------------------------------------------------------------- output


def print_summary(policy: Policy, notes: list[str], out=None) -> None:
    out = out or sys.stdout
    d = policy.data
    max_error, sure = d["target"]["max_error"], d["target"]["sure"]
    w = lambda s="": print(s, file=out)  # noqa: E731
    w(f"howsure · {d['model']} · target: at most {_pct(max_error, 1)} errors · {d['data']['examples']:,} examples")
    if d.get("after"):
        w(f"only what {', then '.join(d['after'])} passed on")
    if not d["questions"]:
        w("too little reached this model to fit it, so it auto-handles nothing")
    for qid, q in d["questions"].items():
        s = q["stats"]
        w()
        w(f"{qid}  ({q['type']}, {s['n']:,} labeled)")
        w(
            f"  accuracy {_pct(s['accuracy_after'])}, calibration error {_pct(s['ece_before'], 1)} -> "
            f"{_pct(s['ece_after'], 1)}  ({describe(q['calibration'])})"
        )
        if q["cutoff"] is None:
            w(f"  no cutoff keeps errors within {_pct(max_error, 1)}: send this question to review")
        else:
            w(
                f"  auto-handle if p >= {q['cutoff']:.3f}  ->  {_pct(s['coverage'])} of cases, "
                f"{_pct(s['error'], 1)} errors on held-out examples"
            )
        if q["safe_cutoff"] is not None:
            w(f"  {_pct(sure)} sure:        p >= {q['safe_cutoff']:.3f}  ->  {_pct(s['safe_coverage'])} of cases")
        else:
            w(f"  {_pct(sure)} sure:        not enough examples yet")
        more = s.get("safe_coverage_with_more")
        if more and q["cutoff"] is not None:
            w(
                f"  more labels: ~{2 * s['n']:,} would likely make ~{_pct(more['2'])} safe, "
                f"~{4 * s['n']:,} ~{_pct(more['4'])}"
            )
        if s.get("score_rounding_mismatch"):
            w(
                f"  note: rounding the model's `score` disagrees with its top level in "
                f"{_pct(s['score_rounding_mismatch'], 1)} of cases; act on howsure.answer"
            )
        other = s["uncommon"]
        w(
            f"  always answering {s['common_answer']!r} would be right {_pct(s['common_accuracy'])} of the time. "
            f"The other {other['n']:,} held-out cases: {other['right']:,} right, {other['wrong']:,} wrong, "
            f"{other['review']:,} to review"
        )
        if other["lost_to_calibration"] or other["gained_from_calibration"]:
            w(
                f"  calibration changed {other['lost_to_calibration']:,} of those from right to wrong, "
                f"{other['gained_from_calibration']:,} from wrong to right"
            )
        for g in s.get("groups", []):
            line = f"  {g['group']} ({g['n']:,}): {_pct(g['coverage'])} auto at {_pct(g['error'], 1)} errors"
            if "unseen_coverage" in g:
                line += f"; fitted without it: {_pct(g['unseen_coverage'])} auto at {_pct(g['unseen_error'], 1)} errors"
            w(line)
        for warning in s["warnings"]:
            w(f"  warning: {warning}")
    for n in notes:
        w(f"note: {n}")


def print_chain_summary(policies: list[Policy], safe: bool) -> None:
    """Estimates for the whole chain per question, from each model's held-out numbers."""
    print("\nThe whole chain (estimates):")
    target = _pct(policies[0].data["target"]["max_error"], 1)
    for qid in dict.fromkeys(qid for p in policies for qid in p.questions):
        left, auto, wrong = 1.0, 0.0, 0.0
        for p in policies:
            s = p.questions.get(qid, {}).get("stats")
            coverage = (s["safe_coverage"] if safe else s["coverage"]) if s else 0.0
            if coverage:
                share = left * coverage
                auto += share
                wrong += share * (0.0 if safe else s["error"])
                left -= share
        if not auto:
            print(f"  {qid}: nothing auto-handled")
            continue
        # Safe cutoffs certify the target, and no tighter number survives picking the cutoff from the same data.
        errors = f"at most {target}" if safe else _pct(wrong / auto, 1)
        print(f"  {qid}: {_pct(auto)} auto-handled, {errors} errors")


def compare_table(policies: list[Policy]) -> str:
    names = []
    for p in policies:
        name = p.model
        while name in names:
            name += "'"
        names.append(name)
    qids: list[str] = []
    for p in policies:
        qids += [q for q in p.questions if q not in qids]
    sure = policies[0].data["target"]["sure"]
    max_error = policies[0].data["target"]["max_error"]
    rows = [
        ("accuracy", lambda s: _pct(s["accuracy_after"], 1)),
        ("calibration error", lambda s: f"{_pct(s['ece_before'], 1)} -> {_pct(s['ece_after'], 1)}"),
        (f"auto-handled at <={_pct(max_error, 1)}", lambda s: _pct(s["coverage"])),
        (f"auto-handled, {_pct(sure)} sure", lambda s: _pct(s["safe_coverage"])),
    ]
    width = max(len(r[0]) for r in rows) + 2
    cols = [max(14, len(n) + 2) for n in names]
    lines = [" " * (width + 10) + "".join(n.rjust(c) for n, c in zip(names, cols))]
    for qid in qids:
        for i, (label, get) in enumerate(rows):
            cells = []
            for p, c in zip(policies, cols):
                q = p.questions.get(qid)
                cells.append((get(q["stats"]) if q else "–").rjust(c))
            head = (qid[:9] if i == 0 else "").ljust(10)
            lines.append(head + label.ljust(width) + "".join(cells))
        lines.append("")
    hashes = {p.data["data"]["hash"] for p in policies}
    targets = {(p.data["target"]["max_error"], p.data["target"]["sure"]) for p in policies}
    if len(hashes) > 1:
        lines.append("warning: these were fitted on different labeled examples, so this isn't a fair comparison.")
    if len(targets) > 1:
        lines.append("warning: these were fitted with different targets (--max-error / --sure).")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------- commands


def cmd_demo(args) -> int:
    os.makedirs(args.out, exist_ok=True)
    records = demo.make_records()
    preds = os.path.join(args.out, "tickets.preds.jsonl")
    with open(preds, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    sample = {
        "answers": {
            "team": {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.82,
                "probabilities": {"billing": 0.88, "technical": 0.07, "other": 0.05},
            },
            "refund": {"type": "noul", "noul": 0.93},
            "urgency": {
                "type": "score",
                "score": 1.4,
                "legend": ["Routine", "Soon", "Urgent"],
                "probabilities": {"0": 0.2, "1": 0.2, "2": 0.6},
                "confidence": 0.5,
            },
        }
    }
    sample_path = os.path.join(args.out, "response.json")
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2)
    print("Demo with a pretend, overconfident model on synthetic support tickets (no real model involved).\n")
    rc = cmd_fit(
        argparse.Namespace(
            preds=[preds],
            max_error=0.05,
            sure=0.95,
            name=None,
            by=None,
            safe=False,
            out=None,
            report=None,
            open=args.open,
        )
    )
    policy_path = os.path.join(args.out, "tickets.policy.json")
    print("\nTry it on one response:")
    print(f"  howsure apply {policy_path} {sample_path}")
    print("\nWith your own model: label at least 500 examples, then")
    print("  howsure run examples.jsonl --url http://localhost:8000")
    return rc


def cmd_run(args) -> int:
    records = load_examples(args.data, args.questions)
    out = args.out or (_stem(args.data) + (f".{_slug(args.model)}" if args.model else "") + ".preds.jsonl")
    res = client.run(records, args.url, out, model=args.model, key=args.key, workers=args.workers)
    print(f"{res.answered:,} answered ({res.new:,} new) -> {out}")
    if res.failed:
        print(f"{len(res.failed):,} failed, first error: {res.failed[0]}")
        print("Rerun the same command to retry them.")
    print(f"Next: howsure fit {out}")
    return 1 if res.failed else 0


def cmd_fit(args) -> int:
    chain = len(args.preds) > 1
    if chain and (args.out or args.report or args.name):
        raise DataError("--out, --report and --name only work with one file. A chain names its files after its inputs.")
    if args.safe and not chain:
        raise DataError("--safe is for fitting a chain. One model's policy already holds both cutoffs.")
    records = [read_jsonl(path) for path in args.preds]
    if chain:
        fitted = fit_chain(records, max_error=args.max_error, sure=args.sure, by=args.by, safe=args.safe)
    else:
        fitted = [fit(records[0], max_error=args.max_error, sure=args.sure, name=args.name, by=args.by)]
    for path, (policy, charts, notes) in zip(args.preds, fitted):
        out = args.out or _stem(path) + ".policy.json"
        rep = (
            args.report or _stem(out[: -len(".policy.json")] if out.endswith(".policy.json") else out) + ".report.html"
        )
        policy.save(out)
        print_summary(policy, notes)
        if policy.questions:
            report.write(rep, policy, charts, notes, policy_path=out)
            print(f"\nSaved {out} and {rep}\n")
            if args.open:
                webbrowser.open("file://" + os.path.abspath(rep))
        else:
            print(f"\nSaved {out}\n")
    if chain:
        print_chain_summary([policy for policy, _, _ in fitted], args.safe)
    return 0


def cmd_gateway(args) -> int:
    from .gateway import load_config, make_server

    if args.quiet:
        logging.getLogger("howsure").setLevel(logging.ERROR)
    gateway = load_config(args.config)
    try:
        server = make_server(gateway, args.host, args.port, quiet=args.quiet)
    except OSError as e:
        raise DataError(f"can't listen on {args.host}:{args.port} ({e.strerror or e})") from None
    print(f"howsure gateway on http://{args.host}:{args.port}. Point your app's base URL here. Ctrl-C to stop.")
    for name, chain in gateway.routes.items():
        mark = " (default)" if name == gateway.default else ""
        print(f"  {name}{mark}: {' -> '.join(m.name for m in chain)}")
    for warning in gateway.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if gateway.key is None and args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            "warning: anyone who can reach this port can spend your model keys; set require_key_env.", file=sys.stderr
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _read_json(path: str):
    if path == "-":
        return json.load(sys.stdin)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cmd_apply(args) -> int:
    policy = Policy.load(args.policy)
    response = _read_json(args.response)
    request = _read_json(args.request) if args.request else None
    print(json.dumps(policy.apply(response, request, safe=args.safe), indent=2, ensure_ascii=False))
    return 0


def cmd_compare(args) -> int:
    sys.stdout.write(compare_table([Policy.load(p) for p in args.policies]))
    return 0


# ---------------------------------------------------------------- entry point


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="howsure",
        description="An open-source gateway for decision models, with calibration: fit each model on your "
        "labels, then serve them behind one endpoint.",
    )
    p.add_argument("--version", action="version", version=f"howsure {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="command")

    s = sub.add_parser("demo", help="see the whole flow on synthetic data, no model needed")
    s.add_argument("--out", default="howsure-demo", help="folder to write into (default: howsure-demo)")
    s.add_argument("--open", action="store_true", help="open the report in a browser")
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("run", help="ask a System One model about your labeled examples")
    s.add_argument("data", help="labeled examples: .jsonl (one request per line) or .csv")
    s.add_argument("--questions", help="questions.json shared by every example (needed for CSV)")
    s.add_argument(
        "--url",
        default="http://localhost:8000",
        help="server base URL, or 'openrouter' (default: http://localhost:8000)",
    )
    s.add_argument("--model", help="model name to send, e.g. typesafe/jev-1.14")
    s.add_argument("--key", help="API key (default: $HOWSURE_API_KEY, or $OPENROUTER_API_KEY for openrouter)")
    s.add_argument("--workers", type=int, default=4, help="parallel requests (default: 4)")
    s.add_argument("--out", help="output file (default: <data>[.<model>].preds.jsonl)")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("fit", help="calibrate and choose cutoffs from `run` output")
    s.add_argument(
        "preds",
        nargs="+",
        help="output of `howsure run` (or any JSONL with labels and answers). "
        "Several files, in order, fit a chain: each model only on what the ones before it pass on",
    )
    s.add_argument(
        "--max-error", type=float, default=0.05, help="most errors you accept on auto-handled cases (default: 0.05)"
    )
    s.add_argument("--sure", type=float, default=0.95, help="certainty for the safe cutoff (default: 0.95)")
    s.add_argument("--name", help="model name to record (default: from the data)")
    s.add_argument(
        "--by",
        metavar="FIELD",
        help="field naming each example's document type, e.g. meta.doc_type: shows results per type, "
        "and warns if a policy wouldn't carry over to a type it hasn't seen",
    )
    s.add_argument(
        "--safe", action="store_true", help='for a chain that will run on safe cutoffs ("safe": true in the gateway)'
    )
    s.add_argument("--out", help="policy file (default: <preds>.policy.json)")
    s.add_argument("--report", help="HTML report (default: <preds>.report.html)")
    s.add_argument("--open", action="store_true", help="open the report in a browser")
    s.set_defaults(func=cmd_fit)

    s = sub.add_parser("gateway", help="serve several models behind one endpoint, chained by how sure they are")
    s.add_argument("config", help="gateway config (JSON): models, their policies, and routes")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8001)
    s.add_argument("--quiet", action="store_true", help="don't log requests")
    s.set_defaults(func=cmd_gateway)

    s = sub.add_parser("apply", help="annotate one saved response (prints JSON)")
    s.add_argument("policy")
    s.add_argument("response", help="response JSON file, or - for stdin")
    s.add_argument("--request", help="the request JSON, to detect questions that changed")
    s.add_argument("--safe", action="store_true", help="use the safe (guaranteed) cutoffs")
    s.set_defaults(func=cmd_apply)

    s = sub.add_parser("compare", help="compare policies of different models fitted on the same examples")
    s.add_argument("policies", nargs="+")
    s.set_defaults(func=cmd_compare)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DataError as e:
        print(f"howsure: {e}", file=sys.stderr)
        return 2
    except client.CallError as e:
        print(f"howsure: model call failed: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"howsure: file not found: {e.filename}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
