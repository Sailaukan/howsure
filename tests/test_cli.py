import json

from howsure import demo
from howsure.cli import main


def test_demo_then_apply_and_compare(tmp_path, capsys):
    out = tmp_path / "d"
    assert main(["demo", "--out", str(out)]) == 0
    policy = out / "tickets.policy.json"
    assert policy.exists() and (out / "tickets.report.html").exists()
    text = capsys.readouterr().out
    assert "auto-handle if p >=" in text and "team" in text

    assert main(["apply", str(policy), str(out / "response.json")]) == 0
    applied = json.loads(capsys.readouterr().out)
    assert "howsure" in applied["answers"]["team"]

    other = tmp_path / "other.preds.jsonl"
    with open(other, "w") as f:
        for rec in demo.make_records(model="other-model", sharpness=1.0):
            f.write(json.dumps(rec) + "\n")
    assert main(["fit", str(other)]) == 0
    assert (tmp_path / "other.policy.json").exists()
    capsys.readouterr()
    assert main(["compare", str(policy), str(tmp_path / "other.policy.json")]) == 0
    table = capsys.readouterr().out
    assert "synthetic-demo" in table and "other-model" in table and "warning" not in table


def test_friendly_errors(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n")
    assert main(["fit", str(bad)]) == 2
    assert "not valid JSON" in capsys.readouterr().err
    assert main(["fit", str(tmp_path / "missing.jsonl")]) == 2
