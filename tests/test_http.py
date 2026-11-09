"""run (client) against a fake System One server."""

import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from howsure import client, demo
from howsure.data import DataError, read_jsonl


def _serve_once(reply, seen=None):
    """A tiny server that gives one kind of reply and notes every request. Returns its URL."""

    class H(BaseHTTPRequestHandler):
        def _handle(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if seen is not None:
                seen.append(body)
            status, obj, *headers = reply(body)
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            for name, value in headers:
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)

        do_POST = do_GET = _handle

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}"


class FakeModel:
    """Answers every question with fixed probabilities; can fail on purpose."""

    def __init__(self):
        self.calls = 0
        self.fail_first = set()
        self.reject = set()
        self.seen = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.calls += 1
                outer.seen.append((body, self.headers.get("Authorization")))
                state = json.dumps(body["state"], sort_keys=True)
                if state in outer.reject:
                    return self._send(422, {"detail": "bad request"})
                if state in outer.fail_first:
                    outer.fail_first.discard(state)
                    return self._send(503, {"detail": "busy"})
                answers = {}
                for qid, q in body["questions"].items():
                    if q["type"] == "noul":
                        answers[qid] = {"type": "noul", "noul": 0.9}
                    else:
                        opts = (
                            list(q["criteria"])
                            if q["type"] == "choice"
                            else [str(i) for i in range(len(q["criteria"]))]
                        )
                        probs = {o: (0.8 if i == 0 else 0.2 / (len(opts) - 1)) for i, o in enumerate(opts)}
                        answers[qid] = {"type": q["type"], "probabilities": probs}
                self._send(200, {"answers": answers, "latency_ms": 1.0})

            def do_GET(self):
                self._send(200, {"ok": True})

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(client, "BACKOFF_SECONDS", 0.0)
    m = FakeModel()
    yield m
    m.close()


def labeled(n=6):
    recs = demo.make_records(n=n)
    for r in recs:
        del r["answers"], r["model"]
    return recs


def test_run_writes_resumes_and_keeps_current_labels(model, tmp_path):
    out = str(tmp_path / "p.jsonl")
    recs = labeled()
    res = client.run(recs, model.url, out, model="m1", log=lambda m: None)
    assert res.answered == 6 and res.new == 6 and model.calls == 6
    assert model.seen[0][0]["model"] == "m1"
    assert all("label" not in q for q in model.seen[0][0]["questions"].values())  # labels never sent

    recs[0]["questions"]["refund"]["label"] = not recs[0]["questions"]["refund"]["label"]
    res = client.run(recs, model.url, out, model="m1", log=lambda m: None)
    assert res.new == 0 and model.calls == 6  # all cached
    lines = read_jsonl(out)
    assert len(lines) == 6
    assert lines[0]["questions"]["refund"]["label"] == recs[0]["questions"]["refund"]["label"]

    client.run(recs, model.url, out, model="m2", log=lambda m: None)
    assert model.calls == 12  # a different model is never served from another model's cache


def test_run_retries_busy_and_reports_rejected(model, tmp_path):
    recs = labeled(3)
    model.fail_first.add(json.dumps(recs[0]["state"], sort_keys=True))
    model.reject.add(json.dumps(recs[1]["state"], sort_keys=True))
    res = client.run(recs, model.url, str(tmp_path / "p.jsonl"), log=lambda m: None)
    assert res.answered == 2
    assert len(res.failed) == 1 and "422" in res.failed[0]


def test_run_sends_key(model, tmp_path):
    client.run(labeled(1), model.url, str(tmp_path / "p.jsonl"), key="sekret", log=lambda m: None)
    assert model.seen[0][1] == "Bearer sekret"


def test_run_rejects_bad_labels_before_calling(model, tmp_path):
    recs = labeled(2)
    recs[0]["questions"]["team"]["label"] = "sales"
    with pytest.raises(DataError):
        client.run(recs, model.url, str(tmp_path / "p.jsonl"), log=lambda m: None)
    assert model.calls == 0


def test_endpoint_presets():
    assert client.endpoint("openrouter") == "https://openrouter.ai/api/v1/systemone"
    assert client.endpoint("http://localhost:8000/") == "http://localhost:8000/v1/systemone"
    assert client.endpoint("http://x/v1/systemone") == "http://x/v1/systemone"


def test_run_records_the_exact_model_version(tmp_path):
    served = _serve_once(
        lambda body: (200, {"model": "vendor/m-20260930", "answers": {"refund": {"type": "noul", "noul": 0.9}}})
    )
    out = str(tmp_path / "p.jsonl")
    client.run(labeled(1), served, out, log=lambda m: None)
    assert read_jsonl(out)[0]["served_model"] == "vendor/m-20260930"
    client.run(labeled(1), served, out, log=lambda m: None)  # and keeps it when resuming
    assert read_jsonl(out)[0]["served_model"] == "vendor/m-20260930"


def test_keys_do_not_follow_redirects(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "BACKOFF_SECONDS", 0.0)
    stolen = []
    elsewhere = _serve_once(lambda body: (200, {"answers": {}}), seen=stolen)
    moved = _serve_once(lambda body: (302, {}, ("Location", elsewhere)))
    with pytest.raises(client.CallError, match="302"):
        client.call(moved + "/v1/systemone", {"state": "x"}, "secret")
    assert stolen == []


def test_a_dropped_response_is_retried_not_a_crash(monkeypatch):
    monkeypatch.setattr(client, "BACKOFF_SECONDS", 0.0)
    calls = []

    def truncated(*args, **kwargs):
        calls.append(1)
        raise http.client.IncompleteRead(b"{")

    monkeypatch.setattr(client, "post_json", truncated)
    with pytest.raises(client.CallError):
        client.call("http://x", {}, None)
    assert len(calls) == client.ATTEMPTS


def test_run_keeps_other_models_answers_in_a_shared_file(model, tmp_path):
    out = str(tmp_path / "p.jsonl")
    recs = labeled(3)
    client.run(recs, model.url, out, model="m1", log=lambda m: None)
    client.run(recs, model.url, out, model="m2", log=lambda m: None)
    assert sorted(line["model"] for line in read_jsonl(out)) == ["m1", "m1", "m1", "m2", "m2", "m2"]


@pytest.mark.parametrize("reply", [b"\xff\xfe not text", b"[1, 2, 3]"], ids=["not-utf8", "not-an-object"])
def test_a_garbled_reply_is_retried_then_reported(monkeypatch, reply):
    monkeypatch.setattr(client, "BACKOFF_SECONDS", 0.0)

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    try:
        with pytest.raises(client.CallError):
            client.call(f"http://127.0.0.1:{server.server_port}/v1/systemone", {"state": "x"}, None)
    finally:
        server.shutdown()
        server.server_close()
