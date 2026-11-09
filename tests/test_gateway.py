"""howsure gateway against fake System One servers, over real HTTP on localhost."""

import http.client
import json
import socket
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from howsure import demo
from howsure import gateway as gateway_module
from howsure.cli import main
from howsure.data import DataError
from howsure.drift import WINDOW
from howsure.gateway import load_config, make_server
from howsure.policy import Policy, fit

TYPES = {"team": "choice", "refund": "noul", "urgency": "score"}
QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "b", "technical": "t"}},
    "refund": {"type": "noul", "instructions": "Refund?"},
}
URGENCY = {"type": "score", "instructions": "How urgent?", "criteria": ["Routine", "Soon", "Urgent"]}
REQUEST = {"model": "tickets", "state": "Charged twice", "questions": QUESTIONS}


def strict_json(data):
    """Parse like a browser's JSON.parse: NaN and Infinity aren't JSON."""

    def refuse(token):
        raise ValueError(f"{token} is not valid JSON")

    return json.loads(data, parse_constant=refuse)


def policy_data(name="m", cutoff=0.9, safe_cutoff=None, questions=("team", "refund")):
    """A policy with identity calibration, so a probability is its own calibrated value."""
    data = {"howsure": 1, "model": name, "target": {"max_error": 0.05, "sure": 0.95}, "questions": {}}
    for qid in questions:
        method = (
            {"method": "platt", "a": 1.0, "b": 0.0}
            if TYPES[qid] == "noul"
            else {"method": "temperature", "temperature": 1.0}
        )
        data["questions"][qid] = {
            "type": TYPES[qid],
            "fingerprint": None,
            "calibration": method,
            "cutoff": cutoff,
            "safe_cutoff": safe_cutoff,
            "stats": {},
        }
    return data


def answers(team=None, refund=None):
    out = {}
    if team is not None:
        out["team"] = {
            "type": "choice",
            "choice": "billing" if team >= 0.5 else "technical",
            "probabilities": {"billing": team, "technical": 1 - team},
        }
    if refund is not None:
        out["refund"] = {"type": "noul", "noul": refund}
    return out


def _serve(server):
    # The default 0.5s poll would add that much to every shutdown.
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()


class Upstream:
    """A fake model server. `reply` is a status + body + any headers, or a callable taking the request."""

    def __init__(self, reply=None, **answer_probs):
        self.reply = reply if reply is not None else (200, {"answers": answers(**answer_probs)})
        self.seen = []
        self.keys = []
        self.gets = []
        self.release = threading.Event()
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.seen.append(body)
                outer.keys.append(self.headers.get("Authorization"))
                self._reply(outer.reply(body) if callable(outer.reply) else outer.reply)

            def do_GET(self):  # where a redirected POST ends up
                outer.gets.append(self.headers.get("Authorization"))
                self._reply(outer.reply({}) if callable(outer.reply) else outer.reply)

            def _reply(self, reply):
                status, obj, *headers = reply
                data = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                for name, value in headers:
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        _serve(self.server)

    @property
    def asked(self):
        return [list(body["questions"]) for body in self.seen]

    def close(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


def closed_port_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}"


class Running:
    def __init__(self, server):
        self.server = server
        self.base = f"http://127.0.0.1:{server.server_port}"

    def call(self, method, path, body=None, key=None):
        data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, strict_json(r.read())
        except urllib.error.HTTPError as e:
            return e.code, strict_json(e.read())

    def post(self, body=REQUEST, key=None):
        return self.call("POST", "/v1/systemone", body, key)

    def raw(self, request: bytes) -> bytes:
        """Send exact bytes and read everything until the gateway closes the connection."""
        with socket.create_connection(("127.0.0.1", self.server.server_port), timeout=5) as s:
            s.sendall(request)
            data = b""
            while chunk := s.recv(4096):
                data += chunk
            return data


@pytest.fixture
def write_config(tmp_path):
    def write(models, **top):
        """models: name -> (upstream or url, policy data, extra settings)."""
        specs = {}
        for name, (upstream, policy, *extra) in models.items():
            path = tmp_path / f"{name}.policy.json"
            path.write_text(json.dumps(policy))
            specs[name] = {
                "url": getattr(upstream, "url", upstream),
                "policy": path.name,
                **(extra[0] if extra else {}),
            }
        config = tmp_path / "gateway.json"
        config.write_text(json.dumps({"models": specs, **top}))
        return str(config)

    return write


@pytest.fixture
def start(write_config):
    """Start a gateway; `tickets` chains the models in the order given."""
    running, upstreams = [], []

    def go(models, **top):
        top.setdefault("routes", {"tickets": list(models)})
        upstreams.extend(u[0] for u in models.values() if isinstance(u[0], Upstream))
        server = make_server(load_config(write_config(models, **top)), port=0, quiet=True)
        _serve(server)
        running.append(server)
        return Running(server)

    yield go
    for server in running:
        server.shutdown()
        server.server_close()
    for upstream in upstreams:
        upstream.close()


# ---------------------------------------------------------------- chains


def test_sure_answers_stop_the_chain(start):
    a, b = Upstream(team=0.95, refund=0.95), Upstream(team=0.95, refund=0.95)
    status, out = start({"a": (a, policy_data()), "b": (b, policy_data())}).post()
    assert status == 200
    assert b.seen == []
    assert out["model"] == "tickets"
    assert out["answers"]["team"]["howsure"]["action"] == "auto"
    assert out["answers"]["team"]["howsure"]["model"] == "a"
    assert "tried" not in out["answers"]["team"]["howsure"]
    assert out["howsure"]["steps"][0]["auto"] == ["team", "refund"]


def test_only_the_unsure_question_moves_on(start):
    a, b = Upstream(team=0.95, refund=0.6), Upstream(refund=0.95)
    _, out = start({"a": (a, policy_data()), "b": (b, policy_data())}).post()
    assert a.asked == [["team", "refund"]] and b.asked == [["refund"]]
    assert b.seen[0]["state"] == "Charged twice"
    team, refund = out["answers"]["team"]["howsure"], out["answers"]["refund"]["howsure"]
    assert (team["model"], team["action"]) == ("a", "auto")
    assert (refund["model"], refund["action"], refund["answer"]) == ("b", "auto", True)
    assert [t["model"] for t in refund["tried"]] == ["a", "b"]
    assert list(out["answers"]) == ["team", "refund"]


def test_each_model_uses_its_own_policy(start):
    a, b = Upstream(team=0.8, refund=0.8), Upstream(team=0.8, refund=0.8)
    _, out = start({"a": (a, policy_data(cutoff=0.9)), "b": (b, policy_data(cutoff=0.7))}).post()
    assert out["answers"]["team"]["howsure"]["model"] == "b"
    assert out["answers"]["team"]["howsure"]["action"] == "auto"


def test_safe_is_set_per_model(start):
    a = Upstream(team=0.85, refund=0.85)
    b = Upstream(team=0.85, refund=0.85)
    gw = start(
        {
            "a": (a, policy_data(cutoff=0.8, safe_cutoff=0.95), {"safe": True}),
            "b": (b, policy_data(cutoff=0.8, safe_cutoff=0.95)),
        }
    )
    _, out = gw.post()
    assert len(a.seen) == 1  # 0.85 is under a's safe cutoff
    assert out["answers"]["team"]["howsure"]["model"] == "b"  # and over b's expected one


def test_upstream_model_name_and_other_fields_are_forwarded(start):
    a = Upstream(team=0.6, refund=0.6)
    b = Upstream(team=0.95, refund=0.95)
    gw = start({"a": (a, policy_data(), {"model": "vendor/a-1"}), "b": (b, policy_data())})
    gw.post({**REQUEST, "images": ["page1"]})
    assert a.seen[0]["model"] == "vendor/a-1" and a.seen[0]["images"] == ["page1"]
    assert "model" not in b.seen[0]  # no name configured: don't leak the route name upstream


def test_nobody_sure_keeps_the_most_probable_answer(start):
    a, b, c = Upstream(refund=0.85), Upstream(refund=0.2), Upstream(refund=0.7)
    models = {n: (u, policy_data(questions=("refund",))) for n, u in (("a", a), ("b", b), ("c", c))}
    _, out = start(models).post({**REQUEST, "questions": {"refund": QUESTIONS["refund"]}})
    block = out["answers"]["refund"]["howsure"]
    assert (block["model"], block["answer"], block["action"]) == ("a", True, "review")
    assert block["probability"] == 0.85
    assert out["answers"]["refund"]["noul"] == 0.85  # the raw answer matches the verdict
    assert [(t["model"], t["answer"]) for t in block["tried"]] == [("a", True), ("b", False), ("c", True)]


def test_models_are_not_asked_what_they_can_never_decide(start):
    drifted = policy_data()
    drifted["questions"]["team"]["fingerprint"] = "from-an-older-wording"
    no_cutoff = policy_data()
    no_cutoff["questions"]["team"]["cutoff"] = None
    partial = policy_data(questions=("refund",))
    a, b, c, d = Upstream(refund=0.6), Upstream(refund=0.6), Upstream(refund=0.6), Upstream(team=0.95, refund=0.95)
    gw = start(
        {"partial": (a, partial), "drifted": (b, drifted), "no_cutoff": (c, no_cutoff), "good": (d, policy_data())}
    )
    _, out = gw.post()
    assert a.asked == b.asked == c.asked == [["refund"]]
    assert d.asked == [["team", "refund"]]
    assert out["answers"]["team"]["howsure"]["model"] == "good"


def test_question_nobody_calibrated_is_asked_once_and_goes_to_review(start):
    a, b = Upstream(team=0.95, refund=0.95), Upstream(team=0.95, refund=0.95)
    gw = start({"a": (a, policy_data(questions=("team",))), "b": (b, policy_data(questions=("team",)))})
    status, out = gw.post()
    assert status == 200
    assert a.asked == [["team", "refund"]] and b.seen == []
    block = out["answers"]["refund"]["howsure"]
    assert block["action"] == "review" and block["reason"] == "no calibration for this question"
    assert (block["model"], block["answer"], block["probability"]) == ("a", None, 0.0)
    assert out["answers"]["refund"]["noul"] == 0.95  # the raw answer is still there for the reviewer


def test_missing_or_unreadable_answers_move_on(start):
    a = Upstream((200, {"answers": {"team": {"type": "choice"}}}))  # no probabilities, and no refund at all
    b = Upstream(team=0.95, refund=0.95)
    _, out = start({"a": (a, policy_data()), "b": (b, policy_data())}).post()
    assert b.asked == [["team", "refund"]]
    assert out["answers"]["team"]["howsure"]["model"] == "b"
    assert out["answers"]["refund"]["howsure"]["model"] == "b"
    assert out["answers"]["team"]["howsure"]["tried"][0]["reason"] == "unreadable answer"


def test_unreadable_answer_does_not_stop_the_search_for_one_to_review(start):
    a = Upstream((200, {"answers": {"team": {"type": "choice"}}}))
    b = Upstream(team=0.7)
    gw = start({"a": (a, policy_data(questions=("team",))), "b": (b, policy_data(questions=("refund",)))})
    _, out = gw.post({**REQUEST, "questions": {"team": QUESTIONS["team"]}})
    assert b.asked == [["team"]]  # b can't decide it, but a gave nothing a reviewer could use
    assert out["answers"]["team"]["howsure"]["model"] == "b"


def test_only_unreadable_answers_is_a_failure(start):
    a = Upstream((200, {"answers": {"team": {"type": "choice"}}}))
    status, out = start({"a": (a, policy_data(questions=("team",)))}).post(
        {**REQUEST, "questions": {"team": QUESTIONS["team"]}}
    )
    assert status == 502
    assert out["howsure"]["steps"][0]["error"] == "no usable answers"


def test_garbage_for_a_question_nobody_calibrated_does_not_reach_a_reviewer(start):
    a = Upstream((200, {"answers": {"team": {"type": "choice", "probabilities": {"hallucinated_team": 1.0}}}}))
    b = Upstream(team=0.7)
    gw = start({"a": (a, policy_data(questions=("refund",))), "b": (b, policy_data(questions=("refund",)))})
    _, out = gw.post({**REQUEST, "questions": {"team": QUESTIONS["team"]}})
    block = out["answers"]["team"]["howsure"]
    assert block["model"] == "b" and block["reason"] == "no calibration for this question"
    assert "don't match" in block["tried"][0]["reason"]


def test_answers_that_drift_go_to_review_and_the_next_model_is_asked(start):
    policy = fit(demo.make_records())[0].data
    question = {"refund": demo.QUESTIONS["refund"]}
    a, b = (
        Upstream(refund=0.9999),
        Upstream(refund=0.5),
    )  # a is far more certain, far more often, than when it was fitted
    gw = start({"a": (a, policy), "b": (b, policy)})
    first = gw.post({"model": "tickets", "state": "x", "questions": question})[1]["answers"]["refund"]["howsure"]
    assert (first["model"], first["action"]) == ("a", "auto")
    for _ in range(WINDOW + 40):
        _, out = gw.post({"model": "tickets", "state": "x", "questions": question})
    block = out["answers"]["refund"]["howsure"]
    assert block["action"] == "review"
    assert "shifted" in block["tried"][0]["reason"]
    assert b.seen  # still worth asking the next model


def test_a_model_cannot_vouch_for_itself(start):
    answer = {"type": "noul", "noul": 0.95, "howsure": {"action": "auto", "answer": True, "probability": 1.0}}
    a = Upstream((200, {"answers": {"refund": answer}}))
    gw = start({"a": (a, policy_data(questions=("team",)))})
    _, out = gw.post({**REQUEST, "questions": {"refund": QUESTIONS["refund"]}})
    assert out["answers"]["refund"]["howsure"]["action"] == "review"


def test_usage_only_totals_what_every_model_reports(start):
    a = Upstream(
        (
            200,
            {"usage": {"input_tokens": 10, "output_tokens": 1, "cost": 0.5}, "answers": answers(team=0.95, refund=0.6)},
        )
    )
    b = Upstream((200, {"usage": {"input_tokens": 4, "output_tokens": 2}, "answers": answers(refund=0.95)}))
    _, out = start({"a": (a, policy_data()), "b": (b, policy_data())}).post()
    assert out["usage"] == {"input_tokens": 14, "output_tokens": 3}  # no cost: a total of 0.5 would understate it
    assert out["howsure"]["steps"][0]["usage"]["cost"] == 0.5


def test_each_step_keeps_its_own_trace_details(start):
    a = Upstream(
        (200, {"id": "or-1", "model": "vendor/a-20260930", "provider": "x", "answers": answers(team=0.95, refund=0.6)})
    )
    b = Upstream((200, {"request_id": "req-2", "model": "b-max", "answers": answers(refund=0.95)}))
    _, out = start({"a": (a, policy_data()), "b": (b, policy_data())}).post()
    first, second = out["howsure"]["steps"]
    assert (first["id"], first["served_model"]) == ("or-1", "vendor/a-20260930")
    assert (second["request_id"], second["served_model"]) == ("req-2", "b-max")
    assert out["model"] == "tickets"
    assert not {"id", "request_id", "provider"} & set(out)  # none of them speaks for the whole response


def test_real_fitted_policies_with_real_fingerprints(start):
    policy = fit(demo.make_records())[0].data
    a, b = Upstream(refund=0.5), Upstream(refund=0.9999)
    gw = start({"a": (a, policy), "b": (b, policy)})
    _, out = gw.post({"model": "tickets", "state": "x", "questions": {"refund": demo.QUESTIONS["refund"]}})
    assert out["answers"]["refund"]["howsure"]["model"] == "b"
    assert out["answers"]["refund"]["howsure"]["action"] == "auto"

    reworded = {"refund": {**demo.QUESTIONS["refund"], "instructions": "Do they want their money back?"}}
    _, out = gw.post({"model": "tickets", "state": "x", "questions": reworded})
    block = out["answers"]["refund"]["howsure"]
    assert block["action"] == "review" and block["reason"] == "question changed since calibration"


# ---------------------------------------------------------------- fallback


@pytest.mark.parametrize(
    "reply",
    [
        (503, {"error": {"type": "overloaded", "message": "busy"}}),
        (529, {"error": {"type": "overloaded"}}),
        (422, {"error": {"type": "invalid_request_error", "message": "too many questions for this model"}}),
        (401, {"error": {"type": "authentication_error"}}),
        (200, b"<html>not json</html>"),
        (200, [1, 2, 3]),
        (200, {"choices": []}),
        (200, {"answers": {}}),
    ],
    ids=["503", "529", "422", "401", "not-json", "not-an-object", "no-answers", "empty-answers"],
)
def test_a_failing_model_falls_back_to_the_next(start, reply):
    a, b = Upstream(reply), Upstream(team=0.95, refund=0.95)
    status, out = start({"a": (a, policy_data()), "b": (b, policy_data())}).post()
    assert status == 200
    assert len(a.seen) == 1  # one try, no retry loop: falling back is the recovery
    assert b.asked == [["team", "refund"]]
    assert out["answers"]["team"]["howsure"]["model"] == "b"
    assert "error" in out["howsure"]["steps"][0]


def test_unreachable_model_falls_back(start):
    b = Upstream(team=0.95, refund=0.95)
    status, out = start({"dead": (closed_port_url(), policy_data()), "b": (b, policy_data())}).post()
    assert status == 200
    assert out["answers"]["team"]["howsure"]["model"] == "b"
    assert out["howsure"]["steps"][0]["model"] == "dead"


def test_stalled_model_times_out_and_falls_back(start):
    def stall(body):
        slow.release.wait(10)
        return 200, {"answers": answers(team=0.99, refund=0.99)}

    slow, b = Upstream(stall), Upstream(team=0.95, refund=0.95)
    status, out = start({"slow": (slow, policy_data(), {"timeout": 0.2}), "b": (b, policy_data())}).post()
    assert status == 200
    assert out["answers"]["team"]["howsure"]["model"] == "b"
    assert "timed out" in out["howsure"]["steps"][0]["error"]


def test_failure_late_in_the_chain_keeps_the_earlier_answer(start):
    a, b = Upstream(team=0.95, refund=0.6), Upstream((503, {"error": "busy"}))
    status, out = start({"a": (a, policy_data()), "b": (b, policy_data())}).post()
    assert status == 200
    block = out["answers"]["refund"]["howsure"]
    assert (block["model"], block["action"], block["probability"]) == ("a", "review", 0.6)
    assert out["howsure"]["steps"][1]["status"] == 503


def test_answer_from_a_model_that_cannot_decide_when_the_one_that_can_fails(start):
    a = Upstream((503, {"error": "busy"}))
    b = Upstream(refund=0.95)
    gw = start({"a": (a, policy_data(questions=("refund",))), "b": (b, policy_data(questions=("team",)))})
    _, out = gw.post({**REQUEST, "questions": {"refund": QUESTIONS["refund"]}})
    block = out["answers"]["refund"]["howsure"]
    assert block["model"] == "b" and block["action"] == "review"  # an answer to review beats none


def test_question_left_without_any_answer_goes_to_review(start):
    a = Upstream(team=0.95)
    b = Upstream((503, {"error": "busy"}))
    gw = start({"a": (a, policy_data(questions=("team",))), "b": (b, policy_data(questions=("refund",)))})
    status, out = gw.post()
    assert status == 200
    assert a.asked == [["team"], ["refund"]] and b.asked == [["refund"]]  # a was healthy, but had no answer
    assert out["answers"]["refund"] == {
        "type": "noul",
        "howsure": {
            "answer": None,
            "probability": 0.0,
            "action": "review",
            "probabilities": {},
            "reason": "no model answered",
        },
    }


@pytest.mark.parametrize(
    "first, second, expected",
    [
        (400, 400, 400),  # the app's request is bad: tell it, don't call it an outage
        (413, 413, 413),
        (422, 422, 422),
        (429, 429, 429),  # the app's own client knows how to back off from these
        (503, 503, 503),
        (504, 504, 504),
        (529, 529, 529),
        (503, 422, 502),
        (401, 401, 502),  # our key is wrong, not the app's
        (402, 402, 502),  # our account is out of credits
        (403, 403, 502),
        (404, 404, 502),
        (407, 407, 502),
        (408, 408, 502),
        (500, 500, 502),
        (301, 301, 502),  # a 3xx with no Location would confuse the app's client
        (308, 308, 502),
    ],
)
def test_when_every_model_fails_the_status_is_meaningful(start, first, second, expected):
    a = Upstream((first, {"error": {"message": "one"}}))
    b = Upstream((second, {"error": {"message": "two"}}))
    status, out = start({"a": (a, policy_data()), "b": (b, policy_data())}).post()
    assert status == expected
    assert out["error"]["type"] == "upstream_error"
    assert f"HTTP {second}" in out["error"]["message"]
    assert [s["status"] for s in out["howsure"]["steps"]] == [first, second]
    assert "answers" not in out


def test_every_model_unreachable_is_a_502(start):
    status, out = start({"dead": (closed_port_url(), policy_data())}).post({**REQUEST, "model": "dead"})
    assert status == 502
    assert out["howsure"]["route"] == "dead"


# ---------------------------------------------------------------- one endpoint


def test_routes_models_and_the_default(start):
    a, b = Upstream(team=0.95, refund=0.95), Upstream(team=0.95, refund=0.95)
    gw = start(
        {"a": (a, policy_data()), "b": (b, policy_data())},
        routes={"tickets": ["a", "b"], "reversed": ["b", "a"]},
        default="reversed",
    )

    assert gw.post({**REQUEST, "model": "a"})[1]["answers"]["team"]["howsure"]["model"] == "a"
    assert gw.post({**REQUEST, "model": "b"})[1]["answers"]["team"]["howsure"]["model"] == "b"
    assert gw.post()[1]["answers"]["team"]["howsure"]["model"] == "a"
    no_model = {k: v for k, v in REQUEST.items() if k != "model"}
    _, out = gw.post(no_model)
    assert out["answers"]["team"]["howsure"]["model"] == "b"
    assert out["model"] == "reversed"


def test_unknown_or_missing_route_names_the_choices_and_calls_nothing(start):
    a = Upstream(team=0.95, refund=0.95)
    gw = start({"a": (a, policy_data())})
    status, out = gw.post({**REQUEST, "model": "drex-latest"})
    assert status == 400 and "drex-latest" in out["error"]["message"] and "tickets" in out["error"]["message"]
    status, out = gw.post({"state": "x", "questions": QUESTIONS})
    assert status == 400 and "no 'model'" in out["error"]["message"]
    assert a.seen == []


@pytest.mark.parametrize(
    "body",
    [
        b"{not json",
        b"[1, 2]",
        b"[" * 200000,
        {"model": "tickets", "state": "x"},
        {"model": "tickets", "questions": {}},
        {"model": "tickets", "questions": ["team"]},
        {"model": "tickets", "questions": {"team": "which?"}},
        {"model": "tickets", "questions": {"team": {"instructions": "Which team?"}}},
    ],
    ids=[
        "not-json",
        "not-object",
        "too-deep",
        "no-questions",
        "empty-questions",
        "questions-list",
        "question-not-object",
        "question-without-type",
    ],
)
def test_bad_requests_are_400s(start, body):
    a = Upstream(team=0.95, refund=0.95)
    status, out = start({"a": (a, policy_data())}).post(body)
    assert status == 400 and out["error"]["type"] == "invalid_request_error"
    assert a.seen == []


def test_health_models_and_other_paths(start):
    a, b = Upstream(), Upstream()
    gw = start({"a": (a, policy_data()), "b": (b, policy_data())}, default="tickets")
    assert gw.call("GET", "/health") == (200, {"status": "ok"})
    status, out = gw.call("GET", "/v1/models")
    assert status == 200 and out["default"] == "tickets"
    assert {"id": "tickets", "chain": ["a", "b"]} in out["data"] and {"id": "a", "chain": ["a"]} in out["data"]
    assert gw.call("GET", "/v1/other")[0] == 404
    assert gw.call("POST", "/v1/other", {"x": 1})[0] == 404
    assert a.seen == b.seen == []


def test_batch_runs_each_request_on_its_own_route(start):
    def by_state(body):
        p = {"sure": 0.95, "unsure": 0.6}.get(body["state"], 0.95)
        return 200, {"answers": answers(team=p, refund=p)}

    a, b = Upstream(by_state), Upstream(team=0.95, refund=0.95)
    gw = start({"a": (a, policy_data()), "b": (b, policy_data())})
    status, out = gw.post(
        {
            "model": "tickets",  # shared by requests that don't name their own
            "requests": [
                {"state": "sure", "questions": QUESTIONS},
                {"state": "unsure", "questions": QUESTIONS},
                {"state": "direct", "model": "b", "questions": QUESTIONS},
                {"state": "bad", "model": "nope", "questions": QUESTIONS},
                "not an object",
            ],
        }
    )
    assert status == 200
    results = out["results"]
    assert results[0]["answers"]["team"]["howsure"]["model"] == "a"
    assert results[1]["answers"]["team"]["howsure"]["model"] == "b"
    assert results[2]["model"] == "b"
    assert results[3]["status"] == 400 and "nope" in results[3]["error"]["message"]
    assert results[4]["status"] == 400


def test_batch_edge_cases(start):
    a = Upstream((503, {"error": "busy"}))
    gw = start({"a": (a, policy_data())})
    assert gw.post({"requests": []}) == (200, {"results": []})
    assert gw.post({"requests": "x"})[0] == 400
    status, out = gw.post({"model": "tickets", "requests": [{"state": "x", "questions": QUESTIONS}]})
    assert status == 200 and out["results"][0]["status"] == 503


# ---------------------------------------------------------------- keys


def test_each_model_gets_its_own_key_and_the_apps_key_stays_home(start, monkeypatch):
    monkeypatch.setenv("KEY_A", "secret-a")
    monkeypatch.setenv("HOWSURE_API_KEY", "must-not-be-sent")
    a = Upstream(team=0.6, refund=0.6)
    b = Upstream(team=0.6, refund=0.6)
    gw = start({"a": (a, policy_data(), {"key_env": "KEY_A"}), "b": (b, policy_data())})
    gw.post(key="the-apps-own-key")
    assert a.keys == ["Bearer secret-a"]
    assert b.keys == [None]


def test_gateway_key_guards_everything_but_health(start, monkeypatch):
    monkeypatch.setenv("GW_KEY", "letmein")
    a = Upstream(team=0.95, refund=0.95)
    gw = start({"a": (a, policy_data())}, require_key_env="GW_KEY")
    assert gw.post()[0] == 401
    assert gw.post(key="wrong")[0] == 401
    assert a.seen == []
    assert gw.call("GET", "/v1/models")[0] == 401
    assert gw.call("GET", "/health")[0] == 200
    assert gw.post(key="letmein")[0] == 200
    assert a.keys == [None]  # the gateway key isn't a model key


def test_keep_alive_survives_answered_requests_and_closes_after_rejected_ones(start, monkeypatch):
    monkeypatch.setenv("GW_KEY", "letmein")
    a = Upstream(team=0.95, refund=0.95)
    gw = start({"a": (a, policy_data())}, require_key_env="GW_KEY")
    conn = http.client.HTTPConnection("127.0.0.1", gw.server.server_port)
    body = json.dumps(REQUEST)
    good = {"Authorization": "Bearer letmein", "Content-Type": "application/json"}
    for _ in range(2):
        conn.request("POST", "/v1/systemone", body, good)
        response = conn.getresponse()
        assert response.status == 200
        response.read()
    conn.request("POST", "/v1/systemone", body, {"Content-Type": "application/json"})
    response = conn.getresponse()
    assert response.status == 401 and response.getheader("Connection") == "close"
    conn.close()


# ---------------------------------------------------------------- config


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda c: c.pop("models"), "'models'"),
        (lambda c: c.update(models={}), "'models'"),
        (lambda c: c.update(modles={}), "modles"),
        (lambda c: c["models"]["a"].pop("policy"), "needs a 'url' and a 'policy'"),
        (lambda c: c["models"]["a"].pop("url"), "needs a 'url' and a 'policy'"),
        (lambda c: c["models"]["a"].update(saef=True), "saef"),
        (lambda c: c["models"]["a"].update(url="localhost:8000"), "http://"),
        (lambda c: c["models"]["a"].update(timeout=0), "timeout"),
        (lambda c: c["models"]["a"].update(timeout=True), "timeout"),
        (lambda c: c["models"]["a"].update(timeout=float("inf")), "timeout"),
        (lambda c: c["models"]["a"].update(timeout=float("nan")), "timeout"),
        (lambda c: c["models"]["a"].update(timeout=1e30), "timeout"),
        (lambda c: c["models"]["a"].update(timeout=3601), "timeout"),
        (lambda c: c["models"]["a"].update(safe="yes"), "safe"),
        (lambda c: c["models"]["a"].update(model=7), "'model'"),
        (lambda c: c["models"]["a"].update(key_env="HOWSURE_TEST_UNSET"), "HOWSURE_TEST_UNSET"),
        (lambda c: c.update(routes=["a"]), "'routes'"),
        (lambda c: c.update(routes={"r": []}), "route 'r'"),
        (lambda c: c.update(routes={"r": "a"}), "route 'r'"),
        (lambda c: c.update(routes={"r": ["a", "ghost"]}), "ghost"),
        (lambda c: c.update(routes={"r": ["a", "a"]}), "twice"),
        (lambda c: c.update(default="ghost"), "ghost"),
        (lambda c: c.update(default=["a"]), "default"),
        (lambda c: c.update(require_key_env="HOWSURE_TEST_UNSET"), "HOWSURE_TEST_UNSET"),
    ],
)
def test_bad_configs_are_caught_at_startup(write_config, change, message):
    path = write_config({"a": ("http://localhost:1", policy_data())})
    with open(path) as f:
        cfg = json.load(f)
    change(cfg)
    with open(path, "w") as f:
        json.dump(cfg, f)
    with pytest.raises(DataError, match=message):
        load_config(path)


def test_config_reads_presets_and_relative_policy_paths(write_config, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path.parent)  # policy paths don't depend on where you start the gateway
    monkeypatch.setenv("OR_KEY", "k")
    gateway = load_config(write_config({"m": ("openrouter", policy_data(), {"model": "x/y", "key_env": "OR_KEY"})}))
    model = gateway.models["m"]
    assert model.url == "https://openrouter.ai/api/v1/systemone"
    assert (model.model, model.key, model.safe, model.timeout) == ("x/y", "k", False, 30.0)
    assert [m.name for m in gateway.routes["m"]] == ["m"]


def test_keys_stay_out_of_reprs(write_config, monkeypatch):
    monkeypatch.setenv("MODEL_KEY", "model-secret")
    monkeypatch.setenv("GW_KEY", "gateway-secret")
    path = write_config({"a": ("http://x", policy_data(), {"key_env": "MODEL_KEY"})}, require_key_env="GW_KEY")
    text = repr(load_config(path))
    assert "model-secret" not in text and "gateway-secret" not in text


def test_route_may_share_a_models_name(write_config):
    path = write_config({"a": ("http://x", policy_data()), "b": ("http://y", policy_data())}, routes={"a": ["a", "b"]})
    assert [m.name for m in load_config(path).routes["a"]] == ["a", "b"]


def test_policy_file_problems_are_friendly(write_config, tmp_path):
    path = write_config({"a": ("http://x", policy_data())})
    (tmp_path / "a.policy.json").unlink()
    with pytest.raises(DataError, match="can't read"):
        load_config(path)
    (tmp_path / "a.policy.json").write_text("{broken")
    with pytest.raises(DataError, match="can't read"):
        load_config(path)
    (tmp_path / "a.policy.json").write_text("[]")
    with pytest.raises(DataError, match="can't read"):
        load_config(path)
    (tmp_path / "a.policy.json").write_text('{"something": "else"}')
    with pytest.raises(DataError, match="howsure policy"):
        load_config(path)


def test_cli_reports_config_problems(tmp_path, capsys):
    bad = tmp_path / "gateway.json"
    bad.write_text("{broken")
    assert main(["gateway", str(bad)]) == 2
    assert "not valid JSON" in capsys.readouterr().err
    assert main(["gateway", str(tmp_path / "missing.json")]) == 2


def test_cli_reports_a_port_in_use(write_config, capsys):
    path = write_config({"a": ("http://x", policy_data())})
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        assert main(["gateway", path, "--port", str(s.getsockname()[1])]) == 2
    assert "can't listen" in capsys.readouterr().err


# ---------------------------------------------------------------- redirects


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_redirects_are_not_followed_so_keys_stay_with_their_model(start, monkeypatch, code):
    monkeypatch.setenv("KEY_A", "secret-a")
    elsewhere = Upstream(team=0.99, refund=0.99)  # following the redirect would accept this as model a's answer
    a = Upstream((code, {}, ("Location", elsewhere.url + "/v1/systemone")))
    b = Upstream(team=0.95, refund=0.95)
    status, out = start({"a": (a, policy_data(), {"key_env": "KEY_A"}), "b": (b, policy_data())}).post()
    assert elsewhere.gets == [] and elsewhere.keys == []
    assert a.keys == ["Bearer secret-a"]
    assert status == 200
    assert out["answers"]["team"]["howsure"]["model"] == "b"
    step = out["howsure"]["steps"][0]
    assert step["status"] == code and "redirects aren't followed" in step["error"]
    assert elsewhere.url not in json.dumps(out)
    elsewhere.close()


def test_redirect_target_goes_to_the_log(write_config):
    a = Upstream((302, {}, ("Location", "https://new.vendor.example/v1/systemone")))
    lines = []
    load_config(write_config({"a": (a, policy_data())})).handle({**REQUEST, "model": "a"}, lines.append)
    assert any("https://new.vendor.example" in line for line in lines)
    a.close()


# ---------------------------------------------------------------- answers that can't be trusted


@pytest.mark.parametrize(
    "probabilities, reason",
    [
        ({"hallucinated_team": 0.5}, "don't match"),
        ({"billing": 0.4}, "don't match"),
        ({"billing": 0.99}, "don't match"),
        ({"billing": 0.99, "technical": 0.01, "other": 0.0}, "don't match"),
        ({"billing": 0.4, "technical": 0.0}, "add up"),
        ({"billing": 0.99, "technical": 0.99}, "add up"),
        ({"billing": 1.5, "technical": -0.5}, "unreadable"),
        ({"billing": "0.9", "technical": "0.1"}, "unreadable"),
        ({"billing": True, "technical": False}, "unreadable"),
        ([0.9, 0.1], "unreadable"),
    ],
)
def test_invented_or_partial_choice_answers_never_settle_a_question(start, probabilities, reason):
    a = Upstream((200, {"answers": {"team": {"type": "choice", "probabilities": probabilities}}}))
    b = Upstream(team=0.95)
    gw = start({"a": (a, policy_data(questions=("team",))), "b": (b, policy_data(questions=("team",)))})
    _, out = gw.post({**REQUEST, "questions": {"team": QUESTIONS["team"]}})
    block = out["answers"]["team"]["howsure"]
    assert (block["model"], block["action"], block["answer"]) == ("b", "auto", "billing")
    assert block["tried"][0]["answer"] is None and reason in block["tried"][0]["reason"]
    assert out["howsure"]["steps"][0]["error"] == "no usable answers"


def test_invented_answer_is_never_returned_even_with_no_other_model(start):
    a = Upstream((200, {"answers": {"team": {"type": "choice", "probabilities": {"hallucinated_team": 0.5}}}}))
    status, out = start({"a": (a, policy_data(questions=("team",)))}).post(
        {**REQUEST, "questions": {"team": QUESTIONS["team"]}}
    )
    assert status == 502 and "answers" not in out


def test_rounded_probabilities_still_count(start):
    a = Upstream(
        (200, {"answers": {"team": {"type": "choice", "probabilities": {"billing": 0.99, "technical": 0.02}}}})
    )
    gw = start({"a": (a, policy_data(questions=("team",)))})
    _, out = gw.post({**REQUEST, "questions": {"team": QUESTIONS["team"]}})
    assert out["answers"]["team"]["howsure"]["action"] == "auto"


@pytest.mark.parametrize("value", [5, -1, True, "0.9", None, [0.9]])
def test_bad_noul_values_never_settle_a_question(start, value):
    a = Upstream((200, {"answers": {"refund": {"type": "noul", "noul": value}}}))
    b = Upstream(refund=0.95)
    gw = start({"a": (a, policy_data(questions=("refund",))), "b": (b, policy_data(questions=("refund",)))})
    _, out = gw.post({**REQUEST, "questions": {"refund": QUESTIONS["refund"]}})
    assert out["answers"]["refund"]["howsure"]["model"] == "b"


@pytest.mark.parametrize(
    "answer, usable",
    [
        ({"probabilities": {"0": 0.1, "1": 0.1, "2": 0.8}}, True),
        ({"probabilities": {"Low": 0.1, "Mid": 0.1, "High": 0.8}, "legend": ["Low", "Mid", "High"]}, True),
        ({"probabilities": {"0": 0.2, "2": 0.8}}, False),
        ({"probabilities": {"0": 0.1, "1": 0.1, "2": 0.7, "3": 0.1}}, False),
        ({"probabilities": {"0": 0.2, "1": 0.0, "Never": 0.8}}, False),
        ({"probabilities": {"0": 0.5, "1": 0.1, "2": 0.1}}, False),
        ({"probabilities": {"²": 0.1, "1": 0.1, "2": 0.8}}, False),
    ],
    ids=["indices", "legend-names", "missing-level", "extra-level", "unknown-name", "does-not-add-up", "odd-digit"],
)
def test_score_answers_must_cover_exactly_the_levels(start, answer, usable):
    a = Upstream((200, {"answers": {"urgency": {"type": "score", **answer}}}))
    status, out = start({"a": (a, policy_data(cutoff=0.7, questions=("urgency",)))}).post(
        {**REQUEST, "questions": {"urgency": URGENCY}}
    )
    if usable:
        assert out["answers"]["urgency"]["howsure"]["action"] == "auto"
    else:
        assert status == 502


@pytest.mark.parametrize(
    "raw",
    [
        b'{"answers": {"refund": {"type": "noul", "noul": NaN}}}',
        b'{"answers": {"refund": {"type": "noul", "noul": Infinity}}}',
        b'{"answers": {"team": {"type": "choice", "probabilities": {"billing": 1e999, "technical": 1}}}}',
        b'{"usage": {"cost": NaN}, "answers": {"refund": {"type": "noul", "noul": 0.95}}}',
        b'{"answers": {"refund": {"type": "noul", "noul": 0.95, "confidence": -Infinity}}}',
    ],
    ids=["noul-nan", "noul-inf", "huge-probability", "usage-nan", "confidence-inf"],
)
def test_non_finite_numbers_fail_the_model_and_never_reach_the_app(start, raw):
    a, b = Upstream((200, raw)), Upstream(team=0.95, refund=0.95)
    status, out = start({"a": (a, policy_data()), "b": (b, policy_data())}).post()  # post parses strictly
    assert status == 200
    assert out["howsure"]["steps"][0]["error"] == "response has numbers that aren't finite"
    assert out["answers"]["refund"]["howsure"]["model"] == "b"
    assert "usage" not in out


def test_upstream_results_field_does_not_cost_the_verdicts(start):
    a = Upstream((200, {"results": [], "answers": answers(team=0.95, refund=0.95)}))
    _, out = start({"a": (a, policy_data())}).post()
    assert out["answers"]["team"]["howsure"]["action"] == "auto"
    assert "results" not in out


# ---------------------------------------------------------------- what apps get to see


PROVIDER_TEXT = {"error": {"message": "Incorrect API key sk-proj-abcd for internal.vendor.example"}}


@pytest.mark.parametrize("code", [401, 402, 403, 404, 429, 500, 503, 529])
def test_provider_error_text_stays_out_of_the_response(start, code):
    a, b = Upstream((code, PROVIDER_TEXT)), Upstream(team=0.95, refund=0.95)
    _, out = start({"a": (a, policy_data()), "b": (b, policy_data())}).post()
    assert out["howsure"]["steps"][0]["error"] == f"HTTP {code}"
    assert "sk-proj" not in json.dumps(out)

    _, out = start({"only": (Upstream((code, PROVIDER_TEXT)), policy_data())}, routes={}).post(
        {**REQUEST, "model": "only"}
    )
    assert "error" in out and "sk-proj" not in json.dumps(out)


def test_provider_error_text_goes_to_the_log_on_one_line(write_config):
    a = Upstream((401, b"key sk-proj-abcd\nwas rejected"))
    lines = []
    load_config(write_config({"a": (a, policy_data())})).handle({**REQUEST, "model": "a"}, lines.append)
    assert lines == ["a: a failed: HTTP 401: key sk-proj-abcd was rejected"]
    a.close()


def test_connection_error_detail_goes_to_the_log_on_one_line(write_config, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("line one\nline two")

    monkeypatch.setattr(gateway_module, "post_json", fail)
    lines = []
    load_config(write_config({"a": ("http://x", policy_data())})).handle({**REQUEST, "model": "a"}, lines.append)
    assert lines == ["a: a failed: line one line two"]


@pytest.mark.parametrize("code", [400, 413, 422])
def test_errors_about_the_apps_own_request_keep_the_providers_words(start, code):
    a = Upstream((code, {"error": {"message": "too many questions"}}))
    status, out = start({"a": (a, policy_data())}).post()
    assert status == code
    assert "too many questions" in out["error"]["message"]


def test_connection_errors_do_not_say_where_the_model_lives(start):
    url = closed_port_url()
    _, out = start({"dead": (url, policy_data())}).post({**REQUEST, "model": "dead"})
    assert out["howsure"]["steps"][0]["error"] == "connection failed"
    assert url.split("//")[1] not in json.dumps(out)


# ---------------------------------------------------------------- crashes


def test_unexpected_error_in_one_model_falls_back(start, monkeypatch):
    apply = Policy.apply

    def explode_for_a(self, *args, **kwargs):
        if self.model == "a":
            raise OverflowError("secret detail")
        return apply(self, *args, **kwargs)

    monkeypatch.setattr(Policy, "apply", explode_for_a)
    a, b = Upstream(team=0.95, refund=0.95), Upstream(team=0.95, refund=0.95)
    status, out = start({"a": (a, policy_data(name="a")), "b": (b, policy_data(name="b"))}).post()
    assert status == 200
    assert out["answers"]["team"]["howsure"]["model"] == "b"
    assert out["howsure"]["steps"][0]["error"] == "internal error"
    assert "secret detail" not in json.dumps(out)


def test_unexpected_error_still_gets_a_json_reply(start, monkeypatch):
    def explode(*args):
        raise RuntimeError("secret detail")

    gw = start({"a": (Upstream(team=0.95, refund=0.95), policy_data())})
    monkeypatch.setattr(gateway_module, "run_chain", explode)
    status, out = gw.post()
    assert status == 500 and out["error"]["type"] == "internal_error"
    assert "secret detail" not in json.dumps(out)


def test_one_crash_in_a_batch_keeps_the_other_results(start, monkeypatch):
    run_chain = gateway_module.run_chain

    def explode_on_boom(route, chain, request, questions, log):
        if request["state"] == "boom":
            raise RuntimeError("secret detail")
        return run_chain(route, chain, request, questions, log)

    gw = start({"a": (Upstream(team=0.95, refund=0.95), policy_data())})
    monkeypatch.setattr(gateway_module, "run_chain", explode_on_boom)
    status, out = gw.post(
        {"model": "tickets", "requests": [{"state": s, "questions": QUESTIONS} for s in ("ok", "boom", "ok")]}
    )
    assert status == 200
    assert [r.get("status") for r in out["results"]] == [None, 500, None]
    assert out["results"][2]["answers"]["team"]["howsure"]["action"] == "auto"


# ---------------------------------------------------------------- dead ends


def test_healthy_model_skipped_earlier_is_asked_when_the_deciding_one_fails(start):
    a, b = Upstream(team=0.95, refund=0.7), Upstream((503, {"error": "busy"}))
    gw = start({"a": (a, policy_data(questions=("team",))), "b": (b, policy_data(questions=("refund",)))})
    status, out = gw.post()
    assert status == 200
    assert a.asked == [["team"], ["refund"]] and b.asked == [["refund"]]
    block = out["answers"]["refund"]["howsure"]
    assert (block["model"], block["action"], block["reason"]) == ("a", "review", "no calibration for this question")
    assert out["answers"]["refund"]["noul"] == 0.7
    assert [s["model"] for s in out["howsure"]["steps"]] == ["a", "b", "a"]


def test_a_failed_model_is_not_asked_twice(start):
    a, b = Upstream((503, {"error": "busy"})), Upstream((503, {"error": "busy"}))
    gw = start({"a": (a, policy_data(questions=("team",))), "b": (b, policy_data(questions=("refund",)))})
    status, _ = gw.post()
    assert status == 503
    assert len(a.seen) == 1 and len(b.seen) == 1


def test_a_model_that_skipped_a_question_is_not_asked_it_again(start):
    a = Upstream(team=0.95)  # asked about both, answers only team
    _, out = start({"a": (a, policy_data())}).post()
    assert a.asked == [["team", "refund"]]
    assert out["answers"]["refund"]["howsure"]["reason"] == "no model answered"


def test_questions_with_an_answer_do_not_get_a_second_ask(start):
    a, b = Upstream(team=0.95, refund=0.6), Upstream((503, {"error": "busy"}))
    start({"a": (a, policy_data()), "b": (b, policy_data())}).post()
    assert a.asked == [["team", "refund"]]


# ---------------------------------------------------------------- limits


def test_oversized_body_is_refused(start, monkeypatch):
    monkeypatch.setattr(gateway_module, "MAX_BODY_BYTES", 10)
    a = Upstream(team=0.95, refund=0.95)
    status, out = start({"a": (a, policy_data())}).post()
    assert status == 413 and out["error"]["type"] == "invalid_request_error"
    assert a.seen == []


def test_oversized_batch_is_refused_before_any_call(start, monkeypatch):
    monkeypatch.setattr(gateway_module, "MAX_BATCH", 2)
    a = Upstream(team=0.95, refund=0.95)
    gw = start({"a": (a, policy_data())})
    request = {"state": "x", "questions": QUESTIONS}
    assert gw.post({"model": "tickets", "requests": [request] * 3})[0] == 413
    assert a.seen == []
    assert gw.post({"model": "tickets", "requests": [request] * 2})[0] == 200


def test_stalled_client_is_disconnected(start, monkeypatch):
    monkeypatch.setattr(gateway_module, "CLIENT_TIMEOUT", 0.1)
    gw = start({"a": (Upstream(team=0.95, refund=0.95), policy_data())})
    assert gw.raw(b"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Length: 1000\r\n\r\nonly-this") == b""


def test_chunked_request_is_told_what_to_do(start):
    gw = start({"a": (Upstream(team=0.95, refund=0.95), policy_data())})
    reply = gw.raw(b"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n")
    assert reply.startswith(b"HTTP/1.1 411") and b"Content-Length" in reply


@pytest.mark.parametrize("length", [b"abc", b"-5"])
def test_invalid_content_length_is_a_400(start, length):
    gw = start({"a": (Upstream(team=0.95, refund=0.95), policy_data())})
    reply = gw.raw(b"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Length: " + length + b"\r\n\r\n")
    assert reply.startswith(b"HTTP/1.1 400") and b"Content-Length" in reply


def test_rejected_body_is_read_before_closing(start, monkeypatch):
    """Closing with unread data can reset the connection before the client reads the 401."""
    monkeypatch.setenv("GW_KEY", "letmein")
    gw = start({"a": (Upstream(team=0.95, refund=0.95), policy_data())}, require_key_env="GW_KEY")
    with socket.create_connection(("127.0.0.1", gw.server.server_port), timeout=5) as s:
        s.sendall(b"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n\r\n")
        data = b""
        while not data.endswith(b"}"):
            data += s.recv(4096)
        assert data.startswith(b"HTTP/1.1 401")
        s.settimeout(0.2)
        with pytest.raises(TimeoutError):
            s.recv(1)  # still open, waiting for the body
        s.settimeout(5)
        s.sendall(b"hello")
        assert s.recv(1) == b""


def test_policy_data_helper_is_identity():
    """The tests above rely on calibrated == raw, so check it."""
    d = Policy(policy_data()).decide({"answers": answers(team=0.8, refund=0.7)})
    assert d["team"].probability == pytest.approx(0.8) and d["refund"].probability == pytest.approx(0.7)
