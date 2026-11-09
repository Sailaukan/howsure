"""A gateway: several decision models behind one endpoint, each with its own policy.

Apps POST System One requests and name a route in ``model``. A route is a chain of models.
Each question moves down the chain until a model's policy marks it ``auto``, and when a
model fails, its questions move on to the next one. The response has one answer per
question, saying which model gave it and whether to act on it or send it to review.

The app's Authorization header is never forwarded: the models may belong to different
providers, so each key only ever goes to its own model's URL.
"""

from __future__ import annotations

import hmac
import http.client
import json
import os
import sys
import time
import traceback
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .client import endpoint, post_json
from .data import SUPPORTED_TYPES, BadAnswer, DataError, answer_probs, option_names
from .drift import DriftMonitor
from .policy import Policy

DEFAULT_TIMEOUT = 30.0  # seconds to wait for data; a stalled model should fall back, not hold up the app
MAX_TIMEOUT = 3600.0
BATCH_WORKERS = 8
MAX_BATCH = 1000  # each request can cost a paid call per model
MAX_BODY_BYTES = 64 * 1024 * 1024  # requests can carry page images
DRAIN_BYTES = 1024 * 1024  # how much of a rejected request's body to read before closing
CLIENT_TIMEOUT = 60  # seconds a client may stall while sending or reading
APP_FAULTS = {400, 413, 422}  # the app's request was rejected, so it needs the provider's reason
PASSED_THROUGH = APP_FAULTS | {429, 503, 504, 529}  # the app can act on these; anything else is a 502


class ModelFailed(Exception):
    """The message is safe to show the app. ``detail`` may hold provider text, so it only goes in the log."""

    def __init__(self, message: str, status: int | None = None, detail: str | None = None):
        super().__init__(message)
        self.status = status
        self.detail = detail or message


def _no_log(line: str) -> None:
    pass


def _error(kind: str, message: str) -> dict:
    return {"error": {"type": kind, "message": message}}


def _review_block(reason: str) -> dict:
    """A verdict for an answer no policy could judge. Same shape as a calibrated one."""
    return {"answer": None, "probability": 0.0, "action": "review", "probabilities": {}, "reason": reason}


def _http_failure(e: urllib.error.HTTPError) -> ModelFailed:
    if 300 <= e.code < 400:
        return ModelFailed(
            f"HTTP {e.code}: redirects aren't followed",
            e.code,
            f"HTTP {e.code} to {e.headers.get('Location')}; point this model's url there",
        )
    try:
        text = " ".join(e.read()[:300].decode("utf-8", "replace").split())
    except (OSError, http.client.HTTPException):
        text = ""
    full = f"HTTP {e.code}: {text}" if text else f"HTTP {e.code}"
    return ModelFailed(full if e.code in APP_FAULTS else f"HTTP {e.code}", e.code, full)


def _uncalibrated_problem(qdef: dict, answer: dict) -> str | None:
    """Why an answer no policy looked at can't go to a reviewer, if it can't."""
    if qdef["type"] not in SUPPORTED_TYPES:
        return None
    try:
        answer_probs(qdef["type"], answer, option_names(qdef))
    except BadAnswer as e:
        return str(e)
    return None


def _rank(block: dict) -> tuple[bool, float]:
    """Order for review answers: calibrated probabilities first, since only those compare across models."""
    return "reason" not in block, block["probability"]


def _sum_usage(responses: list[dict]) -> dict:
    """Totals for what every model reports, since a total missing some models' share would mislead."""
    usages = [r["usage"] if isinstance(r.get("usage"), dict) else {} for r in responses]
    total: dict = {}
    for k in usages[0]:
        values = [u.get(k) for u in usages]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            total[k] = sum(values)
    return total


@dataclass
class Model:
    name: str
    url: str
    policy: Policy
    model: str | None = None  # model name to send upstream
    key: str | None = field(default=None, repr=False)
    safe: bool = False
    timeout: float = DEFAULT_TIMEOUT
    monitor: DriftMonitor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.monitor = DriftMonitor(self.policy)

    def can_auto(self, qid: str, qdef: dict) -> bool:
        return self.policy.can_auto(qid, qdef, self.safe)

    def ask(self, request: dict, questions: dict) -> dict:
        """Ask about some of the request's questions. Returns the response with verdicts added."""
        body = {k: v for k, v in request.items() if k != "model"}
        body["questions"] = questions
        if self.model:
            body["model"] = self.model
        try:
            response = post_json(self.url, body, self.key, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            raise _http_failure(e) from None
        except (OSError, http.client.HTTPException) as e:
            reason = getattr(e, "reason", e)
            message = "timed out" if isinstance(reason, TimeoutError) else "connection failed"
            raise ModelFailed(message, detail=str(reason)) from None
        except ValueError:
            raise ModelFailed("response is not JSON") from None
        if not isinstance(response, dict) or not isinstance(response.get("answers"), dict):
            raise ModelFailed("response has no 'answers'")
        try:
            json.dumps(response, allow_nan=False)  # NaN or Infinity anywhere would make our own response invalid
        except ValueError:
            raise ModelFailed("response has numbers that aren't finite") from None
        response.pop("results", None)  # we only send single requests; Policy.apply would read this as a batch
        for answer in response["answers"].values():
            if isinstance(answer, dict):
                answer.pop("howsure", None)  # only this gateway's policies decide
        return self.policy.apply(response, body, safe=self.safe, monitor=self.monitor)


def run_chain(route: str, chain: list[Model], request: dict, questions: dict, log=_no_log) -> tuple[int, dict]:
    """Send each question down the chain. Returns (HTTP status, System One response)."""
    started = time.monotonic()
    best: dict[str, dict] = {}  # question -> the answer to return so far
    settled: set[str] = set()  # questions some model marked auto
    trail: dict[str, list[dict]] = {qid: [] for qid in questions}
    asked: dict[str, set[str]] = {model.name: set() for model in chain}
    failed: set[str] = set()
    steps: list[dict] = []
    responses: list[dict] = []

    def call(model: Model, ask: dict) -> None:
        if not ask:
            return
        asked[model.name].update(ask)
        step: dict = {"model": model.name, "asked": list(ask)}
        steps.append(step)
        call_started = time.monotonic()
        try:
            response = model.ask(request, ask)
        except Exception as e:  # whatever went wrong, the next model may still answer
            failure = e if isinstance(e, ModelFailed) else ModelFailed("internal error", detail=repr(e))
            step["error"] = str(failure)
            if failure.status is not None:
                step["status"] = failure.status
            failed.add(model.name)
            if failure.detail != str(failure):
                log(f"{route}: {model.name} failed: {' '.join(failure.detail.split())}")
            return
        finally:
            step["latency_ms"] = round((time.monotonic() - call_started) * 1000, 1)

        usable = False
        step["auto"] = []
        for qid, qdef in ask.items():
            answer = response["answers"].get(qid)
            if not isinstance(answer, dict):
                continue
            block = answer.get("howsure")
            unreadable = block is not None and block["answer"] is None
            if block is None:
                problem = _uncalibrated_problem(qdef, answer)
                unreadable = problem is not None
                block = answer["howsure"] = _review_block(problem or "no calibration for this question")
            block["model"] = model.name
            trail[qid].append(
                {k: block[k] for k in ("model", "answer", "probability", "action", "reason") if k in block}
            )
            if unreadable:
                continue  # as good as no answer
            usable = True
            if block["action"] == "auto":
                best[qid] = answer
                settled.add(qid)
                step["auto"].append(qid)
            elif qid not in best or _rank(block) > _rank(best[qid]["howsure"]):
                best[qid] = answer
        if not usable:
            step["error"] = "no usable answers"
            failed.add(model.name)
            return
        responses.append(response)
        for k, name in (("id", "id"), ("request_id", "request_id"), ("model", "served_model"), ("usage", "usage")):
            if k in response:
                step[name] = response[k]

    for i, model in enumerate(chain):
        later = chain[i + 1 :]
        # Only ask a model what it could mark auto, except that a question no remaining
        # model could decide still needs one answer for the reviewer.
        call(
            model,
            {
                qid: q
                for qid, q in questions.items()
                if qid not in settled
                and (model.can_auto(qid, q) or (qid not in best and not any(m.can_auto(qid, q) for m in later)))
            },
        )

    # The model that could decide a question may have failed after the ones before it were
    # skipped. They can't mark it auto, but an answer to review beats none.
    for model in chain:
        if model.name not in failed:
            call(model, {qid: q for qid, q in questions.items() if qid not in best and qid not in asked[model.name]})

    info = {"route": route, "steps": steps}
    if not best:
        codes = {s.get("status") for s in steps}
        status = codes.pop() if len(codes) == 1 and codes <= PASSED_THROUGH else 502
        message = f"howsure gateway: every model in route {route!r} failed; last error: {steps[-1]['error']}"
        return status, {**_error("upstream_error", message), "howsure": info}

    out: dict = {"model": route, "answers": {}}
    for qid, q in questions.items():
        answer = best.get(qid)
        if answer is None:
            answer = {"type": q.get("type"), "howsure": _review_block("no model answered")}
        elif len(trail[qid]) > 1:
            answer["howsure"]["tried"] = trail[qid]
        out["answers"][qid] = answer
    usage = _sum_usage(responses)
    if usage:
        out["usage"] = usage
    out["latency_ms"] = round((time.monotonic() - started) * 1000, 1)
    out["howsure"] = info
    return 200, out


@dataclass
class Gateway:
    models: dict[str, Model]
    routes: dict[str, list[Model]]  # every model is also a route of one
    default: str | None = None
    key: str | None = field(default=None, repr=False)  # what apps must send, if set
    warnings: list[str] = field(default_factory=list)

    def handle(self, body, log=_no_log) -> tuple[int, dict]:
        """A System One request or batch -> (HTTP status, response body). ``log`` gets what apps shouldn't see."""
        if isinstance(body, dict) and "requests" in body:
            return self._batch(body, log)
        return self._one(body, log)

    def _one(self, request, log) -> tuple[int, dict]:
        try:
            return self._route(request, log)
        except Exception:  # still answer, and in a batch keep the other results
            log(f"internal error: {traceback.format_exc()}")
            return 500, _error("internal_error", "howsure gateway: internal error")

    def _route(self, request, log) -> tuple[int, dict]:
        if not isinstance(request, dict):
            return 400, _error("invalid_request_error", "the request must be a JSON object")
        name = request.get("model") or self.default
        chain = self.routes.get(name) if isinstance(name, str) else None
        if chain is None:
            problem = f"unknown model {name!r}" if name else "the request has no 'model' and there's no default route"
            return 400, _error("invalid_request_error", f"{problem}; choose from: {', '.join(sorted(self.routes))}")
        questions = request.get("questions")
        if (
            not isinstance(questions, dict)
            or not questions
            or not all(isinstance(q, dict) and isinstance(q.get("type"), str) for q in questions.values())
        ):
            return 400, _error(
                "invalid_request_error", "'questions' must be an object of named questions, each with a 'type'"
            )
        return run_chain(name, chain, request, questions, log)

    def _batch(self, body: dict, log) -> tuple[int, dict]:
        requests = body["requests"]
        if not isinstance(requests, list):
            return 400, _error("invalid_request_error", "'requests' must be a list")
        if len(requests) > MAX_BATCH:
            return 413, _error("invalid_request_error", f"a batch can have at most {MAX_BATCH} requests")
        if not requests:
            return 200, {"results": []}
        shared = {k: v for k, v in body.items() if k != "requests"}

        def one(request) -> dict:
            status, out = self._one({**shared, **request} if isinstance(request, dict) else request, log)
            return out if status == 200 else {"status": status, **out}

        with ThreadPoolExecutor(max_workers=min(BATCH_WORKERS, len(requests))) as pool:
            return 200, {"results": list(pool.map(one, requests))}


def load_config(path: str) -> Gateway:
    """Read a gateway config. Policy paths are relative to the config file."""

    def bad(message: str) -> DataError:
        return DataError(f"{path}: {message}")

    def secret(name, where: str) -> str:
        if not isinstance(name, str) or not os.environ.get(name):
            raise bad(f"{where}: environment variable {name} is not set")
        return os.environ[name]

    def only(obj: dict, allowed: set, where: str) -> None:
        unknown = set(obj) - allowed
        if unknown:  # a typo like "saef" must not silently loosen a policy
            raise bad(f"{where}: unknown setting(s): {', '.join(sorted(unknown))}")

    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except ValueError as e:
        raise bad(f"not valid JSON ({e})") from None
    if not isinstance(cfg, dict) or not isinstance(cfg.get("models"), dict) or not cfg["models"]:
        raise bad("expected a 'models' object with at least one model")
    only(cfg, {"models", "routes", "default", "require_key_env"}, "config")

    folder = os.path.dirname(os.path.abspath(path))
    models: dict[str, Model] = {}
    for name, spec in cfg["models"].items():
        where = f"model {name!r}"
        if (
            not isinstance(spec, dict)
            or not isinstance(spec.get("url"), str)
            or not isinstance(spec.get("policy"), str)
        ):
            raise bad(f"{where} needs a 'url' and a 'policy'")
        only(spec, {"url", "policy", "model", "key_env", "safe", "timeout"}, where)
        url = endpoint(spec["url"])
        if not url.startswith(("http://", "https://")):
            raise bad(f"{where}: 'url' must start with http:// or https://, or be 'openrouter'")
        timeout = spec.get("timeout", DEFAULT_TIMEOUT)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= MAX_TIMEOUT:
            raise bad(f"{where}: 'timeout' must be above 0 and at most {MAX_TIMEOUT:g} seconds")
        if not isinstance(spec.get("safe", False), bool):
            raise bad(f"{where}: 'safe' must be true or false")
        if not isinstance(spec.get("model", ""), str):
            raise bad(f"{where}: 'model' must be a string")
        policy_path = os.path.join(folder, spec["policy"])
        try:
            policy = Policy.load(policy_path)
        except (ValueError, AttributeError, OSError) as e:
            raise bad(f"{where}: can't read {policy_path} ({e})") from None
        key = secret(spec["key_env"], where) if "key_env" in spec else None
        models[name] = Model(name, url, policy, spec.get("model"), key, spec.get("safe", False), timeout)

    routes = {name: [model] for name, model in models.items()}
    warnings = []
    configured = cfg.get("routes", {})
    if not isinstance(configured, dict):
        raise bad("'routes' must be an object")
    for name, chain in configured.items():
        if not isinstance(chain, list) or not chain or not all(isinstance(m, str) for m in chain):
            raise bad(f"route {name!r} needs a list of model names")
        missing = [m for m in chain if m not in models]
        if missing:
            raise bad(f"route {name!r} uses unknown model(s): {', '.join(missing)}")
        if len(set(chain)) < len(chain):
            raise bad(f"route {name!r} lists a model twice")
        routes[name] = [models[m] for m in chain]
        for i, model in enumerate(routes[name]):
            earlier = [m.policy.model for m in routes[name][:i]]
            after = model.policy.data.get("after")
            if after is None and earlier:
                warnings.append(
                    f"route {name!r}: {model.name}'s policy wasn't fitted for this chain, so it may miss its "
                    "target. Fit the chain with: howsure fit first.preds.jsonl second.preds.jsonl ..."
                )
            elif after is not None and after != earlier:
                raise bad(
                    f"route {name!r}: {model.name}'s policy was fitted after {after or 'nothing'}, "
                    f"but this route puts it after {earlier or 'nothing'}"
                )
            elif after is not None and len(chain) > 1 and model.safe != model.policy.data.get("safe_chain", False):
                fitted_with = "with" if model.safe is False else "without"
                raise bad(
                    f"route {name!r}: {model.name} sets 'safe' to {str(model.safe).lower()}, but its chain was fitted "
                    f"{fitted_with} --safe, so the next model would see a different mix than it was fitted on"
                )

    default = cfg.get("default")
    if default is not None and (not isinstance(default, str) or default not in routes):
        raise bad(f"default {default!r} isn't a route or a model")
    key = secret(cfg["require_key_env"], "require_key_env") if "require_key_env" in cfg else None
    return Gateway(models, routes, default, key, warnings)


def _log_lines(out: dict) -> list[str]:
    lines = []
    for result in out.get("results", [out]):
        info = result.get("howsure") if isinstance(result, dict) else None
        if info:
            parts = [
                f"{s['model']} failed ({s['error']})"
                if "error" in s
                else f"{s['model']} {len(s['auto'])}/{len(s['asked'])} auto"
                for s in info["steps"]
            ]
            lines.append(f"{info['route']}: " + ", ".join(parts))
    return lines


def make_server(
    gateway: Gateway, host: str = "127.0.0.1", port: int = 8001, quiet: bool = False
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = CLIENT_TIMEOUT  # a stalled client gets disconnected instead of holding a thread

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0].rstrip("/")
            if path == "/health":
                self._send(200, {"status": "ok"})
            elif not self._authorized():
                pass
            elif path == "/v1/models":
                routes = [{"id": name, "chain": [m.name for m in chain]} for name, chain in gateway.routes.items()]
                self._send(200, {"default": gateway.default, "data": routes})
            else:
                self._send(
                    404, _error("not_found", "this gateway serves POST /v1/systemone, GET /v1/models and GET /health")
                )

        def do_POST(self) -> None:
            if not self._authorized():
                return
            if not self.path.split("?", 1)[0].rstrip("/").endswith("/systemone"):
                self._send(404, _error("not_found", "this gateway serves POST /v1/systemone"), close=True)
                return
            declared = self.headers.get("Content-Length")
            if declared is None:
                self._send(
                    411,
                    _error("invalid_request_error", "send a Content-Length header; chunked bodies aren't supported"),
                    close=True,
                )
                return
            try:
                length = int(declared)
            except ValueError:
                length = -1
            if length < 0:
                self._send(400, _error("invalid_request_error", "the Content-Length header isn't valid"), close=True)
                return
            if length > MAX_BODY_BYTES:
                self._send(
                    413, _error("invalid_request_error", f"the request body is over {MAX_BODY_BYTES} bytes"), close=True
                )
                return
            try:
                body = json.loads(self.rfile.read(length))
            except (ValueError, RecursionError):
                self._send(400, _error("invalid_request_error", "the request body must be JSON"))
                return
            status, out = gateway.handle(body, lambda line: self.log_message("%s", line))
            self._send(status, out)
            for line in _log_lines(out):
                self.log_message("%s", line)

        def _authorized(self) -> bool:
            if gateway.key is None:
                return True
            sent = self.headers.get("Authorization", "").encode("utf-8", "replace")
            if hmac.compare_digest(sent, f"Bearer {gateway.key}".encode()):
                return True
            self._send(401, _error("authentication_error", "missing or wrong gateway key"), close=True)
            return False

        def _send(self, status: int, obj: dict, close: bool = False) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            if close:  # the request body wasn't read, so this connection can't be reused
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
            if close:
                self._drain()

        def _drain(self) -> None:
            """Closing with unread data resets the connection, which can destroy a reply the client hasn't read yet."""
            try:
                self.rfile.read(max(0, min(int(self.headers.get("Content-Length") or 0), DRAIN_BYTES)))
            except (ValueError, OSError):
                pass

        def log_message(self, fmt: str, *args) -> None:
            if not quiet:
                sys.stderr.write("howsure: " + (fmt % args) + "\n")

    return ThreadingHTTPServer((host, port), Handler)
