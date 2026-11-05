"""Call a System One endpoint (Jev, Drex, Kev, ...) for every labeled example.

Results are appended to the output file as they arrive, so an interrupted run
resumes where it stopped: rerunning the same command only calls the model for
examples it hasn't answered yet.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from .data import DataError, example_id, read_jsonl, request_of, validate_labels

PRESETS = {"openrouter": "https://openrouter.ai/api"}
BACKOFF_SECONDS = 1.0  # first retry delay; doubles each attempt
ATTEMPTS = 4


def base_url(url: str) -> str:
    return PRESETS.get(url, url).rstrip("/")


def endpoint(url: str) -> str:
    base = base_url(url)
    return base if base.endswith("/systemone") else base + "/v1/systemone"


def api_key(key: str | None, url: str) -> str | None:
    if key:
        return key
    if os.environ.get("HOWSURE_API_KEY"):
        return os.environ["HOWSURE_API_KEY"]
    if "openrouter.ai" in base_url(url):
        return os.environ.get("OPENROUTER_API_KEY")
    return None


class CallError(Exception):
    pass


def post_json(url: str, body: dict, key: str | None, timeout: float = 300) -> dict:
    from . import __version__

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"howsure/{__version__}",
        },
    )
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def call(url: str, body: dict, key: str | None) -> dict:
    """POST with retries on rate limits, server errors and dropped connections."""
    last = "no attempt made"
    for attempt in range(ATTEMPTS):
        if attempt:
            time.sleep(min(30.0, BACKOFF_SECONDS * 2 ** (attempt - 1)))
        try:
            return post_json(url, body, key)
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode("utf-8", "replace").strip()
            last = f"HTTP {e.code}: {detail}" if detail else f"HTTP {e.code}"
            if e.code not in (408, 409, 429) and e.code < 500:
                raise CallError(last) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            last = str(getattr(e, "reason", e))
    raise CallError(last)


@dataclass
class RunResult:
    path: str
    answered: int = 0
    new: int = 0
    failed: list = field(default_factory=list)


def _cache_key(rec: dict, model: str | None) -> str:
    return f"{example_id(rec)}|{model or ''}"


def _output_line(rec: dict, resp: dict, model: str | None) -> dict:
    line = {"id": example_id(rec)}
    if model:
        line["model"] = model
    line.update({k: v for k, v in rec.items() if k not in ("id", "model", "answers", "usage", "latency_ms")})
    line["answers"] = resp.get("answers")
    for k in ("usage", "latency_ms"):
        if k in resp:
            line[k] = resp[k]
    return line


def run(
    records: list[dict],
    url: str,
    out_path: str,
    model: str | None = None,
    key: str | None = None,
    workers: int = 4,
    log=None,
) -> RunResult:
    log = log or (lambda msg: print(msg, file=sys.stderr))
    validate_labels(records)
    ep, key = endpoint(url), api_key(key, url)

    cache: dict[str, dict] = {}
    if os.path.exists(out_path):
        for line in read_jsonl(out_path):
            if isinstance(line.get("answers"), dict):
                cache[_cache_key(line, line.get("model"))] = line

    todo, seen = [], set()
    for rec in records:
        ck = _cache_key(rec, model)
        if ck not in cache and ck not in seen:
            seen.add(ck)
            todo.append(rec)
    result = RunResult(out_path)
    if todo:
        log(f"Asking {ep} about {len(todo)} example(s)" + (f" ({len(records) - len(todo)} already done)" if len(todo) < len(records) else ""))

    lock = threading.Lock()
    done = 0
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        with open(out_path, "a", encoding="utf-8") as f:
            futures = {pool.submit(call, ep, request_of(rec, model), key): rec for rec in todo}
            for fut in as_completed(futures):
                rec = futures[fut]
                done += 1
                try:
                    resp = fut.result()
                    if not isinstance(resp.get("answers"), dict):
                        raise CallError("response has no 'answers'")
                except CallError as e:
                    result.failed.append(str(e))
                    continue
                line = _output_line(rec, resp, model)
                with lock:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    f.flush()
                cache[_cache_key(rec, model)] = line
                result.new += 1
                if done % 25 == 0 or done == len(todo):
                    log(f"  {done}/{len(todo)}")
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        log("Stopped. Rerun the same command to continue where it left off.")
        raise
    pool.shutdown()

    # Rewrite in input order, with the labels as they are now (they may have been edited).
    final = []
    for rec in records:
        hit = cache.get(_cache_key(rec, model))
        if hit:
            final.append(_output_line(rec, hit, model))
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for line in final:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)
    result.answered = len(final)
    if result.failed and not final:
        raise DataError(f"Every call failed. First error: {result.failed[0]}")
    return result
