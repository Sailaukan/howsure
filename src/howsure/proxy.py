"""A drop-in proxy: point your app at it instead of the model server.

Every ``POST .../systemone`` response gets a ``howsure`` block per calibrated answer
(calibrated probability and ``action: auto | review``). Everything else, including
errors and other paths, passes through unchanged.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .client import base_url
from .policy import Policy


def make_server(
    policy: Policy,
    upstream: str,
    host: str = "127.0.0.1",
    port: int = 8001,
    key: str | None = None,
    safe: bool = False,
    quiet: bool = False,
) -> ThreadingHTTPServer:
    base = base_url(upstream)
    if base.endswith("/v1/systemone"):
        base = base[: -len("/v1/systemone")]

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            self._forward(None)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            self._forward(self.rfile.read(length) if length else b"")

        def _forward(self, body: bytes | None) -> None:
            headers = {"Accept": self.headers.get("Accept", "application/json")}
            if body is not None:
                headers["Content-Type"] = self.headers.get("Content-Type", "application/json")
            auth = self.headers.get("Authorization") or (f"Bearer {key}" if key else None)
            if auth:
                headers["Authorization"] = auth
            req = urllib.request.Request(base + self.path, data=body, method=self.command, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    status, data, ctype = r.status, r.read(), r.headers.get("Content-Type", "application/json")
            except urllib.error.HTTPError as e:
                status, data, ctype = e.code, e.read(), e.headers.get("Content-Type", "application/json")
            except (urllib.error.URLError, OSError) as e:
                msg = {"error": f"howsure proxy: upstream {base} unreachable ({getattr(e, 'reason', e)})"}
                self._send(502, json.dumps(msg).encode(), "application/json")
                return

            path = self.path.split("?", 1)[0].rstrip("/")
            if status == 200 and self.command == "POST" and path.endswith("/systemone"):
                try:
                    request = json.loads(body) if body else None
                    data = json.dumps(policy.apply(json.loads(data), request, safe=safe)).encode("utf-8")
                    ctype = "application/json"
                except (ValueError, TypeError, KeyError, AttributeError) as e:
                    self.log_message("could not annotate response (%s); passed through", e)
            self._send(status, data, ctype)

        def _send(self, status: int, data: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt: str, *args) -> None:
            if not quiet:
                sys.stderr.write("howsure: " + (fmt % args) + "\n")

    return ThreadingHTTPServer((host, port), Handler)
