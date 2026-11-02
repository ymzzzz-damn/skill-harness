"""Bound image history in outgoing Responses requests without changing artifacts.

Only the documented content lists in ``input`` are visited. Text, function call
arguments, tool schemas, call IDs and opaque reasoning are never interpreted.
The local HTTP proxy has its own token and a single fixed upstream; it performs
no model calls itself, follows no redirects and never retries a request.
"""
from __future__ import annotations
import copy
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import secrets
import select
import socket
import threading
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class ImageContextError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _limit(value: int) -> None:
    if type(value) is not int or not 1 <= value <= 50:
        raise ValueError("max_images must be an integer from 1 to 50")


def prune_response_images(payload: dict[str, Any], max_images: int = 48, *,
                          force_store_false: bool = True,
                          reference_image_count: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an independent payload and an audit containing counts/paths only.

    Keep all initial user-turn reference images and all images in the latest
    image-bearing observation. Fill remaining slots with the most recent older
    images. An overlarge protected set is an error, never silently reduced.
    The default 48 is a local service setting, not a universal API image limit.
    """
    _limit(max_images)
    if reference_image_count is not None and (type(reference_image_count) is not int or reference_image_count < 0):
        raise ValueError("reference_image_count must be a nonnegative integer or None")
    if not isinstance(payload, dict):
        raise ImageContextError("invalid_payload", "Responses payload must be a JSON object")
    for field in ("previous_response_id", "conversation", "prompt"):
        if payload.get(field):
            raise ImageContextError("hidden_history", "Server-side history or prompt references cannot be image-counted locally")
    result = copy.deepcopy(payload)
    items = result.get("input", [])
    if isinstance(items, str):
        items = []
    elif not isinstance(items, list):
        raise ImageContextError("unsupported_input", "Responses input must be text or an explicit item list")
    images: list[tuple[list[Any], int, str, int, bool]] = []
    # Consecutive user messages form one observation turn. This also protects
    # initial reference pictures when the client emits one message per picture.
    initial_turn = True
    seen_user = False
    group = 0
    previous_user = False
    for item_index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        kind = item.get("type", "message" if "role" in item else "")
        if kind in ("item_reference", "computer_call_output"):
            raise ImageContextError("unsupported_image_history", "Item references and computer screenshots require an explicit image-history policy")
        if not seen_user and item.get("role") in ("system", "developer") and kind != "message":
            # Codex may prepend additional_tools developer metadata before its
            # initial user messages. It is not model output or an observation,
            # and its tool schemas are not conversation image content.
            continue
        is_message = kind == "message" or ("role" in item and "content" in item)
        is_user = is_message and item.get("role") == "user"
        if not is_user or not previous_user:
            group += 1
        protected = False
        if is_user:
            if not seen_user:
                seen_user = True
            protected = initial_turn
        elif seen_user:
            initial_turn = False
        elif kind not in ("message",) or item.get("role") not in ("system", "developer"):
            initial_turn = False
        previous_user = is_user
        if is_message:
            blocks, field = item.get("content"), "content"
        elif kind == "function_call_output":
            blocks, field = item.get("output"), "output"
        else:
            # In particular, never recurse into function arguments, reasoning,
            # tool definitions or arbitrary embedded JSON strings.
            continue
        if blocks is None or isinstance(blocks, str):
            continue
        if not isinstance(blocks, list):
            raise ImageContextError("unsupported_content", "Message/tool output content must be text or a content list")
        for block_index, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type in ("computer_screenshot", "screenshot"):
                raise ImageContextError("unsupported_image_history", "Computer screenshot content cannot be replaced with a text content part")
            if block_type in ("input_image", "image", "image_url"):
                images.append((blocks, block_index, f"input[{item_index}].{field}[{block_index}]", group, protected))
    initial = {i for i, entry in enumerate(images) if entry[4]}
    if reference_image_count is not None and len(initial) != reference_image_count:
        raise ImageContextError("initial_reference_mismatch", "The explicit initial reference image count no longer matches the visible initial task turn")
    latest_group = images[-1][3] if images else None
    latest = {i for i, entry in enumerate(images) if entry[3] == latest_group}
    protected = initial | latest
    if len(protected) > max_images:
        raise ImageContextError("protected_images_exceed_limit", "Initial reference images plus the latest observation exceed max_images; none were removed")
    kept = set(protected)
    for index in range(len(images) - 1, -1, -1):
        if len(kept) >= max_images:
            break
        kept.add(index)
    removed_paths = []
    for index, (blocks, block_index, path, _, _) in enumerate(images):
        if index not in kept:
            blocks[block_index] = {"type": "input_text", "text":
                "[Earlier observation image omitted from model context; original image remains in the experiment artifacts.]"}
            removed_paths.append(path)
    if force_store_false:
        result["store"] = False
    audit = {"images_before": len(images), "images_after": len(kept),
             "images_removed": len(removed_paths), "max_images": max_images,
             "initial_images_preserved": len(initial), "latest_observation_images_preserved": len(latest),
             "removed_paths": removed_paths,
             "protected_paths": [images[i][2] for i in sorted(protected)]}
    return result, audit


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


_HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                "te", "trailer", "transfer-encoding", "upgrade"}


def _response_socket(response: Any) -> socket.socket | None:
    current = response
    for _ in range(4):
        sock = getattr(current, "_sock", None)
        if sock is not None:
            return sock
        current = getattr(current, "raw", None) or getattr(current, "fp", None)
        if current is None:
            break
    return None


class _SSETerminal:
    """Observe only top-level SSE completion metadata; never retain/log content."""
    def __init__(self):
        self.pending = b""
        self.dropping_line = False
        self.event_terminal = False
        self.data_terminal = False

    def feed(self, chunk: bytes) -> bool:
        self.pending += chunk
        terminal = False
        names = {"response.completed", "response.failed", "response.incomplete"}
        while b"\n" in self.pending:
            line, self.pending = self.pending.split(b"\n", 1)
            if self.dropping_line:
                self.dropping_line = False
                continue
            line = line.rstrip(b"\r")
            if not line:
                terminal = terminal or self.event_terminal or self.data_terminal
                self.event_terminal = self.data_terminal = False
            elif line.startswith(b"event:"):
                self.event_terminal = line[6:].strip().decode("ascii", errors="ignore") in names
            elif line.startswith(b"data:") and len(line) <= 65536:
                value = line[5:].strip()
                if value == b"[DONE]":
                    self.data_terminal = True
                else:
                    try:
                        metadata = json.loads(value)
                    except (ValueError, UnicodeError):
                        continue
                    if (isinstance(metadata, dict) and isinstance(metadata.get("type"), str)
                            and metadata["type"] in names):
                        self.data_terminal = True
        if len(self.pending) > 65536:
            # Large deltas are forwarded unchanged. Their contents cannot be a
            # short top-level event header; do not retain a second payload copy.
            self.pending = b""
            self.dropping_line = True
        return terminal


class ImageContextProxy:
    """Loopback-only, token-authenticated, fixed-upstream HTTP forwarding context.

    ``base_url`` and ``client_api_key`` are for the model subprocess. The real
    upstream credential remains in this instance's memory and is never logged.
    urllib retains environment proxy discovery and normal TLS certificate checks.
    """
    def __init__(self, upstream_url: str, upstream_api_key: str | None = None, *,
                 log_path: Path | str | None = None, max_images: int = 48,
                 timeout_s: float = 300, max_body_bytes: int = 128 * 1024 * 1024,
                 reference_image_count: int | None = None):
        _limit(max_images)
        parsed = urlsplit(upstream_url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname or
                parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment):
            raise ValueError("upstream_url must be a fixed HTTP(S) API base without credentials/query/fragment")
        if not math.isfinite(timeout_s) or timeout_s <= 0 or type(max_body_bytes) is not int or max_body_bytes < 1:
            raise ValueError("positive timeout_s and max_body_bytes required")
        if reference_image_count is not None and (type(reference_image_count) is not int or reference_image_count < 0):
            raise ValueError("reference_image_count must be a nonnegative integer or None")
        if upstream_api_key is not None and (not isinstance(upstream_api_key, str) or
                                             "\n" in upstream_api_key or "\r" in upstream_api_key):
            raise ValueError("upstream_api_key must be text without newlines, or None")
        self._upstream_url = upstream_url.rstrip("/")
        self._upstream_key = upstream_api_key
        self._prefix = parsed.path.rstrip("/")
        self.max_images, self.timeout_s = max_images, float(timeout_s)
        self.max_body_bytes = max_body_bytes
        self.reference_image_count = reference_image_count
        self.log_path = Path(log_path) if log_path is not None else None
        self.client_api_key = secrets.token_urlsafe(32)
        self.base_url: str | None = None
        self._server = None
        self._thread = None
        self._lock = threading.Lock()
        self._sockets: set[socket.socket] = set()
        self._responses: set[Any] = set()
        self._audit: list[dict[str, Any]] = []
        self._request_index = 0
        self._closing = False
        self._opener = build_opener(_NoRedirects())

    def _record(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._audit.append(copy.deepcopy(record))
            if self.log_path is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def summary(self) -> dict[str, Any]:
        with self._lock:
            audit = copy.deepcopy(self._audit)
            started = self._request_index
        return {"schema": "harness-image-context-v1", "max_images": self.max_images,
                "reference_image_count": self.reference_image_count,
                "requests_started": started, "pending_requests": started - len(audit),
                "requests": len(audit), "upstream_requests": sum(bool(x.get("upstream_requested")) for x in audit),
                "requests_pruned": sum(x.get("images_removed", 0) > 0 for x in audit),
                "images_removed": sum(x.get("images_removed", 0) for x in audit),
                "max_images_before": max((x.get("images_before", 0) for x in audit), default=0),
                "max_images_after": max((x.get("images_after", 0) for x in audit), default=0),
                "errors": [x["error"] for x in audit if x.get("error")],
                "policy": "retain all initial references and latest observation; prune older images only",
                "original_artifacts_modified": False, "automatic_replays": False,
                "audit": audit}

    def __enter__(self):
        if self._server is not None or self._closing:
            raise RuntimeError("proxy instances cannot be restarted")
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format, *args):
                pass  # BaseHTTPRequestHandler would print request URLs/errors.

            def setup(self):
                super().setup()
                self.connection.settimeout(proxy.timeout_s)
                with proxy._lock:
                    proxy._sockets.add(self.connection)

            def finish(self):
                try:
                    super().finish()
                finally:
                    with proxy._lock:
                        proxy._sockets.discard(self.connection)

            def _error(self, status: int, code: str, message: str):
                body = json.dumps({"error": {"type": "image_context_error", "code": code,
                                              "message": message}}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True

            def do_GET(self):
                self._error(405, "unsupported_endpoint", "Only Responses POST endpoints are supported")

            def do_POST(self):
                with proxy._lock:
                    proxy._request_index += 1
                    index = proxy._request_index
                record: dict[str, Any] = {"request": index, "upstream_requested": False}
                upstream = None
                disconnected = threading.Event()
                terminal_forwarded = threading.Event()
                stream_finished = threading.Event()
                watcher = None
                try:
                    expected = "Bearer " + proxy.client_api_key
                    if not hmac.compare_digest(self.headers.get("Authorization", "").encode(), expected.encode()):
                        record["error"] = "local_authentication_failed"
                        self._error(401, record["error"], "Local proxy token required")
                        return
                    endpoints = {proxy._prefix + "/responses": "/responses",
                                 proxy._prefix + "/responses/compact": "/responses/compact"}
                    endpoint = endpoints.get(self.path)
                    if endpoint is None:
                        record["error"] = "unsupported_endpoint"
                        self._error(404, record["error"], "Unsupported local proxy endpoint")
                        return
                    record["endpoint"] = endpoint
                    if (self.headers.get("Content-Encoding", "identity").lower() != "identity" or
                            self.headers.get("Transfer-Encoding")):
                        record["error"] = "unsupported_request_encoding"
                        self._error(415, record["error"], "Send an uncompressed JSON request with Content-Length")
                        return
                    try:
                        length = int(self.headers.get("Content-Length", "-1"))
                    except ValueError:
                        length = -1
                    if not 0 <= length <= proxy.max_body_bytes:
                        record["error"] = "invalid_body_length"
                        self._error(413, record["error"], "Request body length is invalid or too large")
                        return
                    raw = self.rfile.read(length)
                    if len(raw) != length:
                        raise ImageContextError("incomplete_body", "Request body was incomplete")
                    try:
                        payload = json.loads(raw)
                    except (ValueError, UnicodeError):
                        raise ImageContextError("invalid_json", "Request body must be valid JSON") from None
                    payload, counts = prune_response_images(payload, proxy.max_images,
                        force_store_false=endpoint == "/responses",
                        reference_image_count=proxy.reference_image_count)
                    record.update(counts)
                    try:
                        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
                    except ValueError:
                        raise ImageContextError("invalid_json", "Request body contains nonfinite numbers") from None
                    omitted = _HOP_HEADERS | {"host", "authorization", "content-length", "content-encoding",
                                              "api-key", "x-api-key", "cookie"}
                    omitted |= {x.strip().lower() for x in self.headers.get("Connection", "").split(",")}
                    headers = {key: value for key, value in self.headers.items() if key.lower() not in omitted}
                    headers["Content-Type"] = "application/json"
                    if proxy._upstream_key:
                        headers["Authorization"] = "Bearer " + proxy._upstream_key
                    request = Request(proxy._upstream_url + endpoint, data=body, headers=headers, method="POST")
                    record["upstream_requested"] = True
                    try:
                        upstream = proxy._opener.open(request, timeout=proxy.timeout_s)
                    except HTTPError as exc:
                        upstream = exc  # Preserve upstream error status and body.
                    with proxy._lock:
                        proxy._responses.add(upstream)
                    record["upstream_status"] = upstream.status
                    if 300 <= upstream.status < 400:
                        # Even when this proxy does not follow redirects, the
                        # model client might follow a relayed Location using its
                        # original, unpruned request. Keep it on this boundary.
                        record["error"] = "upstream_redirect_rejected"
                        self._error(502, record["error"], "Upstream redirect rejected; request was not replayed")
                        return
                    def watch_client():
                        # A disconnected client may leave an otherwise idle SSE
                        # read blocked. Notice EOF without consuming request data,
                        # then interrupt this request's upstream socket.
                        while not stream_finished.is_set() and not proxy._closing:
                            try:
                                ready, _, _ = select.select([self.connection], [], [], .05)
                                if ready and not self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT):
                                    disconnected.set()
                                    sock = _response_socket(upstream)
                                    if sock is not None:
                                        sock.shutdown(socket.SHUT_RDWR)
                                    return
                            except (OSError, ValueError):
                                return
                    watcher = threading.Thread(target=watch_client, daemon=True)
                    watcher.start()
                    self.send_response(upstream.status)
                    omitted = _HOP_HEADERS | {x.strip().lower() for x in upstream.headers.get("Connection", "").split(",")}
                    for key, value in upstream.headers.items():
                        if key.lower() not in omitted:
                            self.send_header(key, value)
                    # urllib removes HTTP chunk framing. Close-delimited output
                    # preserves SSE payload bytes while permitting immediate flush.
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    read = getattr(upstream, "read1", upstream.read)
                    terminal = (_SSETerminal() if "text/event-stream" in
                                upstream.headers.get("Content-Type", "").lower() else None)
                    while not proxy._closing:
                        chunk = read(65536)
                        if not chunk:
                            break
                        ends_response = terminal.feed(chunk) if terminal is not None else False
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        if ends_response:
                            terminal_forwarded.set()
                except ImageContextError as exc:
                    record["error"] = exc.code
                    self._error(400, exc.code, str(exc))
                except (BrokenPipeError, ConnectionResetError):
                    if not terminal_forwarded.is_set():
                        record["error"] = "client_disconnected"
                    self.close_connection = True
                except (OSError, URLError, TimeoutError):
                    if not terminal_forwarded.is_set():
                        record["error"] = "client_disconnected" if disconnected.is_set() else "upstream_transport_error"
                    if upstream is None:
                        self._error(502, record["error"], "Upstream transport failed; request was not replayed")
                    self.close_connection = True
                finally:
                    stream_finished.set()
                    if watcher is not None:
                        watcher.join(timeout=.1)
                    if upstream is not None:
                        with proxy._lock:
                            proxy._responses.discard(upstream)
                        upstream.close()
                    if terminal_forwarded.is_set():
                        record["stream_terminal_forwarded"] = True
                    elif disconnected.is_set():
                        record["error"] = "client_disconnected"
                    proxy._record(record)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._server.block_on_close = False
        self.base_url = f"http://127.0.0.1:{self._server.server_port}{self._prefix}"
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        kwargs={"poll_interval": 0.05}, daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._closing = True
        if self._server is None:
            return
        self._server.shutdown()
        with self._lock:
            sockets = list(self._sockets)
            responses = list(self._responses)
        for response in responses:
            # Shut down the underlying socket before close(), so another thread
            # blocked in read1() does not hold a buffered-I/O lock indefinitely.
            sock = _response_socket(response)
            if sock is not None:
                sockets.append(sock)
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1)

    def __exit__(self, exc_type, exc, traceback):
        self.close()
