from __future__ import annotations
import copy
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.parse import urlsplit
from harnesslab.image_context import ImageContextError, ImageContextProxy, prune_response_images


def image(index):
    return {"type": "input_image", "image_url": f"data:image/png;base64,IMAGE_SECRET_{index}", "detail": "high"}


def history(total=70):
    items = [{"role": "developer", "content": [{"type": "input_text", "text": "original instructions"}]},
             {"role": "user", "content": [{"type": "input_text", "text": "task"}, *[image(i) for i in range(3)]]}]
    for i in range(3, total):
        items.extend([{"type": "function_call", "call_id": f"call_{i}", "name": "python",
                       "arguments": json.dumps({"code": "print('original tool code')", "example": image(999)})},
                      {"type": "function_call_output", "call_id": f"call_{i}", "output": [
                          {"type": "input_text", "text": f"original physical feedback {i}"}, image(i)]}])
    items.append({"type": "reasoning", "id": "rs_original", "encrypted_content": "OPAQUE_REASONING_UNCHANGED"})
    return {"model": "fake-model", "input": items, "store": True,
            "tools": [{"type": "function", "name": "python", "parameters": {"example": image(998)}}]}


class PruningTests(unittest.TestCase):
    def test_70_images_pruned_to_48_without_mutating_source_or_other_content(self):
        payload = history()
        original = copy.deepcopy(payload)
        outgoing, audit = prune_response_images(payload)
        self.assertEqual(payload, original)
        self.assertEqual((audit["images_before"], audit["images_after"], audit["images_removed"]), (70, 48, 22))
        self.assertEqual(outgoing["input"][1], original["input"][1])
        self.assertEqual(outgoing["input"][-2], original["input"][-2])
        self.assertEqual(outgoing["input"][-1], original["input"][-1])
        self.assertEqual(outgoing["tools"], original["tools"])
        self.assertFalse(outgoing["store"])
        for new, old in zip(outgoing["input"], original["input"]):
            if old.get("type") == "function_call":
                self.assertEqual(new, old)
            if old.get("type") == "function_call_output":
                self.assertEqual(new["call_id"], old["call_id"])
                self.assertEqual(new["output"][0], old["output"][0])
        self.assertIn("omitted", outgoing["input"][3]["output"][1]["text"])
        self.assertNotIn("IMAGE_SECRET", json.dumps(audit))

    def test_initial_turn_split_across_messages_is_all_protected(self):
        payload = {"input": [{"role": "user", "content": [image(i)]} for i in range(3)] +
                   [{"type": "function_call_output", "call_id": "a", "output": [image(3)]},
                    {"role": "user", "content": [image(4), image(5)]}]}
        _, audit = prune_response_images(payload, 5)
        self.assertEqual(audit["initial_images_preserved"], 3)
        self.assertEqual(audit["latest_observation_images_preserved"], 2)
        self.assertEqual(audit["images_removed"], 1)
        with self.assertRaisesRegex(ImageContextError, "none were removed"):
            prune_response_images(payload, 4)

    def test_real_codex_additional_tools_preamble_preserves_initial_three_and_latest(self):
        # Exact item types/roles/content-part layout captured from Codex 0.159.2
        # before its first model output. Tool definitions are opaque metadata.
        shapes = [
            ("additional_tools", "developer", None),
            ("message", "developer", ["input_text"]),
            ("message", "developer", ["input_text", "input_text", "input_text"]),
            ("message", "developer", ["input_text"]),
            ("message", "developer", ["input_text"]),
            ("message", "user", ["input_text"]),
            ("message", "user", ["input_text", "input_image", "input_text", "input_text",
                                  "input_image", "input_text", "input_text", "input_image",
                                  "input_text", "input_text", "input_text"]),
        ]
        initial, image_index = [], 0
        for kind, role, parts in shapes:
            item = {"type": kind, "role": role}
            if parts is None:
                item["tools"] = [{"type": "function", "parameters": {"opaque_example": image(999)}}]
            else:
                item["content"] = []
                for part in parts:
                    if part == "input_image":
                        item["content"].append(image(image_index))
                        image_index += 1
                    else:
                        item["content"].append({"type": "input_text", "text": "original initial text"})
            initial.append(item)
        payload = {"input": initial + history(70)["input"][2:]}
        outgoing, audit = prune_response_images(payload, 4, reference_image_count=3)
        self.assertEqual(audit["images_before"], 70)
        self.assertEqual(audit["images_after"], 4)
        self.assertEqual(audit["initial_images_preserved"], 3)
        self.assertEqual(audit["latest_observation_images_preserved"], 1)
        self.assertEqual(outgoing["input"][:7], initial)
        self.assertEqual(outgoing["input"][-2], payload["input"][-2])

    def test_latest_multi_image_observation_is_never_partially_removed(self):
        payload = history(20)
        payload["input"].insert(-1, {"role": "user", "content": [image(20), image(21), image(22)]})
        outgoing, audit = prune_response_images(payload, 6)
        self.assertEqual(outgoing["input"][-2], payload["input"][-2])
        self.assertEqual(audit["latest_observation_images_preserved"], 3)
        self.assertEqual(audit["images_after"], 6)

    def test_each_payload_is_counted_independently(self):
        _, first = prune_response_images(history(70))
        _, second = prune_response_images(history(8))
        self.assertEqual(first["images_removed"], 22)
        self.assertEqual(second["images_before"], 8)
        self.assertEqual(second["images_removed"], 0)

    def test_hidden_history_and_computer_screenshots_fail_closed(self):
        for field, value in (("previous_response_id", "resp_secret"),
                             ("conversation", {"id": "conv_secret"}), ("prompt", {"id": "prompt_secret"})):
            with self.subTest(field=field), self.assertRaises(ImageContextError):
                prune_response_images({"input": [], field: value})
        for item in ({"type": "computer_call_output", "call_id": "computer", "output": {"type": "computer_screenshot", "image_url": "secret"}},
                     {"type": "item_reference", "id": "msg_unknown"},
                     {"role": "user", "content": [{"type": "computer_screenshot", "image_url": "secret"}]}):
            with self.assertRaises(ImageContextError):
                prune_response_images({"input": [item]})

    def test_strings_and_embedded_argument_json_are_not_interpreted_as_images(self):
        payload = {"input": [{"type": "function_call_output", "call_id": "tool",
                              "output": json.dumps([image(10)])},
                             {"role": "user", "content": "text containing input_image and image_url"}]}
        outgoing, audit = prune_response_images(payload)
        self.assertEqual(audit["images_before"], 0)
        self.assertEqual(outgoing["input"], payload["input"])

    def test_cap_validation_and_oversized_initial_references(self):
        for invalid in (0, 51, -1, True, 4.5):
            with self.assertRaises(ValueError):
                prune_response_images(history(4), invalid)
        with self.assertRaises(ImageContextError):
            prune_response_images({"input": [{"role": "user", "content": [image(i) for i in range(51)]}]}, 50)

    def test_explicit_reference_count_detects_missing_refs_and_no_refs_stays_zero(self):
        _, audit = prune_response_images(history(70), reference_image_count=3)
        self.assertEqual(audit["initial_images_preserved"], 3)
        with self.assertRaisesRegex(ImageContextError, "initial reference image count"):
            prune_response_images(history(70), reference_image_count=4)
        payload = {"input": [{"role": "user", "content": [{"type": "input_text", "text": "no initial picture"}]},
                             {"type": "function_call", "name": "python", "call_id": "a", "arguments": "{}"},
                             {"role": "user", "content": [image(10)]}]}
        _, audit = prune_response_images(payload, reference_image_count=0)
        self.assertEqual(audit["initial_images_preserved"], 0)


class ProxyTests(unittest.TestCase):
    def setUp(self):
        self.captured = []
        self.release_sse = threading.Event()
        self.sse_started = threading.Event()
        outer = self
        class Upstream(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *args): pass
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.captured.append({"path": self.path, "headers": dict(self.headers), "payload": payload})
                mode = payload.get("test_mode", "json")
                if mode in ("sse", "sse-completed-idle"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    try:
                        self.wfile.write(b"data: first\n\n" if mode == "sse" else
                            b'event: response.completed\ndata: {"type":"response.completed"}\n\n')
                        self.wfile.flush()
                        outer.sse_started.set()
                        outer.release_sse.wait(3)
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    self.close_connection = True
                else:
                    status = 429 if mode == "error" else 307 if mode == "redirect" else 200
                    body = b'{"error":"original rate limit body"}' if mode == "error" else b'{"id":"original-response"}'
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("X-Upstream-Marker", "untouched")
                    if mode == "redirect":
                        self.send_header("Location", outer.upstream_url + "/do-not-follow")
                    self.end_headers()
                    self.wfile.write(body)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()
        self.upstream_url = f"http://127.0.0.1:{self.server.server_port}/v1"
        temporary_root = Path(__file__).resolve().parents[2] / "experiments/cpu-tests/tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.log = Path(self.temporary.name) / "audit.jsonl"

    def tearDown(self):
        self.release_sse.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)
        self.temporary.cleanup()

    def connection(self, proxy):
        address = urlsplit(proxy.base_url)
        return http.client.HTTPConnection(address.hostname, address.port, timeout=3)

    def post(self, proxy, payload, *, token=None, suffix="/responses", headers=None):
        connection = self.connection(proxy)
        request_headers = {"Authorization": "Bearer " + (proxy.client_api_key if token is None else token),
                           "Content-Type": "application/json", **(headers or {})}
        connection.request("POST", urlsplit(proxy.base_url).path + suffix,
                           body=json.dumps(payload).encode(), headers=request_headers)
        response = connection.getresponse()
        body = response.read()
        status, response_headers = response.status, dict(response.headers)
        connection.close()
        return status, body, response_headers

    def wait_audit(self, proxy, requests):
        deadline = time.monotonic() + 2
        while proxy.summary()["requests"] < requests and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertEqual(proxy.summary()["requests"], requests)

    def test_json_forwarding_prunes_authenticates_and_logs_no_content(self):
        payload = history()
        with ImageContextProxy(self.upstream_url, "UPSTREAM_SECRET", log_path=self.log) as proxy:
            status, body, headers = self.post(proxy, payload, headers={"Host": "ignored.example", "X-Request-Id": "req"})
            self.assertEqual(status, 200)
            self.assertEqual(body, b'{"id":"original-response"}')
            self.assertEqual(headers["X-Upstream-Marker"], "untouched")
            self.assertEqual(self.captured[0]["path"], "/v1/responses")
            self.assertEqual(self.captured[0]["headers"]["Authorization"], "Bearer UPSTREAM_SECRET")
            self.assertNotEqual(self.captured[0]["headers"]["Host"], "ignored.example")
            self.wait_audit(proxy, 1)
            self.assertEqual(proxy.summary()["images_removed"], 22)
            text = self.log.read_text()
            for secret in ("UPSTREAM_SECRET", proxy.client_api_key, "IMAGE_SECRET", "OPAQUE_REASONING", "original physical feedback"):
                self.assertNotIn(secret, text)
        self.assertTrue(payload["store"])

    def test_sse_stream_flushes_first_event_before_upstream_finishes(self):
        with ImageContextProxy(self.upstream_url, timeout_s=3) as proxy:
            connection = self.connection(proxy)
            connection.request("POST", "/v1/responses", body=json.dumps({"input": [], "test_mode": "sse"}),
                               headers={"Authorization": "Bearer " + proxy.client_api_key})
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertTrue(self.sse_started.wait(1))
            self.assertEqual(response.readline(), b"data: first\n")
            self.release_sse.set()
            self.assertEqual(response.read(), b"\ndata: [DONE]\n\n")
            connection.close()

    def test_upstream_error_is_unchanged_and_redirect_is_rejected_without_location_or_replay(self):
        with ImageContextProxy(self.upstream_url) as proxy:
            status, body, _ = self.post(proxy, {"input": [], "test_mode": "error"})
            self.assertEqual(status, 429)
            self.assertEqual(body, b'{"error":"original rate limit body"}')
            status, body, headers = self.post(proxy, {"input": [], "test_mode": "redirect"})
            self.assertEqual(status, 502)
            self.assertNotIn("Location", headers)
            self.assertEqual(json.loads(body)["error"]["code"], "upstream_redirect_rejected")
            self.assertNotIn(self.upstream_url.encode(), body)
            self.assertNotIn(b"do-not-follow", body)
            self.assertEqual(len(self.captured), 2)
            self.assertTrue(all(item["path"] == "/v1/responses" for item in self.captured))
            self.wait_audit(proxy, 2)
            self.assertIn("upstream_redirect_rejected", proxy.summary()["errors"])

    def test_local_auth_hidden_history_unknown_path_and_encoding_never_reach_upstream(self):
        with ImageContextProxy(self.upstream_url) as proxy:
            self.assertEqual(self.post(proxy, {"input": []}, token="wrong")[0], 401)
            self.assertEqual(self.post(proxy, {"input": [], "previous_response_id": "hidden"})[0], 400)
            self.assertEqual(self.post(proxy, {"input": []}, suffix="/models")[0], 404)
            self.assertEqual(self.post(proxy, {"input": []}, headers={"Content-Encoding": "zstd"})[0], 415)
            self.assertEqual(self.captured, [])

    def test_compact_prunes_input_without_injecting_unsupported_store_field(self):
        payload = history()
        del payload["store"]
        with ImageContextProxy(self.upstream_url) as proxy:
            self.assertEqual(self.post(proxy, payload, suffix="/responses/compact")[0], 200)
            self.assertEqual(self.captured[0]["path"], "/v1/responses/compact")
            self.assertNotIn("store", self.captured[0]["payload"])

    def test_two_requests_have_independent_counts_and_safe_aggregate(self):
        with ImageContextProxy(self.upstream_url) as proxy:
            self.post(proxy, history(70))
            self.post(proxy, history(5))
            self.wait_audit(proxy, 2)
            self.assertEqual([x["images_before"] for x in proxy.summary()["audit"]], [70, 5])
            self.assertEqual(proxy.summary()["images_removed"], 22)
            self.assertEqual(proxy.summary()["upstream_requests"], 2)

    def test_close_interrupts_active_sse_without_waiting_for_idle_timeout(self):
        proxy = ImageContextProxy(self.upstream_url, timeout_s=120).__enter__()
        connection = self.connection(proxy)
        connection.request("POST", "/v1/responses", body=json.dumps({"input": [], "test_mode": "sse"}),
                           headers={"Authorization": "Bearer " + proxy.client_api_key})
        response = connection.getresponse()
        self.assertEqual(response.readline(), b"data: first\n")
        started = time.monotonic()
        proxy.close()
        self.assertLess(time.monotonic() - started, 1)
        connection.close()

    def test_client_disconnect_interrupts_idle_upstream_stream(self):
        with ImageContextProxy(self.upstream_url, timeout_s=120) as proxy:
            connection = self.connection(proxy)
            connection.request("POST", "/v1/responses", body=json.dumps({"input": [], "test_mode": "sse"}),
                               headers={"Authorization": "Bearer " + proxy.client_api_key})
            response = connection.getresponse()
            self.assertEqual(response.readline(), b"data: first\n")
            response.close()
            connection.close()
            self.wait_audit(proxy, 1)
            self.assertIn("client_disconnected", proxy.summary()["errors"])

    def test_client_close_after_forwarded_terminal_is_normal_even_if_upstream_remains_idle(self):
        with ImageContextProxy(self.upstream_url, timeout_s=120) as proxy:
            connection = self.connection(proxy)
            connection.request("POST", "/v1/responses", body=json.dumps({"input": [], "test_mode": "sse-completed-idle"}),
                               headers={"Authorization": "Bearer " + proxy.client_api_key})
            response = connection.getresponse()
            self.assertEqual(response.readline(), b"event: response.completed\n")
            self.assertEqual(response.readline(), b'data: {"type":"response.completed"}\n')
            self.assertEqual(response.readline(), b"\n")
            response.close()
            connection.close()
            self.wait_audit(proxy, 1)
            self.assertEqual(proxy.summary()["errors"], [])
            self.assertTrue(proxy.summary()["audit"][0]["stream_terminal_forwarded"])

    def test_fixed_upstream_base_rejects_credentials_and_query(self):
        for url in ("https://user:secret@example.test/v1", "https://example.test/v1?key=secret", "file:///tmp/model"):
            with self.assertRaises(ValueError):
                ImageContextProxy(url)


if __name__ == "__main__":
    unittest.main()
