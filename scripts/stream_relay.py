"""Relay that lets lm-eval use a chat API route that has to be streamed.

Neuralwatt's standard tier sits behind Cloudflare, which cuts a response that takes over 100 s
(HTTP 524), and lm-eval only sends non-streaming requests. For each request from lm-eval the relay
sets stream=true, forwards the request otherwise unchanged, joins the streamed answer text, and
returns it in the one field lm-eval reads (choices[].message.content). If the stream ends before the
answer finished, lm-eval gets HTTP 502 and re-sends the request, so a cut-off answer is never kept.

run_eval.py starts it with start(); `python scripts/stream_relay.py` runs the self-check.
"""

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

IDLE_TIMEOUT_S = 180  # no data for this long means a dead connection, e.g. after a network change


def join_stream(lines):
    """Answer text and finish_reason of a streamed chat completion (server-sent event lines)."""
    text, finish = "", None
    for line in lines:
        line = line.decode() if isinstance(line, bytes) else line
        if line.startswith("data: {"):
            for choice in json.loads(line[6:]).get("choices") or []:
                text += (choice.get("delta") or {}).get("content") or ""
                finish = choice.get("finish_reason") or finish
    if finish is None:
        raise RuntimeError("stream ended before the answer finished")
    return text, finish


def start(upstream):
    """Run the relay on a free local port for the life of this process; return its URL."""

    class Relay(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body["stream"] = True
            request = urllib.request.Request(
                upstream,
                json.dumps(body).encode(),
                {"Authorization": self.headers["Authorization"], "Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=IDLE_TIMEOUT_S) as response:
                    text, finish = join_stream(response)
                # content is null when there is no answer text, as in a non-streaming response
                message = {"role": "assistant", "content": text or None}
                code, reply = 200, {"choices": [{"index": 0, "message": message, "finish_reason": finish}]}
            except urllib.error.HTTPError as e:  # e.g. 429: passed on, and lm-eval retries
                code, reply = e.code, {"error": e.read().decode(errors="replace")}
            except Exception as e:  # dropped, stalled or cut-off stream: lm-eval retries
                code, reply = 502, {"error": repr(e)}
            data = json.dumps(reply).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Relay)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}/v1/chat/completions"


if __name__ == "__main__":
    chunk = lambda delta, finish=None: "data: " + json.dumps({"choices": [{"delta": delta, "finish_reason": finish}]})
    lines = [chunk({"reasoning_content": "hmm"}), chunk({"content": "the answer "}), chunk({"content": "is (B)"}, "stop"), "data: [DONE]"]
    assert join_stream(lines) == ("the answer is (B)", "stop")
    try:
        join_stream(lines[:2])
        raise AssertionError("a cut-off stream must not pass as a finished answer")
    except RuntimeError:
        pass
    print("stream_relay self-check passed")
