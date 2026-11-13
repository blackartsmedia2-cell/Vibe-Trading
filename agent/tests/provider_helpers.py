"""Shared provider test setup: real ChatLLM initialization and local HTTP wire."""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

from src.providers.chat import ChatLLM
from src.providers.configuration import resolve_config


def chat_with_adapter(adapter, model="test"):
    """Initialize ChatLLM normally while substituting only its SDK adapter."""
    snapshot = resolve_config(model_name=model, environment=dict(os.environ))
    with patch("src.providers.chat.build_llm", return_value=adapter):
        return ChatLLM(snapshot=snapshot)


@pytest.fixture
def provider_server():
    """Serve JSON completions and fragmented SSE tool calls, recording requests."""
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
            frame = {"id": "test", "created": 1, "model": body["model"]}
            message = {"role": "assistant", "content": body["model"]}
            if body.get("stream"):
                frame["object"] = "chat.completion.chunk"
                deltas, reason = [message], "stop"
                if body.get("tools") and body["messages"][-1]["role"] != "tool":
                    deltas = [
                        {"tool_calls": [{"index": 0, "id": "call_quote", "type": "function",
                                         "function": {"name": "quote", "arguments": '{"symbol":'}}]},
                        {"tool_calls": [{"index": 0, "function": {"arguments": '"AAPL"}'}}]},
                    ]
                    reason = "tool_calls"
                events = [dict(frame, choices=[{"index": 0, "delta": d, "finish_reason": reason if not d else None}])
                          for d in deltas + [{}]]
                events.append(dict(frame, choices=[], usage={"prompt_tokens": 17, "completion_tokens": 4, "total_tokens": 21}))
                payload = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
                content_type = "text/event-stream"
            else:
                payload = json.dumps(dict(frame, object="chat.completion", choices=[
                    {"index": 0, "finish_reason": "stop", "message": message}]))
                content_type = "application/json"
            encoded = payload.encode()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", received
        finally:
            server.shutdown()
            thread.join(timeout=5)
