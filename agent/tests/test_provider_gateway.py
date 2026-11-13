"""Gateway contracts and real SDK/HTTP integration through saved settings."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


from src.providers.registry import provider_catalog as catalog
from src.providers.configuration import resolve_config
from src.providers.capabilities import get_provider_capabilities


def test_catalog_projects_the_same_connection_to_every_surface():
    from cli.onboard import PROVIDERS
    from cli._legacy import _PROVIDER_CHOICES
    from src.api.settings_routes import LLM_PROVIDERS

    rows = {r["name"]: r for r in catalog()}
    assert {p.name for p in LLM_PROVIDERS} == set(rows)
    for row in rows.values():
        caps = get_provider_capabilities(row["name"])
        assert (caps.api_key_env, caps.base_url_env) == (row.get("api_key_env"), row["base_url_env"])
    for p in PROVIDERS:
        row = rows[p.key]
        assert (p.default_model, p.base_url) == (row["default_model"], row["default_base_url"])
        assert p.suggested_models[0] == row["default_model"]
    for p in _PROVIDER_CHOICES:
        row = rows[p["provider"]]
        assert (p["model"], p["base_url"]) == (row["default_model"], row["default_base_url"])


def test_model_inference_does_not_change_explicit_adapter_family(monkeypatch):
    route = resolve_config(environment={"LANGCHAIN_PROVIDER": "openrouter", "LANGCHAIN_MODEL_NAME": "anthropic/claude-test"})
    assert route.backend == "openai_compatible"
    assert get_provider_capabilities(route.provider, route.model).name == "openrouter"
    route = resolve_config(environment={"LANGCHAIN_PROVIDER": "openai", "LANGCHAIN_MODEL_NAME": "deepseek-test"})
    assert route.backend == "openai_compatible"
    assert get_provider_capabilities(route.provider, route.model).name == "deepseek"
    from src.providers import diagnostics

    monkeypatch.setattr(diagnostics, "resolve_config", lambda: route)
    monkeypatch.setattr(diagnostics, "_package_version", lambda _: "installed")
    assert diagnostics.provider_diagnostics()["adapter"]["type"] == "openai-compatible"
    assert resolve_config(environment={"LANGCHAIN_PROVIDER": "openai_codex", "LANGCHAIN_MODEL_NAME": "test"}).backend == "openai_codex"
    assert resolve_config(environment={"LANGCHAIN_PROVIDER": "github-copilot", "LANGCHAIN_MODEL_NAME": "test"}).backend == "copilot"
    assert resolve_config(environment={"LANGCHAIN_PROVIDER": "unknown-legacy-relay", "LANGCHAIN_MODEL_NAME": "test"}).backend == "openai_compatible"


def test_catalog_callers_cannot_mutate_gateway_configuration():
    rows = catalog()
    rows[0]["default_model"] = "changed"
    rows[0].setdefault("model_options", []).append("changed")
    assert catalog()[0]["default_model"] != "changed"
    assert "changed" not in catalog()[0].get("model_options", [])


def test_settings_to_real_sdk_stream_and_tool_followup(tmp_path, monkeypatch):
    """Exercise settings write -> gateway -> SDK -> HTTP SSE -> tool follow-up.

    The localhost server implements a deterministic provider wire contract; no
    adapter, HTTP transport, route, or settings implementation is mocked.
    """
    from fastapi.testclient import TestClient
    import api_server
    from src.providers.chat import ChatLLM

    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, body))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            if len(received) == 1:
                deltas = [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_quote",
                                "type": "function",
                                "function": {"name": "quote", "arguments": '{"symbol":'},
                            }
                        ],
                    },
                    {"tool_calls": [{"index": 0, "function": {"arguments": '"AAPL"}'}}]},
                ]
                reason = "tool_calls"
            else:
                deltas = [{"role": "assistant", "content": "Gateway verified."}]
                reason = "stop"
            for delta in deltas + [{}]:
                chunk = {
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "gateway-test",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": reason if not delta else None}],
                }
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            usage = {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gateway-test",
                "choices": [],
                "usage": {"prompt_tokens": 17, "completion_tokens": 4, "total_tokens": 21},
            }
            self.wfile.write(("data: " + json.dumps(usage) + "\n\ndata: [DONE]\n\n").encode())
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env_path = tmp_path / ".env"
    example = tmp_path / ".env.example"
    example.write_text("LANGCHAIN_PROVIDER=ollama\nLANGCHAIN_MODEL_NAME=gateway-test\n")
    monkeypatch.setattr(api_server, "ENV_PATH", env_path)
    monkeypatch.setattr(api_server, "ENV_EXAMPLE_PATH", example)
    monkeypatch.setattr(api_server, "LEGACY_ENV_PATH", tmp_path / "legacy.env", raising=False)
    monkeypatch.delenv("API_AUTH_KEY", raising=False)
    monkeypatch.setenv("VIBE_TRADING_DISABLE_HTTP_PROXY", "1")
    monkeypatch.setenv("LANGCHAIN_USE_RESPONSES_API", "false")
    llm = None
    try:
        with TestClient(api_server.app, client=("127.0.0.1", 50000)) as client:
            response = client.put(
                "/settings/llm",
                json={
                    "provider": "ollama",
                    "model_name": "gateway-test",
                    "base_url": f"http://127.0.0.1:{server.server_port}",
                    "temperature": 0,
                    "timeout_seconds": 10,
                    "max_retries": 0,
                },
            )
            assert response.status_code == 200, response.text
            assert (
                next(p for p in response.json()["providers"] if p["name"] == "ollama")["backend"] == "openai_compatible"
            )
            llm = ChatLLM()
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "quote",
                        "description": "Read a quote",
                        "parameters": {
                            "type": "object",
                            "properties": {"symbol": {"type": "string"}},
                            "required": ["symbol"],
                        },
                    },
                }
            ]
            messages = [{"role": "user", "content": "Quote AAPL"}]
            first = llm.stream_chat(messages, tools=tools)
            assert first.tool_calls[0].arguments == {"symbol": "AAPL"}
            assert first.finish_reason == "tool_calls"
            followup = messages + [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_quote",
                            "type": "function",
                            "function": {"name": "quote", "arguments": '{"symbol":"AAPL"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_quote", "content": '{"price":100}'},
            ]
            chunks = []
            second = llm.stream_chat(followup, tools=tools, on_text_chunk=chunks.append)
            assert second.content == "".join(chunks) == "Gateway verified."
            assert second.usage_metadata["input_tokens"] == 17
            assert second.response_model == "gateway-test"
            assert len(received) == 2
            assert all(path == "/v1/chat/completions" for path, _ in received)
            assert received[1][1]["messages"][-1]["tool_call_id"] == "call_quote"
    finally:
        if llm:
            llm.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_catalog_and_gateway_metadata_do_not_import_sdks():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from src.providers.registry import provider_catalog as catalog; "
            'assert catalog(); assert "langchain_openai" not in sys.modules; '
            'assert "src.providers.llm" not in sys.modules',
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
