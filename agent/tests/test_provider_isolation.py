"""Cross-client isolation through real SDK requests, plus architecture boundaries."""

from __future__ import annotations

import ast
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.providers import adapters
from src.providers.chat import ChatLLM, ProviderStreamError
from src.providers.configuration import resolve_config


def test_old_and_new_clients_keep_their_own_connection_over_real_http(monkeypatch):
    """Defaults can change before construction and during concurrent requests."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, dict(self.headers), body))
            payload = json.dumps(
                {
                    "id": "test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": body["model"]},
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    clients = []
    try:
        snapshots = [
            resolve_config(
                environment={
                    "LANGCHAIN_PROVIDER": "openai",
                    "LANGCHAIN_MODEL_NAME": name,
                    "OPENAI_API_KEY": f"key-{name}",
                    "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/{name}",
                    "LANGCHAIN_TEMPERATURE": str(temperature),
                    "LANGCHAIN_USE_RESPONSES_API": "false",
                    "OPENAI_CUSTOM_HEADERS": f"X-Owner: {name}",
                    "OPENAI_ORG_ID": f"org-{name}",
                    "OPENAI_PROJECT_ID": f"project-{name}",
                    "TIMEOUT_SECONDS": "5",
                    "MAX_RETRIES": "0",
                    "VIBE_TRADING_DISABLE_HTTP_PROXY": "1",
                    "OPENAI_PROXY": "http://127.0.0.1:1",
                }
            )
            for name, temperature in [("before", 0.2), ("after", 0.7)]
        ]
        for key in ("OPENAI_API_KEY", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID"):
            monkeypatch.setenv(key, "unrelated-new-default")
        monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "X-Owner: leaked\nX-Leaked: secret")
        before_build = dict(os.environ)
        clients = [ChatLLM(snapshot=snapshot) for snapshot in snapshots]
        assert dict(os.environ) == before_build
        with ThreadPoolExecutor(2) as pool:
            replies = list(pool.map(lambda c: c.chat([{"role": "user", "content": "hello"}]), clients))
        assert {r.content for r in replies} == {"before", "after"}
        for path, raw_headers, body in seen:
            headers = {key.lower(): value for key, value in raw_headers.items()}
            name = body["model"]
            assert path == f"/{name}/chat/completions"
            assert headers["authorization"] == f"Bearer key-{name}"
            assert headers["x-owner"] == name and "x-leaked" not in headers
            assert headers["openai-organization"] == f"org-{name}"
            assert headers["openai-project"] == f"project-{name}"
            assert body["temperature"] == (0.2 if name == "before" else 0.7)
        assert len(seen) == 2
    finally:
        for client in clients:
            client.close()
        server.shutdown()
        server.server_close()


def test_snapshot_defaults_are_explicit_immutable_and_secret_free(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TEMPERATURE", "1.5")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key")
    snapshot = resolve_config(environment={"LANGCHAIN_MODEL_NAME": "test", "OPENAI_API_KEY": "private-key"})
    assert snapshot.options.langchain_temperature == 0
    assert "private-key" not in repr(snapshot)
    assert "ambient-key" not in repr(snapshot)
    with pytest.raises(FrozenInstanceError):
        snapshot.model = "changed"
    with pytest.raises(TypeError):
        snapshot.environment["OPENAI_API_KEY"] = "changed"
    with pytest.raises(ValidationError):
        snapshot.options.langchain_temperature = 1


def test_stream_errors_keep_original_provider_and_redact_original_key(monkeypatch):
    snapshot = resolve_config(
        environment={
            "LANGCHAIN_PROVIDER": "openrouter",
            "LANGCHAIN_MODEL_NAME": "original-model",
            "OPENROUTER_API_KEY": "old-private-token",
        }
    )

    class Broken:
        def stream(self, *args, **kwargs):
            raise RuntimeError("old-private-token failed")
            yield

    monkeypatch.setattr("src.providers.chat.build_llm", lambda **_: Broken())
    client = ChatLLM(snapshot=snapshot)
    monkeypatch.setenv("LANGCHAIN_PROVIDER", "deepseek")
    monkeypatch.setenv("OPENROUTER_API_KEY", "new-token")
    with pytest.raises(ProviderStreamError) as caught:
        client.stream_chat([{"role": "user", "content": "hello"}])
    assert "provider=openrouter" in str(caught.value)
    assert "model=original-model" in str(caught.value)
    assert "old-private-token" not in str(caught.value)


def test_learned_rejections_do_not_cross_endpoints_or_protocols():
    first, second = [
        adapters.ChatOpenAIWithReasoning(
            model="same-name",
            api_key="test",
            temperature=0,
            base_url=f"https://{name}.invalid/v1",
        )
        for name in ("first", "second")
    ]
    adapters._ANTHROPIC_TEMPERATURE_UNSUPPORTED.clear()
    try:
        error = RuntimeError("`temperature` is deprecated for this model.")
        assert first._remember_temperature_unsupported(error)
        assert not first._remember_temperature_unsupported(error)
        assert "temperature" not in first._get_request_payload("hello")
        assert "temperature" in second._get_request_payload("hello")
        assert adapters._endpoint_model(first) != adapters._endpoint_model(second)
    finally:
        adapters._ANTHROPIC_TEMPERATURE_UNSUPPORTED.clear()


def test_lower_layers_never_import_http_routes_or_the_llm_facade():
    root = Path(__file__).resolve().parents[1]
    paths = [root / "cli/custom_providers.py", root / "src/config/dotenv.py"]
    paths += [p for p in (root / "src/providers").glob("*.py") if p.name not in {"__init__.py", "llm.py"}]
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            modules = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [n.name for n in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            assert not any(m == "api_server" or m.startswith("src.api") or m == "src.providers.llm" for m in modules), (
                path
            )


@pytest.mark.parametrize("provider,module", [("anthropic", "langchain_anthropic"), ("deepseek", "langchain_deepseek")])
def test_native_sdk_credentials_and_headers_are_scoped(monkeypatch, provider, module):
    pytest.importorskip(module)
    from src.providers.gateway import build_llm

    snapshot = resolve_config(
        environment={
            "LANGCHAIN_PROVIDER": provider,
            "LANGCHAIN_MODEL_NAME": "test-model",
            f"{provider.upper()}_API_KEY": "snapshot-key",
            f"{provider.upper()}_BASE_URL": "https://original.invalid",
            "ANTHROPIC_MAX_TOKENS": "128",
        }
    )
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Authorization: wrong\nX-Leaked: secret")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Leaked: secret")
    monkeypatch.setenv("OPENAI_ORG_ID", "wrong-org")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "wrong-token")
    adapter = build_llm(snapshot=snapshot)
    sdk = adapter._client if provider == "anthropic" else adapter.root_client
    assert sdk.api_key == "snapshot-key"
    assert str(sdk.base_url).rstrip("/") == "https://original.invalid"
    assert "X-Leaked" not in sdk._custom_headers
    if provider == "anthropic":
        assert sdk.auth_token is None
        assert "auth_token" not in adapter._get_request_payload("hello")
    else:
        assert sdk.organization is None and sdk.project is None


def test_native_deepseek_does_not_fill_a_missing_snapshot_key_from_the_environment(monkeypatch):
    pytest.importorskip("langchain_deepseek")
    from src.providers.gateway import build_llm

    snapshot = resolve_config(environment={
        "LANGCHAIN_PROVIDER": "deepseek", "LANGCHAIN_MODEL_NAME": "test",
        "DEEPSEEK_BASE_URL": "https://relay.invalid/v1",
    })
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-new-default")
    with pytest.raises(RuntimeError, match="Set DEEPSEEK_API_KEY"):
        build_llm(snapshot=snapshot)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_no_fallback_replays_an_emitted_chunk(monkeypatch, native, asynchronous):
    import asyncio

    error = RuntimeError("`temperature` is deprecated for this model." if native else "stream_options is unsupported")
    calls = []

    def stream(*args, **kwargs):
        calls.append(None)
        yield "already emitted"
        raise error

    async def astream(*args, **kwargs):
        for chunk in stream():
            yield chunk

    if native:

        class Base:
            model = "native-partial"
            _stream = stream
            _astream = astream

        client = adapters._make_temperature_safe_anthropic(Base)()
        iterator = client._astream() if asynchronous else client._stream()
    else:
        monkeypatch.setattr(adapters.ChatOpenAI, "_stream", stream)
        monkeypatch.setattr(adapters.ChatOpenAI, "_astream", astream)
        client = adapters.ChatOpenAIWithReasoning(model="partial", api_key="test")
        iterator = client._astream_with_usage_fallback() if asynchronous else client._stream_with_usage_fallback()
    seen = []
    with pytest.raises(RuntimeError, match="deprecated|unsupported"):
        if asynchronous:

            async def consume():
                async for chunk in iterator:
                    seen.append(chunk)

            asyncio.run(consume())
        else:
            for chunk in iterator:
                seen.append(chunk)
    assert seen == ["already emitted"]
    assert len(calls) == 1


def test_copilot_binding_keeps_explicit_token_and_stored_login_ignores_new_env(monkeypatch):
    from types import SimpleNamespace
    from src.providers import copilot_auth

    client = copilot_auth.CopilotSDKLLM(model="test", token="gho_original")
    monkeypatch.setenv("GH_TOKEN", "gho_new_default")
    assert copilot_auth._client_options(client.bind_tools([])._token) == {"github_token": "gho_original"}
    monkeypatch.setattr(copilot_auth.shutil, "which", lambda _: "gh")
    seen = {}
    def run(*args, **kwargs):
        seen.update(kwargs["env"])
        return SimpleNamespace(returncode=0, stdout="gho_stored")
    monkeypatch.setattr(copilot_auth.subprocess, "run", run)
    assert copilot_auth._client_options("") == {"github_token": "gho_stored"}
    assert "GH_TOKEN" not in seen and "GITHUB_TOKEN" not in seen
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_new_copilot_default")
    monkeypatch.setattr(copilot_auth, "gh_cli_token", lambda **_: "")
    monkeypatch.setenv("COPILOT_SDK_AUTH_TOKEN", "gho_other_sdk_token")
    options = copilot_auth._client_options("")
    assert not {*copilot_auth.TOKEN_ENV_NAMES, "COPILOT_SDK_AUTH_TOKEN"}.intersection(options["env"])
