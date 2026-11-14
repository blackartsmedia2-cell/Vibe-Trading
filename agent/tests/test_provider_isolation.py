"""Credential isolation, frozen settings and streaming retry boundaries."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from src.providers import llm as llm_module
from src.providers.chat import ChatLLM, ProviderStreamError
from src.providers.settings import resolve_config
from tests.provider_helpers import provider_server as provider_server


@pytest.mark.parametrize("explicit_proxy", [False, True])
def test_old_and_new_clients_keep_their_own_connection_over_real_http(monkeypatch, provider_server, explicit_proxy):
    """Defaults can change before construction and during concurrent requests."""
    base_url, seen = provider_server
    with ExitStack() as stack:
        snapshots = [
            resolve_config(
                environment={
                    "LANGCHAIN_PROVIDER": "openai",
                    "LANGCHAIN_MODEL_NAME": name,
                    "OPENAI_API_KEY": f"key-{name}",
                    "OPENAI_BASE_URL": f"http://{name}.invalid/v1" if explicit_proxy else f"{base_url}/{name}",
                    "LANGCHAIN_TEMPERATURE": str(temperature),
                    "LANGCHAIN_USE_RESPONSES_API": "false",
                    "OPENAI_CUSTOM_HEADERS": f"X-Owner: {name}",
                    "OPENAI_ORG_ID": f"org-{name}",
                    "OPENAI_PROJECT_ID": f"project-{name}",
                    "TIMEOUT_SECONDS": "5",
                    "MAX_RETRIES": "0",
                    "VIBE_TRADING_DISABLE_HTTP_PROXY": "0" if explicit_proxy else "1",
                    "OPENAI_PROXY": base_url if explicit_proxy else "http://127.0.0.1:1",
                }
            )
            for name, temperature in [("before", 0.2), ("after", 0.7)]
        ]
        for key in ("OPENAI_API_KEY", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID"):
            monkeypatch.setenv(key, "unrelated-new-default")
        monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "X-Owner: leaked\nX-Leaked: secret")
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            monkeypatch.setenv(name, "http://127.0.0.1:1")
        before_build = dict(os.environ)
        clients = [stack.enter_context(closing(ChatLLM(snapshot=snapshot))) for snapshot in snapshots]
        assert dict(os.environ) == before_build
        with ThreadPoolExecutor(2) as pool:
            replies = list(pool.map(lambda c: c.chat([{"role": "user", "content": "hello"}]), clients))
        assert {r.content for r in replies} == {"before", "after"}
        for path, headers, body in seen:
            name = body["model"]
            assert path == (f"http://{name}.invalid/v1/chat/completions" if explicit_proxy else f"/{name}/chat/completions")
            assert headers.get_all("Authorization") == [f"Bearer key-{name}"]
            assert headers.get_all("X-Owner") == [name] and "X-Leaked" not in headers
            assert headers.get_all("OpenAI-Organization") == [f"org-{name}"]
            assert headers.get_all("OpenAI-Project") == [f"project-{name}"]
            assert body["temperature"] == (0.2 if name == "before" else 0.7)
        assert len(seen) == 2


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


@pytest.mark.parametrize("empty_stream", [False, True])
def test_stream_errors_keep_original_provider_and_redact_original_key(monkeypatch, empty_stream):
    snapshot = resolve_config(
        environment={
            "LANGCHAIN_PROVIDER": "openrouter",
            "LANGCHAIN_MODEL_NAME": "original-model",
            "OPENROUTER_API_KEY": "old-private-token",
        }
    )

    class Broken:
        def stream(self, *args, **kwargs):
            if empty_stream:
                raise ValueError("No generation chunks were returned")
            raise RuntimeError("old-private-token failed")
            yield

        def invoke(self, *args, **kwargs):
            raise RuntimeError("old-private-token failed")

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
        llm_module.ChatOpenAIWithReasoning(
            model="same-name",
            api_key="test",
            temperature=0,
            base_url=f"https://{name}.invalid/v1",
        )
        for name in ("first", "second")
    ]
    llm_module._ANTHROPIC_TEMPERATURE_UNSUPPORTED.clear()
    try:
        error = RuntimeError("`temperature` is deprecated for this model.")
        assert first._remember_temperature_unsupported(error)
        assert not first._remember_temperature_unsupported(error)
        assert "temperature" not in first._get_request_payload("hello")
        assert "temperature" in second._get_request_payload("hello")
        assert llm_module._endpoint_model(first) != llm_module._endpoint_model(second)
    finally:
        llm_module._ANTHROPIC_TEMPERATURE_UNSUPPORTED.clear()


@pytest.mark.parametrize("provider,key", [("anthropic", "snapshot-key"), ("deepseek", "snapshot-key"), ("deepseek", "")])
def test_native_sdk_credentials_and_headers_are_scoped(monkeypatch, provider, key):
    pytest.importorskip(f"langchain_{provider}")
    from src.providers.llm import build_llm

    snapshot = resolve_config(
        environment={
            "LANGCHAIN_PROVIDER": provider,
            "LANGCHAIN_MODEL_NAME": "test-model",
            f"{provider.upper()}_API_KEY": key,
            f"{provider.upper()}_BASE_URL": "https://original.invalid",
            "ANTHROPIC_MAX_TOKENS": "128",
        }
    )
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Authorization: wrong\nX-Leaked: secret")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Leaked: secret")
    monkeypatch.setenv("OPENAI_ORG_ID", "wrong-org")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "wrong-token")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-new-default")
    if not key:
        with pytest.raises(RuntimeError, match="Set DEEPSEEK_API_KEY"):
            build_llm(snapshot=snapshot)
        return
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

        client = llm_module._make_temperature_safe_anthropic(Base)()
        iterator = client._astream() if asynchronous else client._stream()
    else:
        monkeypatch.setattr(llm_module.ChatOpenAI, "_stream", stream)
        monkeypatch.setattr(llm_module.ChatOpenAI, "_astream", astream)
        client = llm_module.ChatOpenAIWithReasoning(model="partial", api_key="test")
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
