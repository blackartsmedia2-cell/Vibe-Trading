"""Gateway contracts and real SDK/HTTP integration through saved settings."""

from __future__ import annotations

from contextlib import closing

from tests.provider_helpers import provider_server as provider_server

from src.providers.registry import provider_catalog as catalog
from src.providers.settings import resolve_config
from src.providers.registry import get_provider_capabilities


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


def test_settings_to_real_sdk_stream_and_tool_followup(tmp_path, monkeypatch, provider_server):
    """Saved settings -> real SDK -> HTTP SSE -> fragmented tool call -> follow-up."""
    from fastapi.testclient import TestClient
    import api_server
    from src.providers.chat import ChatLLM

    base_url, received = provider_server
    for name in ("ENV_PATH", "ENV_EXAMPLE_PATH", "LEGACY_ENV_PATH"):
        monkeypatch.setattr(api_server, name, tmp_path / name, raising=False)
    monkeypatch.delenv("API_AUTH_KEY", raising=False)
    monkeypatch.setenv("VIBE_TRADING_DISABLE_HTTP_PROXY", "1")
    monkeypatch.setenv("LANGCHAIN_USE_RESPONSES_API", "false")
    with TestClient(api_server.app, client=("127.0.0.1", 50000)) as client:
        response = client.put("/settings/llm", json={
            "provider": "ollama", "model_name": "gateway-test", "base_url": base_url,
            "temperature": 0, "timeout_seconds": 10, "max_retries": 0,
        })
        assert response.status_code == 200, response.text
        provider = next(p for p in response.json()["providers"] if p["name"] == "ollama")
        assert provider["backend"] == "openai_compatible"
        with closing(ChatLLM()) as llm:
            parameters = {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}
            function = {"name": "quote", "description": "Read a quote", "parameters": parameters}
            tools = [{"type": "function", "function": function}]
            messages = [{"role": "user", "content": "Quote AAPL"}]
            first = llm.stream_chat(messages, tools=tools)
            assert first.tool_calls[0].arguments == {"symbol": "AAPL"}
            assert first.finish_reason == "tool_calls"
            tool_call = {
                "id": "call_quote", "type": "function",
                "function": {"name": "quote", "arguments": '{"symbol":"AAPL"}'},
            }
            followup = messages + [
                {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                {"role": "tool", "tool_call_id": "call_quote", "content": '{"price":100}'},
            ]
            chunks = []
            second = llm.stream_chat(followup, tools=tools, on_text_chunk=chunks.append)
            assert second.content == "".join(chunks) == "gateway-test"
            assert second.usage_metadata["input_tokens"] == 17
            assert second.response_model == "served-gateway-test"
    assert len(received) == 2
    assert all(path == "/v1/chat/completions" for path, _, _ in received)
    assert received[1][2]["messages"][-1]["tool_call_id"] == "call_quote"


def test_catalog_only_provider_reaches_settings_and_real_sdk(tmp_path, monkeypatch, provider_server):
    """A new compatible provider needs data only, including its secret allowlist."""
    import json
    from src.providers import registry
    from src.providers.chat import ChatLLM
    from src.providers.settings import secret_names
    from src.api.settings_routes import _load_llm_providers

    base_url, received = provider_server
    row = {
        "name": "catalog-test", "label": "Catalog Test", "api_key_env": "CATALOG_TEST_KEY",
        "base_url_env": "CATALOG_TEST_URL", "default_model": "catalog-model",
        "default_base_url": base_url, "api_key_required": True,
        "setup": {"order": 100}, "onboarding": {"order": 100},
    }
    path = tmp_path / "providers.json"
    path.write_text(json.dumps([*registry._catalog(), row]))
    monkeypatch.setattr(registry, "CATALOG_PATH", path)
    registry._catalog.cache_clear()
    registry.get_provider_capabilities.cache_clear()
    try:
        assert "catalog-test" in registry.provider_names()
        assert _load_llm_providers()[-1].name == "catalog-test"
        assert registry.setup_choices()[-1]["key_env"] == "CATALOG_TEST_KEY"
        assert registry.onboarding_choices()[-1]["key"] == "catalog-test"
        assert "CATALOG_TEST_KEY" in secret_names()
        config = resolve_config(environment={
            "LANGCHAIN_PROVIDER": "catalog-test", "LANGCHAIN_MODEL_NAME": "catalog-model",
            "CATALOG_TEST_KEY": "catalog-secret", "OPENAI_API_KEY": "wrong-secret",
            "LANGCHAIN_USE_RESPONSES_API": "false", "VIBE_TRADING_DISABLE_HTTP_PROXY": "1",
        })
        with closing(ChatLLM(snapshot=config)) as llm:
            assert llm.chat([{"role": "user", "content": "hello"}]).content == "catalog-model"
        assert received[0][1]["Authorization"] == "Bearer catalog-secret"
    finally:
        registry._catalog.cache_clear()
        registry.get_provider_capabilities.cache_clear()
