"""Adapter construction for the in-process gateway.

Compatibility helpers remain in llm.py so existing adapters and patch points
keep their behavior. No protocol fallback or retry policy is changed here.
"""

from __future__ import annotations

from typing import Any, Optional
from src.providers.gateway import ProviderRoute


def build_adapter(route: ProviderRoute, *, model_name: Optional[str] = None, callbacks: Any = None) -> Any:
    """Construct the configured LangChain chat model.

    Args:
        route: Resolved provider, model, backend and wire capabilities.
        model_name: Model name; defaults to LANGCHAIN_MODEL_NAME.
        callbacks: Optional LangChain callbacks.

    Returns:
        Provider-specific LangChain chat model.

    Raises:
        RuntimeError: If langchain-openai is missing or LANGCHAIN_MODEL_NAME is unset.
        ValueError: If a Moonshot kimi-k3.1 configuration carries a
            ``LANGCHAIN_REASONING_EFFORT`` outside the published
            low/high/max set.
    """
    from src.providers import llm as compat

    name = route.model
    provider = route.provider
    caps = route.capabilities
    temperature = compat.get_env_config().llm.langchain_temperature
    if route.backend == "openai_codex":
        from src.providers.openai_codex import OpenAICodexLLM

        effort = compat.get_env_config().llm.langchain_reasoning_effort.strip().lower()
        return OpenAICodexLLM(
            model=name,
            temperature=temperature,
            timeout=compat.get_env_config().llm.timeout_seconds,
            reasoning_effort=effort or None,
        )

    if route.backend == "copilot":
        from src.providers.copilot_auth import CopilotSDKLLM

        return CopilotSDKLLM(
            model=name,
            timeout=compat.get_env_config().llm.timeout_seconds,
        )

    if route.backend == "claude_cli":
        from src.providers.claude_cli import ClaudeCliLLM

        llm_cfg = compat.get_env_config().llm
        return ClaudeCliLLM(
            model=name,
            timeout=llm_cfg.timeout_seconds,
            reasoning_effort=llm_cfg.langchain_reasoning_effort.strip().lower(),
        )

    if route.backend == "anthropic":
        return compat._build_anthropic(
            model=name,
            temperature=temperature,
            callbacks=callbacks,
            effort=compat.get_env_config().llm.langchain_reasoning_effort.strip().lower(),
        )

    if route.backend == "deepseek":
        adapter_mode = compat._deepseek_adapter_mode()
        if adapter_mode != "openai-compatible":
            native_llm = compat._build_native_deepseek(
                model=name,
                temperature=temperature,
                callbacks=callbacks,
            )
            if native_llm is not None:
                return native_llm
            if adapter_mode == "native":
                raise RuntimeError("VIBE_TRADING_DEEPSEEK_ADAPTER=native requires langchain-deepseek")

    if compat.ChatOpenAI is None:
        raise RuntimeError("langchain-openai is not installed")
    # MiniMax requires temperature in (0.0, 1.0] — clamp to 0.01 when the
    # default 0.0 is used to avoid an API validation error.
    if provider == "minimax" and temperature <= 0.0:
        temperature = 0.01
    # Kimi reasoning models reject any temperature other than 1
    # ("invalid temperature: only 1 is allowed for this model").
    if (
        caps.name in {"moonshot", "kimi-coding"}
        and compat._KIMI_FORCED_TEMPERATURE_RE.match(name)
        and temperature != 1.0
    ):
        compat.logger.info("Forcing temperature=1.0 for %s (provider requirement)", name)
        temperature = 1.0
    effort = compat.get_env_config().llm.langchain_reasoning_effort.strip().lower()
    # Moonshot/DeepSeek official APIs emit reasoning by default and ignore this field.
    configured_responses_api = compat.get_env_config().llm.langchain_use_responses_api
    use_responses_api = compat.uses_responses_api(provider, configured_responses_api)
    creds = compat.get_llm_credentials(provider, name)
    if provider == "custom":
        name = model_name or creds["model"]
    compat._validate_moonshot_reasoning_effort(caps, name, effort)
    api_key = creds["api_key"]
    compat._validate_authorization_credential(
        api_key,
        source=compat._credential_env_source(caps.api_key_env),
    )
    kwargs: dict[str, Any] = {
        "model": name,
        "api_key": api_key or None,
        "base_url": creds["base_url"] or None,
        "temperature": temperature,
        "timeout": compat.get_env_config().llm.timeout_seconds,
        "max_retries": compat.get_env_config().llm.max_retries,
        "callbacks": callbacks,
        # Ask for real token usage on streamed calls (issue #1224); endpoints
        # that reject stream_options self-heal in _stream_with_usage_fallback.
        "stream_usage": True,
        "output_version": "responses/v1" if use_responses_api else None,
        "use_responses_api": use_responses_api,
        "reasoning": ({"effort": effort} if use_responses_api and effort else None),
        "extra_body": (
            {"reasoning": {"effort": effort}}
            if effort and not use_responses_api and caps.openrouter_reasoning_body
            else None
        ),
        "reasoning_effort": (
            effort
            if (effort and not use_responses_api and compat._sends_top_level_reasoning_effort(caps, name))
            else None
        ),
        "vibe_provider": provider,
        "vibe_api_key": api_key,
    }
    langchain_max_tokens = compat.get_env_config().llm.langchain_max_tokens
    if langchain_max_tokens:
        kwargs["max_tokens"] = langchain_max_tokens
    if caps.default_headers:
        headers = dict(caps.default_headers)
        if caps.name in {"moonshot", "kimi-coding"}:
            custom_ua = compat.get_env_config().llm.moonshot_user_agent.strip()
            if custom_ua:
                headers["User-Agent"] = custom_ua
        compat._validate_explicit_headers(headers, source=f"{caps.name} provider configuration")
        kwargs["default_headers"] = headers
    disable_proxy = compat.get_env_config().llm.vibe_trading_disable_http_proxy
    if provider == "custom":
        # Imported here: the guard needs httpx/httpcore/anyio, and the
        # background-tool lifecycle checks import this module without them.
        from src.providers.endpoint_guard import build_public_only_clients, proxy_applies
    if provider == "custom" and (disable_proxy or not proxy_applies(creds["base_url"])):
        # A saved custom endpoint is re-checked on every connection, so a
        # host that re-resolves to a private address later is refused.
        sync_client, async_client = build_public_only_clients()
        kwargs["http_client"] = sync_client
        kwargs["http_async_client"] = async_client
        kwargs["vibe_owned_http_clients"] = (sync_client, async_client)
    elif disable_proxy:
        sync_client, async_client = compat._build_proxy_free_http_clients()
        kwargs["http_client"] = sync_client
        kwargs["http_async_client"] = async_client
        kwargs["vibe_owned_http_clients"] = (sync_client, async_client)
    return compat.ChatOpenAIWithReasoning(**kwargs)
