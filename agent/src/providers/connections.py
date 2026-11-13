"""Connection testing and activation shared by CLI and HTTP entry points."""

from __future__ import annotations

import ipaddress
import socket
import time
from typing import Any
from urllib.parse import urlsplit

import httpx

from src.providers.configuration import CONFIG_LOCK
from src.providers.custom_profiles import STORE_LOCK, CustomProviderStore
from src.providers.endpoint_guard import is_non_public_address, proxy_applies, public_only_async_client
from src.providers.settings import SettingsStore

_PROVIDER_REQUEST_TIMEOUT_S = 60.0


def _validate_custom_base_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Provider base URL must be an HTTP(S) URL without embedded credentials")
    hostname = parsed.hostname.lower()
    if hostname in {"localhost", "localhost.localdomain", "metadata.google.internal"}:
        raise ValueError("Provider base URL cannot target local or metadata services")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and is_non_public_address(address):
        raise ValueError("Provider base URL cannot target local or private network addresses")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    return normalized


def _reject_private_dns_targets(base_url: str) -> None:
    hostname = urlsplit(base_url).hostname
    if not hostname:
        raise ValueError("Provider base URL must include a hostname")
    try:
        results = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError("Provider hostname could not be resolved") from exc
    for result in results:
        address = ipaddress.ip_address(result[4][0])
        if is_non_public_address(address, resolved=True):
            raise ValueError("Provider hostname resolves to a local or private network address")


def _provider_async_client(base_url: str) -> httpx.AsyncClient:
    """Client for the provider test: every direct connection is re-checked.

    The name is resolved again when the request connects, and only to public
    addresses, so a host that re-resolves after the check above (DNS
    rebinding) or redirects is still refused. Through a proxy the proxy
    resolves the name, so the default client is used.
    """
    kwargs = {"timeout": _PROVIDER_REQUEST_TIMEOUT_S, "follow_redirects": False}
    if proxy_applies(base_url):
        return httpx.AsyncClient(**kwargs)
    return public_only_async_client(**kwargs)


async def _perform_completion_test(*, base_url: str, model: str, api_key: str) -> dict[str, Any]:
    started = time.perf_counter()
    normalized_base_url = _validate_custom_base_url(base_url)
    import asyncio

    await asyncio.to_thread(_reject_private_dns_targets, normalized_base_url)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: PROVIDER_TEST_OK"}],
        "max_tokens": 32,
        "temperature": 0,
    }
    try:
        async with _provider_async_client(normalized_base_url) as client:
            response = await client.post(f"{normalized_base_url}/chat/completions", headers=headers, json=body)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise ValueError(f"Provider returned HTTP {exc.response.status_code}") from exc
    except httpx.ConnectError as exc:
        raise ValueError(f"Provider test failed: {exc}") from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise ValueError("Provider test failed: endpoint did not return valid JSON") from exc
    try:
        content = payload["choices"][0]["message"].get("content", "")
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Provider test failed: response had no chat completion") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Provider test failed: response had no text")
    return {
        "response_preview": str(content).replace(api_key, "[redacted]").strip()[:500],
        "latency_ms": round((time.perf_counter() - started) * 1000),
    }


def activate_custom_provider(
    profile_id: str, *, store: CustomProviderStore, settings: SettingsStore | None = None
) -> dict[str, Any]:
    """Activate one vault profile and roll back metadata if settings cannot persist."""
    settings = settings or SettingsStore()
    with CONFIG_LOCK, STORE_LOCK:
        profile = store.get(profile_id)
        updates = {"LANGCHAIN_PROVIDER": "custom", "LANGCHAIN_MODEL_NAME": profile["model"]}
        previous = store._read()
        store.activate(profile["id"])
        try:
            settings.persist(updates, apply=True)
        except Exception:
            store._write(previous)
            raise
    return {"status": "ok", "active_id": profile["id"], "model": profile["model"], "base_url": profile["base_url"]}
