"""Shared, credential-free provider catalog for runtime, CLI and Web settings.

Protocol exceptions live in adapters; declarative provider facts live in
``llm_providers.json``. Reading this module never loads credentials or SDKs.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

CATALOG_PATH = Path(__file__).with_name("llm_providers.json")
DEFAULT_REASONING_EFFORTS = ("none", "low", "medium", "high", "max")
_BACKENDS = {"openai_compatible", "openai_codex", "copilot", "claude_cli", "anthropic", "deepseek"}


@lru_cache(maxsize=1)
def _catalog() -> tuple[dict[str, Any], ...]:
    """Load and validate the bundled catalog, failing visibly on invalid data."""
    raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("LLM provider catalog must be a non-empty list")
    seen: set[str] = set()
    for row in raw:
        for field in ("name", "label", "base_url_env", "default_model", "default_base_url"):
            if not isinstance(row.get(field), str):
                raise ValueError(f"Invalid provider catalog field: {field}")
        if row.get("backend", "openai_compatible") not in _BACKENDS:
            raise ValueError(f"Unknown provider backend: {row['name']}")
        for name in (row["name"], *row.get("aliases", []), *row.get("runtime_variants", {})):
            if name in seen:
                raise ValueError(f"Duplicate LLM provider name: {name}")
            seen.add(name)
    return tuple(raw)


def provider_catalog() -> list[dict[str, Any]]:
    """Return independent public metadata records without runtime-only fields."""
    private = {"capabilities", "aliases", "runtime_variants", "setup", "onboarding"}
    return [
        {"backend": "openai_compatible", "automatic_model_list": False,
         **deepcopy({k: v for k, v in row.items() if k not in private})}
        for row in _catalog()
    ]


def provider_backend(provider: str) -> str:
    """Resolve adapter family, retaining generic compatibility for unknown names."""
    name = provider.strip().lower().replace("_", "-")
    for row in _catalog():
        if name == row["name"] or name in row.get("aliases", ()):
            return row.get("backend", "openai_compatible")
    return "openai_compatible"


def capability_records(user_agent: str) -> dict[str, dict[str, Any]]:
    """Project capability constructor arguments, including legacy runtime aliases.

    Args:
        user_agent: Installed application's User-Agent header value.

    Returns:
        Fresh capability mappings keyed by accepted provider name.
    """
    records = {}
    for row in _catalog():
        caps = {
            "name": row["name"],
            "api_key_env": row.get("api_key_env"),
            "base_url_env": row["base_url_env"],
            **deepcopy(row.get("capabilities", {})),
        }
        if "default_headers" in caps:
            caps["default_headers"] = {
                k: v.replace("{user_agent}", user_agent) for k, v in caps["default_headers"].items()
            }
        if "reasoning_effort_models" in caps:
            caps["reasoning_effort_models"] = tuple(caps["reasoning_effort_models"])
        for name in (row["name"], *row.get("aliases", ())):
            records[name] = deepcopy(caps)
        records.update(deepcopy(row.get("runtime_variants", {})))
    # Historical import compatibility; normal lookup already normalizes underscores.
    records["openai_codex"] = deepcopy(records["openai-codex"])
    return records


def setup_choices() -> list[dict[str, Any]]:
    """Project legacy setup options without duplicating models or endpoints."""
    rows = sorted((r for r in _catalog() if r.get("setup")), key=lambda r: r["setup"]["order"])
    return [
        {
            "provider": r["name"],
            "label": r["setup"].get("label", r["label"]),
            "key_env": r.get("api_key_env"),
            "base_env": r["base_url_env"],
            "model": r["default_model"],
            "base_url": r["default_base_url"],
            "key_prefix": r["setup"].get("key_prefix"),
            "key_placeholder": r["setup"].get("key_placeholder"),
        }
        for r in rows
    ]


def onboarding_choices() -> list[dict[str, Any]]:
    """Project first-launch wizard options while preserving its ordering."""
    rows = sorted((r for r in _catalog() if r.get("onboarding")), key=lambda r: r["onboarding"]["order"])
    return [
        {
            "key": r["name"],
            "label": r["onboarding"].get("label", r["label"]),
            "description": r["onboarding"].get("description", ""),
            "default_model": r["default_model"],
            "key_env": r.get("api_key_env"),
            "base_env": r["base_url_env"],
            "base_url": r["default_base_url"],
            "key_prefix": r["onboarding"].get("key_prefix"),
            "suggested_models": tuple(dict.fromkeys([r["default_model"], *r.get("model_options", [])])),
        }
        for r in rows
    ]
