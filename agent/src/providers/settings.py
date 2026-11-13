"""Shared provider settings persistence and process-default updates."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from src.config.accessor import get_env_value, reset_env_config
from src.config.dotenv import _read_env_values, _write_env_values
from src.providers.configuration import AGENT_DIR, CONFIG_LOCK
from src.providers.registry import provider_catalog


def secret_names() -> set[str]:
    """Names whose values belong to the desktop credential store."""
    return {"TUSHARE_TOKEN", "GILDATA_TOKEN"} | {p["api_key_env"] for p in provider_catalog() if p.get("api_key_env")}


@dataclass(frozen=True)
class SettingsStore:
    """Canonical dotenv settings; paths can be supplied by an embedding host."""

    path: Path = field(default_factory=lambda: Path.home() / ".vibe-trading" / ".env")
    legacy: Path = AGENT_DIR / ".env"
    example: Path = AGENT_DIR / ".env.example"

    def read(self) -> dict[str, str]:
        """Read existing settings without creating files or exposing vault keys."""
        values = next((_read_env_values(p) for p in (self.path, self.legacy, self.example) if p.exists()), {})
        if get_env_value("VIBE_TRADING_DESKTOP_SECURE_CREDENTIALS") == "1":
            for key in secret_names():
                if get_env_value(key):
                    values[key] = get_env_value(key)
                else:
                    values.pop(key, None)
        return values

    def persist(self, updates: Mapping[str, str], *, apply: bool = False) -> dict[str, str]:
        """Upsert privately with legacy migration, preserving unrelated settings."""
        with CONFIG_LOCK:
            merged = dict(updates)
            if not self.path.exists() and self.legacy != self.path and self.legacy.exists():
                merged = {**_read_env_values(self.legacy), **merged}
            if get_env_value("VIBE_TRADING_DESKTOP_SECURE_CREDENTIALS") == "1":
                merged.update({key: "" for key in secret_names()})
            _write_env_values(self.path, merged)
            if apply:
                self.apply(updates)
            return self.read()

    def apply(self, updates: Mapping[str, str]) -> None:
        """Change defaults for future clients; existing snapshots remain unchanged."""
        with CONFIG_LOCK:
            for key, value in updates.items():
                if value:
                    os.environ[key] = str(value)
                else:
                    os.environ.pop(key, None)
            reset_env_config()
