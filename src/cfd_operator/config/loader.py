"""YAML configuration loader with simple CLI overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union

import yaml

from .schemas import ProjectConfig


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _parse_override_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        pass
    if raw.startswith("[") or raw.startswith("{"):
        return yaml.safe_load(raw)
    return raw


def apply_overrides(config: dict[str, Any], overrides: Optional[List[str]]) -> dict[str, Any]:
    if not overrides:
        return config

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected key=value")
        key_path, raw_value = item.split("=", 1)
        value = _parse_override_value(raw_value)
        parts = key_path.split(".")
        cursor = config
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return config


def load_config(config_path: Union[str, Path], overrides: Optional[List[str]] = None) -> ProjectConfig:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data = apply_overrides(data, overrides)
    return ProjectConfig(**data)
