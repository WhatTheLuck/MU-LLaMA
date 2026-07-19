"""Small YAML configuration loader with recursive ``base`` inheritance."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_config(path: str | Path, _stack: Iterable[Path] = ()) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()
    stack = tuple(_stack)
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Cyclic config inheritance: {chain}")
    with path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    base_ref = current.pop("base", None)
    result: Dict[str, Any] = {}
    if base_ref:
        refs = base_ref if isinstance(base_ref, list) else [base_ref]
        for ref in refs:
            base_path = Path(ref)
            if not base_path.is_absolute():
                base_path = path.parent / base_path
            result = deep_merge(result, load_config(base_path, (*stack, path)))
    result = deep_merge(result, current)
    result["_config_path"] = str(path)
    return _expand(result)


def dump_config(config: Dict[str, Any], path: str | Path) -> None:
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, allow_unicode=True, sort_keys=False)

