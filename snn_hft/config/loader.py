"""YAML loading with inheritance and command-line overrides.

A config file may name a parent with ``inherits: <path>`` (relative to the file).
Mappings are merged recursively; any other value, including lists, replaces the
parent's value. Overrides use dotted paths, e.g. ``models.0.core.lif.threshold=1.2``;
the value is parsed as YAML.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from snn_hft.config.schema import ExperimentConfig

INHERITS_KEY = "inherits"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_raw(path: str | Path, _chain: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read a YAML file and resolve its ``inherits`` chain into one mapping."""
    path = Path(path).resolve()
    if path in _chain:
        cycle = " -> ".join(str(p) for p in (*_chain, path))
        raise ValueError(f"cyclic 'inherits': {cycle}")
    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    parent = raw.pop(INHERITS_KEY, None)
    if parent is None:
        return raw
    base = load_raw(path.parent / parent, (*_chain, path))
    return deep_merge(base, raw)


def apply_override(raw: dict[str, Any], assignment: str) -> None:
    """Apply one ``dotted.path=value`` assignment in place."""
    if "=" not in assignment:
        raise ValueError(f"override {assignment!r} must look like key.path=value")
    dotted, _, text = assignment.partition("=")
    keys = dotted.strip().split(".")
    value = yaml.safe_load(text)
    node: Any = raw
    for key in keys[:-1]:
        node = _child(node, key, create=True)
    last = keys[-1]
    if isinstance(node, list):
        node[_index(node, last)] = value
    else:
        node[last] = value


def _index(node: list, key: str) -> int:
    try:
        idx = int(key)
    except ValueError:
        raise KeyError(f"expected a list index, got {key!r}") from None
    if not -len(node) <= idx < len(node):
        raise KeyError(f"list index {idx} out of range")
    return idx


def _child(node: Any, key: str, create: bool) -> Any:
    if isinstance(node, list):
        return node[_index(node, key)]
    if not isinstance(node, dict):
        raise KeyError(f"cannot descend into {type(node).__name__} at {key!r}")
    if key not in node:
        if not create:
            raise KeyError(key)
        node[key] = {}
    return node[key]


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> ExperimentConfig:
    raw = load_raw(path)
    for assignment in overrides:
        apply_override(raw, assignment)
    return ExperimentConfig.model_validate(raw)


def to_yaml(model: BaseModel) -> str:
    return yaml.safe_dump(model.model_dump(mode="json"), sort_keys=False, allow_unicode=True)


def dump_config(model: BaseModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_yaml(model))
