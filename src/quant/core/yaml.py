"""Strict YAML parsing shared by configuration loaders."""

from typing import Any

import yaml
from yaml.nodes import MappingNode

from quant.core.errors import ConfigError


class StrictLoader(yaml.SafeLoader):
    """Safe YAML with duplicate mapping keys refused at every level."""

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[Any, Any]:
        self.flatten_mapping(node)
        keys: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                if key in keys:
                    raise ConfigError(f"duplicate YAML key: {key!r}")
                keys.add(key)
            except TypeError as exc:
                raise ConfigError("unhashable YAML mapping key") from exc
        return super().construct_mapping(node, deep=deep)


def strict_load(data: str | bytes) -> Any:
    """Read safe YAML; malformed syntax, constructors and timestamps are ConfigError."""
    try:
        return yaml.load(data, Loader=StrictLoader)  # noqa: S506 - SafeLoader subclass
    except (yaml.YAMLError, ValueError) as exc:
        raise ConfigError(f"invalid YAML: {exc}") from exc
