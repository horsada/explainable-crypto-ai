from __future__ import annotations
from typing import Callable, Dict, Type
from excrypto.features.base import Feature

_REGISTRY: Dict[str, Type[Feature]] = {}

def register_feature(name: str) -> Callable[[Type[Feature]], Type[Feature]]:
    def _decorator(cls: Type[Feature]) -> Type[Feature]:
        _REGISTRY[name] = cls
        return cls
    return _decorator

def get_feature_cls(name: str) -> Type[Feature]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Feature '{name}' not found. Available: {list(_REGISTRY.keys())}")

def list_features() -> Dict[str, Type[Feature]]:
    return dict(_REGISTRY)
