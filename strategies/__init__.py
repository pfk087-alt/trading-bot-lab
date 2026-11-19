"""Estrategias experimentales. Ninguna se asume rentable."""

from .base import (
    ParameterSpec,
    Strategy,
    StrategyError,
    build_signal_frame,
    expand_parameter_grid,
)
from .registry import (
    STRATEGY_REGISTRY,
    create_strategy,
    default_params,
    get_strategy_class,
    list_strategies,
    strategy_catalog,
)

__all__ = [
    "ParameterSpec",
    "Strategy",
    "StrategyError",
    "build_signal_frame",
    "expand_parameter_grid",
    "STRATEGY_REGISTRY",
    "create_strategy",
    "default_params",
    "get_strategy_class",
    "list_strategies",
    "strategy_catalog",
]
