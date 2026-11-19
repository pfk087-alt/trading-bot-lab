"""Registro central de estrategias."""

from __future__ import annotations

from typing import Any, Type

from .atr_breakout import ATRBreakoutStrategy
from .base import (
    ParameterSpec,
    Strategy,
    StrategyError,
    StrategyInfo,
    expand_parameter_grid,
)
from .breakout import SupportResistanceBreakoutStrategy
from .ema_cross import EMACrossStrategy
from .rsi_trend import RSITrendStrategy
from .trend_pullback import TrendPullbackStrategy

STRATEGY_REGISTRY: dict[str, Type[Strategy]] = {
    cls.key: cls
    for cls in (
        EMACrossStrategy,
        RSITrendStrategy,
        SupportResistanceBreakoutStrategy,
        ATRBreakoutStrategy,
        TrendPullbackStrategy,
    )
}


def list_strategies() -> dict[str, Type[Strategy]]:
    return dict(STRATEGY_REGISTRY)


def get_strategy_class(key: str) -> Type[Strategy]:
    try:
        return STRATEGY_REGISTRY[key]
    except KeyError as exc:
        raise StrategyError(
            f"Estrategia desconocida: {key}. Disponibles: {', '.join(STRATEGY_REGISTRY)}"
        ) from exc


def create_strategy(key: str, params: dict[str, Any] | None = None) -> Strategy:
    return get_strategy_class(key)(**(params or {}))


def strategy_catalog() -> list[dict[str, Any]]:
    return [
        StrategyInfo(
            key=cls.key,
            name=cls.name,
            description=cls.description,
            parameter_specs=cls.parameter_specs(),
        ).to_dict()
        for cls in STRATEGY_REGISTRY.values()
    ]


def default_params(key: str) -> dict[str, Any]:
    return get_strategy_class(key).default_params()


__all__ = [
    "ATRBreakoutStrategy",
    "EMACrossStrategy",
    "ParameterSpec",
    "RSITrendStrategy",
    "STRATEGY_REGISTRY",
    "Strategy",
    "StrategyError",
    "SupportResistanceBreakoutStrategy",
    "TrendPullbackStrategy",
    "create_strategy",
    "default_params",
    "expand_parameter_grid",
    "get_strategy_class",
    "list_strategies",
    "strategy_catalog",
]
