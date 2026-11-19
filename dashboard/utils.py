"""Utilidades compartidas del panel."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pandas as pd

from trading_bot_lab.config import AppConfig
from trading_bot_lab.data.resampling import infer_timeframe, resample_ohlcv
from trading_bot_lab.data.validation import DataValidationError, validate_ohlcv
from trading_bot_lab.from trading_bot_lab.bot_types import import Timeframe


def money(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/D"
    return f"{float(value):,.2f}"


def pct(value: Any, decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/D"
    return f"{float(value):,.{decimals}f}%"


def number(value: Any, decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/D"
    return f"{float(value):,.{decimals}f}"


def build_config(
    base: AppConfig,
    *,
    initial_capital: float,
    risk_per_trade_pct: float,
    max_daily_loss_pct: float,
    max_drawdown_pct: float,
    max_notional_exposure_pct: float,
    close_on_max_drawdown: bool,
    spread_pips: float,
    commission_per_lot_per_side: float,
    slippage_pips: float,
    stop_mode: str,
    stop_value: float,
    take_profit_mode: str,
    take_profit_value: float,
    trailing_stop_atr_mult: float,
    max_bars_in_trade: int,
    warmup_bars: int,
) -> AppConfig:
    return replace(
        base,
        account=replace(base.account, initial_capital=float(initial_capital)),
        risk=replace(
            base.risk,
            risk_per_trade_pct=float(risk_per_trade_pct),
            max_daily_loss_pct=float(max_daily_loss_pct),
            max_drawdown_pct=float(max_drawdown_pct),
            max_notional_exposure_pct=float(max_notional_exposure_pct),
            close_on_max_drawdown=bool(close_on_max_drawdown),
            max_open_positions=1,
        ),
        costs=replace(
            base.costs,
            spread_pips=float(spread_pips),
            commission_per_lot_per_side=float(commission_per_lot_per_side),
            slippage_pips=float(slippage_pips),
        ),
        execution=replace(
            base.execution,
            stop_mode=stop_mode,
            stop_value=float(stop_value),
            take_profit_mode=take_profit_mode,
            take_profit_value=float(take_profit_value),
            trailing_stop_atr_mult=float(trailing_stop_atr_mult),
            max_bars_in_trade=int(max_bars_in_trade),
            warmup_bars=int(warmup_bars),
        ),
    )


def normalize_data(raw: pd.DataFrame, symbol: str, timeframe: str | Timeframe) -> pd.DataFrame:
    """Valida y, si hace falta, agrega a la temporalidad solicitada."""
    target = Timeframe.parse(timeframe)
    clean = validate_ohlcv(raw, symbol=symbol, timeframe=target.value)
    source = infer_timeframe(clean)
    if source is None:
        raise DataValidationError(
            "No se pudo inferir la temporalidad. Compruebe que los timestamps sean regulares."
        )
    if target.minutes < source.minutes:
        raise DataValidationError(
            f"Los datos son de {source.value} y se pide {target.value}. "
            "No se puede inventar informacion intra-barra."
        )
    if target.minutes > source.minutes:
        clean = resample_ohlcv(clean, target, symbol=symbol)
    clean.attrs["inferred_timeframe"] = source.value
    return clean


def default_grid(strategy_key: str) -> str:
    grids: dict[str, dict[str, list[Any]]] = {
        "ema_cross": {"fast_period": [10, 20, 30], "slow_period": [50, 100]},
        "rsi_trend": {"rsi_period": [7, 14], "oversold": [25, 30], "overbought": [70, 75]},
        "sr_breakout": {"lookback": [10, 20, 40]},
        "atr_breakout": {"atr_mult": [1.0, 1.5, 2.0], "exit_period": [10, 20]},
        "trend_pullback": {"pullback_period": [30, 50], "rsi_pullback": [45, 50]},
    }
    return json.dumps(grids.get(strategy_key, {}), indent=2)


def parse_grid(text: str) -> dict[str, list[Any]]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON invalido: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("La rejilla debe ser un objeto JSON {parametro: [valores]}.")
    result: dict[str, list[Any]] = {}
    for key, values in value.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"'{key}' debe ser una lista no vacia.")
        result[str(key)] = values
    return result


def grid_size(grid: dict[str, list[Any]]) -> int:
    size = 1
    for values in grid.values():
        size *= len(values)
    return size
