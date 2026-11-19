"""Modelos del motor de backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from trading_bot_lab.data.instruments import Instrument
from trading_bot_lab.from trading_bot_lab.bot_types import import Direction


@dataclass
class PendingOrder:
    """Orden decidida al cierre de una barra y ejecutada en la siguiente apertura."""

    kind: Literal["entry", "exit", "reverse"]
    direction: Direction
    signal_time: pd.Timestamp
    stop_distance: float = 0.0
    take_profit_distance: float = 0.0
    reason: str = "signal"


@dataclass
class Position:
    direction: Direction
    lots: float
    entry_time: pd.Timestamp
    entry_bar_index: int
    entry_raw_price: float
    entry_price: float
    stop_price: float
    take_profit_price: float | None
    entry_commission: float
    entry_conversion_rate: float
    equity_at_entry: float
    stop_distance: float
    initial_stop_price: float
    bars_held: int = 0
    mae: float = 0.0
    mfe: float = 0.0

    def is_long(self) -> bool:
        return self.direction is Direction.LONG


@dataclass(frozen=True)
class Trade:
    trade_id: int
    symbol: str
    direction: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price_raw: float
    entry_price: float
    exit_price_raw: float
    exit_price: float
    lots: float
    gross_pnl_quote: float
    gross_pnl_account: float
    costs_account: float
    net_pnl_account: float
    return_pct: float
    exit_reason: str
    bars_held: int
    stop_price: float
    initial_stop_price: float
    take_profit_price: float | None
    mae: float
    mfe: float
    entry_conversion_rate: float
    exit_conversion_rate: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "entry_price_raw": self.entry_price_raw,
            "entry_price": self.entry_price,
            "exit_price_raw": self.exit_price_raw,
            "exit_price": self.exit_price,
            "lots": self.lots,
            "gross_pnl_quote": self.gross_pnl_quote,
            "gross_pnl_account": self.gross_pnl_account,
            "costs_account": self.costs_account,
            "net_pnl_account": self.net_pnl_account,
            "return_pct": self.return_pct,
            "exit_reason": self.exit_reason,
            "bars_held": self.bars_held,
            "stop_price": self.stop_price,
            "initial_stop_price": self.initial_stop_price,
            "take_profit_price": self.take_profit_price,
            "mae": self.mae,
            "mfe": self.mfe,
            "entry_conversion_rate": self.entry_conversion_rate,
            "exit_conversion_rate": self.exit_conversion_rate,
        }

    @classmethod
    def columns(cls) -> list[str]:
        return [
            "trade_id", "symbol", "direction", "entry_time", "exit_time",
            "entry_price_raw", "entry_price", "exit_price_raw", "exit_price",
            "lots", "gross_pnl_quote", "gross_pnl_account", "costs_account",
            "net_pnl_account", "return_pct", "exit_reason", "bars_held",
            "stop_price", "initial_stop_price", "take_profit_price", "mae", "mfe",
            "entry_conversion_rate", "exit_conversion_rate",
        ]


@dataclass
class BacktestResult:
    run_id: str
    symbol: str
    timeframe: str
    strategy_key: str
    strategy_name: str
    params: dict[str, Any]
    settings: dict[str, Any]
    instrument: Instrument
    initial_capital: float
    final_equity: float
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, Any]
    signals: pd.DataFrame
    data: pd.DataFrame
    start: pd.Timestamp
    end: pd.Timestamp
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "strategy_key": self.strategy_key,
            "strategy_name": self.strategy_name,
            "params": self.params,
            "initial_capital": self.initial_capital,
            "final_equity": self.final_equity,
            "start": str(self.start),
            "end": str(self.end),
            "metrics": self.metrics,
            "warnings": list(self.warnings),
            "diagnostics": dict(self.diagnostics),
        }


class BacktestError(RuntimeError):
    """Error en la simulacion."""
