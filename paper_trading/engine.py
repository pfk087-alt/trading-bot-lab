"""Modo paper trading: simulacion barra a barra sin dinero real ni broker.

Este modulo NO usa el motor de backtesting. Comparte unicamente primitivas de
riesgo, costes e indicadores. No existe ninguna conexion para enviar ordenes.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from trading_bot_lab.backtesting.costs import CostModel
from trading_bot_lab.config import AppConfig
from trading_bot_lab.data.instruments import Instrument, get_instrument
from trading_bot_lab.indicators import atr
from trading_bot_lab.risk import RiskManager
from trading_bot_lab.strategies.base import Strategy
from trading_bot_lab.types import Direction, Timeframe


class PaperTradingError(RuntimeError):
    """Error del modo paper trading."""


@dataclass
class PaperTrade:
    trade_id: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    lots: float
    entry_price: float
    exit_price: float
    net_pnl: float
    costs: float
    exit_reason: str
    bars_held: int
    return_pct: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "direction": self.direction,
            "lots": self.lots,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "net_pnl": self.net_pnl,
            "costs": self.costs,
            "exit_reason": self.exit_reason,
            "bars_held": self.bars_held,
            "return_pct": self.return_pct,
        }


@dataclass
class PaperState:
    session_id: str
    symbol: str
    timeframe: str
    strategy_key: str
    strategy_name: str
    params: dict[str, Any]
    initial_capital: float
    cash: float
    equity: float
    started_at: pd.Timestamp
    last_bar_time: pd.Timestamp | None = None
    position: dict[str, Any] | None = None
    pending: dict[str, Any] | None = None
    trades: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    blocked_entries: dict[str, int] = field(default_factory=dict)
    bars_processed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "strategy_key": self.strategy_key,
            "strategy_name": self.strategy_name,
            "params": self.params,
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "equity": self.equity,
            "started_at": str(self.started_at),
            "last_bar_time": str(self.last_bar_time) if self.last_bar_time is not None else None,
            "position": (
                {**self.position, "entry_time": str(self.position["entry_time"])}
                if self.position is not None
                else None
            ),
            "pending": self.pending,
            "trades": [
                {**trade, "entry_time": str(trade["entry_time"]), "exit_time": str(trade["exit_time"])}
                for trade in self.trades
            ],
            "equity_curve": [
                {**row, "timestamp": str(row["timestamp"])} for row in self.equity_curve
            ],
            "blocked_entries": self.blocked_entries,
            "bars_processed": self.bars_processed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PaperState":
        trades = []
        for trade in data.get("trades", []):
            trades.append(
                {
                    **trade,
                    "entry_time": pd.Timestamp(trade["entry_time"]),
                    "exit_time": pd.Timestamp(trade["exit_time"]),
                }
            )
        equity = [
            {
                **row,
                "timestamp": pd.Timestamp(row["timestamp"]),
            }
            for row in data.get("equity_curve", [])
        ]
        return cls(
            session_id=data["session_id"],
            symbol=data["symbol"],
            timeframe=data["timeframe"],
            strategy_key=data["strategy_key"],
            strategy_name=data["strategy_name"],
            params=data["params"],
            initial_capital=float(data["initial_capital"]),
            cash=float(data["cash"]),
            equity=float(data["equity"]),
            started_at=pd.Timestamp(data["started_at"]),
            last_bar_time=pd.Timestamp(data["last_bar_time"]) if data.get("last_bar_time") else None,
            position=data.get("position"),
            pending=data.get("pending"),
            trades=trades,
            equity_curve=equity,
            blocked_entries=data.get("blocked_entries", {}),
            bars_processed=int(data.get("bars_processed", 0)),
        )


class PaperTradingEngine:
    """Sesion de paper trading alimentada por barras (CSV replay o feed propio)."""

    def __init__(
        self,
        config: AppConfig,
        strategy: Strategy,
        *,
        symbol: str,
        timeframe: str | Timeframe,
        instrument: Instrument | None = None,
    ) -> None:
        self.config = config
        self.strategy = strategy
        self.symbol = symbol.upper()
        self.timeframe = Timeframe.parse(timeframe)
        self.instrument = instrument or get_instrument(symbol)
        self.costs = CostModel(config.costs)
        self.state = self._new_state()
        self._buffer: list[dict[str, Any]] = []
        self._risk = RiskManager(config.risk, config.account.initial_capital)

    def _new_state(self) -> PaperState:
        now = pd.Timestamp.now(tz="UTC")
        return PaperState(
            session_id=uuid.uuid4().hex[:12],
            symbol=self.symbol,
            timeframe=self.timeframe.value,
            strategy_key=self.strategy.key,
            strategy_name=self.strategy.name,
            params=dict(self.strategy.parameters),
            initial_capital=float(self.config.account.initial_capital),
            cash=float(self.config.account.initial_capital),
            equity=float(self.config.account.initial_capital),
            started_at=now,
        )

    def reset(self) -> PaperState:
        self.state = self._new_state()
        self._buffer = []
        self._risk = RiskManager(self.config.risk, self.config.account.initial_capital)
        return self.state

    def on_bar(
        self,
        timestamp: pd.Timestamp,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float = 0.0,
    ) -> dict[str, Any]:
        """Procesa una barra cerrada y devuelve un resumen del evento."""
        timestamp = pd.Timestamp(timestamp)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        if self.state.last_bar_time is not None and timestamp <= self.state.last_bar_time:
            raise PaperTradingError("Las barras deben llegar en orden estrictamente creciente.")
        if min(open_price, high_price, low_price, close_price) <= 0:
            raise PaperTradingError("Precios invalidos")
        if high_price < max(open_price, close_price) or low_price > min(open_price, close_price):
            raise PaperTradingError("OHLC inconsistente")

        event: dict[str, Any] = {"timestamp": timestamp, "actions": []}
        self._risk.on_bar(timestamp, self.state.equity)

        pending = self.state.pending
        self.state.pending = None
        if pending is not None:
            if pending["kind"] in {"exit", "reverse"} and self.state.position is not None:
                reason = "reverse" if pending["kind"] == "reverse" else pending.get("reason", "signal")
                self._close(timestamp, open_price, reason)
                event["actions"].append(f"exit:{reason}")
            if pending["kind"] in {"entry", "reverse"} and self.state.position is None:
                entered = self._enter(
                    timestamp,
                    open_price,
                    Direction(pending["direction"]),
                    float(pending["stop_distance"]),
                    float(pending.get("take_profit_distance", 0.0)),
                )
                if entered:
                    event["actions"].append(f"entry:{pending['direction']}")

        if self.state.position is not None:
            exit_raw, exit_reason = self._stop_or_target_hit(open_price, high_price, low_price)
            if exit_raw is not None:
                self._close(timestamp, exit_raw, exit_reason)
                event["actions"].append(f"exit:{exit_reason}")

        # Marcar a mercado.
        if self.state.position is not None:
            position = self.state.position
            direction = Direction(position["direction"])
            mark = self.costs.exit_price(close_price, direction, self.instrument)
            conversion = self.instrument.conversion_rate(close_price)
            unrealized = (
                (mark - position["entry_price"])
                * direction.sign
                * self.instrument.contract_size
                * position["lots"]
                * conversion
            )
            equity = self.state.cash + unrealized - self.costs.commission(position["lots"])
        else:
            equity = self.state.cash
        self.state.equity = float(equity)
        self.state.equity_curve.append(
            {"timestamp": timestamp, "equity": float(equity), "in_position": self.state.position is not None}
        )

        # Anadir la barra cerrada y recalcular senales (causal).
        self._buffer.append(
            {
                "timestamp": timestamp,
                "open": float(open_price),
                "high": float(high_price),
                "low": float(low_price),
                "close": float(close_price),
                "volume": float(volume),
            }
        )
        self.state.last_bar_time = timestamp
        self.state.bars_processed += 1

        buffer = self._buffer_frame()
        if len(buffer) > self.config.execution.warmup_bars + 2:
            signals = self.strategy.generate_signals(buffer)
            self._schedule_next(signals.iloc[-1], timestamp, close_price, buffer)

        event["equity"] = self.state.equity
        return event

    def run_replay(
        self,
        data: pd.DataFrame,
        *,
        delay_seconds: float = 0.0,
        progress: Callable[[int, int], None] | None = None,
    ) -> PaperState:
        """Reproduce un CSV historico como si llegara en tiempo real."""
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(data.columns)
        if missing:
            raise PaperTradingError(f"Faltan columnas: {sorted(missing)}")
        self.reset()
        total = len(data)
        for number, (timestamp, row) in enumerate(data.iterrows(), start=1):
            self.on_bar(
                timestamp,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row.get("volume", 0.0)),
            )
            if progress is not None:
                progress(number, total)
            if delay_seconds > 0:
                time.sleep(delay_seconds)
        return self.state

    def _buffer_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self._buffer).set_index("timestamp").sort_index()
        frame.index = pd.to_datetime(frame.index, utc=True)
        return frame

    def _schedule_next(
        self,
        signal: pd.Series,
        timestamp: pd.Timestamp,
        close_price: float,
        buffer: pd.DataFrame,
    ) -> None:
        if self.state.position is not None:
            position = self.state.position
            direction = Direction(position["direction"])
            if direction is Direction.LONG:
                if bool(signal["short_entry"]):
                    self._queue_reverse(timestamp, close_price, buffer, Direction.SHORT)
                elif bool(signal["long_exit"]):
                    self.state.pending = {"kind": "exit", "direction": direction.value, "reason": "signal"}
            else:
                if bool(signal["long_entry"]):
                    self._queue_reverse(timestamp, close_price, buffer, Direction.LONG)
                elif bool(signal["short_exit"]):
                    self.state.pending = {"kind": "exit", "direction": direction.value, "reason": "signal"}

            if (
                self.state.pending is None
                and self.config.execution.max_bars_in_trade > 0
                and (self.state.bars_processed - int(position["entry_bar"])) >= self.config.execution.max_bars_in_trade
            ):
                self.state.pending = {"kind": "exit", "direction": direction.value, "reason": "time_stop"}

            if self.config.execution.trailing_stop_atr_mult > 0 and len(buffer) > self.config.execution.atr_period:
                atr_value = float(
                    atr(buffer["high"], buffer["low"], buffer["close"], self.config.execution.atr_period).iloc[-1]
                )
                if np.isfinite(atr_value):
                    distance = self.config.execution.trailing_stop_atr_mult * atr_value
                    if direction is Direction.LONG:
                        position["stop_price"] = max(position["stop_price"], close_price - distance)
                    else:
                        position["stop_price"] = min(position["stop_price"], close_price + distance)
        else:
            if bool(signal["long_entry"]) and not bool(signal["short_entry"]):
                self._queue_entry(timestamp, close_price, buffer, Direction.LONG)
            elif bool(signal["short_entry"]) and not bool(signal["long_entry"]):
                self._queue_entry(timestamp, close_price, buffer, Direction.SHORT)

    def _queue_entry(
        self,
        timestamp: pd.Timestamp,
        close_price: float,
        buffer: pd.DataFrame,
        direction: Direction,
    ) -> None:
        stop_distance = self._stop_distance(close_price, buffer)
        if stop_distance is None:
            self.state.blocked_entries["stop_invalido"] = (
                self.state.blocked_entries.get("stop_invalido", 0) + 1
            )
            return
        self.state.pending = {
            "kind": "entry",
            "direction": direction.value,
            "stop_distance": stop_distance,
            "take_profit_distance": self._take_profit_distance(close_price, buffer, stop_distance) or 0.0,
        }

    def _queue_reverse(
        self,
        timestamp: pd.Timestamp,
        close_price: float,
        buffer: pd.DataFrame,
        direction: Direction,
    ) -> None:
        stop_distance = self._stop_distance(close_price, buffer)
        if stop_distance is None:
            return
        self.state.pending = {
            "kind": "reverse",
            "direction": direction.value,
            "stop_distance": stop_distance,
            "take_profit_distance": self._take_profit_distance(close_price, buffer, stop_distance) or 0.0,
        }

    def _stop_distance(self, close_price: float, buffer: pd.DataFrame) -> float | None:
        mode = self.config.execution.stop_mode
        value = self.config.execution.stop_value
        if mode == "none":
            return None
        if mode == "atr":
            if len(buffer) <= self.config.execution.atr_period:
                return None
            atr_value = float(
                atr(buffer["high"], buffer["low"], buffer["close"], self.config.execution.atr_period).iloc[-1]
            )
            if not np.isfinite(atr_value) or atr_value <= 0:
                return None
            distance = atr_value * value
        elif mode == "pips":
            distance = value * self.instrument.pip_size
        elif mode == "percent":
            distance = close_price * value / 100.0
        else:
            return None
        return float(distance) if distance > 0 else None

    def _take_profit_distance(
        self, close_price: float, buffer: pd.DataFrame, stop_distance: float
    ) -> float | None:
        mode = self.config.execution.take_profit_mode
        value = self.config.execution.take_profit_value
        if mode == "none":
            return None
        if mode == "rr":
            return stop_distance * value if value > 0 else None
        if mode == "atr":
            if len(buffer) <= self.config.execution.atr_period:
                return None
            atr_value = float(
                atr(buffer["high"], buffer["low"], buffer["close"], self.config.execution.atr_period).iloc[-1]
            )
            return atr_value * value if np.isfinite(atr_value) and value > 0 else None
        if mode == "pips":
            return value * self.instrument.pip_size if value > 0 else None
        if mode == "percent":
            return close_price * value / 100.0 if value > 0 else None
        return None

    def _enter(
        self,
        timestamp: pd.Timestamp,
        raw_price: float,
        direction: Direction,
        stop_distance: float,
        take_profit_distance: float,
    ) -> bool:
        sizing = self._risk.size_position(
            equity=self.state.cash,
            stop_distance=stop_distance,
            instrument=self.instrument,
            price=raw_price,
            open_positions=0,
            has_pending_order=False,
        )
        if not sizing.allowed:
            key = sizing.reason.split(":")[0]
            self.state.blocked_entries[key] = self.state.blocked_entries.get(key, 0) + 1
            return False
        entry_price = self.costs.entry_price(raw_price, direction, self.instrument)
        commission = self.costs.commission(sizing.lots)
        self.state.cash -= commission
        self.state.position = {
            "direction": direction.value,
            "lots": sizing.lots,
            "entry_time": timestamp,
            "entry_raw_price": raw_price,
            "entry_price": entry_price,
            "equity_at_entry": float(self.state.cash),
            "stop_price": entry_price - direction.sign * stop_distance,
            "take_profit_price": entry_price + direction.sign * take_profit_distance
            if take_profit_distance > 0
            else None,
            "entry_commission": commission,
            "entry_bar": self.state.bars_processed,
        }
        return True

    def _close(self, timestamp: pd.Timestamp, raw_price: float, reason: str) -> None:
        position = self.state.position
        if position is None:
            return
        direction = Direction(position["direction"])
        exit_price = self.costs.exit_price(raw_price, direction, self.instrument)
        conversion = self.instrument.conversion_rate(raw_price)
        units = self.instrument.contract_size * position["lots"]
        exit_commission = self.costs.commission(position["lots"])
        net_pnl = (
            (exit_price - position["entry_price"])
            * direction.sign
            * units
            * conversion
            - position["entry_commission"]
            - exit_commission
        )
        gross = (raw_price - position["entry_raw_price"]) * direction.sign * units * conversion
        self.state.cash += (
            (exit_price - position["entry_price"]) * direction.sign * units * conversion
            - exit_commission
        )
        costs = gross - net_pnl
        equity_at_entry = float(position.get("equity_at_entry", self.state.cash))
        self.state.trades.append(
            PaperTrade(
                trade_id=len(self.state.trades) + 1,
                entry_time=position["entry_time"],
                exit_time=timestamp,
                direction=direction.value,
                lots=position["lots"],
                entry_price=position["entry_price"],
                exit_price=exit_price,
                net_pnl=net_pnl,
                costs=costs,
                exit_reason=reason,
                bars_held=max(1, self.state.bars_processed - int(position["entry_bar"]) + 1),
                return_pct=(net_pnl / equity_at_entry * 100.0) if equity_at_entry > 0 else 0.0,
            ).as_dict()
        )
        self.state.position = None
        self._risk.register_trade_result(net_pnl)
        self.state.equity = self.state.cash

    def _stop_or_target_hit(
        self, open_price: float, high_price: float, low_price: float
    ) -> tuple[float | None, str]:
        position = self.state.position
        if position is None:
            return None, ""
        direction = Direction(position["direction"])
        stop = float(position["stop_price"])
        target = position["take_profit_price"]
        if direction is Direction.LONG:
            if open_price <= stop:
                return open_price, "stop_gap"
            if target is not None and open_price >= target:
                return open_price, "take_profit_gap"
            if low_price <= stop:
                return stop, "stop_loss"
            if target is not None and high_price >= target:
                return target, "take_profit"
        else:
            if open_price >= stop:
                return open_price, "stop_gap"
            if target is not None and open_price <= target:
                return open_price, "take_profit_gap"
            if high_price >= stop:
                return stop, "stop_loss"
            if target is not None and low_price <= target:
                return target, "take_profit"
        return None, ""

    def summary(self) -> dict[str, Any]:
        state = self.state
        equity = pd.DataFrame(state.equity_curve)
        trades = pd.DataFrame(state.trades)
        metrics: dict[str, Any] = {}
        if not equity.empty:
            from trading_bot_lab.analytics import compute_metrics

            equity = equity.set_index("timestamp")
            trade_frame = trades.copy()
            if not trade_frame.empty:
                trade_frame = trade_frame.rename(
                    columns={"net_pnl": "net_pnl_account", "costs": "costs_account"}
                )
                trade_frame["gross_pnl_account"] = (
                    trade_frame["net_pnl_account"] + trade_frame["costs_account"]
                )
            metrics = compute_metrics(
                equity,
                trade_frame if not trade_frame.empty else pd.DataFrame(columns=["net_pnl_account"]),
                initial_capital=state.initial_capital,
                annualization_days=self.config.backtest.annualization_days,
            )
        return {
            "session_id": state.session_id,
            "symbol": state.symbol,
            "timeframe": state.timeframe,
            "strategy": state.strategy_name,
            "bars_processed": state.bars_processed,
            "equity": state.equity,
            "trades": len(state.trades),
            "return_pct": (state.equity / state.initial_capital - 1.0) * 100.0,
            "metrics": metrics,
            "blocked_entries": state.blocked_entries,
        }
