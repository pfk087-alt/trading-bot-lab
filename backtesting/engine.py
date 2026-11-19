"""Motor de backtesting event-driven con ejecucion en la apertura siguiente.

Garantias principales:
- Las senales se calculan con el cierre de la barra t y se ejecutan en la
  apertura de t+1 (sin look-ahead).
- Stops y take profits se evaluan con high/low de la barra; si ambos se tocan
  en la misma barra se asume el stop (criterio pesimista).
- Spread, slippage y comision se aplican en cada operacion.
- El riesgo se dimensiona por distancia al stop y se aplican limites diarios
  y de drawdown.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from trading_bot_lab.analytics.metrics import compute_metrics
from trading_bot_lab.config import AppConfig
from trading_bot_lab.data.instruments import Instrument, get_instrument
from trading_bot_lab.data.validation import validate_ohlcv
from trading_bot_lab.indicators import atr
from trading_bot_lab.risk import RiskManager
from trading_bot_lab.strategies.base import Strategy, StrategyError
from trading_bot_lab.types import Direction, Timeframe

from .costs import CostModel
from .models import BacktestError, BacktestResult, PendingOrder, Position, Trade

SIGNAL_COLUMNS = ["long_entry", "short_entry", "long_exit", "short_exit"]


class BacktestEngine:
    """Simulador de una estrategia sobre un unico instrumento."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.costs = CostModel(config.costs)

    # ------------------------------------------------------------------ API
    def run(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        *,
        symbol: str,
        timeframe: str | Timeframe | None = None,
        instrument: Instrument | None = None,
        warmup_bars: int | None = None,
        run_id: str | None = None,
    ) -> BacktestResult:
        tf = Timeframe.parse(timeframe or self.config.backtest.default_timeframe)
        instrument = instrument or get_instrument(symbol)
        clean = validate_ohlcv(data, symbol=symbol, timeframe=tf.value)
        warmup = self.config.execution.warmup_bars if warmup_bars is None else int(warmup_bars)
        if warmup < 0:
            raise BacktestError("warmup_bars no puede ser negativo")
        if len(clean) <= warmup + 2:
            raise BacktestError(
                f"Datos insuficientes: {len(clean)} barras y warmup={warmup}. "
                "Aporte mas historico o reduzca warmup_bars."
            )

        try:
            signals = strategy.generate_signals(clean)
        except StrategyError:
            raise
        except Exception as exc:  # pragma: no cover - proteccion defensiva
            raise BacktestError(f"Error generando senales: {exc}") from exc
        signals = self._validate_signals(signals, clean.index)

        atr_values = atr(
            clean["high"], clean["low"], clean["close"], self.config.execution.atr_period
        ).to_numpy(dtype=float)

        initial = float(self.config.account.initial_capital)
        cash = initial
        position: Position | None = None
        pending: PendingOrder | None = None
        trades: list[Trade] = []
        equity_rows: list[dict[str, Any]] = []
        risk = RiskManager(self.config.risk, initial)
        diagnostics: dict[str, Any] = {
            "blocked_entries": {},
            "skipped_invalid_stop": 0,
            "bars_total": int(len(clean)),
            "warmup_bars": warmup,
            "open_position_at_end": False,
        }
        warnings: list[str] = []
        peak_equity = initial
        mark_equity = initial

        index = clean.index
        opens = clean["open"].to_numpy(dtype=float)
        highs = clean["high"].to_numpy(dtype=float)
        lows = clean["low"].to_numpy(dtype=float)
        closes = clean["close"].to_numpy(dtype=float)
        long_entry = signals["long_entry"].to_numpy(dtype=bool)
        short_entry = signals["short_entry"].to_numpy(dtype=bool)
        long_exit = signals["long_exit"].to_numpy(dtype=bool)
        short_exit = signals["short_exit"].to_numpy(dtype=bool)

        for i in range(len(clean)):
            timestamp = index[i]
            open_price = opens[i]
            high_price = highs[i]
            low_price = lows[i]
            close_price = closes[i]
            atr_value = atr_values[i]

            risk.on_bar(timestamp, mark_equity)

            # 1) Ejecutar la orden pendiente en la apertura de esta barra.
            if pending is not None:
                order = pending
                pending = None
                if order.kind in {"exit", "reverse"} and position is not None:
                    reason = "reverse" if order.kind == "reverse" else order.reason
                    trade, cash = self._close_position(
                        position, timestamp, open_price, reason, cash, i, instrument, len(trades) + 1
                    )
                    trades.append(trade)
                    risk.register_trade_result(trade.net_pnl_account)
                    position = None

                if order.kind in {"entry", "reverse"} and position is None:
                    position, cash = self._try_enter(
                        direction=order.direction,
                        stop_distance=order.stop_distance,
                        take_profit_distance=order.take_profit_distance,
                        timestamp=timestamp,
                        bar_index=i,
                        raw_price=open_price,
                        cash=cash,
                        risk=risk,
                        instrument=instrument,
                        diagnostics=diagnostics,
                    )

            # 2) Actualizar recorrido favorable/adverso y comprobar stops.
            if position is not None:
                if position.is_long():
                    position.mae = min(position.mae, low_price - position.entry_price)
                    position.mfe = max(position.mfe, high_price - position.entry_price)
                else:
                    position.mae = min(position.mae, position.entry_price - high_price)
                    position.mfe = max(position.mfe, position.entry_price - low_price)

                exit_raw, exit_reason = self._stop_or_target_hit(position, open_price, high_price, low_price)
                if exit_raw is not None:
                    trade, cash = self._close_position(
                        position, timestamp, exit_raw, exit_reason, cash, i, instrument, len(trades) + 1
                    )
                    trades.append(trade)
                    risk.register_trade_result(trade.net_pnl_account)
                    position = None

            # 3) Marcar a mercado al cierre y registrar la curva de capital.
            if position is not None:
                mark_price = self.costs.exit_price(close_price, position.direction, instrument)
                conversion = instrument.conversion_rate(close_price)
                unrealized = (
                    (mark_price - position.entry_price)
                    * position.direction.sign
                    * instrument.contract_size
                    * position.lots
                    * conversion
                )
                equity = cash + unrealized - self.costs.commission(position.lots)
            else:
                unrealized = 0.0
                equity = cash

            peak_equity = max(peak_equity, equity)
            drawdown = equity / peak_equity - 1.0
            risk.state.peak_equity = max(risk.state.peak_equity, equity)
            equity_rows.append(
                {
                    "timestamp": timestamp,
                    "equity": equity,
                    "cash": cash,
                    "unrealized": unrealized,
                    "drawdown": drawdown,
                    "in_position": position is not None,
                }
            )
            mark_equity = equity

            # 4) Proteccion de drawdown maximo: cierre inmediato al cierre.
            if (
                position is not None
                and self.config.risk.close_on_max_drawdown
                and risk.max_drawdown_breached(equity)
            ):
                trade, cash = self._close_position(
                    position, timestamp, close_price, "max_drawdown", cash, i, instrument, len(trades) + 1
                )
                trades.append(trade)
                risk.register_trade_result(trade.net_pnl_account)
                position = None
                equity = cash
                equity_rows[-1].update(
                    {"equity": equity, "cash": cash, "unrealized": 0.0, "in_position": False}
                )
                peak_equity = max(peak_equity, equity)
                equity_rows[-1]["drawdown"] = equity / peak_equity - 1.0
                mark_equity = equity

            # 5) Trailing stop (basado en el cierre, efectivo en barras siguientes).
            if (
                position is not None
                and self.config.execution.trailing_stop_atr_mult > 0
                and np.isfinite(atr_value)
            ):
                distance = self.config.execution.trailing_stop_atr_mult * atr_value
                if position.is_long():
                    position.stop_price = max(position.stop_price, close_price - distance)
                else:
                    position.stop_price = min(position.stop_price, close_price + distance)

            # 6) Decidir la orden para la proxima apertura usando el cierre actual.
            if i >= warmup and i < len(clean) - 1:
                desired: str | None = None
                direction: Direction | None = None
                reason = "signal"
                if position is not None:
                    if position.is_long():
                        if short_entry[i]:
                            desired, direction = "reverse", Direction.SHORT
                        elif long_exit[i]:
                            desired, direction = "exit", Direction.LONG
                    else:
                        if long_entry[i]:
                            desired, direction = "reverse", Direction.LONG
                        elif short_exit[i]:
                            desired, direction = "exit", Direction.SHORT
                else:
                    if long_entry[i] and not short_entry[i]:
                        desired, direction = "entry", Direction.LONG
                    elif short_entry[i] and not long_entry[i]:
                        desired, direction = "entry", Direction.SHORT

                if (
                    desired is None
                    and position is not None
                    and self.config.execution.max_bars_in_trade > 0
                    and (i - position.entry_bar_index + 1) >= self.config.execution.max_bars_in_trade
                ):
                    desired, direction, reason = "exit", position.direction, "time_stop"

                if desired == "exit" and position is not None:
                    pending = PendingOrder("exit", position.direction, timestamp, reason=reason)
                elif desired in {"entry", "reverse"} and direction is not None:
                    stop_distance = self._stop_distance(close_price, atr_value, instrument)
                    take_profit_distance = self._take_profit_distance(
                        close_price, atr_value, stop_distance, instrument
                    )
                    if stop_distance is None or stop_distance <= 0:
                        diagnostics["skipped_invalid_stop"] += 1
                    else:
                        pending = PendingOrder(
                            "entry" if desired == "entry" else "reverse",
                            direction,
                            timestamp,
                            stop_distance=stop_distance,
                            take_profit_distance=take_profit_distance or 0.0,
                            reason=reason,
                        )

        # 7) Cierre final de la posicion abierta.
        if position is not None:
            if self.config.execution.close_at_end:
                last_i = len(clean) - 1
                trade, cash = self._close_position(
                    position, index[last_i], closes[last_i], "end_of_data", cash, last_i,
                    instrument, len(trades) + 1,
                )
                trades.append(trade)
                risk.register_trade_result(trade.net_pnl_account)
                position = None
                equity_rows[-1].update(
                    {"equity": cash, "cash": cash, "unrealized": 0.0, "in_position": False}
                )
                peak = max(float(row["equity"]) for row in equity_rows)
                equity_rows[-1]["drawdown"] = cash / peak - 1.0
            else:
                diagnostics["open_position_at_end"] = True
                warnings.append("Quedo una posicion abierta al final; no se cerro por configuracion.")

        equity_curve = pd.DataFrame(equity_rows).set_index("timestamp")
        equity_curve.index.name = "timestamp"
        equity_curve["equity"] = equity_curve["equity"].astype(float)
        equity_curve["cash"] = equity_curve["cash"].astype(float)
        equity_curve["unrealized"] = equity_curve["unrealized"].astype(float)
        equity_curve["drawdown"] = equity_curve["drawdown"].astype(float)
        equity_curve["in_position"] = equity_curve["in_position"].astype(bool)
        equity_curve.attrs["symbol"] = symbol

        if trades:
            trades_frame = pd.DataFrame([trade.as_dict() for trade in trades], columns=Trade.columns())
        else:
            trades_frame = pd.DataFrame(columns=Trade.columns())

        final_equity = float(equity_curve["equity"].iloc[-1])
        metrics = compute_metrics(
            equity_curve,
            trades_frame,
            initial_capital=initial,
            annualization_days=self.config.backtest.annualization_days,
            bars_per_year=Timeframe.parse(tf).bars_per_day * 252,
        )
        diagnostics["blocked_entries"] = dict(risk.state.blocked_entries)
        diagnostics["trades_executed"] = int(len(trades_frame))
        diagnostics["final_equity"] = final_equity
        diagnostics["total_costs"] = float(trades_frame["costs_account"].sum()) if not trades_frame.empty else 0.0

        if trades_frame.empty:
            warnings.append(
                "No se ejecuto ninguna operacion. Revise warmup, filtros, tamano de lote y costes."
            )
        if len(clean) < warmup + 500:
            warnings.append("Muestra corta: los resultados pueden no ser estadisticamente robustos.")
        if instrument.account_rate_mode == "fixed" and instrument.quote_currency != self.config.app.base_currency:
            warnings.append(
                f"Conversion {instrument.quote_currency}->{self.config.app.base_currency} "
                f"mediante tasa fija {instrument.fixed_account_rate}. Es una aproximacion."
            )
        if len(trades_frame) < self.config.backtest.min_trades_for_ranking:
            warnings.append(
                f"Menos de {self.config.backtest.min_trades_for_ranking} operaciones: "
                "no usar para concluir que existe ventaja estadistica."
            )

        return BacktestResult(
            run_id=run_id or uuid.uuid4().hex[:12],
            symbol=symbol.upper(),
            timeframe=tf.value,
            strategy_key=strategy.key,
            strategy_name=strategy.name,
            params=dict(strategy.parameters),
            settings=asdict(self.config),
            instrument=instrument,
            initial_capital=initial,
            final_equity=final_equity,
            equity_curve=equity_curve,
            trades=trades_frame,
            metrics=metrics,
            signals=signals,
            data=clean,
            start=index[0],
            end=index[-1],
            warnings=warnings,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _validate_signals(signals: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
        missing = set(SIGNAL_COLUMNS) - set(signals.columns)
        if missing:
            raise BacktestError(f"La estrategia no devolvio las columnas: {sorted(missing)}")
        if not signals.index.equals(index):
            raise BacktestError("El indice de senales no coincide con el de datos")
        out = signals[SIGNAL_COLUMNS].fillna(False).astype(bool)
        return out

    def _try_enter(
        self,
        *,
        direction: Direction,
        stop_distance: float,
        take_profit_distance: float,
        timestamp: pd.Timestamp,
        bar_index: int,
        raw_price: float,
        cash: float,
        risk: RiskManager,
        instrument: Instrument,
        diagnostics: dict[str, Any],
    ) -> tuple[Position | None, float]:
        sizing = risk.size_position(
            equity=cash,
            stop_distance=stop_distance,
            instrument=instrument,
            price=raw_price,
            open_positions=0,
            has_pending_order=False,
        )
        if not sizing.allowed:
            key = sizing.reason.split(":")[0]
            diagnostics["blocked_entries"][key] = diagnostics["blocked_entries"].get(key, 0) + 1
            return None, cash
        entry_price = self.costs.entry_price(raw_price, direction, instrument)
        conversion = instrument.conversion_rate(raw_price)
        commission = self.costs.commission(sizing.lots)
        stop_price = entry_price - direction.sign * stop_distance
        take_profit_price = (
            entry_price + direction.sign * take_profit_distance
            if take_profit_distance and take_profit_distance > 0
            else None
        )
        position = Position(
            direction=direction,
            lots=sizing.lots,
            entry_time=timestamp,
            entry_bar_index=bar_index,
            entry_raw_price=raw_price,
            entry_price=entry_price,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            entry_commission=commission,
            entry_conversion_rate=conversion,
            equity_at_entry=cash,
            stop_distance=stop_distance,
            initial_stop_price=stop_price,
        )
        cash -= commission
        return position, cash

    def _close_position(
        self,
        position: Position,
        timestamp: pd.Timestamp,
        raw_price: float,
        reason: str,
        cash: float,
        bar_index: int,
        instrument: Instrument,
        trade_id: int,
    ) -> tuple[Trade, float]:
        exit_price = self.costs.exit_price(raw_price, position.direction, instrument)
        conversion = instrument.conversion_rate(raw_price)
        units = instrument.contract_size * position.lots
        sign = position.direction.sign
        gross_quote = (raw_price - position.entry_raw_price) * sign * units
        gross_account = gross_quote * conversion
        exit_commission = self.costs.commission(position.lots)
        net_pnl = (
            (exit_price - position.entry_price) * sign * units * conversion
            - position.entry_commission
            - exit_commission
        )
        cash += (exit_price - position.entry_price) * sign * units * conversion - exit_commission
        costs_account = gross_account - net_pnl
        trade = Trade(
            trade_id=trade_id,
            symbol=instrument.symbol,
            direction=position.direction.value,
            entry_time=position.entry_time,
            exit_time=timestamp,
            entry_price_raw=position.entry_raw_price,
            entry_price=position.entry_price,
            exit_price_raw=raw_price,
            exit_price=exit_price,
            lots=position.lots,
            gross_pnl_quote=gross_quote,
            gross_pnl_account=gross_account,
            costs_account=costs_account,
            net_pnl_account=net_pnl,
            return_pct=(net_pnl / position.equity_at_entry * 100.0)
            if position.equity_at_entry > 0
            else 0.0,
            exit_reason=reason,
            bars_held=max(1, bar_index - position.entry_bar_index + 1),
            stop_price=position.stop_price,
            initial_stop_price=position.initial_stop_price,
            take_profit_price=position.take_profit_price,
            mae=position.mae,
            mfe=position.mfe,
            entry_conversion_rate=position.entry_conversion_rate,
            exit_conversion_rate=conversion,
        )
        return trade, cash

    @staticmethod
    def _stop_or_target_hit(
        position: Position,
        open_price: float,
        high_price: float,
        low_price: float,
    ) -> tuple[float | None, str]:
        stop = position.stop_price
        target = position.take_profit_price
        if position.is_long():
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

    def _stop_distance(
        self, close_price: float, atr_value: float, instrument: Instrument
    ) -> float | None:
        mode = self.config.execution.stop_mode
        value = self.config.execution.stop_value
        if mode == "none":
            return None
        if mode == "atr":
            if not np.isfinite(atr_value) or atr_value <= 0:
                return None
            distance = atr_value * value
        elif mode == "pips":
            distance = value * instrument.pip_size
        elif mode == "percent":
            distance = close_price * value / 100.0
        else:
            return None
        return float(distance) if np.isfinite(distance) and distance > 0 else None

    def _take_profit_distance(
        self,
        close_price: float,
        atr_value: float,
        stop_distance: float | None,
        instrument: Instrument,
    ) -> float | None:
        mode = self.config.execution.take_profit_mode
        value = self.config.execution.take_profit_value
        if mode == "none":
            return None
        if mode == "rr":
            return stop_distance * value if stop_distance and value > 0 else None
        if mode == "atr":
            if not np.isfinite(atr_value) or atr_value <= 0 or value <= 0:
                return None
            return float(atr_value * value)
        if mode == "pips":
            return float(value * instrument.pip_size) if value > 0 else None
        if mode == "percent":
            return float(close_price * value / 100.0) if value > 0 else None
        return None
