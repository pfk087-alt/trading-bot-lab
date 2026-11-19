"""Gestion de riesgo por operacion, diaria y global."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trading_bot_lab.config import RiskConfig
from trading_bot_lab.data.instruments import Instrument


@dataclass
class SizingResult:
    allowed: bool
    lots: float = 0.0
    risk_amount: float = 0.0
    risk_per_lot: float = 0.0
    notional: float = 0.0
    reason: str = ""


@dataclass
class RiskState:
    peak_equity: float
    day_start_equity: float
    current_day: object | None = None
    daily_realized_pnl: float = 0.0
    trades_today: int = 0
    blocked_entries: dict[str, int] = field(default_factory=dict)


class RiskManager:
    """Aplica riesgo por trade, perdida diaria, drawdown y exposure.

    No conoce brokers ni ordenes reales. El motor de backtesting y el modo
    paper trading usan esta misma clase.
    """

    def __init__(self, config: RiskConfig, initial_capital: float) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital debe ser > 0")
        self.config = config
        self.state = RiskState(
            peak_equity=float(initial_capital),
            day_start_equity=float(initial_capital),
        )

    def on_bar(self, timestamp: pd.Timestamp, equity: float) -> None:
        """Actualiza el dia y el maximo de equity. Llamar al inicio de cada barra."""
        day = pd.Timestamp(timestamp).date()
        if self.state.current_day != day:
            self.state.current_day = day
            self.state.day_start_equity = float(equity)
            self.state.daily_realized_pnl = 0.0
            self.state.trades_today = 0
        self.state.peak_equity = max(self.state.peak_equity, float(equity))

    def current_drawdown_pct(self, equity: float) -> float:
        peak = max(self.state.peak_equity, 1e-12)
        return (float(equity) / peak - 1.0) * 100.0

    def daily_pnl_pct(self) -> float:
        base = max(self.state.day_start_equity, 1e-12)
        return self.state.daily_realized_pnl / base * 100.0

    def max_drawdown_breached(self, equity: float) -> bool:
        return self.current_drawdown_pct(equity) <= -abs(self.config.max_drawdown_pct)

    def daily_loss_breached(self) -> bool:
        return self.daily_pnl_pct() <= -abs(self.config.max_daily_loss_pct)

    def can_open_trade(
        self,
        *,
        equity: float,
        open_positions: int,
        has_pending_order: bool = False,
    ) -> tuple[bool, str]:
        if has_pending_order:
            return self._block("orden_duplicada")
        if open_positions >= self.config.max_open_positions:
            return self._block("max_posiciones_simultaneas")
        if self.max_drawdown_breached(equity):
            return self._block("drawdown_maximo")
        if self.daily_loss_breached():
            return self._block("perdida_diaria_maxima")
        return True, ""

    def _block(self, reason: str) -> tuple[bool, str]:
        self.state.blocked_entries[reason] = self.state.blocked_entries.get(reason, 0) + 1
        return False, reason

    def size_position(
        self,
        *,
        equity: float,
        stop_distance: float,
        instrument: Instrument,
        price: float,
        open_positions: int = 0,
        has_pending_order: bool = False,
    ) -> SizingResult:
        """Calcula lotes para arriesgar el % configurado hasta el stop."""
        allowed, reason = self.can_open_trade(
            equity=equity,
            open_positions=open_positions,
            has_pending_order=has_pending_order,
        )
        if not allowed:
            return SizingResult(False, reason=reason)
        if stop_distance <= 0 or price <= 0:
            self.state.blocked_entries["stop_invalido"] = (
                self.state.blocked_entries.get("stop_invalido", 0) + 1
            )
            return SizingResult(False, reason="stop_invalido")

        conversion_rate = instrument.conversion_rate(price)
        risk_amount = float(equity) * self.config.risk_per_trade_pct / 100.0
        risk_per_lot = stop_distance * instrument.contract_size * conversion_rate
        if risk_per_lot <= 0:
            return SizingResult(False, reason="riesgo_por_lote_invalido")

        lots = instrument.round_lots(risk_amount / risk_per_lot)
        notional_per_lot = price * instrument.contract_size * conversion_rate
        max_notional = float(equity) * self.config.max_notional_exposure_pct / 100.0
        if notional_per_lot > 0:
            lots_by_exposure = instrument.round_lots(max_notional / notional_per_lot)
            lots = min(lots, lots_by_exposure)

        if lots < instrument.min_lot:
            self.state.blocked_entries["lote_minimo"] = (
                self.state.blocked_entries.get("lote_minimo", 0) + 1
            )
            return SizingResult(
                allowed=False,
                reason=(
                    "lote_minimo: el riesgo configurado es demasiado pequeno para el stop. "
                    "Aumente capital/riesgo o reduzca el stop."
                ),
            )

        return SizingResult(
            allowed=True,
            lots=lots,
            risk_amount=risk_amount,
            risk_per_lot=risk_per_lot,
            notional=lots * notional_per_lot,
        )

    def register_trade_result(self, net_pnl: float) -> None:
        self.state.daily_realized_pnl += float(net_pnl)
        self.state.trades_today += 1

    def summary(self) -> dict[str, object]:
        return {
            "peak_equity": self.state.peak_equity,
            "day_start_equity": self.state.day_start_equity,
            "daily_realized_pnl": self.state.daily_realized_pnl,
            "trades_today": self.state.trades_today,
            "blocked_entries": dict(self.state.blocked_entries),
        }
