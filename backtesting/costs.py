"""Modelo de costes: spread, comision y slippage.

Convencion:
- ``spread_pips`` es el spread total de ida y vuelta. Se aplica la mitad en
  cada lado.
- ``slippage_pips`` es por lado.
- ``commission_per_lot_per_side`` se cobra en la divisa de la cuenta.
"""

from __future__ import annotations

from dataclasses import dataclass

from trading_bot_lab.config import CostConfig
from trading_bot_lab.data.instruments import Instrument
from trading_bot_lab.types import Direction


@dataclass(frozen=True)
class CostBreakdown:
    spread: float
    slippage: float
    commission: float

    @property
    def total(self) -> float:
        return self.spread + self.slippage + self.commission


class CostModel:
    def __init__(self, config: CostConfig) -> None:
        self.config = config

    def half_spread_price(self, instrument: Instrument) -> float:
        return self.config.spread_pips * instrument.pip_size / 2.0

    def slippage_price(self, instrument: Instrument) -> float:
        return self.config.slippage_pips * instrument.pip_size

    def entry_price(self, raw_price: float, direction: Direction, instrument: Instrument) -> float:
        penalty = self.half_spread_price(instrument) + self.slippage_price(instrument)
        return raw_price + penalty if direction is Direction.LONG else raw_price - penalty

    def exit_price(self, raw_price: float, direction: Direction, instrument: Instrument) -> float:
        # La salida de un largo es una venta (recibe el bid) y viceversa.
        penalty = self.half_spread_price(instrument) + self.slippage_price(instrument)
        return raw_price - penalty if direction is Direction.LONG else raw_price + penalty

    def commission(self, lots: float) -> float:
        return self.config.commission_per_lot_per_side * float(lots)

    def estimate_costs(
        self,
        *,
        instrument: Instrument,
        lots: float,
        conversion_rate: float = 1.0,
    ) -> CostBreakdown:
        """Coste total estimado de ida y vuelta en divisa de cuenta."""
        units_value = instrument.contract_size * float(lots) * float(conversion_rate)
        return CostBreakdown(
            spread=self.config.spread_pips * instrument.pip_size * units_value,
            slippage=2.0 * self.config.slippage_pips * instrument.pip_size * units_value,
            commission=2.0 * self.commission(lots),
        )
