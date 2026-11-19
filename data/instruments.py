"""Especificaciones de instrumentos financieros soportados."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from trading_bot_lab.config import PROJECT_ROOT, resolve_project_path

DEFAULT_INSTRUMENTS_PATH = PROJECT_ROOT / "config" / "instruments.yaml"


class InstrumentError(ValueError):
    """Especificacion de instrumento invalida."""


@dataclass(frozen=True)
class Instrument:
    """Contrato del instrumento usado para dimensionar posiciones y costes."""

    symbol: str
    name: str
    asset_class: str
    quote_currency: str
    pip_size: float
    contract_size: float
    min_lot: float = 0.01
    lot_step: float = 0.01
    max_lot: float = 100.0
    account_rate_mode: str = "one"
    fixed_account_rate: float = 1.0

    def __post_init__(self) -> None:
        if self.pip_size <= 0 or self.contract_size <= 0:
            raise InstrumentError(f"{self.symbol}: pip_size y contract_size deben ser > 0")
        if self.min_lot <= 0 or self.lot_step <= 0 or self.max_lot < self.min_lot:
            raise InstrumentError(f"{self.symbol}: limites de lote invalidos")
        if self.account_rate_mode not in {"one", "inverse", "fixed"}:
            raise InstrumentError(
                f"{self.symbol}: account_rate_mode debe ser one|inverse|fixed"
            )
        if self.account_rate_mode == "fixed" and self.fixed_account_rate <= 0:
            raise InstrumentError(f"{self.symbol}: fixed_account_rate debe ser > 0")

    @property
    def is_forex(self) -> bool:
        return self.asset_class == "forex"

    def conversion_rate(self, price: float) -> float:
        """Factor para convertir 1 unidad de divisa cotizada a divisa de cuenta."""
        if self.account_rate_mode == "one":
            return 1.0
        if self.account_rate_mode == "inverse":
            if price <= 0:
                raise InstrumentError(f"{self.symbol}: precio invalido para conversion inversa")
            return 1.0 / price
        return self.fixed_account_rate

    def pip_value_per_lot(self) -> float:
        """Valor de 1 pip por lote, en divisa cotizada."""
        return self.pip_size * self.contract_size

    def round_lots(self, lots: float) -> float:
        """Redondea hacia abajo al step de lote y aplica maximos."""
        if lots <= 0:
            return 0.0
        steps = int(lots / self.lot_step + 1e-9)
        rounded = steps * self.lot_step
        rounded = min(rounded, self.max_lot)
        return round(rounded, 8)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "asset_class": self.asset_class,
            "quote_currency": self.quote_currency,
            "pip_size": self.pip_size,
            "contract_size": self.contract_size,
            "min_lot": self.min_lot,
            "lot_step": self.lot_step,
            "max_lot": self.max_lot,
            "account_rate_mode": self.account_rate_mode,
            "fixed_account_rate": self.fixed_account_rate,
        }


def load_instruments(path: str | Path | None = None) -> dict[str, Instrument]:
    target = resolve_project_path(path) if path else DEFAULT_INSTRUMENTS_PATH
    if not target.exists():
        raise InstrumentError(f"No existe el archivo de instrumentos: {target}")
    with target.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise InstrumentError("instruments.yaml debe contener un objeto YAML")

    instruments: dict[str, Instrument] = {}
    for symbol, spec in raw.items():
        if not isinstance(spec, dict):
            raise InstrumentError(f"Instrumento {symbol}: especificacion invalida")
        instruments[str(symbol).upper()] = Instrument(symbol=str(symbol).upper(), **spec)
    if not instruments:
        raise InstrumentError("No hay instrumentos definidos")
    return instruments


def list_instruments() -> dict[str, Instrument]:
    return load_instruments()


def get_instrument(symbol: str, path: str | Path | None = None) -> Instrument:
    instruments = load_instruments(path)
    key = symbol.strip().upper()
    if key not in instruments:
        raise InstrumentError(
            f"Instrumento no soportado: {symbol}. Disponibles: {', '.join(instruments)}"
        )
    return instruments[key]
