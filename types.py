"""Tipos compartidos por todos los modulos."""

from __future__ import annotations

from enum import Enum
from typing import Iterable


class Timeframe(str, Enum):
    """Temporalidades soportadas por el laboratorio."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"

    @property
    def minutes(self) -> int:
        return {
            Timeframe.M1: 1,
            Timeframe.M5: 5,
            Timeframe.M15: 15,
            Timeframe.M30: 30,
            Timeframe.H1: 60,
            Timeframe.H4: 240,
        }[self]

    @property
    def pandas_rule(self) -> str:
        """Regla de remuestreo de pandas."""
        return {
            Timeframe.M1: "1min",
            Timeframe.M5: "5min",
            Timeframe.M15: "15min",
            Timeframe.M30: "30min",
            Timeframe.H1: "1h",
            Timeframe.H4: "4h",
        }[self]

    @property
    def bars_per_day(self) -> int:
        """Barras por dia natural (mercado 24h). Aproximacion documentada."""
        return 1440 // self.minutes

    @classmethod
    def parse(cls, value: "str | Timeframe") -> "Timeframe":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        aliases = {
            "1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m",
            "60min": "1h", "60m": "1h", "240m": "4h", "240min": "4h",
        }
        normalized = aliases.get(normalized, normalized)
        for tf in cls:
            if tf.value == normalized:
                return tf
        valid = ", ".join(tf.value for tf in cls)
        raise ValueError(f"Temporalidad no soportada: {value!r}. Valores validos: {valid}")

    @classmethod
    def values(cls) -> Iterable[str]:
        return (tf.value for tf in cls)


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        return 1 if self is Direction.LONG else -1

    @property
    def opposite(self) -> "Direction":
        return Direction.SHORT if self is Direction.LONG else Direction.LONG


SIGNAL_COLUMNS = ("long_entry", "short_entry", "long_exit", "short_exit")
