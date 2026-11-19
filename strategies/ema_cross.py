"""Estrategia 1: cruce de medias moviles exponenciales (EMA)."""

from __future__ import annotations

import pandas as pd

from trading_bot_lab.indicators import cross_above, cross_below, ema

from .base import ParameterSpec, Strategy, StrategyError, build_signal_frame


class EMACrossStrategy(Strategy):
    key = "ema_cross"
    name = "Cruce de EMA"
    description = (
        "Entra cuando la EMA rapida cruza a la lenta en la direccion de la "
        "tendencia de largo plazo (EMA de filtro). Sale en el cruce contrario."
    )
    params = (
        ParameterSpec("fast_period", "EMA rapida", "int", 20, 2, 200, 1,
                      help="Periodo de la media rapida."),
        ParameterSpec("slow_period", "EMA lenta", "int", 50, 5, 500, 1,
                      help="Periodo de la media lenta."),
        ParameterSpec("use_trend_filter", "Filtro de tendencia", "bool", True,
                      help="Exige que el precio este del lado correcto de la EMA de filtro."),
        ParameterSpec("trend_period", "EMA de filtro", "int", 200, 10, 1000, 10,
                      help="Periodo de la EMA de tendencia."),
        ParameterSpec("allow_short", "Permitir cortos", "bool", True,
                      help="Si se desactiva, solo opera al alza."),
    )

    def validate_parameters(self) -> None:
        if self.param("fast_period") >= self.param("slow_period"):
            raise StrategyError("fast_period debe ser menor que slow_period")

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate_data(data)
        close = data["close"]
        fast = ema(close, self.param("fast_period"))
        slow = ema(close, self.param("slow_period"))
        trend = ema(close, self.param("trend_period"))

        long_cross = cross_above(fast, slow)
        short_cross = cross_below(fast, slow)
        if self.param("use_trend_filter"):
            long_cross &= close > trend
            short_cross &= close < trend

        allow_short = self.param("allow_short")
        return build_signal_frame(
            data.index,
            long_entry=long_cross,
            short_entry=short_cross if allow_short else False,
            long_exit=cross_below(fast, slow),
            short_exit=cross_above(fast, slow) if allow_short else False,
        )
