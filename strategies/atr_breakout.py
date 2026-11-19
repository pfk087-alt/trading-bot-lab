"""Estrategia 4: breakout de volatilidad con ATR."""

from __future__ import annotations

import pandas as pd

from trading_bot_lab.indicators import atr, cross_above, cross_below, ema

from .base import ParameterSpec, Strategy, build_signal_frame


class ATRBreakoutStrategy(Strategy):
    key = "atr_breakout"
    name = "Breakout ATR"
    description = (
        "Entra cuando el movimiento de la barra supera un multiplo del ATR "
        "reciente, filtrando por tendencia. Sale al perder la EMA de salida."
    )
    params = (
        ParameterSpec("atr_period", "Periodo ATR", "int", 14, 2, 100, 1,
                      help="Periodo del ATR de Wilder."),
        ParameterSpec("atr_mult", "Multiplo ATR", "float", 1.5, 0.1, 10.0, 0.1,
                      help="Movimiento minimo en multiplos de ATR."),
        ParameterSpec("use_trend_filter", "Filtro de tendencia", "bool", True,
                      help="Exige alineacion con la EMA de tendencia."),
        ParameterSpec("trend_period", "EMA de tendencia", "int", 100, 10, 1000, 10,
                      help="Periodo de la EMA de tendencia."),
        ParameterSpec("exit_period", "EMA de salida", "int", 20, 2, 500, 1,
                      help="EMA usada como salida."),
        ParameterSpec("allow_short", "Permitir cortos", "bool", True,
                      help="Si se desactiva, solo opera al alza."),
    )

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate_data(data)
        close = data["close"]
        atr_values = atr(
            data["high"], data["low"], close, self.param("atr_period")
        )
        trend = ema(close, self.param("trend_period"))
        exit_ema = ema(close, self.param("exit_period"))

        # Se usa el ATR de la barra anterior para que el umbral sea conocido
        # antes de observar el movimiento de la barra actual.
        threshold = self.param("atr_mult") * atr_values.shift(1)
        move = close - close.shift(1)
        long_entry = move > threshold
        short_entry = move < -threshold
        if self.param("use_trend_filter"):
            long_entry &= close > trend
            short_entry &= close < trend

        allow_short = self.param("allow_short")
        return build_signal_frame(
            data.index,
            long_entry=long_entry,
            short_entry=short_entry if allow_short else False,
            long_exit=cross_below(close, exit_ema),
            short_exit=cross_above(close, exit_ema) if allow_short else False,
        )
