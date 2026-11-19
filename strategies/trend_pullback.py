"""Estrategia 5: tendencia con retroceso (pullback) y recuperacion."""

from __future__ import annotations

import pandas as pd

from trading_bot_lab.indicators import cross_above, cross_below, ema, rsi

from .base import ParameterSpec, Strategy, build_signal_frame


class TrendPullbackStrategy(Strategy):
    key = "trend_pullback"
    name = "Tendencia + retroceso"
    description = (
        "En tendencia, espera un retroceso hacia la EMA de pullback y entra "
        "cuando el precio la recupera, opcionalmente con RSI debil. Sale al "
        "perder la EMA de tendencia."
    )
    params = (
        ParameterSpec("trend_period", "EMA de tendencia", "int", 200, 20, 1000, 10,
                      help="Filtro direccional de largo plazo."),
        ParameterSpec("pullback_period", "EMA de retroceso", "int", 50, 5, 300, 1,
                      help="Media que el precio debe tocar y recuperar."),
        ParameterSpec("touch_lookback", "Barras de toque", "int", 5, 1, 50, 1,
                      help="Ventana para considerar valido el retroceso."),
        ParameterSpec("use_rsi_filter", "Filtro RSI", "bool", True,
                      help="Exige RSI debil durante el retroceso."),
        ParameterSpec("rsi_period", "Periodo RSI", "int", 14, 2, 100, 1,
                      help="Periodo del RSI."),
        ParameterSpec("rsi_pullback", "RSI de retroceso", "float", 50.0, 5.0, 90.0, 1.0,
                      help="Umbral de RSI para validar el retroceso."),
        ParameterSpec("allow_short", "Permitir cortos", "bool", True,
                      help="Si se desactiva, solo opera al alza."),
    )

    def validate_parameters(self) -> None:
        if self.param("pullback_period") >= self.param("trend_period"):
            raise ValueError("pullback_period debe ser menor que trend_period")

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate_data(data)
        close = data["close"]
        trend = ema(close, self.param("trend_period"))
        pullback = ema(close, self.param("pullback_period"))
        rsi_values = rsi(close, self.param("rsi_period"))

        touch_lookback = self.param("touch_lookback")
        # El toque debe haber ocurrido antes de la barra de recuperacion.
        touched_long = (
            (data["low"] <= pullback)
            .rolling(touch_lookback, min_periods=1)
            .max()
            .astype(bool)
            .shift(1)
        )
        touched_short = (
            (data["high"] >= pullback)
            .rolling(touch_lookback, min_periods=1)
            .max()
            .astype(bool)
            .shift(1)
        )

        long_entry = (close > trend) & touched_long & cross_above(close, pullback)
        short_entry = (close < trend) & touched_short & cross_below(close, pullback)
        if self.param("use_rsi_filter"):
            long_entry &= rsi_values.shift(1) < self.param("rsi_pullback")
            short_entry &= rsi_values.shift(1) > (100.0 - self.param("rsi_pullback"))

        allow_short = self.param("allow_short")
        return build_signal_frame(
            data.index,
            long_entry=long_entry,
            short_entry=short_entry if allow_short else False,
            long_exit=cross_below(close, trend),
            short_exit=cross_above(close, trend) if allow_short else False,
        )
