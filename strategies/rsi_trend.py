"""Estrategia 2: RSI con confirmacion de tendencia."""

from __future__ import annotations

import pandas as pd

from trading_bot_lab.indicators import cross_above, cross_below, ema, rsi

from .base import ParameterSpec, Strategy, StrategyError, build_signal_frame


class RSITrendStrategy(Strategy):
    key = "rsi_trend"
    name = "RSI + tendencia"
    description = (
        "Compra cuando el RSI sale de sobreventa dentro de una tendencia alcista "
        "y vende cuando sale de sobrecompra dentro de una tendencia bajista. "
        "Salida al alcanzar el nivel neutro."
    )
    params = (
        ParameterSpec("rsi_period", "Periodo RSI", "int", 14, 2, 100, 1,
                      help="Periodo del RSI de Wilder."),
        ParameterSpec("oversold", "Sobreventa", "float", 30.0, 1.0, 49.0, 1.0,
                      help="Nivel de sobreventa."),
        ParameterSpec("overbought", "Sobrecompra", "float", 70.0, 51.0, 99.0, 1.0,
                      help="Nivel de sobrecompra."),
        ParameterSpec("exit_level", "Nivel de salida", "float", 50.0, 1.0, 99.0, 1.0,
                      help="Salida al cruzar este nivel neutro."),
        ParameterSpec("use_trend_filter", "Filtro de tendencia", "bool", True,
                      help="Exige alineacion con la EMA de tendencia."),
        ParameterSpec("trend_period", "EMA de filtro", "int", 200, 10, 1000, 10,
                      help="Periodo de la EMA de tendencia."),
        ParameterSpec("allow_short", "Permitir cortos", "bool", True,
                      help="Si se desactiva, solo opera al alza."),
    )

    def validate_parameters(self) -> None:
        if self.param("oversold") >= self.param("overbought"):
            raise StrategyError("oversold debe ser menor que overbought")
        if not self.param("oversold") < self.param("exit_level") < self.param("overbought"):
            raise StrategyError("exit_level debe estar entre oversold y overbought")

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate_data(data)
        close = data["close"]
        rsi_values = rsi(close, self.param("rsi_period"))
        trend = ema(close, self.param("trend_period"))

        long_entry = cross_above(rsi_values, self.param("oversold"))
        short_entry = cross_below(rsi_values, self.param("overbought"))
        if self.param("use_trend_filter"):
            long_entry &= close > trend
            short_entry &= close < trend

        long_exit = cross_above(rsi_values, self.param("exit_level"))
        short_exit = cross_below(rsi_values, self.param("exit_level"))
        allow_short = self.param("allow_short")
        return build_signal_frame(
            data.index,
            long_entry=long_entry,
            short_entry=short_entry if allow_short else False,
            long_exit=long_exit,
            short_exit=short_exit if allow_short else False,
        )
