"""Estrategia 3: ruptura de soportes y resistencias (Donchian)."""

from __future__ import annotations

import pandas as pd

from trading_bot_lab.indicators import donchian

from .base import ParameterSpec, Strategy, build_signal_frame


class SupportResistanceBreakoutStrategy(Strategy):
    key = "sr_breakout"
    name = "Ruptura S/R"
    description = (
        "Opera rupturas del maximo/minimo de las ultimas N barras, con un "
        "pequeno margen de confirmacion. Sale al volver al punto medio del canal."
    )
    params = (
        ParameterSpec("lookback", "Barras del canal", "int", 20, 3, 500, 1,
                      help="Ventana de maximos/minimos previos."),
        ParameterSpec("buffer_pct", "Margen de ruptura (%)", "float", 0.0, 0.0, 5.0, 0.05,
                      help="Porcentaje adicional sobre el nivel para confirmar la ruptura."),
        ParameterSpec("allow_short", "Permitir cortos", "bool", True,
                      help="Si se desactiva, solo opera rupturas al alza."),
    )

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate_data(data)
        close = data["close"]
        lookback = self.param("lookback")
        channels = donchian(data["high"], data["low"], lookback)

        # shift(1): el nivel de ruptura excluye la barra actual.
        upper = channels["dc_upper"].shift(1)
        lower = channels["dc_lower"].shift(1)
        mid = (upper + lower) / 2.0
        buffer = self.param("buffer_pct") / 100.0

        long_entry = close > upper * (1.0 + buffer)
        short_entry = close < lower * (1.0 - buffer)
        long_exit = close < mid
        short_exit = close > mid
        allow_short = self.param("allow_short")
        return build_signal_frame(
            data.index,
            long_entry=long_entry,
            short_entry=short_entry if allow_short else False,
            long_exit=long_exit,
            short_exit=short_exit if allow_short else False,
        )
