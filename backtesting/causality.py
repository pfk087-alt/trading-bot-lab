"""Verificacion de causalidad de estrategias.

Compara las senales calculadas con todo el historico contra las calculadas
truncando los datos en cada punto de control. Si una estrategia usa datos
futuros, las senales difieren y se lanza un error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_bot_lab.strategies.base import Strategy

SIGNAL_COLUMNS = ["long_entry", "short_entry", "long_exit", "short_exit"]


class CausalityError(AssertionError):
    """La estrategia usa informacion futura."""


def verify_causality(
    strategy: Strategy,
    data: pd.DataFrame,
    *,
    checks: int = 6,
    min_history: int = 30,
) -> None:
    if checks < 1:
        raise ValueError("checks debe ser >= 1")
    if len(data) < min_history + 2:
        raise ValueError("Datos insuficientes para verificar causalidad")

    full = strategy.generate_signals(data)
    positions = np.linspace(min_history, len(data) - 1, num=min(checks, len(data) - min_history), dtype=int)

    for position in positions:
        subset = data.iloc[: position + 1]
        partial = strategy.generate_signals(subset)
        expected = full.iloc[position][SIGNAL_COLUMNS].astype(bool)
        actual = partial.iloc[-1][SIGNAL_COLUMNS].astype(bool)
        if not expected.equals(actual):
            differences = [
                column for column in SIGNAL_COLUMNS if bool(expected[column]) != bool(actual[column])
            ]
            raise CausalityError(
                f"Look-ahead detectado en {strategy.key} en {subset.index[-1]} "
                f"(campos: {differences}). Las senales deben ser causales."
            )
