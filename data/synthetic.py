"""Generador de datos SINTETICOS para pruebas y demostracion.

ADVERTENCIA: estos precios NO son datos de mercado. Se generan con un paseo
aleatorio reproducible y solo sirven para probar el software, nunca para
concluir que una estrategia es rentable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_bot_lab.types import Timeframe

from .validation import validate_ohlcv


def generate_synthetic_ohlcv(
    *,
    symbol: str = "SYNTH",
    timeframe: str | Timeframe = "1h",
    periods: int = 3000,
    start: str = "2020-01-01",
    start_price: float = 1.10,
    volatility: float = 0.0025,
    drift: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Genera OHLCV sintetico reproducible (GBM con ruido intrabarra).

    No representa ningun activo real. El atributo ``synthetic`` queda marcado
    en ``DataFrame.attrs`` para que el panel lo muestre con claridad.
    """
    if periods < 10:
        raise ValueError("periods debe ser >= 10")
    if volatility <= 0 or start_price <= 0:
        raise ValueError("volatility y start_price deben ser > 0")

    tf = Timeframe.parse(timeframe)
    rng = np.random.default_rng(seed)
    index = pd.date_range(
        start=pd.Timestamp(start, tz="UTC"), periods=periods, freq=tf.pandas_rule
    )

    shocks = rng.normal(drift, volatility, size=periods)
    closes = start_price * np.exp(np.cumsum(shocks))
    opens = np.empty(periods)
    opens[0] = start_price
    opens[1:] = closes[:-1]

    intrabar = np.abs(rng.normal(0, volatility * 0.55, size=periods))
    highs = np.maximum(opens, closes) * np.exp(intrabar)
    lows = np.minimum(opens, closes) * np.exp(-intrabar)
    volumes = rng.integers(500, 5000, size=periods).astype(float)

    data = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=index,
    )
    clean = validate_ohlcv(data, symbol=symbol, timeframe=tf.value)
    clean.attrs["synthetic"] = True
    clean.attrs["source"] = "generador sintetico (NO es dato de mercado)"
    return clean
