"""Remuestreo temporal y deteccion de temporalidad."""

from __future__ import annotations

from typing import Any

import pandas as pd

from trading_bot_lab.types import Timeframe

from .validation import DataValidationError, validate_ohlcv


def infer_timeframe(df: pd.DataFrame) -> Timeframe | None:
    """Infiere la temporalidad a partir de la mediana de diferencias."""
    if len(df.index) < 2:
        return None
    deltas = df.index.to_series().diff().dropna()
    if deltas.empty:
        return None
    minutes = deltas.median().total_seconds() / 60.0
    if minutes <= 0:
        return None
    candidates = sorted(Timeframe, key=lambda tf: tf.minutes)
    match = min(candidates, key=lambda tf: abs(tf.minutes - minutes))
    if abs(match.minutes - minutes) <= max(1.0, minutes * 0.05):
        return match
    return None


def resample_ohlcv(
    df: pd.DataFrame,
    target: str | Timeframe,
    *,
    symbol: str = "",
) -> pd.DataFrame:
    """Agrega barras a una temporalidad mayor.

    No se permite remuestrear al alza (por ejemplo, de 1h a 5m) para evitar
    fabricar informacion intra-barra.
    """
    tf = Timeframe.parse(target)
    clean = validate_ohlcv(df, symbol=symbol)
    source = infer_timeframe(clean)
    if source is None:
        raise DataValidationError(
            "No se pudo inferir la temporalidad de origen. Revise los timestamps."
        )
    if tf.minutes < source.minutes:
        raise DataValidationError(
            f"No se puede remuestrear de {source.value} a {tf.value}: "
            "solo se permite agregar a temporalidades mayores."
        )
    if tf.minutes == source.minutes:
        result = clean.copy()
        result.attrs["timeframe"] = tf.value
        return result

    aggregated = clean.resample(tf.pandas_rule, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    aggregated = aggregated.dropna(subset=["open", "high", "low", "close"])
    aggregated["volume"] = aggregated["volume"].fillna(0.0)
    if aggregated.empty:
        raise DataValidationError("El remuestreo produjo un DataFrame vacio.")

    aggregated.attrs["symbol"] = symbol
    aggregated.attrs["timeframe"] = tf.value
    aggregated.attrs["validation_warnings"] = []
    return aggregated


def timeframe_summary(df: pd.DataFrame) -> dict[str, Any]:
    tf = infer_timeframe(df)
    return {
        "inferred_timeframe": tf.value if tf else None,
        "rows": int(len(df)),
        "start": df.index[0] if len(df) else None,
        "end": df.index[-1] if len(df) else None,
    }
