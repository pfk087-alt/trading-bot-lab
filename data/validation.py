"""Validacion estricta de datos OHLCV.

Reglas:
- Indice temporal unico, ordenado y en UTC.
- Precios finitos y positivos.
- high >= max(open, close) y low <= min(open, close).
- high >= low.
- Volumen no negativo (si falta, se rellena con 0).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("trading_bot_lab.data.validation")

REQUIRED_COLUMNS = ("open", "high", "low", "close")
COLUMN_ALIASES = {
    "date": "timestamp",
    "datetime": "timestamp",
    "time": "timestamp",
    "bar_time": "timestamp",
    "timestamp": "timestamp",
    "tickvol": "volume",
    "tick_volume": "volume",
    "vol": "volume",
    "volume": "volume",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
}


class DataValidationError(ValueError):
    """Los datos no cumplen el contrato OHLCV."""


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        renamed[col] = COLUMN_ALIASES.get(key, key)
    out = df.rename(columns=renamed)
    if out.columns.duplicated().any():
        dupes = out.columns[out.columns.duplicated()].tolist()
        raise DataValidationError(f"Columnas duplicadas tras normalizar: {dupes}")
    return out


def validate_ohlcv(
    df: pd.DataFrame,
    *,
    symbol: str = "",
    timeframe: str | None = None,
    min_rows: int = 2,
    allow_duplicates: bool = True,
) -> pd.DataFrame:
    """Valida y normaliza un DataFrame OHLCV. Lanza DataValidationError si es invalido."""
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("Se esperaba un pandas.DataFrame.")
    if df.empty:
        raise DataValidationError("El DataFrame esta vacio.")

    out = _normalize_columns(df.copy())
    warnings: list[str] = []

    if not isinstance(out.index, pd.DatetimeIndex):
        if "timestamp" not in out.columns:
            raise DataValidationError(
                "Falta una columna de fecha (date/datetime/time/timestamp) o un indice temporal."
            )
        out = out.set_index("timestamp")
    elif "timestamp" in out.columns:
        out = out.drop(columns=["timestamp"])

    try:
        index = pd.to_datetime(out.index, errors="coerce", utc=True)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"No se pudo interpretar el indice temporal: {exc}") from exc

    nat_mask = pd.isna(index)
    if nat_mask.any():
        raise DataValidationError(
            f"Hay {int(nat_mask.sum())} fechas invalidas en el indice temporal."
        )
    out.index = index
    out.index.name = "timestamp"

    if out.index.has_duplicates:
        duplicate_count = int(out.index.duplicated(keep="last").sum())
        if not allow_duplicates:
            raise DataValidationError(f"Hay {duplicate_count} timestamps duplicados.")
        warnings.append(f"Se eliminaron {duplicate_count} timestamps duplicados (se conserva el ultimo).")
        out = out[~out.index.duplicated(keep="last")]

    out = out.sort_index()

    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise DataValidationError(f"Faltan columnas OHLC obligatorias: {missing}")

    if "volume" not in out.columns:
        out["volume"] = 0.0
        warnings.append("No habia columna de volumen; se relleno con 0.")

    out = out[[*REQUIRED_COLUMNS, "volume"]]
    for col in out.columns:
        converted = pd.to_numeric(out[col], errors="coerce")
        if converted.isna().any():
            count = int(converted.isna().sum())
            raise DataValidationError(f"La columna '{col}' contiene {count} valores no numericos o vacios.")
        out[col] = converted.astype(float)

    values = out.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DataValidationError("Los datos contienen valores infinitos.")

    if (out[list(REQUIRED_COLUMNS)] <= 0).any().any():
        bad = int((out[list(REQUIRED_COLUMNS)] <= 0).sum().sum())
        raise DataValidationError(f"Hay {bad} precios <= 0.")

    tolerance = 1e-10 * out["high"].abs().clip(lower=1.0)
    bad_high = out["high"] + tolerance < out[["open", "close", "low"]].max(axis=1)
    if bad_high.any():
        first = out.index[bad_high][0]
        raise DataValidationError(
            f"OHLC inconsistente en {first}: high no es el maximo de la barra."
        )
    bad_low = out["low"] - tolerance > out[["open", "close", "high"]].min(axis=1)
    if bad_low.any():
        first = out.index[bad_low][0]
        raise DataValidationError(
            f"OHLC inconsistente en {first}: low no es el minimo de la barra."
        )
    if (out["volume"] < 0).any():
        raise DataValidationError("El volumen no puede ser negativo.")

    if len(out) < min_rows:
        raise DataValidationError(f"Se necesitan al menos {min_rows} barras; hay {len(out)}.")

    out.index = out.index.tz_convert("UTC")
    out.attrs["symbol"] = symbol
    out.attrs["timeframe"] = timeframe
    out.attrs["validation_warnings"] = warnings
    for message in warnings:
        logger.warning("%s%s", f"[{symbol}] " if symbol else "", message)

    return out


def inspect_ohlcv(df: pd.DataFrame, symbol: str = "") -> dict[str, Any]:
    """Genera un informe legible sin lanzar excepciones de validacion."""
    report: dict[str, Any] = {"symbol": symbol, "valid": False, "errors": [], "warnings": []}
    try:
        clean = validate_ohlcv(df, symbol=symbol)
    except DataValidationError as exc:
        report["errors"].append(str(exc))
        return report

    deltas = clean.index.to_series().diff().dropna()
    median_delta = deltas.median() if len(deltas) else pd.Timedelta(0)
    expected = 0
    if median_delta > pd.Timedelta(0):
        expected = int((clean.index[-1] - clean.index[0]) / median_delta) + 1
    max_gap = deltas.max() if len(deltas) else pd.Timedelta(0)

    report.update(
        {
            "valid": True,
            "rows": int(len(clean)),
            "start": clean.index[0],
            "end": clean.index[-1],
            "median_bar": median_delta,
            "missing_bars": max(0, expected - len(clean)),
            "max_gap": max_gap,
            "warnings": list(clean.attrs.get("validation_warnings", [])),
            "price_min": float(clean["low"].min()),
            "price_max": float(clean["high"].max()),
        }
    )
    return report
