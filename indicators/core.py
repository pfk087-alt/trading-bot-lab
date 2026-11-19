"""Indicadores tecnicos con implementacion causal.

Todos los indicadores usan exclusivamente datos presentes y pasados. Las
funciones devuelven series alineadas con el indice de entrada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    if period < 1:
        raise ValueError("period debe ser >= 1")
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    if period < 1:
        raise ValueError("period debe ser >= 1")
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _wilder(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    previous_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    if period < 1:
        raise ValueError("period debe ser >= 1")
    return _wilder(true_range(high, low, close), period)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    if period < 1:
        raise ValueError("period debe ser >= 1")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    average_gain = _wilder(gain, period)
    average_loss = _wilder(loss, period)
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + relative_strength))
    result = result.where(average_loss != 0.0, 100.0)
    result = result.where(~((average_gain == 0.0) & (average_loss == 0.0)), 50.0)
    return result


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    if not 0 < fast < slow:
        raise ValueError("Se requiere 0 < fast < slow")
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    line = fast_ema - slow_ema
    signal_line = ema(line, signal)
    return pd.DataFrame(
        {"macd": line, "macd_signal": signal_line, "macd_hist": line - signal_line}
    )


def bollinger(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    if period < 2:
        raise ValueError("period debe ser >= 2")
    middle = sma(close, period)
    deviation = close.rolling(window=period, min_periods=period).std(ddof=0)
    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": middle + num_std * deviation,
            "bb_lower": middle - num_std * deviation,
            "bb_width": (2 * num_std * deviation) / middle,
        }
    )


def donchian(
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
) -> pd.DataFrame:
    if period < 1:
        raise ValueError("period debe ser >= 1")
    upper = high.rolling(window=period, min_periods=period).max()
    lower = low.rolling(window=period, min_periods=period).min()
    return pd.DataFrame(
        {"dc_upper": upper, "dc_lower": lower, "dc_mid": (upper + lower) / 2.0}
    )


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """ADX/DI de Wilder."""
    if period < 2:
        raise ValueError("period debe ser >= 2")
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )
    atr_value = atr(high, low, close, period).replace(0.0, np.nan)
    plus_di = 100.0 * _wilder(plus_dm, period) / atr_value
    minus_di = 100.0 * _wilder(minus_dm, period) / atr_value
    denominator = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denominator
    return pd.DataFrame(
        {
            "adx": _wilder(dx, period),
            "plus_di": plus_di,
            "minus_di": minus_di,
        }
    )


def cross_above(left: pd.Series, right: pd.Series | float) -> pd.Series:
    """True en la barra donde left cruza por encima de right."""
    if not isinstance(right, pd.Series):
        right = pd.Series(float(right), index=left.index)
    return (left > right) & (left.shift(1) <= right.shift(1))


def cross_below(left: pd.Series, right: pd.Series | float) -> pd.Series:
    """True en la barra donde left cruza por debajo de right."""
    if not isinstance(right, pd.Series):
        right = pd.Series(float(right), index=left.index)
    return (left < right) & (left.shift(1) >= right.shift(1))
