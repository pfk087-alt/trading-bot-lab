"""Metricas de rendimiento ajustadas por costes.

Convenciones:
- ``max_drawdown_pct`` es negativo (p.ej. -12.5).
- ``profit_factor`` es None si no hay perdidas (JSON no admite infinito).
- Sharpe/Sortino se calculan con retornos diarios y anualizacion configurable.
- La exposicion es el porcentaje de barras con posicion abierta.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _safe(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _consecutive(series: pd.Series, positive: bool) -> int:
    best = current = 0
    for value in series:
        condition = value > 0 if positive else value < 0
        current = current + 1 if condition else 0
        best = max(best, current)
    return best


def compute_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    initial_capital: float,
    annualization_days: int = 252,
    bars_per_year: int | None = None,
) -> dict[str, Any]:
    """Calcula el informe cuantitativo completo de una simulacion."""
    if initial_capital <= 0:
        raise ValueError("initial_capital debe ser > 0")
    if "equity" not in equity_curve.columns or equity_curve.empty:
        raise ValueError("equity_curve debe contener la columna 'equity' y no estar vacia")

    equity = equity_curve["equity"].astype(float).sort_index()
    final_equity = float(equity.iloc[-1])
    net_profit = final_equity - float(initial_capital)
    return_pct = net_profit / float(initial_capital) * 100.0

    running_peak = equity.cummax()
    drawdown_series = equity / running_peak - 1.0
    max_drawdown_pct = float(drawdown_series.min() * 100.0)
    max_drawdown_abs = float((running_peak - equity).max())

    start = pd.Timestamp(equity.index[0])
    end = pd.Timestamp(equity.index[-1])
    elapsed_days = max((end - start).total_seconds() / 86400.0, 1.0 / 24.0)
    years = max(elapsed_days / 365.25, 1.0 / 365.25)
    cagr = (final_equity / float(initial_capital)) ** (1.0 / years) - 1.0

    daily = equity.resample("1D").last().dropna()
    daily_returns = daily.pct_change(fill_method=None).dropna()
    volatility = float(daily_returns.std(ddof=1)) if len(daily_returns) > 1 else 0.0
    mean_return = float(daily_returns.mean()) if len(daily_returns) else 0.0
    sharpe = None
    if len(daily_returns) > 1 and volatility > 1e-12:
        sharpe = mean_return / volatility * math.sqrt(annualization_days)

    downside = daily_returns[daily_returns < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = None
    if len(daily_returns) > 1 and downside_std > 1e-12:
        sortino = mean_return / downside_std * math.sqrt(annualization_days)

    calmar = None
    if max_drawdown_pct < 0:
        calmar = cagr / abs(max_drawdown_pct / 100.0)

    exposure_pct = None
    if "in_position" in equity_curve.columns and len(equity_curve):
        exposure_pct = float(equity_curve["in_position"].astype(bool).mean() * 100.0)

    stats: dict[str, Any] = {
        "initial_capital": float(initial_capital),
        "final_equity": final_equity,
        "net_profit": net_profit,
        "return_pct": return_pct,
        "cagr_pct": cagr * 100.0,
        "start": str(start),
        "end": str(end),
        "days": elapsed_days,
        "max_drawdown_pct": max_drawdown_pct,
        "max_drawdown_abs": max_drawdown_abs,
        "sharpe": _safe(sharpe),
        "sortino": _safe(sortino),
        "calmar": _safe(calmar),
        "recovery_factor": _safe(net_profit / max_drawdown_abs) if max_drawdown_abs > 0 else None,
        "volatility_annual_pct": _safe(volatility * math.sqrt(annualization_days) * 100.0),
        "exposure_pct": exposure_pct,
        "trades": int(len(trades)),
        "long_trades": int((trades["direction"] == "long").sum()) if not trades.empty else 0,
        "short_trades": int((trades["direction"] == "short").sum()) if not trades.empty else 0,
    }

    if trades.empty:
        stats.update(
            {
                "win_rate_pct": 0.0,
                "profit_factor": None,
                "expectancy": 0.0,
                "avg_trade": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "payoff_ratio": None,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "avg_bars_held": 0.0,
                "total_costs": 0.0,
                "costs_pct_of_gross": None,
                "max_consecutive_wins": 0,
                "max_consecutive_losses": 0,
                "sqn": None,
                "avg_return_pct": 0.0,
            }
        )
        return stats

    required = {"net_pnl_account", "direction"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"trades carece de columnas: {sorted(missing)}")

    pnl = trades["net_pnl_account"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-12 else None
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    payoff = avg_win / abs(avg_loss) if avg_loss < 0 else None
    costs = float(trades["costs_account"].sum()) if "costs_account" in trades else 0.0
    trade_std = float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
    sqn = math.sqrt(len(pnl)) * float(pnl.mean()) / trade_std if trade_std > 1e-12 else None

    stats.update(
        {
            "win_rate_pct": float(len(wins) / len(pnl) * 100.0),
            "profit_factor": _safe(profit_factor),
            "expectancy": float(pnl.mean()),
            "avg_trade": float(pnl.mean()),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "payoff_ratio": _safe(payoff),
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "largest_win": float(pnl.max()),
            "largest_loss": float(pnl.min()),
            "avg_bars_held": float(trades["bars_held"].mean())
            if "bars_held" in trades
            else 0.0,
            "total_costs": costs,
            "costs_pct_of_gross": (costs / gross_profit * 100.0) if gross_profit > 0 else None,
            "max_consecutive_wins": _consecutive(pnl, True),
            "max_consecutive_losses": _consecutive(pnl, False),
            "sqn": _safe(sqn),
            "avg_return_pct": float(trades["return_pct"].mean())
            if "return_pct" in trades
            else 0.0,
        }
    )
    return stats


def compute_monthly_returns(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """Retornos mensuales para el mapa de calor del panel."""
    equity = equity_curve["equity"].astype(float).sort_index()
    monthly = equity.resample("ME").last().dropna()
    returns = monthly.pct_change(fill_method=None).dropna() * 100.0
    if monthly.empty:
        return pd.DataFrame()
    # Incluye el primer mes como retorno desde la primera barra hasta su cierre.
    first_month = monthly.index[0]
    first_return = (monthly.iloc[0] / equity.iloc[0] - 1.0) * 100.0
    returns = pd.concat(
        [pd.Series([first_return], index=pd.DatetimeIndex([first_month])), returns]
    )
    frame = pd.DataFrame(
        {
            "year": returns.index.year,
            "month": returns.index.month,
            "return_pct": returns.values,
        }
    )
    return frame.pivot_table(index="year", columns="month", values="return_pct", aggfunc="first")


def metrics_to_frame(results: dict[str, Any]) -> pd.DataFrame:
    """Convierte un diccionario {nombre: metricas} en tabla comparativa."""
    rows = []
    for name, metrics in results.items():
        row = {"strategy": name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def objective_score(metrics: dict[str, Any], objective: str, min_trades: int = 1) -> tuple:
    """Puntuacion para optimizacion. Prioriza muestras suficientes.

    Devuelve una tupla comparable; nunca usa datos fuera de muestra.
    """
    trades = int(metrics.get("trades", 0) or 0)
    meets_minimum = trades >= max(1, min_trades)
    value: float | None
    if objective == "sharpe":
        value = metrics.get("sharpe")
    elif objective == "sortino":
        value = metrics.get("sortino")
    elif objective == "calmar":
        value = metrics.get("calmar")
    elif objective == "profit_factor":
        value = metrics.get("profit_factor")
    elif objective == "expectancy":
        value = metrics.get("expectancy")
    elif objective == "net_profit":
        value = metrics.get("net_profit")
    elif objective == "return_pct":
        value = metrics.get("return_pct")
    else:
        raise ValueError(f"Objetivo de optimizacion desconocido: {objective}")

    numeric = float(value) if value is not None and math.isfinite(float(value)) else float("-inf")
    net = float(metrics.get("net_profit", 0.0) or 0.0)
    drawdown = abs(float(metrics.get("max_drawdown_pct", 0.0) or 0.0))
    return (1 if meets_minimum else 0, numeric, net, -drawdown)
