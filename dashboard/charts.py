"""Graficos Plotly del panel."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from trading_bot_lab.analytics.metrics import compute_monthly_returns

ACCENT = "#4f8cff"
GREEN = "#2ecc8f"
RED = "#ff5d73"
AMBER = "#f2b84b"
MUTED = "#8ea0bd"
GRID = "rgba(142,160,189,0.12)"
TEMPLATE = "plotly_dark"


def _base_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        template=TEMPLATE,
        height=height,
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbe6f6", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
        xaxis=dict(gridcolor=GRID, zeroline=False),
        yaxis=dict(gridcolor=GRID, zeroline=False),
    )
    return fig


def equity_figure(equity: pd.Series, initial_capital: float, title: str = "Evolucion del capital") -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            mode="lines",
            name="Capital",
            line=dict(color=ACCENT, width=2.2),
            fill="tozeroy",
            fillcolor="rgba(79,140,255,0.10)",
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Capital: %{y:,.2f}<extra></extra>",
        )
    )
    figure.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color=MUTED,
        annotation_text="Capital inicial",
        annotation_position="bottom right",
    )
    figure.update_layout(title=title, yaxis_title="Capital")
    return _base_layout(figure)


def drawdown_figure(equity: pd.Series, title: str = "Curva de drawdown") -> go.Figure:
    peak = equity.cummax()
    drawdown = (equity / peak - 1.0) * 100.0
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode="lines",
            name="Drawdown",
            line=dict(color=RED, width=1.8),
            fill="tozeroy",
            fillcolor="rgba(255,93,115,0.16)",
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Drawdown: %{y:.2f}%<extra></extra>",
        )
    )
    figure.update_layout(title=title, yaxis_title="%", yaxis_ticksuffix="%")
    return _base_layout(figure, height=260)


def price_trade_figure(data: pd.DataFrame, trades: pd.DataFrame, symbol: str) -> go.Figure:
    frame = data
    if len(frame) > 5000:
        frame = frame.iloc[-5000:]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["close"],
            mode="lines",
            name=symbol,
            line=dict(color="#a9c1e8", width=1.3),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:,.4f}<extra></extra>",
        )
    )
    if not trades.empty:
        entries = trades.copy()
        entries["entry_time"] = pd.to_datetime(entries["entry_time"], utc=True)
        entries["exit_time"] = pd.to_datetime(entries["exit_time"], utc=True)
        entries = entries[entries["entry_time"] >= frame.index[0]]
        for direction, color, symbol_shape in (
            ("long", GREEN, "triangle-up"),
            ("short", RED, "triangle-down"),
        ):
            subset = entries[entries["direction"] == direction]
            if subset.empty:
                continue
            figure.add_trace(
                go.Scatter(
                    x=subset["entry_time"],
                    y=subset["entry_price_raw"],
                    mode="markers",
                    name=f"Entrada {direction}",
                    marker=dict(color=color, size=9, symbol=symbol_shape),
                    customdata=subset[["lots", "net_pnl_account"]],
                    hovertemplate=(
                        f"Entrada {direction}<br>%{{x|%Y-%m-%d %H:%M}}<br>"
                        "Precio: %{y:,.4f}<br>Lotes: %{customdata[0]:.2f}<br>"
                        "PnL: %{customdata[1]:,.2f}<extra></extra>"
                    ),
                )
            )
        exits = entries[entries["exit_time"] >= frame.index[0]]
        figure.add_trace(
            go.Scatter(
                x=exits["exit_time"],
                y=exits["exit_price_raw"],
                mode="markers",
                name="Salidas",
                marker=dict(color=AMBER, size=5, symbol="x"),
                customdata=exits[["net_pnl_account", "exit_reason"]],
                hovertemplate=(
                    "Salida<br>%{x|%Y-%m-%d %H:%M}<br>Precio: %{y:,.4f}<br>"
                    "PnL: %{customdata[0]:,.2f}<br>%{customdata[1]}<extra></extra>"
                ),
            )
        )
    figure.update_layout(title="Precio y operaciones", yaxis_title="Precio")
    return _base_layout(figure, height=430)


def monthly_heatmap_figure(equity: pd.DataFrame) -> go.Figure | None:
    monthly = compute_monthly_returns(equity)
    if monthly.empty:
        return None
    monthly = monthly.reindex(columns=list(range(1, 13)))
    months = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    values = monthly.to_numpy(dtype=float)
    text = [[("" if pd.isna(value) else f"{value:.1f}") for value in row] for row in values]
    limit = max(1.0, float(pd.Series(values.ravel()).abs().max(skipna=True) or 1.0))
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=months,
            y=[str(year) for year in monthly.index],
            colorscale="RdYlGn",
            zmid=0,
            zmin=-limit,
            zmax=limit,
            text=text,
            texttemplate="%{text}",
            hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>",
            colorbar=dict(title="%"),
        )
    )
    figure.update_layout(title="Retornos mensuales (%)")
    return _base_layout(figure, height=280)


def comparison_figure(results: dict[str, pd.DataFrame]) -> go.Figure:
    figure = go.Figure()
    colors = [ACCENT, GREEN, AMBER, RED, "#b388ff", "#4dd0e1", "#ff9e80"]
    for index, (name, equity) in enumerate(results.items()):
        series = equity["equity"].astype(float)
        normalized = series / series.iloc[0] * 100.0
        figure.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized.values,
                mode="lines",
                name=name,
                line=dict(width=2, color=colors[index % len(colors)]),
                hovertemplate="%{x|%Y-%m-%d}<br>Base 100: %{y:.2f}<extra></extra>",
            )
        )
    figure.add_hline(y=100, line_dash="dash", line_color=MUTED)
    figure.update_layout(title="Comparacion (base 100)", yaxis_title="Indice")
    return _base_layout(figure, height=390)
