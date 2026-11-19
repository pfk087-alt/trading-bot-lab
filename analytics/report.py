"""Informe HTML reproducible de un backtest."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd

from trading_bot_lab.backtesting.models import BacktestResult

from .metrics import compute_monthly_returns

DISCLAIMER = (
    "Resultados de simulacion historica. No constituyen asesoramiento financiero "
    "ni garantizan rentabilidad futura. Los resultados dependen de la calidad de "
    "los datos, de los costes asumidos y de las decisiones de modelado."
)


def _fmt(value: Any, suffix: str = "", decimals: int = 2) -> str:
    if value is None:
        return "N/D"
    if isinstance(value, str):
        return html.escape(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if pd.isna(number):
        return "N/D"
    return f"{number:,.{decimals}f}{suffix}"


def _kpi_rows(metrics: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        ("Beneficio neto", _fmt(metrics.get("net_profit")) + " " + "USD"),
        ("Rentabilidad", _fmt(metrics.get("return_pct"), "%")),
        ("Operaciones", str(metrics.get("trades", 0))),
        ("Win rate", _fmt(metrics.get("win_rate_pct"), "%")),
        ("Profit factor", _fmt(metrics.get("profit_factor"))),
        ("Expectativa / trade", _fmt(metrics.get("expectancy")) + " USD"),
        ("Drawdown maximo", _fmt(abs(metrics.get("max_drawdown_pct", 0.0)), "%")),
        ("Sharpe (diario anualizado)", _fmt(metrics.get("sharpe"))),
        ("Sortino", _fmt(metrics.get("sortino"))),
        ("Costes totales", _fmt(metrics.get("total_costs")) + " USD"),
        ("Exposicion", _fmt(metrics.get("exposure_pct"), "%")),
        ("Duracion media", _fmt(metrics.get("avg_bars_held"), " barras", 1)),
    ]


def generate_html_report(result: BacktestResult, *, data_source: str = "") -> str:
    metrics = result.metrics
    kpis = "".join(
        f'<div class="kpi"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{value}</div></div>'
        for label, value in _kpi_rows(metrics)
    )

    params_rows = "".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in result.params.items()
    )
    warnings = "".join(f"<li>{html.escape(message)}</li>" for message in result.warnings) or "<li>Ninguna.</li>"

    monthly = compute_monthly_returns(result.equity_curve)
    monthly_html = monthly.round(2).to_html(classes="data", border=0) if not monthly.empty else "<p>Sin datos mensuales.</p>"

    trades = result.trades.copy()
    if not trades.empty:
        display = trades[
            [
                "trade_id", "direction", "entry_time", "exit_time", "entry_price_raw",
                "exit_price_raw", "lots", "net_pnl_account", "costs_account",
                "exit_reason", "bars_held",
            ]
        ].head(500).copy()
        display["entry_time"] = pd.to_datetime(display["entry_time"]).dt.strftime("%Y-%m-%d %H:%M")
        display["exit_time"] = pd.to_datetime(display["exit_time"]).dt.strftime("%Y-%m-%d %H:%M")
        trades_html = display.round(4).to_html(classes="data", border=0, index=False)
        if len(trades) > 500:
            trades_html += f"<p>Mostradas 500 de {len(trades)} operaciones.</p>"
    else:
        trades_html = "<p>No hubo operaciones.</p>"

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>TRADING BOT LAB - {html.escape(result.strategy_name)}</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0;
         background: #f3f5f9; color: #172033; }}
  .wrap {{ max-width: 1180px; margin: 0 auto; padding: 32px 24px 60px; }}
  header {{ background: linear-gradient(135deg, #0b1220, #16294a); color: white;
            padding: 28px 32px; border-radius: 18px; margin-bottom: 24px; }}
  header h1 {{ margin: 0 0 6px; font-size: 28px; letter-spacing: .3px; }}
  header p {{ margin: 0; opacity: .8; }}
  h2 {{ margin: 28px 0 12px; font-size: 20px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap: 12px; }}
  .kpi {{ background: white; border-radius: 14px; padding: 16px; box-shadow: 0 6px 18px rgba(15,23,42,.06); }}
  .kpi .label {{ font-size: 12px; text-transform: uppercase; letter-spacing: .6px; color: #64748b; }}
  .kpi .value {{ font-size: 20px; font-weight: 700; margin-top: 6px; }}
  table.data {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px;
                overflow: hidden; font-size: 13px; }}
  table.data th, table.data td {{ padding: 8px 10px; border-bottom: 1px solid #eef2f7; text-align: right; }}
  table.data th:first-child, table.data td:first-child {{ text-align: left; }}
  table.data th {{ background: #eef4ff; color: #24406b; }}
  .panel {{ background: white; border-radius: 14px; padding: 18px; box-shadow: 0 6px 18px rgba(15,23,42,.06); }}
  .muted {{ color: #64748b; font-size: 13px; }}
  .warning {{ background: #fff7e8; border: 1px solid #f0d49a; border-radius: 12px; padding: 14px 18px; }}
  footer {{ margin-top: 36px; font-size: 12px; color: #64748b; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>TRADING BOT LAB</h1>
    <p>{html.escape(result.strategy_name)} &middot; {html.escape(result.symbol)} &middot;
       {html.escape(result.timeframe)} &middot; {result.start:%Y-%m-%d} a {result.end:%Y-%m-%d}</p>
  </header>

  <h2>Resumen ejecutivo</h2>
  <div class="kpis">{kpis}</div>

  <h2>Configuracion de la simulacion</h2>
  <div class="panel">
    <p><strong>Run ID:</strong> {html.escape(result.run_id)} &nbsp;
       <strong>Fuente:</strong> {html.escape(data_source or "no especificada")}</p>
    <table class="data"><tr><th>Parametro</th><th>Valor</th></tr>{params_rows}</table>
    <p class="muted">Capital inicial: {_fmt(result.initial_capital)} USD. Los costes, el
    apalancamiento y los limites de riesgo se detallan en la configuracion del run.</p>
  </div>

  <h2>Retornos mensuales (%)</h2>
  <div class="panel">{monthly_html}</div>

  <h2>Operaciones</h2>
  <div class="panel">{trades_html}</div>

  <h2>Advertencias de robustez</h2>
  <div class="warning"><ul>{warnings}</ul></div>

  <h2>Metodologia</h2>
  <div class="panel">
    <p>Senales calculadas al cierre de cada barra y ejecutadas en la apertura siguiente.
    Stops y objetivos se evaluan con el rango de la barra; si ambos se tocan, se asume el
    stop. Spread, slippage y comision se aplican en todas las operaciones. El tamano de
    posicion se deriva del riesgo configurado hasta el stop.</p>
    <p>Para validar una ventaja estadistica, compare este resultado con pruebas fuera de
    muestra y walk-forward. Un unico backtest no demuestra rentabilidad futura.</p>
  </div>

  <footer>{html.escape(DISCLAIMER)}</footer>
</div>
</body>
</html>"""


def save_html_report(result: BacktestResult, path: str | Path, *, data_source: str = "") -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(generate_html_report(result, data_source=data_source), encoding="utf-8")
    return target
