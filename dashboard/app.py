"""Panel Streamlit de TRADING BOT LAB.

Ejecutar desde la raiz del proyecto:
    streamlit run trading_bot_lab/dashboard/app.py
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from trading_bot_lab.analytics.store import ResultsStore
from trading_bot_lab.backtesting import BacktestEngine, verify_causality
from trading_bot_lab.config import PROJECT_ROOT, load_config
from trading_bot_lab.dashboard import sections
from trading_bot_lab.dashboard.utils import build_config, normalize_data
from trading_bot_lab.data import MarketDataStore, get_instrument, list_instruments
from trading_bot_lab.data.synthetic import generate_synthetic_ohlcv
from trading_bot_lab.strategies import create_strategy, strategy_catalog
from trading bot lab.bot types import Timeframe

logger = logging.getLogger("trading_bot_lab.dashboard")

st.set_page_config(
    page_title="TRADING BOT LAB",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
    .block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d172a 0%, #0a1322 100%);
        border-right: 1px solid rgba(120,150,200,.14);
    }
    .hero {
        background: radial-gradient(circle at 10% 0%, rgba(79,140,255,.22), transparent 42%),
                    linear-gradient(135deg, #0d172a 0%, #12233f 100%);
        border: 1px solid rgba(120,150,200,.18);
        border-radius: 20px;
        padding: 26px 30px;
        margin-bottom: 18px;
        box-shadow: 0 18px 45px rgba(0,0,0,.28);
    }
    .hero h1 { margin: 0; font-size: 30px; letter-spacing: .5px; color: #f4f8ff; }
    .hero p { margin: 7px 0 0; color: #9fb3d1; font-size: 14px; }
    .pill {
        display: inline-block; padding: 4px 10px; border-radius: 999px;
        background: rgba(79,140,255,.14); color: #9cc2ff; font-size: 11px;
        letter-spacing: .8px; text-transform: uppercase; margin-bottom: 10px;
        border: 1px solid rgba(79,140,255,.28);
    }
    .data-strip {
        display: flex; gap: 10px; flex-wrap: wrap; margin: 6px 0 16px;
    }
    .data-chip {
        background: rgba(18,28,46,.85); border: 1px solid rgba(120,150,200,.18);
        color: #c8d7ef; padding: 7px 12px; border-radius: 10px; font-size: 12.5px;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(160deg, rgba(24,38,62,.95), rgba(15,25,43,.95));
        border: 1px solid rgba(120,150,200,.16);
        border-radius: 14px;
        padding: 14px 15px;
        box-shadow: 0 10px 26px rgba(0,0,0,.18);
    }
    [data-testid="stMetricLabel"] { color: #93a8c7 !important; }
    [data-testid="stMetricValue"] { color: #f2f7ff !important; font-size: 1.35rem !important; }
    div[data-testid="stTabs"] button { font-weight: 600; letter-spacing: .2px; }
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #3f7cf5, #5b9bff);
        border: 0; font-weight: 700;
    }
    .disclaimer {
        margin-top: 26px; padding: 16px 18px; border-radius: 12px;
        border: 1px solid rgba(242,184,75,.25);
        background: rgba(242,184,75,.06); color: #d8c9a4; font-size: 12.5px;
    }
    code { color: #9cc2ff; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_base_config():
    return load_config()


@st.cache_resource(show_spinner=False)
def get_market_store(database: str) -> MarketDataStore:
    return MarketDataStore(database)


@st.cache_resource(show_spinner=False)
def get_results_store(database: str) -> ResultsStore:
    return ResultsStore(database)


@st.cache_data(show_spinner=False)
def parse_uploaded_csv(content: bytes, filename: str) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(
                io.BytesIO(content), sep=None, engine="python", encoding=encoding
            )
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise ValueError(f"No se pudo leer el CSV {filename}: {last_error}")


@st.cache_data(show_spinner=False)
def synthetic_data(
    symbol: str,
    timeframe: str,
    periods: int,
    seed: int,
    start_price: float,
    volatility: float,
) -> pd.DataFrame:
    return generate_synthetic_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        periods=periods,
        seed=seed,
        start_price=start_price,
        volatility=volatility,
    )


for key, default in {
    "bt_result": None,
    "bt_label": "",
    "comparison_results": {},
    "comparison_errors": [],
    "comparison_label": "",
    "oos_result": None,
    "wf_result": None,
    "paper_state": None,
    "paper_summary": {},
    "paper_path": "",
    "paper_meta": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

base_config = get_base_config()
market_store = get_market_store(str(base_config.database_path))
results_store = get_results_store(str(base_config.database_path))
instruments = list_instruments()
instrument_symbols = list(instruments)

raw_data: pd.DataFrame | None = None
data_label = ""
synthetic = False
data_error: str | None = None
csv_stored = st.session_state.get("csv_bytes")

with st.sidebar:
    st.markdown("## 📈 TRADING BOT LAB")
    st.caption("Laboratorio cuantitativo · sin ejecucion real")

    st.markdown("### 1 · Datos")
    source = st.radio(
        "Origen de datos",
        ["Subir CSV", "Base de datos SQLite", "Demo sintetica"],
        key="data_source",
    )

    if source == "Subir CSV":
        symbol = st.selectbox("Instrumento", instrument_symbols, index=0, key="csv_symbol")
        timeframe = st.selectbox(
            "Temporalidad", list(Timeframe.values()), index=4, key="csv_timeframe"
        )
        uploaded = st.file_uploader(
            "Archivo CSV OHLCV", type=["csv"], key="csv_uploader",
            help="Columnas: date/datetime/time/timestamp, open, high, low, close, volume.",
        )
        if uploaded is not None:
            csv_stored = uploaded.getvalue()
            st.session_state["csv_bytes"] = csv_stored
            st.session_state["csv_name"] = uploaded.name
        if csv_stored:
            st.caption(f"CSV: {st.session_state.get('csv_name', 'archivo cargado')}")
            if st.button("Olvidar CSV", key="forget_csv"):
                st.session_state.pop("csv_bytes", None)
                st.session_state.pop("csv_name", None)
                st.rerun()
            try:
                raw_data = parse_uploaded_csv(
                    csv_stored, st.session_state.get("csv_name", "subida.csv")
                )
                data_label = st.session_state.get("csv_name", "CSV local")
            except Exception as exc:
                data_error = str(exc)
        else:
            st.info("Carga un CSV para empezar. No se descargan datos automaticamente.")

    elif source == "Base de datos SQLite":
        series = market_store.list_series()
        if series.empty:
            symbol, timeframe = "EURUSD", "1h"
            st.info("La base local esta vacia. Importa un CSV desde la pestana Datos.")
        else:
            labels = [
                f"{row.symbol} · {row.timeframe} · {int(row.bars)} barras"
                for row in series.itertuples(index=False)
            ]
            chosen = st.selectbox("Serie almacenada", labels, key="db_series")
            row = series.iloc[labels.index(chosen)]
            symbol = str(row["symbol"])
            timeframe = str(row["timeframe"])
            try:
                raw_data = market_store.load_bars(symbol, timeframe)
                data_label = f"SQLite · {symbol} {timeframe}"
            except Exception as exc:
                data_error = str(exc)
    else:
        symbol = st.selectbox("Instrumento", instrument_symbols, index=0, key="synth_symbol")
        timeframe = st.selectbox(
            "Temporalidad", list(Timeframe.values()), index=4, key="synth_timeframe"
        )
        periods = st.slider("Barras sinteticas", 500, 8000, 3000, 100, key="synth_periods")
        seed = st.number_input("Semilla", 1, 999999, 42, 1, key="synth_seed")
        instrument_preview = instruments[symbol]
        start_price = 4000.0 if instrument_preview.asset_class == "index" else 1.10
        volatility = 0.004 if instrument_preview.asset_class == "index" else 0.0022
        raw_data = synthetic_data(
            symbol, timeframe, int(periods), int(seed), start_price, volatility
        )
        synthetic = True
        data_label = "SINTETICO (demo, no es dato de mercado)"
        st.warning("Datos sinteticos solo para probar el software. No son precios reales.")

    data: pd.DataFrame | None = None
    if raw_data is not None and data_error is None:
        try:
            data = normalize_data(raw_data, symbol, timeframe)
            st.caption(
                f"{len(data):,} barras · {data.index[0]:%Y-%m-%d} → {data.index[-1]:%Y-%m-%d}"
            )
        except Exception as exc:
            data_error = str(exc)

    st.markdown("### 2 · Cuenta y riesgo")
    with st.expander("Capital, riesgo y limites", expanded=True):
        initial_capital = st.number_input(
            "Capital inicial (USD)", 100.0, 100_000_000.0,
            float(base_config.account.initial_capital), 1000.0, key="initial_capital",
        )
        risk_per_trade = st.number_input(
            "Riesgo por operacion (%)", 0.01, 10.0,
            float(base_config.risk.risk_per_trade_pct), 0.1, format="%.2f",
            key="risk_per_trade",
        )
        max_daily_loss = st.number_input(
            "Perdida diaria maxima (%)", 0.0, 100.0,
            float(base_config.risk.max_daily_loss_pct), 0.5, key="max_daily_loss",
        )
        max_drawdown = st.number_input(
            "Drawdown maximo (%)", 1.0, 100.0,
            float(base_config.risk.max_drawdown_pct), 1.0, key="max_drawdown",
        )
        max_exposure = st.number_input(
            "Exposicion nocional maxima (%)", 10.0, 5000.0,
            float(base_config.risk.max_notional_exposure_pct), 50.0, key="max_exposure",
        )
        close_on_dd = st.checkbox(
            "Cerrar posicion al alcanzar drawdown maximo", value=False, key="close_on_dd"
        )

    st.markdown("### 3 · Costes")
    with st.expander("Spread, comision y slippage", expanded=True):
        spread = st.number_input(
            "Spread (pips, ida y vuelta)", 0.0, 100.0,
            float(base_config.costs.spread_pips), 0.1, format="%.2f", key="spread",
        )
        commission = st.number_input(
            "Comision por lote y lado (USD)", 0.0, 100.0,
            float(base_config.costs.commission_per_lot_per_side), 0.1,
            format="%.2f", key="commission",
        )
        slippage = st.number_input(
            "Slippage por lado (pips)", 0.0, 50.0,
            float(base_config.costs.slippage_pips), 0.1, format="%.2f", key="slippage",
        )

    st.markdown("### 4 · Ejecucion")
    with st.expander("Stops, objetivos y warmup", expanded=True):
        stop_mode = st.selectbox(
            "Modo de stop loss", ["atr", "pips", "percent", "none"],
            format_func=lambda value: {
                "atr": "ATR", "pips": "Pips", "percent": "% del precio", "none": "Sin stop"
            }[value],
            key="stop_mode",
        )
        stop_value = st.number_input(
            "Valor del stop", 0.01, 10000.0, float(base_config.execution.stop_value), 0.05,
            format="%.4f", key="stop_value",
            help="Multiplo ATR, pips o porcentaje segun el modo.",
        )
        tp_mode = st.selectbox(
            "Modo de take profit", ["rr", "atr", "pips", "percent", "none"],
            format_func=lambda value: {
                "rr": "Ratio riesgo/beneficio", "atr": "ATR", "pips": "Pips",
                "percent": "% del precio", "none": "Sin objetivo",
            }[value],
            key="tp_mode",
        )
        tp_value = st.number_input(
            "Valor del take profit", 0.01, 10000.0,
            float(base_config.execution.take_profit_value), 0.05,
            format="%.4f", key="tp_value",
        )
        trailing = st.number_input(
            "Trailing stop (multiplo ATR, 0=off)", 0.0, 20.0,
            float(base_config.execution.trailing_stop_atr_mult), 0.1, key="trailing",
        )
        max_bars = st.number_input(
            "Cerrar tras N barras (0=off)", 0, 5000,
            int(base_config.execution.max_bars_in_trade), 1, key="max_bars",
        )
        warmup = st.number_input(
            "Barras de warmup", 0, 5000,
            int(base_config.execution.warmup_bars), 10, key="warmup",
        )
        verify = st.checkbox(
            "Verificar ausencia de look-ahead", value=True, key="verify_causality"
        )
        if stop_mode == "none":
            st.warning("Sin stop no se puede dimensionar el riesgo: no se abriran operaciones.")

    st.markdown("### 5 · Estrategia")
    catalog = strategy_catalog()
    name_to_key = {entry["name"]: entry["key"] for entry in catalog}
    strategy_name = st.selectbox(
        "Estrategia", list(name_to_key), key="strategy_name_select"
    )
    strategy_key = name_to_key[strategy_name]
    entry = next(item for item in catalog if item["key"] == strategy_key)
    st.caption(entry["description"])

    strategy_params: dict[str, object] = {}
    for spec in entry["parameters"]:
        widget_key = f"param_{strategy_key}_{spec['name']}"
        label = spec["label"]
        if spec["kind"] == "bool":
            strategy_params[spec["name"]] = st.checkbox(
                label, value=bool(spec["default"]), help=spec.get("help", ""), key=widget_key
            )
        elif spec["kind"] == "choice":
            options = spec.get("choices", [])
            default_index = options.index(spec["default"]) if spec["default"] in options else 0
            strategy_params[spec["name"]] = st.selectbox(
                label, options, index=default_index, help=spec.get("help", ""), key=widget_key
            )
        elif spec["kind"] == "int":
            strategy_params[spec["name"]] = st.number_input(
                label,
                min_value=int(spec["minimum"] or 0),
                max_value=int(spec["maximum"] or 10**6),
                value=int(spec["default"]),
                step=int(spec["step"] or 1),
                help=spec.get("help", ""),
                key=widget_key,
            )
        else:
            step = float(spec["step"] or 0.1)
            strategy_params[spec["name"]] = st.number_input(
                label,
                min_value=float(spec["minimum"] or 0.0),
                max_value=float(spec["maximum"] or 10**9),
                value=float(spec["default"]),
                step=step,
                format="%.4f" if step < 0.01 else "%.2f",
                help=spec.get("help", ""),
                key=widget_key,
            )

    st.divider()
    save_run = st.checkbox("Guardar resultado en SQLite", value=True, key="save_run")
    run_clicked = st.sidebar.button(
        "▶ Ejecutar backtest",
        type="primary",
        width="stretch",
        disabled=data is None,
        key="run_backtest",
    )

st.markdown(
    """
    <div class="hero">
      <div class="pill">Research · Backtesting · Paper trading</div>
      <h1>TRADING BOT LAB</h1>
      <p>Laboratorio modular para descubrir, probar y validar ventajas estadisticas
      en Forex e indices, siempre despues de costes. Ninguna estrategia se asume rentable.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if data_error:
    st.error(f"No se pudieron preparar los datos: {data_error}")
if synthetic:
    st.warning(
        "Estas usando datos SINTETICOS de demostracion. No representan ningun mercado real "
        "y no permiten concluir que una estrategia sea rentable."
    )

if data is not None:
    st.markdown(
        f"""
        <div class="data-strip">
          <div class="data-chip">Mercado: <b>{symbol}</b></div>
          <div class="data-chip">Temporalidad: <b>{timeframe}</b></div>
          <div class="data-chip">Barras: <b>{len(data):,}</b></div>
          <div class="data-chip">Desde: <b>{data.index[0]:%Y-%m-%d}</b></div>
          <div class="data-chip">Hasta: <b>{data.index[-1]:%Y-%m-%d}</b></div>
          <div class="data-chip">Fuente: <b>{data_label}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

config = build_config(
    base_config,
    initial_capital=initial_capital,
    risk_per_trade_pct=risk_per_trade,
    max_daily_loss_pct=max_daily_loss,
    max_drawdown_pct=max_drawdown,
    max_notional_exposure_pct=max_exposure,
    close_on_max_drawdown=close_on_dd,
    spread_pips=spread,
    commission_per_lot_per_side=commission,
    slippage_pips=slippage,
    stop_mode=stop_mode,
    stop_value=stop_value,
    take_profit_mode=tp_mode,
    take_profit_value=tp_value,
    trailing_stop_atr_mult=trailing,
    max_bars_in_trade=int(max_bars),
    warmup_bars=int(warmup),
)

if run_clicked and data is not None:
    try:
        strategy = create_strategy(strategy_key, strategy_params)
        with st.spinner("Ejecutando simulacion..."):
            if verify:
                verify_causality(strategy, data, checks=5)
            result = BacktestEngine(config).run(
                data, strategy, symbol=symbol, timeframe=timeframe
            )
        st.session_state["bt_result"] = result
        st.session_state["bt_label"] = data_label
        if save_run:
            results_store.save_result(
                result, data_source=("synthetic-demo" if synthetic else data_label)
            )
        st.success(
            f"Backtest completado: {result.metrics.get('trades', 0)} operaciones, "
            f"beneficio neto {result.metrics.get('net_profit', 0):,.2f} USD."
        )
    except Exception as exc:
        logger.exception("Error en backtest")
        st.error(f"No se pudo ejecutar el backtest: {exc}")

tab_backtest, tab_compare, tab_validation, tab_data, tab_history, tab_paper = st.tabs(
    ["Backtest", "Comparacion", "Validacion OOS", "Datos", "Historial", "Paper trading"]
)

with tab_backtest:
    existing = st.session_state.get("bt_result")
    if existing is not None:
        sections.render_result(
            existing, st.session_state.get("bt_label", ""), key_prefix="bt_result"
        )
    elif data is None:
        st.info("Carga un CSV, selecciona una serie local o usa la demo sintetica.")
        st.markdown(
            """
            **Formato CSV esperado**

            ```
            timestamp,open,high,low,close,volume
            2024-01-01 00:00:00,1.1032,1.1041,1.1027,1.1038,1200
            ```

            La zona horaria se interpreta como UTC si no tiene offset. No se descargan
            datos de mercado automaticamente y no se inventan precios.
            """
        )
    else:
        st.info("Configura la estrategia y pulsa **Ejecutar backtest** en la barra lateral.")

with tab_compare:
    sections.render_comparison(data, config, symbol, timeframe)

with tab_validation:
    sections.render_validation(
        data, config, symbol, timeframe, strategy_key, strategy_name
    )

with tab_data:
    sections.render_data(
        data,
        symbol=symbol,
        timeframe=timeframe,
        data_label=data_label,
        synthetic=synthetic,
        market_store=market_store,
    )

with tab_history:
    sections.render_history(results_store)

with tab_paper:
    sections.render_paper(
        data, config, strategy_key, strategy_name, symbol, timeframe
    )

st.markdown(
    """
    <div class="disclaimer">
      <b>Aviso:</b> TRADING BOT LAB es una herramienta de investigacion. Todos los
      resultados son simulaciones sobre datos historicos o sinteticos. No hay conexion
      con brokers ni envio de ordenes reales. Rentabilidad pasada o simulada no garantiza
      rentabilidad futura. Verifique los datos, los costes y la robustez fuera de muestra
      antes de cualquier uso comercial.
    </div>
    """,
    unsafe_allow_html=True,
)
