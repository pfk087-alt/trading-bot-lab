"""Secciones del panel: resultados, comparacion, validacion, datos, historial y paper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from trading_bot_lab.analytics.report import generate_html_report
from trading_bot_lab.analytics.store import ResultsStore
from trading_bot_lab.analytics.validation import (
    train_test_evaluate,
    walk_forward,
)
from trading_bot_lab.backtesting import BacktestEngine, BacktestResult
from trading_bot_lab.config import PROJECT_ROOT, AppConfig
from trading_bot_lab.dashboard import charts
from trading_bot_lab.dashboard.utils import (
    default_grid,
    grid_size,
    money,
    number,
    parse_grid,
    pct,
)
from trading_bot_lab.data import MarketDataStore, inspect_ohlcv
from trading_bot_lab.data.instruments import get_instrument
from trading_bot_lab.paper_trading import PaperSessionStore, PaperTradingEngine
from trading_bot_lab.strategies import create_strategy, strategy_catalog


def _metric_value(value: Any, formatter=money) -> str:
    return formatter(value)


def render_result(result: BacktestResult, data_label: str, *, key_prefix: str = "bt") -> None:
    metrics = result.metrics
    st.markdown(
        f"#### {result.strategy_name} · {result.symbol} · {result.timeframe}"
    )
    st.caption(
        f"{result.start:%Y-%m-%d %H:%M} → {result.end:%Y-%m-%d %H:%M} · "
        f"fuente: {data_label or 'no especificada'} · run {result.run_id}"
    )

    row1 = st.columns(6)
    row1[0].metric("Beneficio neto", money(metrics.get("net_profit")), pct(metrics.get("return_pct")))
    row1[1].metric("Rentabilidad", pct(metrics.get("return_pct")))
    row1[2].metric("Operaciones", str(metrics.get("trades", 0)))
    row1[3].metric("Win rate", pct(metrics.get("win_rate_pct")))
    row1[4].metric(
        "Profit factor",
        "N/D" if metrics.get("profit_factor") is None else number(metrics.get("profit_factor")),
    )
    row1[5].metric("Expectativa / trade", money(metrics.get("expectancy")))

    row2 = st.columns(6)
    row2[0].metric("Drawdown maximo", pct(abs(metrics.get("max_drawdown_pct", 0.0))))
    row2[1].metric("Sharpe", number(metrics.get("sharpe")))
    row2[2].metric("Sortino", number(metrics.get("sortino")))
    row2[3].metric("Costes totales", money(metrics.get("total_costs")))
    row2[4].metric("Exposicion", pct(metrics.get("exposure_pct")))
    row2[5].metric("Capital final", money(metrics.get("final_equity")))

    if result.warnings:
        with st.expander(f"Advertencias de robustez ({len(result.warnings)})", expanded=True):
            for message in result.warnings:
                st.warning(message)
    if result.diagnostics.get("blocked_entries"):
        with st.expander("Entradas bloqueadas por riesgo"):
            st.json(result.diagnostics["blocked_entries"])

    if metrics.get("trades", 0) < 30:
        st.error(
            "Muestra insuficiente para afirmar una ventaja estadistica. "
            "No use este resultado aislado para decisiones comerciales."
        )

    st.plotly_chart(
        charts.equity_figure(
            result.equity_curve["equity"], result.initial_capital, "Evolucion del capital"
        ),
        width="stretch",
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            charts.drawdown_figure(result.equity_curve["equity"]), width="stretch"
        )
    with right:
        heatmap = charts.monthly_heatmap_figure(result.equity_curve)
        if heatmap is None:
            st.info("No hay suficientes meses para el mapa de retornos.")
        else:
            st.plotly_chart(heatmap, width="stretch")

    st.plotly_chart(
        charts.price_trade_figure(result.data, result.trades, result.symbol),
        width="stretch",
    )

    st.markdown("#### Historial de operaciones")
    if result.trades.empty:
        st.info("No se ejecuto ninguna operacion.")
    else:
        display = result.trades.copy()
        display["entry_time"] = pd.to_datetime(display["entry_time"]).dt.strftime("%Y-%m-%d %H:%M")
        display["exit_time"] = pd.to_datetime(display["exit_time"]).dt.strftime("%Y-%m-%d %H:%M")
        columns = [
            "trade_id", "direction", "entry_time", "exit_time", "entry_price_raw",
            "exit_price_raw", "lots", "net_pnl_account", "costs_account", "return_pct",
            "exit_reason", "bars_held",
        ]
        st.dataframe(
            display[columns].round(4),
            width="stretch",
            hide_index=True,
            height=360,
        )

    downloads = st.columns(3)
    report_html = generate_html_report(result, data_source=data_label)
    downloads[0].download_button(
        "Informe HTML",
        data=report_html.encode("utf-8"),
        file_name=f"informe_{result.run_id}.html",
        mime="text/html",
        key=f"{key_prefix}_html",
        width="stretch",
    )
    downloads[1].download_button(
        "Operaciones CSV",
        data=result.trades.to_csv(index=False).encode("utf-8-sig") if not result.trades.empty else b"",
        file_name=f"trades_{result.run_id}.csv",
        mime="text/csv",
        key=f"{key_prefix}_trades",
        width="stretch",
    )
    downloads[2].download_button(
        "Metricas JSON",
        data=json.dumps(result.to_summary(), ensure_ascii=False, indent=2, default=str).encode("utf-8"),
        file_name=f"metricas_{result.run_id}.json",
        mime="application/json",
        key=f"{key_prefix}_json",
        width="stretch",
    )

    with st.expander("Parametros y configuracion"):
        st.write("**Parametros de estrategia**")
        st.json(result.params)
        st.write("**Ejecucion y riesgo**")
        st.json(
            {
                "account": result.settings.get("account"),
                "risk": result.settings.get("risk"),
                "costs": result.settings.get("costs"),
                "execution": result.settings.get("execution"),
            }
        )
        st.json({"diagnostics": result.diagnostics, "instrument": result.instrument.to_dict()})


def render_comparison(
    data: pd.DataFrame | None,
    config: AppConfig,
    symbol: str,
    timeframe: str,
    *,
    key_prefix: str = "cmp",
) -> None:
    st.markdown("#### Comparacion de estrategias")
    st.caption(
        "Todas las estrategias se ejecutan con sus parametros por defecto, los mismos "
        "datos, costes y riesgo. Es una comparacion in-sample: no sustituye la validacion OOS."
    )
    if data is None:
        st.info("Cargue datos para comparar estrategias.")
        return

    catalog = strategy_catalog()
    names = {entry["name"]: entry["key"] for entry in catalog}
    selected = st.multiselect(
        "Estrategias a comparar",
        options=list(names),
        default=list(names)[:3],
        key=f"{key_prefix}_selected",
    )
    if st.button("Comparar estrategias", type="primary", key=f"{key_prefix}_run"):
        results: dict[str, BacktestResult] = {}
        progress = st.progress(0.0, text="Ejecutando comparacion...")
        errors: list[str] = []
        for index, name in enumerate(selected, start=1):
            key = names[name]
            try:
                strategy = create_strategy(key, {})
                results[key] = BacktestEngine(config).run(
                    data, strategy, symbol=symbol, timeframe=timeframe
                )
            except Exception as exc:
                errors.append(f"{name}: {exc}")
            progress.progress(index / max(1, len(selected)), text=f"Ejecutando {name}...")
        progress.empty()
        st.session_state["comparison_results"] = results
        st.session_state["comparison_errors"] = errors
        st.session_state["comparison_label"] = (
            f"{symbol} · {timeframe} · {len(data)} barras"
        )

    results = st.session_state.get("comparison_results", {})
    errors = st.session_state.get("comparison_errors", [])
    for error in errors:
        st.error(error)
    if not results:
        st.info("Seleccione estrategias y pulse comparar.")
        return

    rows = []
    for key, result in results.items():
        metrics = result.metrics
        rows.append(
            {
                "Estrategia": result.strategy_name,
                "Beneficio neto": metrics.get("net_profit"),
                "Retorno %": metrics.get("return_pct"),
                "Operaciones": metrics.get("trades"),
                "Win rate %": metrics.get("win_rate_pct"),
                "Profit factor": metrics.get("profit_factor"),
                "Expectativa": metrics.get("expectancy"),
                "Max DD %": abs(metrics.get("max_drawdown_pct", 0.0) or 0.0),
                "Sharpe": metrics.get("sharpe"),
                "Costes": metrics.get("total_costs"),
            }
        )
    table = pd.DataFrame(rows)
    st.dataframe(
        table.round(3),
        width="stretch",
        hide_index=True,
        height=min(420, 60 + 35 * len(table)),
    )
    st.plotly_chart(
        charts.comparison_figure({result.strategy_name: result.equity_curve for result in results.values()}),
        width="stretch",
    )
    st.warning(
        "Comparar muchas estrategias sobre los mismos datos aumenta la probabilidad de "
        "encontrar resultados positivos por azar. Confirme con datos fuera de muestra."
    )


def render_validation(
    data: pd.DataFrame | None,
    config: AppConfig,
    symbol: str,
    timeframe: str,
    strategy_key: str,
    strategy_name: str,
) -> None:
    st.markdown("#### Validacion fuera de muestra y walk-forward")
    st.caption(
        "Los parametros se optimizan exclusivamente en el tramo de entrenamiento. "
        "El tramo de test se evalua una sola vez con los parametros congelados."
    )
    if data is None:
        st.info("Cargue datos para validar.")
        return
    if len(data) < 150:
        st.warning("Muy pocas barras para una validacion creible. Aporte mas historico.")
        return

    grid_text = st.text_area(
        "Rejilla de parametros (JSON)",
        value=default_grid(strategy_key),
        height=140,
        key=f"grid_{strategy_key}",
        help="Cada clave debe ser un parametro de la estrategia y cada valor una lista.",
    )
    try:
        grid = parse_grid(grid_text)
        combinations = grid_size(grid)
    except ValueError as exc:
        st.error(str(exc))
        return
    if combinations > config.backtest.max_optimization_combinations:
        st.error(
            f"{combinations} combinaciones superan el limite "
            f"{config.backtest.max_optimization_combinations}. Reduzca la rejilla."
        )
        return
    st.caption(f"Estrategia: {strategy_name} · {combinations} combinaciones por ventana.")

    control1, control2, control3 = st.columns(3)
    train_ratio = control1.slider("Proporcion de entrenamiento", 0.5, 0.9, 0.7, 0.05)
    min_trades = control2.number_input("Minimo de operaciones", 1, 500, 10, 5)
    objective = control3.selectbox(
        "Objetivo de optimizacion",
        ["sharpe", "sortino", "calmar", "profit_factor", "expectancy", "net_profit"],
        index=0,
    )

    buttons = st.columns(2)
    run_oos = buttons[0].button("Ejecutar train/test", type="primary", key="btn_oos")
    run_wf = buttons[1].button("Ejecutar walk-forward", key="btn_wf")

    if run_oos:
        with st.spinner("Optimizando en train y evaluando en test..."):
            try:
                result = train_test_evaluate(
                    data,
                    strategy_key=strategy_key,
                    param_grid=grid,
                    config=config,
                    symbol=symbol,
                    timeframe=timeframe,
                    train_ratio=train_ratio,
                    objective=objective,
                    min_trades=int(min_trades),
                )
                st.session_state["oos_result"] = result
                st.session_state["oos_meta"] = {
                    "strategy": strategy_name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                }
            except Exception as exc:
                st.error(f"No se pudo completar la validacion: {exc}")

    if run_wf:
        with st.spinner("Ejecutando walk-forward (puede tardar)..."):
            try:
                result = walk_forward(
                    data,
                    strategy_key=strategy_key,
                    param_grid=grid,
                    config=config,
                    symbol=symbol,
                    timeframe=timeframe,
                    n_folds=3,
                    train_ratio=0.6,
                    anchored=True,
                    objective=objective,
                    min_trades=int(min_trades),
                )
                st.session_state["wf_result"] = result
                st.session_state["wf_meta"] = {
                    "strategy": strategy_name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                }
            except Exception as exc:
                st.error(f"No se pudo completar el walk-forward: {exc}")

    oos = st.session_state.get("oos_result")
    if oos is not None:
        st.divider()
        st.markdown("##### Resultado train/test")
        meta = st.session_state.get("oos_meta", {})
        st.caption(
            f"{meta.get('strategy', strategy_name)} · {meta.get('symbol', symbol)} · "
            f"{meta.get('timeframe', timeframe)} · split {oos.split_time:%Y-%m-%d %H:%M}"
        )
        st.write("**Mejores parametros (elegidos solo en train):**")
        st.json(oos.best_params)
        comparison = pd.DataFrame(
            [
                {
                    "Tramo": "Entrenamiento",
                    "Beneficio neto": oos.train_metrics.get("net_profit"),
                    "Retorno %": oos.train_metrics.get("return_pct"),
                    "Operaciones": oos.train_metrics.get("trades"),
                    "Sharpe": oos.train_metrics.get("sharpe"),
                    "Max DD %": abs(oos.train_metrics.get("max_drawdown_pct", 0.0) or 0.0),
                    "Profit factor": oos.train_metrics.get("profit_factor"),
                },
                {
                    "Tramo": "Test (OOS)",
                    "Beneficio neto": oos.test_metrics.get("net_profit"),
                    "Retorno %": oos.test_metrics.get("return_pct"),
                    "Operaciones": oos.test_metrics.get("trades"),
                    "Sharpe": oos.test_metrics.get("sharpe"),
                    "Max DD %": abs(oos.test_metrics.get("max_drawdown_pct", 0.0) or 0.0),
                    "Profit factor": oos.test_metrics.get("profit_factor"),
                },
            ]
        )
        st.dataframe(comparison.round(3), width="stretch", hide_index=True)
        st.plotly_chart(
            charts.comparison_figure(
                {
                    "Train": oos.train_result.equity_curve,
                    "Test OOS": oos.test_result.equity_curve,
                }
            ),
            width="stretch",
        )
        if oos.degradation_pct is not None:
            st.metric("Degradacion del retorno train → test", pct(oos.degradation_pct))
        st.warning(
            "Un unico split puede tener suerte o mala suerte. El walk-forward ofrece "
            "una vision mas robusta de la consistencia."
        )

    wf = st.session_state.get("wf_result")
    if wf is not None:
        st.divider()
        st.markdown("##### Walk-forward (OOS agregado)")
        meta = st.session_state.get("wf_meta", {})
        st.caption(
            f"{meta.get('strategy', strategy_name)} · {meta.get('symbol', symbol)} · "
            f"{meta.get('timeframe', timeframe)} · {wf.n_folds} folds · {wf.objective}"
        )
        summary = wf.summary()
        cards = st.columns(5)
        cards[0].metric("Retorno OOS", pct(summary["oos_return_pct"]))
        cards[1].metric("Beneficio OOS", money(summary["oos_net_profit"]))
        cards[2].metric("Sharpe OOS", number(summary["oos_sharpe"]))
        cards[3].metric("Folds positivos", f"{summary['positive_folds']}/{summary['folds']}")
        cards[4].metric("Parametros estables", pct(summary["stable_params_pct"]))
        st.dataframe(
            wf.param_history,
            width="stretch",
            hide_index=True,
            height=min(360, 70 + 35 * len(wf.param_history)),
        )
        st.plotly_chart(
            charts.equity_figure(wf.oos_equity["equity"], wf.initial_capital, "Capital OOS encadenado"),
            width="stretch",
        )
        if wf.consistency_pct < 60:
            st.error(
                "Menos del 60% de los folds son positivos: la estrategia no muestra "
                "consistencia fuera de muestra."
            )
        elif summary["oos_net_profit"] <= 0:
            st.error("El resultado OOS agregado es negativo tras costes.")
        else:
            st.info(
                "Resultado OOS positivo, pero esto no garantiza rentabilidad futura. "
                "Revise sensibilidad de parametros y costes."
            )


def render_data(
    data: pd.DataFrame | None,
    *,
    symbol: str,
    timeframe: str,
    data_label: str,
    synthetic: bool,
    market_store: MarketDataStore,
) -> None:
    st.markdown("#### Datos de mercado")
    if data is None:
        st.info("No hay datos cargados.")
        return

    report = inspect_ohlcv(data, symbol)
    cards = st.columns(5)
    cards[0].metric("Barras", f"{len(data):,}")
    cards[1].metric("Inicio", f"{data.index[0]:%Y-%m-%d}")
    cards[2].metric("Fin", f"{data.index[-1]:%Y-%m-%d}")
    cards[3].metric("Precio minimo", f"{data['low'].min():,.4f}")
    cards[4].metric("Precio maximo", f"{data['high'].max():,.4f}")
    st.caption(
        f"Simbolo {symbol} · temporalidad {timeframe} · fuente: {data_label}"
    )
    if synthetic:
        st.warning(
            "Datos SINTETICOS de demostracion. No son precios de mercado y no sirven "
            "para concluir que una estrategia es rentable."
        )
    for warning in report.get("warnings", []):
        st.warning(warning)
    if report.get("missing_bars", 0):
        st.info(f"Huecos detectados (informativos): {report['missing_bars']} barras.")

    st.dataframe(data.tail(300), width="stretch", height=320)
    st.download_button(
        "Descargar datos procesados (CSV)",
        data=data.to_csv().encode("utf-8-sig"),
        file_name=f"{symbol}_{timeframe}.csv",
        mime="text/csv",
        key="download_data",
    )

    st.divider()
    st.markdown("##### Guardar en la base de datos local")
    if synthetic:
        st.caption("La importacion de datos sinteticos esta deshabilitada para no contaminar la base.")
    else:
        if st.button("Guardar serie en SQLite", key="save_series"):
            try:
                inserted = market_store.save_bars(symbol, timeframe, data)
                st.success(f"Guardadas {inserted} barras de {symbol} {timeframe}.")
                st.rerun()
            except Exception as exc:
                st.error(f"No se pudo guardar: {exc}")

    st.markdown("##### Series almacenadas")
    series = market_store.list_series()
    if series.empty:
        st.info("La base de datos local esta vacia.")
        return
    st.dataframe(series, width="stretch", hide_index=True)
    labels = [
        f"{row.symbol} · {row.timeframe} · {int(row.bars)} barras"
        for row in series.itertuples(index=False)
    ]
    selected = st.selectbox("Serie a eliminar", labels, key="delete_series_select")
    confirm = st.checkbox("Confirmo que quiero eliminar esta serie", key="confirm_delete_series")
    if st.button("Eliminar serie", disabled=not confirm, key="delete_series"):
        row = series.iloc[labels.index(selected)]
        deleted = market_store.delete_series(row["symbol"], row["timeframe"])
        st.success(f"Eliminadas {deleted} barras.")
        st.rerun()


def render_history(results_store: ResultsStore, *, key_prefix: str = "hist") -> None:
    st.markdown("#### Historial de backtests")
    runs = results_store.list_runs()
    if runs.empty:
        st.info("Todavia no hay resultados guardados.")
        return
    display = runs[
        ["created_at", "run_id", "symbol", "timeframe", "strategy_name",
         "initial_capital", "final_equity", "return_pct", "data_source"]
    ].copy()
    st.dataframe(display, width="stretch", hide_index=True)
    selected = st.selectbox(
        "Seleccionar run",
        runs["run_id"].tolist(),
        format_func=lambda run_id: (
            f"{run_id} · "
            f"{runs.loc[runs['run_id'] == run_id, 'strategy_name'].iloc[0]} · "
            f"{runs.loc[runs['run_id'] == run_id, 'symbol'].iloc[0]}"
        ),
        key=f"{key_prefix}_select",
    )
    if not selected:
        return
    try:
        loaded = results_store.load_run(selected)
    except Exception as exc:
        st.error(f"No se pudo cargar el run: {exc}")
        return
    run = loaded["run"]
    metrics = run["metrics"]
    cards = st.columns(5)
    cards[0].metric("Beneficio neto", money(metrics.get("net_profit")))
    cards[1].metric("Retorno", pct(metrics.get("return_pct")))
    cards[2].metric("Operaciones", str(metrics.get("trades", 0)))
    cards[3].metric("Max DD", pct(abs(metrics.get("max_drawdown_pct", 0.0))))
    cards[4].metric("Sharpe", number(metrics.get("sharpe")))

    st.plotly_chart(
        charts.equity_figure(
            loaded["equity"]["equity"], run["initial_capital"], "Capital del run guardado"
        ),
        width="stretch",
    )
    st.dataframe(loaded["trades"], width="stretch", hide_index=True, height=320)

    if not loaded["equity"].empty:
        reconstruction = BacktestResult(
            run_id=run["run_id"],
            symbol=run["symbol"],
            timeframe=run["timeframe"],
            strategy_key=run["strategy_key"],
            strategy_name=run["strategy_name"],
            params=run["params"],
            settings=run["settings"],
            instrument=get_instrument(run["symbol"]),
            initial_capital=float(run["initial_capital"]),
            final_equity=float(run["final_equity"]),
            equity_curve=loaded["equity"],
            trades=loaded["trades"],
            metrics=metrics,
            signals=pd.DataFrame(),
            data=pd.DataFrame(),
            start=pd.Timestamp(run["start_ts"]),
            end=pd.Timestamp(run["end_ts"]),
            warnings=run.get("warnings", []),
        )
        html = generate_html_report(reconstruction, data_source=run.get("data_source", ""))
        st.download_button(
            "Descargar informe HTML",
            data=html.encode("utf-8"),
            file_name=f"informe_{run['run_id']}.html",
            mime="text/html",
            key=f"{key_prefix}_download",
        )

    confirm = st.checkbox("Confirmo que quiero borrar este run", key=f"{key_prefix}_confirm")
    if st.button("Borrar run", disabled=not confirm, key=f"{key_prefix}_delete"):
        results_store.delete_run(selected)
        st.success("Run eliminado.")
        st.rerun()


def render_paper(
    data: pd.DataFrame | None,
    config: AppConfig,
    strategy_key: str,
    strategy_name: str,
    symbol: str,
    timeframe: str,
) -> None:
    st.markdown("#### Paper trading (replay)")
    st.caption(
        "Simulacion barra a barra con datos historicos. Este modulo esta separado del "
        "motor de backtesting y NO tiene conexion con ningun broker ni envia ordenes reales."
    )
    if data is None:
        st.info("Cargue datos para iniciar una sesion de paper trading.")
        return
    max_bars = len(data)
    if max_bars < 300:
        st.warning("Se recomiendan al menos 300 barras para el replay.")
    bars = st.slider(
        "Barras a reproducir (desde el final)",
        min_value=min(200, max_bars),
        max_value=max_bars,
        value=min(max(300, config.execution.warmup_bars + 200), max_bars),
        step=50,
        key="paper_bars",
    )
    if st.button("Iniciar replay de paper trading", type="primary", key="paper_run"):
        subset = data.iloc[-bars:]
        if len(subset) <= config.execution.warmup_bars + 2:
            st.error("El tramo es mas corto que el warmup. Reduzca warmup o aumente barras.")
        else:
            with st.spinner("Reproduciendo barras..."):
                try:
                    strategy = create_strategy(strategy_key, {})
                    engine = PaperTradingEngine(
                        config, strategy, symbol=symbol, timeframe=timeframe
                    )
                    state = engine.run_replay(subset)
                    store = PaperSessionStore(PROJECT_ROOT / "paper_trading" / "sessions")
                    path = store.save(state)
                    st.session_state["paper_state"] = state
                    st.session_state["paper_summary"] = engine.summary()
                    st.session_state["paper_path"] = str(path)
                    st.session_state["paper_meta"] = {
                        "strategy": strategy_name,
                        "symbol": symbol,
                        "timeframe": timeframe,
                    }
                except Exception as exc:
                    st.error(f"No se pudo completar el replay: {exc}")

    state = st.session_state.get("paper_state")
    if state is None:
        return
    summary = st.session_state.get("paper_summary", {})
    meta = st.session_state.get("paper_meta", {})
    st.divider()
    st.markdown(f"##### Sesion {state.session_id}")
    st.caption(
        f"{meta.get('strategy', strategy_name)} · {meta.get('symbol', symbol)} · "
        f"{meta.get('timeframe', timeframe)} · guardada en "
        f"{st.session_state.get('paper_path', '')}"
    )
    cards = st.columns(5)
    cards[0].metric("Capital", money(state.equity))
    cards[1].metric("Retorno", pct(summary.get("return_pct")))
    cards[2].metric("Operaciones", str(summary.get("trades", 0)))
    cards[3].metric("Barras", str(state.bars_processed))
    cards[4].metric("Posicion abierta", "Si" if state.position else "No")
    if state.position:
        st.warning("La sesion termino con una posicion abierta; el replay no la cerro.")
    equity = pd.DataFrame(state.equity_curve)
    if not equity.empty:
        equity = equity.set_index("timestamp")
        st.plotly_chart(
            charts.equity_figure(equity["equity"], state.initial_capital, "Capital paper"),
            width="stretch",
        )
    trades = pd.DataFrame(state.trades)
    if trades.empty:
        st.info("No se ejecutaron operaciones en el replay.")
    else:
        st.dataframe(trades, width="stretch", hide_index=True, height=320)
