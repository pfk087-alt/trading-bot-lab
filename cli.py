"""CLI de TRADING BOT LAB.

Ejemplos:
    python -m trading_bot_lab.cli list-strategies
    python -m trading_bot_lab.cli validate-csv data/raw/EURUSD.csv
    python -m trading_bot_lab.cli backtest --csv data/raw/EURUSD.csv \
        --symbol EURUSD --timeframe 1h --strategy ema_cross
    python -m trading_bot_lab.cli walkforward --csv data/raw/EURUSD.csv \
        --symbol EURUSD --timeframe 1h --strategy ema_cross \
        --grid '{"fast_period":[10,20],"slow_period":[50,100]}'

No ejecuta ordenes reales ni se conecta a brokers.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd

from trading_bot_lab.analytics.report import save_html_report
from trading_bot_lab.analytics.store import ResultsStore
from trading_bot_lab.analytics.validation import walk_forward
from trading_bot_lab.backtesting import BacktestEngine, verify_causality
from trading_bot_lab.config import AppConfig, ConfigError, load_config
from trading_bot_lab.data import (
    CSVProvider,
    DataRequest,
    DataValidationError,
    ProviderError,
    inspect_ohlcv,
)
from trading_bot_lab.data.synthetic import generate_synthetic_ohlcv
from trading_bot_lab.logging_config import setup_logging
from trading_bot_lab.paper_trading import PaperSessionStore, PaperTradingEngine
from trading_bot_lab.strategies import (
    create_strategy,
    default_params,
    get_strategy_class,
    strategy_catalog,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default))


def _load_config(args: argparse.Namespace) -> AppConfig:
    config = load_config(args.config)
    account = replace(
        config.account,
        initial_capital=float(args.capital)
        if getattr(args, "capital", None) is not None
        else config.account.initial_capital,
    )
    risk = replace(
        config.risk,
        risk_per_trade_pct=float(args.risk)
        if getattr(args, "risk", None) is not None
        else config.risk.risk_per_trade_pct,
    )
    costs = replace(
        config.costs,
        spread_pips=float(args.spread)
        if getattr(args, "spread", None) is not None
        else config.costs.spread_pips,
        commission_per_lot_per_side=float(args.commission)
        if getattr(args, "commission", None) is not None
        else config.costs.commission_per_lot_per_side,
        slippage_pips=float(args.slippage)
        if getattr(args, "slippage", None) is not None
        else config.costs.slippage_pips,
    )
    execution = replace(
        config.execution,
        warmup_bars=int(args.warmup)
        if getattr(args, "warmup", None) is not None
        else config.execution.warmup_bars,
        stop_mode=args.stop if getattr(args, "stop", None) else config.execution.stop_mode,
        stop_value=float(args.stop_value)
        if getattr(args, "stop_value", None) is not None
        else config.execution.stop_value,
        take_profit_mode=args.tp
        if getattr(args, "tp", None)
        else config.execution.take_profit_mode,
        take_profit_value=float(args.tp_value)
        if getattr(args, "tp_value", None) is not None
        else config.execution.take_profit_value,
    )
    return replace(config, account=account, risk=risk, costs=costs, execution=execution)


def _params_from_args(strategy_key: str, pairs: list[str]) -> dict[str, Any]:
    params = default_params(strategy_key)
    specs = {spec.name: spec for spec in get_strategy_class(strategy_key).parameter_specs()}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Parametro invalido '{item}'. Use nombre=valor.")
        name, raw = item.split("=", 1)
        name = name.strip()
        if name not in specs:
            raise ValueError(f"Parametro desconocido: {name}")
        params[name] = specs[name].coerce(raw)
    return params


def _load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    if getattr(args, "synthetic", False):
        print(
            "ADVERTENCIA: datos sinteticos. No son precios de mercado.",
            file=sys.stderr,
        )
        config = load_config(args.config)
        start_price = 4000.0 if args.symbol.upper() in {"SPX500", "NAS100", "GER40"} else 1.10
        data = generate_synthetic_ohlcv(
            symbol=args.symbol,
            timeframe=args.timeframe,
            periods=int(args.periods),
            start_price=start_price,
            volatility=0.004 if start_price > 100 else 0.0022,
            seed=int(args.seed),
        )
        return data, "synthetic-demo"
    if not args.csv:
        raise ValueError("Debe indicar --csv o usar --synthetic.")
    provider = CSVProvider(args.csv)
    request = DataRequest(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start=getattr(args, "start", None),
        end=getattr(args, "end", None),
    )
    data = provider.fetch(request)
    return data, str(args.csv)


def _output_dir(value: str | None) -> Path:
    path = Path(value) if value else Path("reports")
    path.mkdir(parents=True, exist_ok=True)
    return path


def command_list_strategies(_: argparse.Namespace) -> int:
    for entry in strategy_catalog():
        print(f"\n{entry['key']}: {entry['name']}")
        print(f"  {entry['description']}")
        for spec in entry["parameters"]:
            print(
                f"    - {spec['name']} ({spec['kind']}, default={spec['default']}, "
                f"rango={spec['minimum']}..{spec['maximum']})"
            )
    return 0


def command_validate_csv(args: argparse.Namespace) -> int:
    try:
        raw = pd.read_csv(args.csv, sep=None, engine="python", encoding="utf-8-sig")
        report = inspect_ohlcv(raw, args.symbol)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        print(f"ERROR: no se pudo leer el CSV: {exc}", file=sys.stderr)
        return 2
    _print_json(report)
    return 0 if report.get("valid") else 1


def command_backtest(args: argparse.Namespace) -> int:
    config = _load_config(args)
    data, data_source = _load_data(args)
    params = _params_from_args(args.strategy, args.param)
    strategy = create_strategy(args.strategy, params)
    if args.check_causality:
        verify_causality(strategy, data, checks=5)
    result = BacktestEngine(config).run(
        data, strategy, symbol=args.symbol, timeframe=args.timeframe
    )

    if args.save:
        ResultsStore(config.database_path).save_result(result, data_source=data_source)

    output = _output_dir(args.output)
    save_html_report(result, output / f"informe_{result.run_id}.html", data_source=data_source)
    result.trades.to_csv(output / f"trades_{result.run_id}.csv", index=False)
    result.equity_curve.to_csv(output / f"equity_{result.run_id}.csv")
    (output / f"metricas_{result.run_id}.json").write_text(
        json.dumps(result.to_summary(), ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _print_json(result.to_summary())
    return 0


def command_walkforward(args: argparse.Namespace) -> int:
    config = _load_config(args)
    data, _ = _load_data(args)
    try:
        grid = json.loads(args.grid)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--grid debe ser JSON valido: {exc}") from exc
    result = walk_forward(
        data,
        strategy_key=args.strategy,
        param_grid=grid,
        config=config,
        symbol=args.symbol,
        timeframe=args.timeframe,
        n_folds=int(args.folds),
        train_ratio=float(args.train_ratio),
        anchored=not args.rolling,
        objective=args.objective,
        min_trades=int(args.min_trades),
    )
    output = _output_dir(args.output)
    (output / "wf_summary.json").write_text(
        json.dumps(result.summary(), ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    result.param_history.to_csv(output / "wf_folds.csv", index=False)
    result.oos_equity.to_csv(output / "wf_oos_equity.csv")
    result.oos_trades.to_csv(output / "wf_oos_trades.csv", index=False)
    _print_json(result.summary())
    return 0


def command_paper(args: argparse.Namespace) -> int:
    config = _load_config(args)
    data, _ = _load_data(args)
    if args.bars > 0:
        data = data.iloc[-int(args.bars):]
    params = _params_from_args(args.strategy, args.param)
    strategy = create_strategy(args.strategy, params)
    engine = PaperTradingEngine(
        config, strategy, symbol=args.symbol, timeframe=args.timeframe
    )
    state = engine.run_replay(data)
    store = PaperSessionStore(args.session_dir)
    path = store.save(state)
    trades_path = store.export_trades_csv(state)
    summary = engine.summary()
    summary["session_file"] = str(path)
    summary["trades_file"] = str(trades_path)
    _print_json(summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trading-bot-lab",
        description="Laboratorio de backtesting. No ejecuta ordenes reales.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-strategies", help="Lista estrategias y parametros")

    validate = subparsers.add_parser("validate-csv", help="Valida un CSV OHLCV")
    validate.add_argument("csv")
    validate.add_argument("--symbol", default="")

    def add_common(sub: argparse.ArgumentParser, *, data_required: bool = False) -> None:
        sub.add_argument("--csv", help="Ruta del CSV OHLCV")
        sub.add_argument("--synthetic", action="store_true", help="Usar datos sinteticos de demo")
        sub.add_argument("--periods", type=int, default=3000)
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--symbol", default="EURUSD")
        sub.add_argument("--timeframe", default="1h")
        sub.add_argument("--start")
        sub.add_argument("--end")
        sub.add_argument("--config")
        sub.add_argument("--strategy", default="ema_cross")
        sub.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
        sub.add_argument("--capital", type=float)
        sub.add_argument("--risk", type=float)
        sub.add_argument("--spread", type=float)
        sub.add_argument("--commission", type=float)
        sub.add_argument("--slippage", type=float)
        sub.add_argument("--warmup", type=int)
        sub.add_argument("--stop", choices=["atr", "pips", "percent", "none"])
        sub.add_argument("--stop-value", type=float)
        sub.add_argument("--tp", choices=["rr", "atr", "pips", "percent", "none"])
        sub.add_argument("--tp-value", type=float)
        sub.add_argument("--output", default="reports")

    backtest = subparsers.add_parser("backtest", help="Ejecuta un backtest")
    add_common(backtest)
    backtest.add_argument("--save", action="store_true", help="Guardar en SQLite")
    backtest.add_argument("--check-causality", action="store_true")
    backtest.set_defaults(func=command_backtest)

    wf = subparsers.add_parser("walkforward", help="Ejecuta walk-forward OOS")
    add_common(wf)
    wf.add_argument("--grid", required=True, help="Rejilla JSON de parametros")
    wf.add_argument("--folds", type=int, default=3)
    wf.add_argument("--train-ratio", type=float, default=0.6)
    wf.add_argument("--rolling", action="store_true", help="Ventana rolling en vez de anclada")
    wf.add_argument(
        "--objective",
        default="sharpe",
        choices=["sharpe", "sortino", "calmar", "profit_factor", "expectancy", "net_profit"],
    )
    wf.add_argument("--min-trades", type=int, default=10)
    wf.set_defaults(func=command_walkforward)

    paper = subparsers.add_parser("paper", help="Replay de paper trading")
    add_common(paper)
    paper.add_argument("--bars", type=int, default=0, help="Usar solo las ultimas N barras")
    paper.add_argument("--session-dir", default="paper_trading/sessions")
    paper.set_defaults(func=command_paper)

    subparsers.choices["list-strategies"].set_defaults(func=command_list_strategies)
    validate.set_defaults(func=command_validate_csv)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(getattr(args, "config", None))
        setup_logging(config.log_path, config.logging.level)
        return int(args.func(args))
    except (ConfigError, DataValidationError, ProviderError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrumpido.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
