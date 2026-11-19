"""Separacion cronologica, optimizacion in-sample y walk-forward.

Regla fundamental: los parametros se eligen SOLO con el tramo de entrenamiento.
El tramo de validacion no participa en la seleccion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from trading_bot_lab.config import AppConfig
from trading_bot_lab.strategies import (
    create_strategy,
    expand_parameter_grid,
    get_strategy_class,
)
from trading_bot_lab.types import Timeframe

from .metrics import compute_metrics, objective_score


class ValidationError(ValueError):
    """Configuracion de validacion invalida."""


@dataclass
class OptimizationResult:
    best_params: dict[str, Any]
    best_metrics: dict[str, Any]
    best_result: Any
    trials: pd.DataFrame
    combinations: int


@dataclass
class OutOfSampleResult:
    best_params: dict[str, Any]
    train_result: Any
    test_result: Any
    train_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    split_time: pd.Timestamp

    @property
    def degradation_pct(self) -> float | None:
        train_return = float(self.train_metrics.get("return_pct", 0.0) or 0.0)
        test_return = float(self.test_metrics.get("return_pct", 0.0) or 0.0)
        if abs(train_return) < 1e-12:
            return None
        return (test_return - train_return) / abs(train_return) * 100.0


@dataclass
class WalkForwardFold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict[str, Any]
    train_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    test_result: Any


@dataclass
class WalkForwardResult:
    folds: list[WalkForwardFold]
    oos_equity: pd.DataFrame
    oos_trades: pd.DataFrame
    oos_metrics: dict[str, Any]
    param_history: pd.DataFrame
    initial_capital: float
    objective: str
    train_ratio: float
    n_folds: int
    anchored: bool
    notes: list[str] = field(default_factory=list)

    @property
    def positive_folds(self) -> int:
        return sum(1 for fold in self.folds if fold.test_metrics.get("net_profit", 0) > 0)

    @property
    def consistency_pct(self) -> float:
        return self.positive_folds / len(self.folds) * 100.0 if self.folds else 0.0

    @property
    def stable_params_pct(self) -> float:
        if not self.folds:
            return 0.0
        counts: dict[str, int] = {}
        for fold in self.folds:
            key = json.dumps(fold.best_params, sort_keys=True)
            counts[key] = counts.get(key, 0) + 1
        return max(counts.values()) / len(self.folds) * 100.0

    def summary(self) -> dict[str, Any]:
        return {
            "folds": len(self.folds),
            "positive_folds": self.positive_folds,
            "consistency_pct": self.consistency_pct,
            "stable_params_pct": self.stable_params_pct,
            "oos_net_profit": self.oos_metrics.get("net_profit"),
            "oos_return_pct": self.oos_metrics.get("return_pct"),
            "oos_sharpe": self.oos_metrics.get("sharpe"),
            "oos_max_drawdown_pct": self.oos_metrics.get("max_drawdown_pct"),
            "objective": self.objective,
            "train_ratio": self.train_ratio,
            "anchored": self.anchored,
        }


def chronological_split(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Divide los datos por fecha, sin mezclar pasado y futuro."""
    if not 0.05 <= train_ratio <= 0.95:
        raise ValidationError("train_ratio debe estar entre 0.05 y 0.95")
    if len(data) < 20:
        raise ValidationError("Se necesitan al menos 20 barras para dividir.")
    split_at = int(len(data) * train_ratio)
    if split_at < 5 or len(data) - split_at < 5:
        raise ValidationError("La division deja un tramo demasiado corto.")
    train = data.iloc[:split_at].copy()
    test = data.iloc[split_at:].copy()
    if train.index[-1] >= test.index[0]:
        raise ValidationError("La division cronologica no es estricta.")
    return train, test, test.index[0]


def optimize_parameters(
    data: pd.DataFrame,
    *,
    strategy_key: str,
    param_grid: dict[str, list[Any]],
    config: AppConfig,
    symbol: str,
    timeframe: str | Timeframe,
    objective: str = "sharpe",
    min_trades: int | None = None,
    instrument: Any = None,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> OptimizationResult:
    """Optimiza parametros SOLO sobre ``data`` (que debe ser in-sample)."""
    from trading_bot_lab.backtesting.engine import BacktestEngine

    strategy_class = get_strategy_class(strategy_key)
    combinations = expand_parameter_grid(strategy_class.parameter_specs(), param_grid)
    limit = config.backtest.max_optimization_combinations
    if len(combinations) > limit:
        raise ValidationError(
            f"La rejilla tiene {len(combinations)} combinaciones; el limite es {limit}. "
            "Reduzca la rejilla para mitigar el sobreajuste."
        )
    minimum_trades = (
        config.backtest.min_trades_for_ranking if min_trades is None else int(min_trades)
    )
    engine = BacktestEngine(config)

    rows: list[dict[str, Any]] = []
    best_result = None
    best_score: tuple | None = None
    best_params: dict[str, Any] = {}
    best_metrics: dict[str, Any] = {}

    for index, combo in enumerate(combinations, start=1):
        params = {**strategy_class.default_params(), **combo}
        row: dict[str, Any] = {"combination": index, **combo}
        try:
            strategy = create_strategy(strategy_key, params)
            result = engine.run(
                data,
                strategy,
                symbol=symbol,
                timeframe=timeframe,
                instrument=instrument,
            )
            metrics = result.metrics
            score = objective_score(metrics, objective, minimum_trades)
            row.update(
                {
                    "trades": metrics.get("trades"),
                    "net_profit": metrics.get("net_profit"),
                    "return_pct": metrics.get("return_pct"),
                    "sharpe": metrics.get("sharpe"),
                    "max_drawdown_pct": metrics.get("max_drawdown_pct"),
                    "profit_factor": metrics.get("profit_factor"),
                    "objective_value": score[1],
                    "meets_min_trades": bool(score[0]),
                    "error": "",
                }
            )
            if best_score is None or score > best_score:
                best_score = score
                best_result = result
                best_params = dict(result.params)
                best_metrics = dict(metrics)
        except Exception as exc:  # se conserva el error para auditoria
            row.update(
                {
                    "trades": 0,
                    "net_profit": 0.0,
                    "return_pct": 0.0,
                    "sharpe": None,
                    "max_drawdown_pct": 0.0,
                    "profit_factor": None,
                    "objective_value": None,
                    "meets_min_trades": False,
                    "error": str(exc),
                }
            )
        rows.append(row)
        if progress is not None:
            progress(index, len(combinations), row)

    if best_result is None:
        errors = sorted({row["error"] for row in rows if row["error"]})
        raise ValidationError(f"Ninguna combinacion pudo ejecutarse: {errors[:3]}")

    return OptimizationResult(
        best_params=best_params,
        best_metrics=best_metrics,
        best_result=best_result,
        trials=pd.DataFrame(rows),
        combinations=len(combinations),
    )


def train_test_evaluate(
    data: pd.DataFrame,
    *,
    strategy_key: str,
    param_grid: dict[str, list[Any]],
    config: AppConfig,
    symbol: str,
    timeframe: str | Timeframe,
    train_ratio: float = 0.7,
    objective: str = "sharpe",
    min_trades: int | None = None,
    instrument: Any = None,
) -> OutOfSampleResult:
    """Optimiza en train, congela parametros y evalua una sola vez en test."""
    from trading_bot_lab.backtesting.engine import BacktestEngine

    train, test, split_time = chronological_split(data, train_ratio)
    optimization = optimize_parameters(
        train,
        strategy_key=strategy_key,
        param_grid=param_grid,
        config=config,
        symbol=symbol,
        timeframe=timeframe,
        objective=objective,
        min_trades=min_trades,
        instrument=instrument,
    )
    engine = BacktestEngine(config)
    test_result = engine.run(
        test,
        create_strategy(strategy_key, optimization.best_params),
        symbol=symbol,
        timeframe=timeframe,
        instrument=instrument,
    )
    return OutOfSampleResult(
        best_params=optimization.best_params,
        train_result=optimization.best_result,
        test_result=test_result,
        train_metrics=optimization.best_metrics,
        test_metrics=test_result.metrics,
        split_time=split_time,
    )


def walk_forward(
    data: pd.DataFrame,
    *,
    strategy_key: str,
    param_grid: dict[str, list[Any]],
    config: AppConfig,
    symbol: str,
    timeframe: str | Timeframe,
    n_folds: int = 3,
    train_ratio: float = 0.6,
    anchored: bool = True,
    objective: str = "sharpe",
    min_trades: int | None = None,
    instrument: Any = None,
) -> WalkForwardResult:
    """Walk-forward con optimizacion en cada ventana de train.

    Cada bloque de test es estrictamente posterior a su train. Los resultados
    OOS se encadenan para construir una curva de capital agregada.
    """
    from trading_bot_lab.backtesting.engine import BacktestEngine

    if n_folds < 2:
        raise ValidationError("n_folds debe ser >= 2")
    if not 0.2 <= train_ratio <= 0.9:
        raise ValidationError("train_ratio debe estar entre 0.2 y 0.9")
    if len(data) < 100:
        raise ValidationError("Se recomiendan al menos 100 barras para walk-forward.")

    total = len(data)
    initial_train = int(total * train_ratio)
    remaining = total - initial_train
    if remaining < n_folds * 20:
        raise ValidationError(
            "Muy pocas barras para el numero de folds. Reduzca folds o aporte mas historico."
        )

    blocks = np.array_split(np.arange(initial_train, total), n_folds)
    engine = BacktestEngine(config)
    folds: list[WalkForwardFold] = []
    parameter_rows: list[dict[str, Any]] = []
    notes: list[str] = [
        "Los costes se aplican en todas las ventanas.",
        "La rejilla se optimiza en cada train y se evalua una vez en su test.",
    ]

    for fold_number, block in enumerate(blocks, start=1):
        test_start_idx = int(block[0])
        test_end_idx = int(block[-1]) + 1
        train_start_idx = 0 if anchored else max(0, test_start_idx - initial_train)
        train_data = data.iloc[train_start_idx:test_start_idx]
        test_data = data.iloc[test_start_idx:test_end_idx]
        if len(test_data) <= config.execution.warmup_bars + 2:
            raise ValidationError(
                f"El fold {fold_number} tiene {len(test_data)} barras de test y warmup="
                f"{config.execution.warmup_bars}. Reduzca folds, warmup o aporte mas datos."
            )

        optimization = optimize_parameters(
            train_data,
            strategy_key=strategy_key,
            param_grid=param_grid,
            config=config,
            symbol=symbol,
            timeframe=timeframe,
            objective=objective,
            min_trades=min_trades,
            instrument=instrument,
        )
        test_result = engine.run(
            test_data,
            create_strategy(strategy_key, optimization.best_params),
            symbol=symbol,
            timeframe=timeframe,
            instrument=instrument,
        )
        folds.append(
            WalkForwardFold(
                fold=fold_number,
                train_start=train_data.index[0],
                train_end=train_data.index[-1],
                test_start=test_data.index[0],
                test_end=test_data.index[-1],
                best_params=optimization.best_params,
                train_metrics=optimization.best_metrics,
                test_metrics=test_result.metrics,
                test_result=test_result,
            )
        )
        parameter_rows.append(
            {
                "fold": fold_number,
                "train_start": train_data.index[0],
                "train_end": train_data.index[-1],
                "test_start": test_data.index[0],
                "test_end": test_data.index[-1],
                "params": json.dumps(optimization.best_params, sort_keys=True),
                "train_net_profit": optimization.best_metrics.get("net_profit"),
                "train_sharpe": optimization.best_metrics.get("sharpe"),
                "test_net_profit": test_result.metrics.get("net_profit"),
                "test_sharpe": test_result.metrics.get("sharpe"),
                "test_return_pct": test_result.metrics.get("return_pct"),
                "test_trades": test_result.metrics.get("trades"),
            }
        )

    oos_equity, oos_trades = _chain_oos_results(folds, config.account.initial_capital)
    oos_metrics = compute_metrics(
        oos_equity,
        oos_trades,
        initial_capital=config.account.initial_capital,
        annualization_days=config.backtest.annualization_days,
    )
    notes.append(
        "La curva OOS encadena el capital de cada fold; las operaciones se escalan "
        "al capital de inicio de su fold."
    )
    return WalkForwardResult(
        folds=folds,
        oos_equity=oos_equity,
        oos_trades=oos_trades,
        oos_metrics=oos_metrics,
        param_history=pd.DataFrame(parameter_rows),
        initial_capital=config.account.initial_capital,
        objective=objective,
        train_ratio=train_ratio,
        n_folds=n_folds,
        anchored=anchored,
        notes=notes,
    )


def _chain_oos_results(
    folds: list[WalkForwardFold],
    initial_capital: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    equity_parts: list[pd.Series] = []
    position_parts: list[pd.Series] = []
    trade_parts: list[pd.DataFrame] = []
    running_capital = float(initial_capital)

    for fold in folds:
        equity = fold.test_result.equity_curve["equity"].astype(float).copy()
        scale = running_capital / initial_capital
        scaled = equity * scale
        scaled.name = "equity"
        equity_parts.append(scaled)
        if "in_position" in fold.test_result.equity_curve.columns:
            position_parts.append(fold.test_result.equity_curve["in_position"].astype(bool))

        trades = fold.test_result.trades.copy()
        if not trades.empty:
            for column in ("gross_pnl_account", "costs_account", "net_pnl_account"):
                if column in trades.columns:
                    trades[column] = trades[column].astype(float) * scale
            trades["fold"] = fold.fold
            trade_parts.append(trades)

        running_capital = float(scaled.iloc[-1])

    combined_equity = pd.concat(equity_parts).sort_index()
    combined_equity = combined_equity[~combined_equity.index.duplicated(keep="last")]
    combined_in_position = (
        pd.concat(position_parts).sort_index()
        if position_parts
        else pd.Series(False, index=combined_equity.index)
    )
    combined_in_position = combined_in_position[
        ~combined_in_position.index.duplicated(keep="last")
    ].reindex(combined_equity.index, fill_value=False)
    peak = combined_equity.cummax()
    equity_frame = pd.DataFrame(
        {
            "equity": combined_equity,
            "drawdown": combined_equity / peak - 1.0,
            "in_position": combined_in_position.astype(bool),
        }
    )
    equity_frame.index.name = "timestamp"

    if trade_parts:
        combined_trades = pd.concat(trade_parts, ignore_index=True)
    else:
        combined_trades = pd.DataFrame(columns=["net_pnl_account", "costs_account", "direction"])
    return equity_frame, combined_trades
