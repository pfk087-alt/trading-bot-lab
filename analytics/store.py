"""Persistencia de backtests, operaciones y curvas de capital en SQLite."""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_bot_lab.backtesting.models import BacktestResult, Trade

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    strategy_key TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    params_json TEXT NOT NULL,
    settings_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    initial_capital REAL NOT NULL,
    final_equity REAL NOT NULL,
    start_ts TEXT NOT NULL,
    end_ts TEXT NOT NULL,
    data_source TEXT,
    warnings_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trades (
    run_id TEXT NOT NULL,
    trade_id INTEGER NOT NULL,
    symbol TEXT,
    direction TEXT,
    entry_time TEXT,
    exit_time TEXT,
    entry_price_raw REAL,
    entry_price REAL,
    exit_price_raw REAL,
    exit_price REAL,
    lots REAL,
    gross_pnl_quote REAL,
    gross_pnl_account REAL,
    costs_account REAL,
    net_pnl_account REAL,
    return_pct REAL,
    exit_reason TEXT,
    bars_held INTEGER,
    stop_price REAL,
    initial_stop_price REAL,
    take_profit_price REAL,
    mae REAL,
    mfe REAL,
    entry_conversion_rate REAL,
    exit_conversion_rate REAL,
    PRIMARY KEY (run_id, trade_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS equity (
    run_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    equity REAL NOT NULL,
    drawdown REAL,
    in_position INTEGER,
    PRIMARY KEY (run_id, ts),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);
"""

TRADE_COLUMNS = Trade.columns()
MONETARY_COLUMNS = {
    "gross_pnl_quote", "gross_pnl_account", "costs_account", "net_pnl_account"
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Tipo no serializable: {type(value)!r}")


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=_json_default, allow_nan=False)


class ResultsStore:
    """Repositorio de resultados. No almacena credenciales."""

    def __init__(self, database: str | Path) -> None:
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def save_result(self, result: BacktestResult, *, data_source: str = "") -> str:
        trades = result.trades.copy()
        if not trades.empty:
            for column in ("entry_time", "exit_time"):
                trades[column] = pd.to_datetime(trades[column], utc=True).astype(str)

        equity = result.equity_curve.reset_index()
        equity["timestamp"] = pd.to_datetime(equity["timestamp"], utc=True).astype(str)
        equity = equity[["timestamp", "equity", "drawdown", "in_position"]]

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO runs
                (run_id, created_at, symbol, timeframe, strategy_key, strategy_name,
                 params_json, settings_json, metrics_json, initial_capital, final_equity,
                 start_ts, end_ts, data_source, warnings_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.run_id,
                    datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    result.symbol,
                    result.timeframe,
                    result.strategy_key,
                    result.strategy_name,
                    _dumps(result.params),
                    _dumps(result.settings),
                    _dumps(result.metrics),
                    float(result.initial_capital),
                    float(result.final_equity),
                    str(result.start),
                    str(result.end),
                    data_source,
                    _dumps(result.warnings),
                ),
            )
            connection.execute("DELETE FROM trades WHERE run_id = ?", (result.run_id,))
            connection.execute("DELETE FROM equity WHERE run_id = ?", (result.run_id,))
            if not trades.empty:
                columns = ["run_id", *TRADE_COLUMNS]
                connection.executemany(
                    f"""
                    INSERT INTO trades ({", ".join(columns)})
                    VALUES ({", ".join(["?"] * len(columns))})
                    """,
                    [
                        (result.run_id, *(row[column] for column in TRADE_COLUMNS))
                        for _, row in trades.iterrows()
                    ],
                )
            connection.executemany(
                "INSERT INTO equity (run_id, ts, equity, drawdown, in_position) VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        result.run_id,
                        row.timestamp,
                        float(row.equity),
                        float(row.drawdown),
                        int(bool(row.in_position)),
                    )
                    for row in equity.itertuples(index=False)
                ],
            )
        return result.run_id

    def list_runs(self, limit: int = 200) -> pd.DataFrame:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, created_at, symbol, timeframe, strategy_name,
                       initial_capital, final_equity, start_ts, end_ts, data_source
                FROM runs ORDER BY created_at DESC LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        columns = [
            "run_id", "created_at", "symbol", "timeframe", "strategy_name",
            "initial_capital", "final_equity", "start_ts", "end_ts", "data_source",
        ]
        frame = pd.DataFrame(rows, columns=columns)
        if not frame.empty:
            frame["return_pct"] = (
                frame["final_equity"] / frame["initial_capital"] - 1.0
            ) * 100.0
            frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
        return frame

    def load_run(self, run_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            run = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if run is None:
                raise KeyError(f"No existe el run {run_id}")
            run_columns = [description[0] for description in connection.execute(
                "SELECT * FROM runs LIMIT 0"
            ).description]
            trades_rows = connection.execute(
                "SELECT * FROM trades WHERE run_id = ? ORDER BY trade_id", (run_id,)
            ).fetchall()
            trades_columns = [description[0] for description in connection.execute(
                "SELECT * FROM trades LIMIT 0"
            ).description]
            equity_rows = connection.execute(
                "SELECT ts, equity, drawdown, in_position FROM equity WHERE run_id = ? ORDER BY ts",
                (run_id,),
            ).fetchall()

        run_dict = dict(zip(run_columns, run))
        run_dict["params"] = json.loads(run_dict.pop("params_json"))
        run_dict["settings"] = json.loads(run_dict.pop("settings_json"))
        run_dict["metrics"] = json.loads(run_dict.pop("metrics_json"))
        run_dict["warnings"] = json.loads(run_dict.pop("warnings_json"))

        trades = pd.DataFrame(trades_rows, columns=trades_columns)
        if not trades.empty:
            trades = trades.drop(columns=["run_id"])
            for column in ("entry_time", "exit_time"):
                trades[column] = pd.to_datetime(trades[column], utc=True)
        equity = pd.DataFrame(equity_rows, columns=["timestamp", "equity", "drawdown", "in_position"])
        if not equity.empty:
            equity["timestamp"] = pd.to_datetime(equity["timestamp"], utc=True)
            equity = equity.set_index("timestamp")
            equity["in_position"] = equity["in_position"].astype(bool)
        return {"run": run_dict, "trades": trades, "equity": equity}

    def delete_run(self, run_id: str) -> bool:
        with self._connect() as connection:
            connection.execute("DELETE FROM equity WHERE run_id = ?", (run_id,))
            connection.execute("DELETE FROM trades WHERE run_id = ?", (run_id,))
            cursor = connection.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            return cursor.rowcount > 0

    def count_runs(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM runs").fetchone()
        return int(row[0])
