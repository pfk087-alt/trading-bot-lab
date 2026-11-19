"""Almacenamiento de barras de mercado en SQLite."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from trading_bot_lab.from trading_bot_lab.bot_types import import Timeframe

from .validation import validate_ohlcv

SCHEMA = """
CREATE TABLE IF NOT EXISTS market_bars (
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    ts TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    PRIMARY KEY (symbol, timeframe, ts)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_market_bars_ts ON market_bars(ts);
"""


class MarketDataStore:
    """Repositorio local de barras OHLCV validadas."""

    def __init__(self, database: str | Path) -> None:
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def save_bars(self, symbol: str, timeframe: str | Timeframe, data: pd.DataFrame) -> int:
        tf = Timeframe.parse(timeframe)
        clean = validate_ohlcv(data, symbol=symbol, timeframe=tf.value)
        rows = [
            (
                symbol.upper(),
                tf.value,
                timestamp.isoformat(),
                float(row.open),
                float(row.high),
                float(row.low),
                float(row.close),
                float(row.volume),
            )
            for timestamp, row in clean.iterrows()
        ]
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO market_bars
                (symbol, timeframe, ts, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def load_bars(
        self,
        symbol: str,
        timeframe: str | Timeframe,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        tf = Timeframe.parse(timeframe)
        query = (
            "SELECT ts, open, high, low, close, volume FROM market_bars "
            "WHERE symbol = ? AND timeframe = ?"
        )
        params: list[object] = [symbol.upper(), tf.value]
        if start is not None:
            ts = pd.Timestamp(start)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            query += " AND ts >= ?"
            params.append(ts.isoformat())
        if end is not None:
            ts = pd.Timestamp(end)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            query += " AND ts <= ?"
            params.append(ts.isoformat())
        query += " ORDER BY ts"

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()

        if not rows:
            empty = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"],
                index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
            )
            return empty

        frame = pd.DataFrame(
            rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.set_index("timestamp").sort_index()
        frame.attrs["symbol"] = symbol.upper()
        frame.attrs["timeframe"] = tf.value
        return frame

    def list_series(self) -> pd.DataFrame:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT symbol, timeframe, COUNT(*) AS bars, MIN(ts) AS start, MAX(ts) AS end
                FROM market_bars
                GROUP BY symbol, timeframe
                ORDER BY symbol, timeframe
                """
            ).fetchall()
        columns = ["symbol", "timeframe", "bars", "start", "end"]
        frame = pd.DataFrame(rows, columns=columns)
        if not frame.empty:
            frame["start"] = pd.to_datetime(frame["start"], utc=True)
            frame["end"] = pd.to_datetime(frame["end"], utc=True)
        return frame

    def delete_series(self, symbol: str, timeframe: str | Timeframe) -> int:
        tf = Timeframe.parse(timeframe)
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM market_bars WHERE symbol = ? AND timeframe = ?",
                (symbol.upper(), tf.value),
            )
            return int(cursor.rowcount)

    def coverage(self, symbol: str, timeframe: str | Timeframe) -> dict[str, object]:
        tf = Timeframe.parse(timeframe)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*), MIN(ts), MAX(ts)
                FROM market_bars WHERE symbol = ? AND timeframe = ?
                """,
                (symbol.upper(), tf.value),
            ).fetchone()
        return {"bars": int(row[0] or 0), "start": row[1], "end": row[2]}
