"""Interfaz de proveedores de datos historicos.

El laboratorio incluye un proveedor CSV local. No se incluyen datos de mercado
inventados ni descargas no verificadas. Cualquier proveedor futuro debe
implementar DataProvider y documentar sus limitaciones.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from trading_bot_lab.types import Timeframe

from .resampling import infer_timeframe, resample_ohlcv
from .validation import DataValidationError, validate_ohlcv


class ProviderError(RuntimeError):
    """Error de adquisicion de datos."""


@dataclass(frozen=True)
class DataRequest:
    symbol: str
    timeframe: str | Timeframe
    start: str | pd.Timestamp | None = None
    end: str | pd.Timestamp | None = None


class DataProvider(ABC):
    """Contrato minimo de un proveedor historico."""

    name: str = "abstract"
    limitations: str = ""

    @abstractmethod
    def fetch(self, request: DataRequest) -> pd.DataFrame:
        """Devuelve OHLCV validado y en UTC."""


def _utc_timestamp(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def read_csv_ohlcv(
    path: str | Path,
    *,
    separator: str | None = None,
) -> pd.DataFrame:
    """Lee un CSV OHLCV con deteccion de separador y codificacion.

    Formatos aceptados: columnas date/datetime/time/timestamp + open, high, low,
    close (volume opcional). La zona horaria se interpreta como UTC si no se
    especifica offset.
    """
    target = Path(path)
    if not target.exists():
        raise ProviderError(f"No existe el archivo CSV: {target}")
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(
                target,
                sep=separator,
                engine="python" if separator is None else "c",
                encoding=encoding,
            )
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.ParserError as exc:
            last_error = exc
    raise ProviderError(f"No se pudo leer el CSV {target}: {last_error}")


class CSVProvider(DataProvider):
    """Proveedor local basado en CSV. No descarga datos."""

    name = "csv"
    limitations = (
        "Depende de la calidad del archivo: zona horaria, huecos, fines de semana, "
        "ajustes por splits/dividendos y precision de precios. El CSV debe ser "
        "exportado por una fuente fiable y auditado antes de usarlo."
    )

    def __init__(self, path: str | Path, *, separator: str | None = None) -> None:
        self.path = Path(path)
        self.separator = separator

    def fetch(self, request: DataRequest) -> pd.DataFrame:
        try:
            raw = read_csv_ohlcv(self.path, separator=self.separator)
            data = validate_ohlcv(raw, symbol=request.symbol)
        except DataValidationError as exc:
            raise ProviderError(f"CSV invalido: {exc}") from exc

        start = _utc_timestamp(request.start)
        end = _utc_timestamp(request.end)
        if start is not None:
            data = data[data.index >= start]
        if end is not None:
            data = data[data.index <= end]
        if data.empty:
            raise ProviderError("El filtro de fechas no devolvio ninguna barra.")

        target_tf = Timeframe.parse(request.timeframe)
        source_tf = infer_timeframe(data)
        if source_tf is not None and target_tf.minutes > source_tf.minutes:
            data = resample_ohlcv(data, target_tf, symbol=request.symbol)
        elif source_tf is not None and target_tf.minutes < source_tf.minutes:
            raise ProviderError(
                f"El CSV tiene barras de {source_tf.value} y se solicito {target_tf.value}. "
                "No se puede fabricar informacion intra-barra."
            )
        data.attrs["provider"] = self.name
        data.attrs["source"] = str(self.path)
        return data


class DataFrameProvider(DataProvider):
    """Proveedor en memoria, util para pruebas y para el panel."""

    name = "dataframe"
    limitations = "Datos aportados por el usuario; su calidad no es verificable por el sistema."

    def __init__(self, data: pd.DataFrame, *, source_label: str = "memoria") -> None:
        self.data = data.copy()
        self.source_label = source_label

    def fetch(self, request: DataRequest) -> pd.DataFrame:
        data = validate_ohlcv(self.data, symbol=request.symbol)
        if request.start is not None:
            data = data[data.index >= _utc_timestamp(request.start)]
        if request.end is not None:
            data = data[data.index <= _utc_timestamp(request.end)]
        target_tf = Timeframe.parse(request.timeframe)
        source_tf = infer_timeframe(data)
        if source_tf is not None and source_tf.minutes < target_tf.minutes:
            data = resample_ohlcv(data, target_tf, symbol=request.symbol)
        data.attrs["provider"] = self.name
        data.attrs["source"] = self.source_label
        return data


PROVIDER_LIMITATIONS: dict[str, dict[str, Any]] = {
    "csv": {
        "name": "CSV local",
        "needs_network": False,
        "adjustments": "Depende del exportador (splits, dividendos, rollover de futuros).",
        "timezone": "Se asume UTC si el archivo no incluye offset.",
        "quality": "El sistema valida OHLC y duplicados, pero no la fuente.",
        "cost": "Gratuito; el usuario obtiene y verifica los datos.",
    },
    "future_broker_api": {
        "name": "API de broker (no implementada)",
        "needs_network": True,
        "adjustments": "Depende del broker.",
        "timezone": "Normalmente UTC o zona del servidor del broker.",
        "quality": "Puede incluir huecos y precios de bid/ask distintos.",
        "cost": "Requiere cuenta y credenciales; nunca incluirlas en el codigo.",
    },
    "future_vendor": {
        "name": "Proveedor institucional (no implementado)",
        "needs_network": True,
        "adjustments": "Alta calidad, con ajustes documentados.",
        "timezone": "Verificar contrato.",
        "quality": "Alta, sujeta a licencia de uso.",
        "cost": "Suscripcion de pago.",
    },
}
