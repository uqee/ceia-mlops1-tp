"""Núcleo de la API: arma las features de un aviso, predice y registra.

No conoce FastAPI, MLflow ni PostgreSQL. De dónde sale el modelo y dónde se
guardan las predicciones lo deciden los adaptadores que implementan los puertos
`FuenteModelo` y `RegistroPredicciones`, y que se inyectan en `app.py`.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from airbnb_rio.config import FEAT_CAT
from airbnb_rio.features import COLUMNAS_CRUDAS, construir_features

log = logging.getLogger(__name__)

COLUMNAS_TEXTO = ["amenities", "bathrooms_text", *FEAT_CAT]
COLUMNAS_NUMERICAS = [c for c in COLUMNAS_CRUDAS if c not in COLUMNAS_TEXTO]


class ModeloNoDisponible(Exception):
    """No hay modelo para servir (todavía no hay champion o no se pudo cargar)."""


@dataclass(frozen=True)
class ModeloCargado:
    modelo: Any  # pipeline completo: .predict(DataFrame de FEATURES) -> BRL
    nombre: str
    alias: str
    version: str
    test_mae: float | None


@dataclass(frozen=True)
class Prediccion:
    precio_brl: float
    version_modelo: str


class FuenteModelo(Protocol):
    def obtener(self) -> ModeloCargado:
        """Devuelve el modelo a servir o lanza ModeloNoDisponible."""


class RegistroPredicciones(Protocol):
    def guardar(self, aviso: dict, precio_brl: float, version_modelo: str) -> None:
        """Persiste una predicción hecha."""


class ServicioPrediccion:
    def __init__(self, fuente: FuenteModelo, registro: RegistroPredicciones):
        self._fuente = fuente
        self._registro = registro

    def modelo_actual(self) -> ModeloCargado:
        return self._fuente.obtener()

    def predecir(self, aviso: dict) -> Prediccion:
        """Precio por noche (BRL) de un aviso con las columnas crudas de Inside Airbnb.

        El log de la predicción es secundario: si falla, se avisa en el log de
        la aplicación y la predicción se devuelve igual.
        """
        cargado = self._fuente.obtener()
        X = construir_features(_a_dataframe(aviso))
        # El modelo ya devuelve BRL: el exp() del log(precio) vive adentro.
        precio = float(cargado.modelo.predict(X)[0])

        try:
            self._registro.guardar(aviso, precio, cargado.version)
        except Exception:
            log.exception("No se pudo registrar la predicción")

        return Prediccion(precio_brl=precio, version_modelo=cargado.version)


def _a_dataframe(aviso: dict) -> pd.DataFrame:
    """Una fila con el mismo formato que el CSV de Inside Airbnb.

    `amenities` vuelve a ser la lista JSON en texto del CSV, y las numéricas se
    pasan a float para que un null quede como NaN (lo que imputa el modelo).
    """
    fila = {**aviso, "amenities": json.dumps(aviso.get("amenities", []))}
    df = pd.DataFrame([fila], columns=COLUMNAS_CRUDAS)
    df[COLUMNAS_NUMERICAS] = df[COLUMNAS_NUMERICAS].astype(float)
    return df
