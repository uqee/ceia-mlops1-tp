"""Adaptador de salida: el modelo a servir sale del Model Registry de MLflow.

Sirve la versión que tiene el alias (`champion`) y cada `ttl_segundos` vuelve a
preguntar a qué versión apunta. Si `train_model` promovió una nueva, la carga
sola, sin reiniciar el contenedor. Si MLflow no responde, sigue sirviendo la que
ya tenía cargada.
"""

import logging
import threading
import time

import mlflow
from mlflow import MlflowClient

from api.servicio import ModeloCargado, ModeloNoDisponible

log = logging.getLogger(__name__)


class FuenteMlflow:
    def __init__(
        self,
        nombre: str,
        alias: str,
        ttl_segundos: float = 60,
        cliente=None,
        cargar=mlflow.sklearn.load_model,
        reloj=time.monotonic,
    ):
        # Nada de I/O acá: el cliente se crea recién en la primera consulta.
        self._nombre = nombre
        self._alias = alias
        self._ttl = ttl_segundos
        self._cliente = cliente
        self._cargar = cargar
        self._reloj = reloj
        self._cargado: ModeloCargado | None = None
        self._ultimo_chequeo = float("-inf")
        # FastAPI atiende requests en paralelo: una sola recarga a la vez.
        self._lock = threading.Lock()

    def obtener(self) -> ModeloCargado:
        with self._lock:
            if self._reloj() - self._ultimo_chequeo >= self._ttl:
                self._ultimo_chequeo = self._reloj()
                self._actualizar()
            if self._cargado is None:
                raise ModeloNoDisponible(
                    f"No hay modelo con alias '{self._alias}' en '{self._nombre}' todavía"
                )
            return self._cargado

    def _actualizar(self) -> None:
        """Carga la versión del alias si cambió. Ante cualquier error, deja la que había."""
        try:
            if self._cliente is None:
                self._cliente = MlflowClient()
            info = self._cliente.get_model_version_by_alias(self._nombre, self._alias)
            version = str(info.version)  # según el backend de MLflow viene int o str
            if self._cargado is not None and self._cargado.version == version:
                return

            modelo = self._cargar(f"models:/{self._nombre}/{version}")
            test_mae = info.tags.get("test_mae")
            self._cargado = ModeloCargado(
                modelo=modelo,
                nombre=self._nombre,
                alias=self._alias,
                version=version,
                test_mae=float(test_mae) if test_mae is not None else None,
            )
            log.info("Sirviendo %s v%s (%s)", self._nombre, version, self._alias)
        except Exception as error:
            actual = f"v{self._cargado.version}" if self._cargado else "ninguno"
            log.warning("No se pudo actualizar el modelo (sigue %s): %s", actual, error)
