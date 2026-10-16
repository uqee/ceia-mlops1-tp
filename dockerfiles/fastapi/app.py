"""Punto de entrada (`uvicorn app:app`): conecta el núcleo con los adaptadores reales.

Variables de entorno (las define docker-compose.yaml):
    MLFLOW_TRACKING_URI  -- servidor de MLflow (la lee MLflow solo).
    PREDICCIONES_DB_URL  -- base de PostgreSQL donde se registran las predicciones.
"""

import logging
import os
from contextlib import asynccontextmanager

from airbnb_rio.config import ALIAS_CHAMPION, NOMBRE_MODELO
from api.fuente_mlflow import FuenteMlflow
from api.registro_postgres import RegistroPostgres
from api.rutas import crear_app
from api.servicio import ModeloNoDisponible, ServicioPrediccion

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

registro = RegistroPostgres(os.environ["PREDICCIONES_DB_URL"])
servicio = ServicioPrediccion(FuenteMlflow(NOMBRE_MODELO, ALIAS_CHAMPION), registro)


@asynccontextmanager
async def lifespan(app):
    # Ninguno de los dos pasos impide arrancar: sin tabla, la API predice igual
    # (sin registrar); sin champion, /predict responde 503 hasta que haya uno.
    try:
        registro.inicializar()
    except Exception:
        log.exception("No se pudo crear la tabla de predicciones")
    try:
        servicio.modelo_actual()
    except ModeloNoDisponible as error:
        log.warning("%s: /predict responde 503 hasta que train_model promueva uno", error)
    yield


app = crear_app(servicio, lifespan)
