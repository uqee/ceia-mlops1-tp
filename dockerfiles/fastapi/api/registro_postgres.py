"""Adaptador de salida: cada predicción queda guardada en PostgreSQL.

Tabla `predicciones` en la base `api_db` (la crea `dockerfiles/postgres/api.sql`).
Guarda el aviso tal como llegó, para poder auditar o medir drift más adelante.
Abre una conexión por predicción: alcanza para el volumen de este TP.
"""

import psycopg
from psycopg.types.json import Jsonb

CREAR_TABLA = """
CREATE TABLE IF NOT EXISTS predicciones (
    id             bigserial PRIMARY KEY,
    creado         timestamptz NOT NULL DEFAULT now(),
    version_modelo text NOT NULL,
    entrada        jsonb NOT NULL,
    precio_brl     double precision NOT NULL
)
"""

INSERTAR = """
INSERT INTO predicciones (version_modelo, entrada, precio_brl) VALUES (%s, %s, %s)
"""


class RegistroPostgres:
    def __init__(self, url: str, timeout_segundos: int = 5):
        self._url = url
        self._timeout = timeout_segundos

    def _conectar(self) -> psycopg.Connection:
        return psycopg.connect(self._url, connect_timeout=self._timeout)

    def inicializar(self) -> None:
        """Crea la tabla si no existe. Se llama al arrancar la API."""
        with self._conectar() as conexion:
            conexion.execute(CREAR_TABLA)

    def guardar(self, aviso: dict, precio_brl: float, version_modelo: str) -> None:
        with self._conectar() as conexion:
            conexion.execute(INSERTAR, (version_modelo, Jsonb(aviso), precio_brl))
