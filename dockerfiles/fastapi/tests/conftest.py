"""Adaptadores falsos: los tests no necesitan MLflow, PostgreSQL ni un modelo real."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.rutas import crear_app
from api.servicio import ModeloCargado, ModeloNoDisponible, ServicioPrediccion


class ModeloFalso:
    """Devuelve siempre el mismo precio y guarda lo que recibió."""

    def __init__(self, precio: float = 300.0):
        self.precio = precio
        self.recibido = None

    def predict(self, X):
        self.recibido = X
        return np.full(len(X), self.precio)


class FuenteFalsa:
    def __init__(self, modelo=None):
        self.modelo = modelo

    def obtener(self) -> ModeloCargado:
        if self.modelo is None:
            raise ModeloNoDisponible("Todavía no hay champion")
        return ModeloCargado(self.modelo, "airbnb_rio_price", "champion", "3", 249.6)


class RegistroFalso:
    def __init__(self, falla: bool = False):
        self.falla = falla
        self.filas = []

    def guardar(self, aviso: dict, precio_brl: float, version_modelo: str) -> None:
        if self.falla:
            raise RuntimeError("PostgreSQL caído")
        self.filas.append((aviso, precio_brl, version_modelo))


def cliente_con(fuente, registro) -> TestClient:
    return TestClient(crear_app(ServicioPrediccion(fuente, registro)))


@pytest.fixture
def modelo() -> ModeloFalso:
    return ModeloFalso()


@pytest.fixture
def registro() -> RegistroFalso:
    return RegistroFalso()


@pytest.fixture
def cliente(modelo, registro) -> TestClient:
    return cliente_con(FuenteFalsa(modelo), registro)
