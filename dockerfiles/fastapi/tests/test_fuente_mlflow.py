from types import SimpleNamespace

import pytest

from api.fuente_mlflow import FuenteMlflow
from api.servicio import ModeloNoDisponible


class ClienteFalso:
    """Imita MlflowClient.get_model_version_by_alias."""

    def __init__(self, version=None):
        self.version = version
        self.caido = False
        self.consultas = 0

    def get_model_version_by_alias(self, nombre, alias):
        self.consultas += 1
        if self.caido:
            raise ConnectionError("MLflow no responde")
        if self.version is None:
            raise Exception(f"Registered model alias {alias} not found.")
        return SimpleNamespace(version=self.version, tags={"test_mae": "250.5"})


class Reloj:
    def __init__(self):
        self.ahora = 0.0

    def __call__(self) -> float:
        return self.ahora


@pytest.fixture
def reloj() -> Reloj:
    return Reloj()


@pytest.fixture
def cargas() -> list:
    return []


def fuente(cliente, reloj, cargas) -> FuenteMlflow:
    def cargar(uri):
        cargas.append(uri)
        return f"modelo de {uri}"

    return FuenteMlflow(
        "airbnb_rio_price", "champion", ttl_segundos=60,
        cliente=cliente, cargar=cargar, reloj=reloj,
    )


def test_carga_la_version_del_alias(reloj, cargas):
    cargado = fuente(ClienteFalso("1"), reloj, cargas).obtener()

    assert cargado.version == "1"
    assert cargado.test_mae == 250.5
    assert cargas == ["models:/airbnb_rio_price/1"]


def test_no_consulta_mlflow_dentro_del_ttl(reloj, cargas):
    cliente = ClienteFalso("1")
    f = fuente(cliente, reloj, cargas)

    f.obtener()
    reloj.ahora = 59
    f.obtener()

    assert cliente.consultas == 1


def test_recarga_solo_si_cambia_la_version(reloj, cargas):
    cliente = ClienteFalso("1")
    f = fuente(cliente, reloj, cargas)
    f.obtener()

    reloj.ahora = 60
    f.obtener()
    assert len(cargas) == 1

    cliente.version = "2"
    reloj.ahora = 120
    assert f.obtener().version == "2"
    assert cargas[-1] == "models:/airbnb_rio_price/2"


def test_sin_champion_lanza_modelo_no_disponible(reloj, cargas):
    with pytest.raises(ModeloNoDisponible):
        fuente(ClienteFalso(version=None), reloj, cargas).obtener()


def test_si_mlflow_no_responde_sigue_con_el_modelo_cargado(reloj, cargas):
    cliente = ClienteFalso("1")
    f = fuente(cliente, reloj, cargas)
    f.obtener()

    cliente.caido = True
    reloj.ahora = 60

    assert f.obtener().version == "1"


def test_normaliza_la_version_numerica(reloj, cargas):
    assert fuente(ClienteFalso(4), reloj, cargas).obtener().version == "4"
