import json
import math

import pytest

from airbnb_rio.config import FEATURES
from airbnb_rio.features import COLUMNAS_CRUDAS
from api.esquemas import EJEMPLO_AVISO, AvisoEntrada
from conftest import FuenteFalsa, RegistroFalso, cliente_con

OPCIONALES_EN_NULL = {
    campo: None
    for campo, info in AvisoEntrada.model_fields.items()
    if not info.is_required() and campo != "amenities"
}


def test_el_esquema_cubre_las_columnas_crudas_del_entrenamiento():
    assert set(AvisoEntrada.model_fields) == set(COLUMNAS_CRUDAS)


def test_estado(cliente):
    assert cliente.get("/").json() == {"estado": "ok"}


def test_modelo_informa_la_version_en_produccion(cliente):
    respuesta = cliente.get("/modelo")

    assert respuesta.status_code == 200
    assert respuesta.json() == {
        "nombre": "airbnb_rio_price", "alias": "champion", "version": "3", "test_mae": 249.6,
    }


def test_predict_devuelve_precio_y_version_y_lo_registra(cliente, registro):
    respuesta = cliente.post("/predict", json=EJEMPLO_AVISO)

    assert respuesta.status_code == 200
    assert respuesta.json() == {"precio_brl": 300.0, "version_modelo": "3"}
    assert len(registro.filas) == 1
    aviso, precio, version = registro.filas[0]
    assert aviso["neighbourhood_cleansed"] == "Copacabana"
    assert (precio, version) == (300.0, "3")


def test_predict_le_pasa_al_modelo_las_mismas_features_que_el_entrenamiento(cliente, modelo):
    aviso = {
        **EJEMPLO_AVISO,
        "bedrooms": None,
        "bathrooms_text": "1 shared half-bath",
        "amenities": ["Wifi", "Outdoor pool"],
        "number_of_reviews": 0,
        "reviews_per_month": None,
        "review_scores_rating": None,
        "hosts_time_as_host_years": 2,
        "hosts_time_as_host_months": 6,
    }

    assert cliente.post("/predict", json=aviso).status_code == 200

    X = modelo.recibido
    assert list(X.columns) == FEATURES
    fila = X.iloc[0]
    assert math.isnan(fila["bedrooms"])  # lo imputa el modelo
    assert fila["sin_dormitorios"] == 1
    assert fila["bano_compartido"] == 1
    assert fila["tiene_pileta"] == 1
    assert fila["tiene_aire"] == 0
    assert fila["sin_resenas"] == 1
    assert fila["review_scores_rating"] == 0
    assert fila["antiguedad_host"] == 2.5


def test_predict_acepta_los_opcionales_en_null_y_sin_amenities(cliente):
    aviso = {**EJEMPLO_AVISO, **OPCIONALES_EN_NULL}
    del aviso["amenities"]

    assert cliente.post("/predict", json=aviso).status_code == 200


def test_predict_rechaza_room_type_desconocido(cliente):
    respuesta = cliente.post("/predict", json={**EJEMPLO_AVISO, "room_type": "Castillo"})

    assert respuesta.status_code == 422


def test_predict_rechaza_campos_que_no_son_del_aviso(cliente):
    respuesta = cliente.post("/predict", json={**EJEMPLO_AVISO, "precio": 383.2})

    assert respuesta.status_code == 422


def test_sin_champion_responde_503():
    cliente = cliente_con(FuenteFalsa(modelo=None), RegistroFalso())

    assert cliente.post("/predict", json=EJEMPLO_AVISO).status_code == 503
    assert cliente.get("/modelo").status_code == 503


def test_predict_responde_aunque_falle_el_registro(modelo):
    cliente = cliente_con(FuenteFalsa(modelo), RegistroFalso(falla=True))

    respuesta = cliente.post("/predict", json=EJEMPLO_AVISO)

    assert respuesta.status_code == 200
    assert respuesta.json()["precio_brl"] == 300.0


@pytest.mark.parametrize("valor", [float("nan"), float("inf"), float("-inf")])
def test_predict_rechaza_numeros_no_finitos_con_422(cliente, valor):
    # json.dumps los escribe como NaN / Infinity, que no es JSON estándar.
    cuerpo = json.dumps({**EJEMPLO_AVISO, "bedrooms": valor})
    respuesta = cliente.post(
        "/predict", content=cuerpo, headers={"Content-Type": "application/json"},
    )

    assert respuesta.status_code == 422
    assert respuesta.json()["detail"][0]["loc"] == ["body", "bedrooms"]
