"""Adaptador de entrada: endpoints HTTP sobre el ServicioPrediccion.

Los endpoints son `def` (no `async`): predecir es CPU y el log en PostgreSQL es
bloqueante, así que FastAPI los corre en su pool de threads.
"""

import math

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.esquemas import AvisoEntrada, ModeloInfo, PrediccionSalida
from api.servicio import ModeloNoDisponible, ServicioPrediccion

DESCRIPCION = """
Predice el **precio por noche en reales (BRL)** de un aviso de Airbnb en Río de Janeiro.

Sirve la versión del modelo `airbnb_rio_price` con alias `champion` en MLflow. Es un
XGBoost entrenado sobre el snapshot de Inside Airbnb del 24/06/2026 (MAE en test de
~250 BRL). Cuando `train_model` promueve una versión nueva, la API la empieza a usar
en menos de un minuto.

El aviso se manda con las columnas crudas de Inside Airbnb; la API le aplica las
mismas transformaciones que el entrenamiento.

**Limitación conocida:** el error relativo es mayor en los extremos (~46 % en el
quintil más barato y ~37 % en el más caro, contra ~23 % en el medio).
"""


def crear_app(servicio: ServicioPrediccion, lifespan=None) -> FastAPI:
    app = FastAPI(
        title="Precio de Airbnb en Río",
        description=DESCRIPCION,
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    def aviso_invalido(request: Request, error: RequestValidationError) -> JSONResponse:
        # El 422 por defecto repite el valor recibido. Si es NaN o Infinity (JSON no
        # estándar, pero el parser lo acepta) no se puede serializar y sale un 500.
        detalle = jsonable_encoder(error.errors(), custom_encoder={float: _float_json})
        return JSONResponse(status_code=422, content={"detail": detalle})

    @app.exception_handler(ModeloNoDisponible)
    def sin_modelo(request: Request, error: ModeloNoDisponible) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(error)})

    @app.get("/", summary="Estado del servicio")
    def estado() -> dict:
        return {"estado": "ok"}

    @app.get(
        "/modelo",
        summary="Modelo en producción",
        responses={503: {"description": "Todavía no hay champion"}},
    )
    def modelo() -> ModeloInfo:
        actual = servicio.modelo_actual()
        return ModeloInfo(
            nombre=actual.nombre,
            alias=actual.alias,
            version=actual.version,
            test_mae=actual.test_mae,
        )

    @app.post(
        "/predict",
        summary="Precio por noche de un aviso",
        responses={503: {"description": "Todavía no hay champion"}},
    )
    def predecir(aviso: AvisoEntrada) -> PrediccionSalida:
        prediccion = servicio.predecir(aviso.model_dump())
        return PrediccionSalida(
            precio_brl=round(prediccion.precio_brl, 2),
            version_modelo=prediccion.version_modelo,
        )

    return app


def _float_json(valor: float) -> float | str:
    return valor if math.isfinite(valor) else str(valor)
