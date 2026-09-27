"""Pipeline de entrenamiento: preprocesador + XGBoost + transformación del target.

El objeto que devuelve `construir_modelo` es lo que se registra en MLflow y lo
que va a servir la API. Contiene TODO: imputación, escalado, one-hot,
agrupación de categorías raras, el modelo y la vuelta de log(precio) a BRL.
Quien lo use sólo tiene que pasarle un DataFrame con las columnas de FEATURES.
"""

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from airbnb_rio.config import (
    FEAT_BIN,
    FEAT_CAT,
    FEAT_NUM,
    UMBRAL_CATEGORIAS,
    XGB_FIJOS,
    XGB_PARAMS,
)


def construir_preprocesador() -> ColumnTransformer:
    """Imputa y escala numéricas, codifica categóricas, deja pasar las binarias.

    Igual que la sección 3.4 del TP, con una diferencia: la agrupación de
    categorías raras de la sección 3.3 (barrios con menos de 200 avisos ->
    "Otros") ahora la hace el propio OneHotEncoder con `min_frequency`. Así
    queda ajustada sobre train y guardada dentro del modelo, en vez de ser un
    paso manual que la API tendría que repetir.
    """
    numericas = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])
    categoricas = OneHotEncoder(
        drop="first",
        min_frequency=UMBRAL_CATEGORIAS,
        handle_unknown="infrequent_if_exist",  # barrio nuevo -> "Otros", no error
        sparse_output=False,
    )
    return ColumnTransformer([
        ("num", numericas, FEAT_NUM),
        ("cat", categoricas, FEAT_CAT),
        ("bin", "passthrough", FEAT_BIN),
    ])


def construir_modelo(params: dict | None = None) -> TransformedTargetRegressor:
    """Arma el modelo completo, listo para `.fit(X, y)` con y en BRL.

    El TransformedTargetRegressor entrena sobre log(precio) (decisión de la
    sección 3.7) y, al predecir, aplica exp() para devolver BRL. Por eso nadie
    fuera de este archivo necesita saber que existe un logaritmo.

    Args:
        params: hiperparámetros de XGBoost. Por defecto, los de Optuna del TP.
    """
    pipeline = Pipeline([
        ("prep", construir_preprocesador()),
        ("model", XGBRegressor(**(params or XGB_PARAMS), **XGB_FIJOS)),
    ])
    return TransformedTargetRegressor(
        regressor=pipeline, func=np.log, inverse_func=np.exp, check_inverse=False,
    )


def metricas(y_real, y_pred) -> dict[str, float]:
    """Métricas de regresión, siempre medidas en BRL. MAE es la principal."""
    return {
        "mae": float(mean_absolute_error(y_real, y_pred)),
        "rmse": float(root_mean_squared_error(y_real, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_real, y_pred)),
        "r2": float(r2_score(y_real, y_pred)),
    }
