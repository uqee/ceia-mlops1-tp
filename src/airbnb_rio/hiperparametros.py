"""De dónde salen los hiperparámetros con que entrena `train_model`.

De la búsqueda de `tune_hyperparams` con mejor MAE de CV hecha sobre el mismo
train y, si no hay ninguna, de `config.XGB_PARAMS`. Una búsqueda hecha sobre
otro train no cuenta: su MAE de CV no es comparable.

Lo usan los dos DAGs: `train_model` para entrenar y `tune_hyperparams` para
saber si su búsqueda cambia los hiperparámetros (si no, no dispara un
reentrenamiento que daría el mismo modelo).
"""

import hashlib
from dataclasses import dataclass

import mlflow
import pandas as pd

from airbnb_rio.config import ARCHIVO_MEJORES_PARAMS, EXPERIMENTO_BUSQUEDA, XGB_PARAMS


@dataclass(frozen=True)
class Hiperparametros:
    params: dict
    origen: str  # "config" u "optuna:<run_id>"
    cv_mae: float | None  # mejor_cv_mae de esa búsqueda; None si salen de config


def huella(train: pd.DataFrame) -> str:
    """Identifica el contenido de train: cambia si cambia cualquier fila o columna."""
    hashes = pd.util.hash_pandas_object(train, index=False).to_numpy()
    return hashlib.sha256(hashes.tobytes()).hexdigest()[:16]


def vigentes(huella_train: str) -> Hiperparametros:
    """Los hiperparámetros que usaría hoy `train_model` para este train."""
    # search_runs devuelve un DataFrame vacío si el experimento todavía no existe.
    busquedas = mlflow.search_runs(
        experiment_names=[EXPERIMENTO_BUSQUEDA],
        filter_string=(
            "tags.tipo = 'busqueda' and attributes.status = 'FINISHED' "
            f"and tags.datos_train = '{huella_train}'"
        ),
        order_by=["metrics.mejor_cv_mae ASC"],
        max_results=1,
    )
    if busquedas.empty:
        return Hiperparametros(XGB_PARAMS, "config", None)

    mejor = busquedas.iloc[0]
    params = mlflow.artifacts.load_dict(f"runs:/{mejor['run_id']}/{ARCHIVO_MEJORES_PARAMS}")
    return Hiperparametros(params, f"optuna:{mejor['run_id']}", float(mejor["metrics.mejor_cv_mae"]))
