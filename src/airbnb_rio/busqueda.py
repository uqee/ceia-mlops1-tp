"""Búsqueda de hiperparámetros de XGBoost (sección 9.3 del TP de AMq1).

Mismo espacio de búsqueda que el notebook, con una diferencia: allá se medía el
MAE sobre log(precio); acá el modelo completo (con el TransformedTargetRegressor)
se evalúa en BRL, la misma métrica con la que `train_model` compara contra el
champion.

No sabe nada de MLflow ni de S3: el DAG `tune_hyperparams` lee los datos y
registra cada trial.
"""

import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

from airbnb_rio.config import CV_FOLDS, SEMILLA
from airbnb_rio.modelo import construir_modelo


def sugerir_params(trial) -> dict:
    """Pide a Optuna un punto del espacio de búsqueda de la sección 9.3."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
    }


def mae_cv(params: dict, X: pd.DataFrame, y: pd.Series) -> float:
    """MAE promedio (BRL) del modelo completo con `params`, por CV sobre train.

    Sólo se le pasa train: test queda intacto para la comparación
    champion/challenger.
    """
    folds = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEMILLA)
    scores = cross_val_score(
        construir_modelo(params), X, y, cv=folds, scoring="neg_mean_absolute_error",
    )
    return float(-scores.mean())
