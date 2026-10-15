"""
### Búsqueda de hiperparámetros de XGBoost con Optuna

Repite la sección 9.3 del TP de AMq1 (TPE, CV 3-fold sobre train) y la deja
registrada en MLflow, en el experimento `airbnb_rio_price_optuna`:

- un run padre `optuna_busqueda` por ejecución del DAG, con el mejor MAE de CV
  (`mejor_cv_mae`) y los hiperparámetros ganadores en `mejores_params.json`;
- un run anidado por trial, con sus hiperparámetros y su `cv_mae`.

El primer trial siempre es el de los hiperparámetros actuales
(`config.XGB_PARAMS`), así que la búsqueda nunca termina peor que la base.

Se dispara a mano (parámetro `n_trials`, 20 por defecto). Al terminar actualiza
el Asset `airbnb_rio_mejores_params`, que dispara `train_model`: el modelo se
reentrena con los mejores hiperparámetros y pasa por champion/challenger.
"""

import pendulum
from airflow.sdk import Asset, Param, dag, task

ASSET_PARAMS = Asset("airbnb_rio_mejores_params")


@dag(
    dag_id="tune_hyperparams",
    schedule=None,
    start_date=pendulum.datetime(2026, 9, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    params={
        "n_trials": Param(20, type="integer", minimum=1, description="Trials de Optuna"),
    },
    doc_md=__doc__,
    tags=["airbnb", "train", "optuna"],
)
def tune_hyperparams():

    @task(outlets=[ASSET_PARAMS])
    def buscar(**context) -> dict:
        """Corre el estudio de Optuna y registra cada trial como run anidado."""
        import mlflow
        import optuna

        from airbnb_rio import s3
        from airbnb_rio.busqueda import mae_cv, sugerir_params
        from airbnb_rio.config import (
            ARCHIVO_MEJORES_PARAMS,
            CV_FOLDS,
            EXPERIMENTO_BUSQUEDA,
            FEATURES,
            KEY_TRAIN,
            SEMILLA,
            TARGET,
            XGB_PARAMS,
        )

        n_trials = context["params"]["n_trials"]
        train = s3.leer_parquet(KEY_TRAIN)
        X, y = train[FEATURES], train[TARGET]

        def objetivo(trial) -> float:
            params = sugerir_params(trial)
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                mae = mae_cv(params, X, y)
                mlflow.log_params(params)
                mlflow.log_metric("cv_mae", mae)
            return mae

        mlflow.set_experiment(EXPERIMENTO_BUSQUEDA)
        with mlflow.start_run(run_name="optuna_busqueda") as padre:
            mlflow.set_tag("tipo", "busqueda")
            mlflow.log_params({"n_trials": n_trials, "cv_folds": CV_FOLDS, "n_train": len(X)})

            estudio = optuna.create_study(
                direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEMILLA),
            )
            estudio.enqueue_trial(XGB_PARAMS)
            estudio.optimize(objetivo, n_trials=n_trials)

            mlflow.log_metric("mejor_cv_mae", estudio.best_value)
            mlflow.set_tag("mejor_trial", estudio.best_trial.number)
            mlflow.log_dict(estudio.best_params, ARCHIVO_MEJORES_PARAMS)

        print(f"Mejor trial: {estudio.best_trial.number} (CV MAE {estudio.best_value:.1f})")
        print(f"Mejores hiperparámetros: {estudio.best_params}")
        return {"run_id": padre.info.run_id, "mejor_cv_mae": estudio.best_value}

    buscar()


tune_hyperparams()
