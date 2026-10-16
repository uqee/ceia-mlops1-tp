"""
### Búsqueda de hiperparámetros de XGBoost con Optuna

Repite la sección 9.3 del TP de AMq1 (TPE, CV 3-fold sobre train) y la deja
registrada en MLflow, en el experimento `airbnb_rio_price_optuna`:

- un run padre `optuna_busqueda` por ejecución del DAG, con el mejor MAE de CV
  (`mejor_cv_mae`) y los hiperparámetros ganadores en `mejores_params.json`;
- un run anidado por trial, con sus hiperparámetros y su `cv_mae`.

El primer trial siempre es el de los hiperparámetros actuales
(`config.XGB_PARAMS`), así que la búsqueda nunca termina peor que la base.

```
buscar -> hay_mejora -> publicar
```

Se dispara a mano (parámetro `n_trials`, 20 por defecto). Si la búsqueda cambia
los hiperparámetros que usa `train_model` (mejor MAE de CV que la mejor búsqueda
anterior sobre el mismo train, y no ganó el trial de la base), `publicar`
actualiza el Asset `airbnb_rio_mejores_params`, que dispara `train_model`: el
modelo se reentrena y pasa por champion/challenger. Si no, `hay_mejora` corta
ahí y no se reentrena un modelo que daría igual.

Cada búsqueda queda etiquetada con la huella del train que usó (`datos_train`):
`train_model` sólo considera búsquedas sobre su mismo train.
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

    @task
    def buscar(**context) -> dict:
        """Corre el estudio de Optuna y registra cada trial como run anidado."""
        import mlflow
        import optuna

        from airbnb_rio import hiperparametros, s3
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
        huella = hiperparametros.huella(train)
        # Lo que usaría train_model hoy: contra eso se mide si la búsqueda mejora.
        previos = hiperparametros.vigentes(huella)

        def objetivo(trial) -> float:
            params = sugerir_params(trial)
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                mae = mae_cv(params, X, y)
                mlflow.log_params(params)
                mlflow.log_metric("cv_mae", mae)
            return mae

        mlflow.set_experiment(EXPERIMENTO_BUSQUEDA)
        with mlflow.start_run(run_name="optuna_busqueda") as padre:
            mlflow.set_tags({"tipo": "busqueda", "datos_train": huella})
            mlflow.log_params({"n_trials": n_trials, "cv_folds": CV_FOLDS, "n_train": len(X)})

            estudio = optuna.create_study(
                direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEMILLA),
            )
            estudio.enqueue_trial(XGB_PARAMS)
            estudio.optimize(objetivo, n_trials=n_trials)

            mlflow.log_metric("mejor_cv_mae", estudio.best_value)
            mlflow.set_tag("mejor_trial", estudio.best_trial.number)
            mlflow.log_dict(estudio.best_params, ARCHIVO_MEJORES_PARAMS)

            mejora = estudio.best_params != previos.params and (
                previos.cv_mae is None or estudio.best_value < previos.cv_mae
            )
            mlflow.set_tag("mejora", str(mejora).lower())

        print(f"Mejor trial: {estudio.best_trial.number} (CV MAE {estudio.best_value:.1f})")
        print(f"Mejores hiperparámetros: {estudio.best_params}")
        previo = f"{previos.cv_mae:.1f}" if previos.cv_mae is not None else "sin búsqueda"
        print(f"Vigentes: {previos.origen} (CV MAE {previo}). ¿Mejora? {mejora}")
        return {
            "run_id": padre.info.run_id,
            "mejor_cv_mae": estudio.best_value,
            "mejora": mejora,
        }

    @task.short_circuit
    def hay_mejora(resultado: dict) -> bool:
        """Corta acá si la búsqueda no cambia los hiperparámetros de train_model."""
        return resultado["mejora"]

    @task(outlets=[ASSET_PARAMS])
    def publicar(resultado: dict) -> str:
        """Actualiza el Asset: dispara train_model con los hiperparámetros nuevos."""
        print(f"Hiperparámetros nuevos en optuna:{resultado['run_id']}")
        return resultado["run_id"]

    resultado = buscar()
    hay_mejora(resultado) >> publicar(resultado)


tune_hyperparams()
