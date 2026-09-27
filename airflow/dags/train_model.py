"""
### Entrenamiento del modelo de precio

Entrena XGBoost con los hiperparámetros que encontró Optuna en el TP de AMq1, lo
registra en MLflow y decide si reemplaza al modelo en producción.

```
entrenar -> evaluar -> promover
```

- `entrenar`: ajusta el pipeline completo sobre train y lo registra como una
  versión nueva de `airbnb_rio_price` en el Model Registry.
- `evaluar`: mide MAE, RMSE, MAPE y R² en test (en BRL) y los guarda en el run.
- `promover`: si el MAE es mejor que el del `champion` actual (o no hay
  champion), la versión nueva pasa a ser `champion`. Si no, queda como
  `challenger` y el champion no se toca.

Corre solo cada vez que `etl_listings` actualiza `s3://data/final/train.parquet`.
"""

import pendulum
from airflow.sdk import Asset, dag, task

DATASET_TRAIN = Asset("s3://data/final/train.parquet")


@dag(
    dag_id="train_model",
    schedule=[DATASET_TRAIN],
    start_date=pendulum.datetime(2026, 9, 1, tz="UTC"),
    catchup=False,
    # Un run a la vez: `promover` lee el champion, compara y reasigna el alias.
    # Dos runs en paralelo podrían leer el mismo champion y el último en
    # terminar pisaría al mejor.
    max_active_runs=1,
    doc_md=__doc__,
    tags=["airbnb", "train"],
)
def train_model():

    @task
    def entrenar() -> dict:
        """Ajusta el modelo sobre train y lo registra en MLflow."""
        import mlflow

        from airbnb_rio import s3
        from airbnb_rio.config import (
            EXPERIMENTO_MLFLOW,
            FEATURES,
            KEY_TRAIN,
            NOMBRE_MODELO,
            TARGET,
            XGB_PARAMS,
        )
        from airbnb_rio.modelo import construir_modelo

        train = s3.leer_parquet(KEY_TRAIN)
        X, y = train[FEATURES], train[TARGET]

        mlflow.set_experiment(EXPERIMENTO_MLFLOW)
        with mlflow.start_run(run_name="xgboost_optuna_amq1") as run:
            modelo = construir_modelo().fit(X, y)

            mlflow.log_params(XGB_PARAMS)
            mlflow.log_params({"n_train": len(X), "target": "log(precio)"})
            info = mlflow.sklearn.log_model(
                modelo,
                name="model",
                registered_model_name=NOMBRE_MODELO,
                input_example=X.head(3),
                # skops en vez de pickle: al cargar no ejecuta código arbitrario,
                # sólo reconstruye los tipos declarados como confiables.
                serialization_format="skops",
                skops_trusted_types=[
                    "numpy.dtype",
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBRegressor",
                ],
            )

        print(f"Registrado {NOMBRE_MODELO} v{info.registered_model_version}")
        return {
            "run_id": run.info.run_id,
            "model_uri": info.model_uri,
            "version": str(info.registered_model_version),
        }

    @task
    def evaluar(entrenado: dict) -> dict:
        """Mide el modelo en test y guarda las métricas en el run y en la versión."""
        import mlflow
        from mlflow import MlflowClient

        from airbnb_rio import s3
        from airbnb_rio.config import FEATURES, KEY_TEST, NOMBRE_MODELO, TARGET
        from airbnb_rio.modelo import metricas

        test = s3.leer_parquet(KEY_TEST)
        modelo = mlflow.sklearn.load_model(entrenado["model_uri"])
        resultado = metricas(test[TARGET], modelo.predict(test[FEATURES]))

        with mlflow.start_run(run_id=entrenado["run_id"]):
            mlflow.log_metrics({f"test_{k}": v for k, v in resultado.items()})

        # Se guarda también en la versión del registry, para poder comparar
        # versiones sin tener que ir a buscar el run de cada una.
        MlflowClient().set_model_version_tag(
            NOMBRE_MODELO, entrenado["version"], "test_mae", str(resultado["mae"])
        )
        print(f"Test: {resultado}")
        return {**entrenado, **resultado}

    @task
    def promover(evaluado: dict) -> str:
        """Champion/challenger: la versión nueva sólo reemplaza si mejora el MAE."""
        from mlflow import MlflowClient
        from mlflow.exceptions import MlflowException

        from airbnb_rio.config import ALIAS_CHALLENGER, ALIAS_CHAMPION, NOMBRE_MODELO

        client = MlflowClient()
        try:
            champion = client.get_model_version_by_alias(NOMBRE_MODELO, ALIAS_CHAMPION)
            mae_champion = float(champion.tags["test_mae"])
        except (MlflowException, KeyError):
            champion, mae_champion = None, float("inf")

        version, mae = evaluado["version"], evaluado["mae"]
        if mae < mae_champion:
            client.set_registered_model_alias(NOMBRE_MODELO, ALIAS_CHAMPION, version)
            antes = f"v{champion.version} ({mae_champion:.1f})" if champion else "ninguno"
            print(f"Nuevo champion: v{version} (MAE {mae:.1f}). Antes: {antes}")
            return ALIAS_CHAMPION

        client.set_registered_model_alias(NOMBRE_MODELO, ALIAS_CHALLENGER, version)
        print(f"v{version} (MAE {mae:.1f}) no mejora al champion "
              f"v{champion.version} (MAE {mae_champion:.1f}); queda como challenger")
        return ALIAS_CHALLENGER

    promover(evaluar(entrenar()))


train_model()
