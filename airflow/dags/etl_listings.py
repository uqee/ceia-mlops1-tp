"""
### ETL de avisos de Airbnb en Río de Janeiro

Parte en tasks las secciones 2 y 3 del TP de AMq1. Cada task lee su entrada del
data lake (MinIO) y escribe su salida ahí mismo; nada viaja en memoria entre
tasks, así que cualquiera se puede reintentar sola.

```
obtener_raw -> limpiar -> construir_features -> separar_train_test
```

| Task | Lee | Escribe |
| --- | --- | --- |
| `obtener_raw` | Inside Airbnb (sólo si falta) | `s3://data/raw/listings.csv.gz` |
| `limpiar` | raw | `s3://data/interim/listings_limpio.parquet` |
| `construir_features` | limpio | `s3://data/interim/listings_features.parquet` |
| `separar_train_test` | features | `s3://data/final/{train,test}.parquet` |

Al terminar, `separar_train_test` actualiza el Asset `s3://data/final/train.parquet`,
y eso dispara el DAG `train_model`.
"""

import pendulum
from airflow.sdk import Asset, dag, task

DATASET_TRAIN = Asset("s3://data/final/train.parquet")


@dag(
    dag_id="etl_listings",
    schedule="@weekly",
    start_date=pendulum.datetime(2026, 9, 1, tz="UTC"),
    catchup=False,
    doc_md=__doc__,
    tags=["airbnb", "etl"],
    default_args={"retries": 1, "retry_delay": pendulum.duration(minutes=1)},
)
def etl_listings():
    # Los imports pesados (pandas, sklearn) van dentro de cada task: Airflow
    # parsea este archivo seguido y no queremos cargarlos en cada parseo.

    @task
    def obtener_raw() -> str:
        """Asegura que el CSV crudo esté en el data lake.

        Si ya está (lo normal), no hace nada. Si falta, baja el snapshot fijo
        de Inside Airbnb y lo sube. Devuelve la key del objeto.
        """
        import tempfile
        import urllib.request

        from airbnb_rio import s3
        from airbnb_rio.config import KEY_RAW, URL_RAW

        if s3.existe(KEY_RAW):
            print(f"Ya existe s3://data/{KEY_RAW}, no se descarga.")
            return KEY_RAW

        print(f"Descargando {URL_RAW}")
        with tempfile.NamedTemporaryFile(suffix=".csv.gz") as tmp:
            urllib.request.urlretrieve(URL_RAW, tmp.name)
            print(f"Subido a {s3.subir_archivo(tmp.name, KEY_RAW)}")
        return KEY_RAW

    @task
    def limpiar(key_raw: str) -> str:
        """Fuga, parseo del precio y corte de precios extremos (sección 2)."""
        from airbnb_rio import features, s3
        from airbnb_rio.config import KEY_LIMPIO

        raw = s3.leer_csv(key_raw, low_memory=False)
        limpio = features.limpiar(raw)
        print(f"{len(raw):,} avisos crudos -> {len(limpio):,} tras la limpieza")
        s3.escribir_parquet(limpio, KEY_LIMPIO)
        return KEY_LIMPIO

    @task
    def construir_features(key_limpio: str) -> str:
        """Banderas de faltantes, amenities y antigüedad del host (sección 3.1)."""
        from airbnb_rio import features, s3
        from airbnb_rio.config import KEY_FEATURES

        df = features.construir_features(s3.leer_parquet(key_limpio))
        print(f"{df.shape[1] - 1} features para {len(df):,} avisos")
        s3.escribir_parquet(df, KEY_FEATURES)
        return KEY_FEATURES

    @task(outlets=[DATASET_TRAIN])
    def separar_train_test(key_features: str) -> dict:
        """Split 70/30 con la misma semilla del TP, para que las métricas sean comparables."""
        from sklearn.model_selection import train_test_split

        from airbnb_rio import s3
        from airbnb_rio.config import KEY_TEST, KEY_TRAIN, SEMILLA, TAMANO_TEST

        df = s3.leer_parquet(key_features)
        train, test = train_test_split(df, test_size=TAMANO_TEST, random_state=SEMILLA)
        s3.escribir_parquet(train, KEY_TRAIN)
        s3.escribir_parquet(test, KEY_TEST)
        print(f"train: {len(train):,}  test: {len(test):,}")
        return {"train": KEY_TRAIN, "test": KEY_TEST}

    separar_train_test(construir_features(limpiar(obtener_raw())))


etl_listings()
