"""Paquete compartido del TP: precio por noche de Airbnb en Río de Janeiro.

Lo usan los DAGs de Airflow (y, en la entrega final, la API de FastAPI). Tener
una sola copia de la limpieza, las features y el pipeline evita que el código de
entrenamiento y el de inferencia se desincronicen.

Módulos:
    config    -- constantes: columnas, hiperparámetros, rutas en S3, nombres.
    features  -- limpieza del CSV crudo y construcción de features (secc. 2-3 del TP de AMq1).
    modelo    -- preprocesador, pipeline completo y métricas.
    s3        -- lectura y escritura en el data lake (MinIO).
"""
