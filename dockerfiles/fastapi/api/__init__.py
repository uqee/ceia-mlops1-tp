"""API del modelo de precio de Airbnb Río.

    esquemas           -- contratos JSON de entrada y salida.
    servicio           -- núcleo: predicción y puertos (FuenteModelo, RegistroPredicciones).
    rutas              -- endpoints HTTP (adaptador de entrada).
    fuente_mlflow      -- modelo champion desde el registry de MLflow (adaptador de salida).
    registro_postgres  -- log de predicciones en PostgreSQL (adaptador de salida).
"""
