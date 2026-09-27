"""Limpieza y construcción de features (secciones 2 y 3.1 del TP de AMq1).

Las dos funciones son puras: reciben un DataFrame y devuelven otro, sin leer ni
escribir nada. Así se pueden probar solas y el DAG decide dónde guardar el
resultado.
"""

import json

import pandas as pd

from airbnb_rio.config import (
    AMENITIES,
    BANDERAS_FALTANTES,
    COLS_FUGA,
    COLS_RESENAS,
    CORTE_PRECIO,
    FEAT_CAT,
    FEATURES,
    TARGET,
)

# Columnas del CSV crudo que hacen falta para armar las features. Todo lo demás
# (90 columnas en total) se descarta en la limpieza.
COLUMNAS_CRUDAS = sorted({
    "accommodates", "bedrooms", "beds", "bathrooms", "bathrooms_text",
    "minimum_nights", "maximum_nights",
    "number_of_reviews", *COLS_RESENAS,
    "availability_30", "availability_365",
    "estimated_occupancy_l365d", "calculated_host_listings_count",
    "hosts_time_as_host_years", "hosts_time_as_host_months",
    "latitude", "longitude", "amenities", *FEAT_CAT,
})


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la limpieza de la sección 2 del TP.

    1. Descarta las columnas con fuga de información (2.2).
    2. Parsea el precio de texto ("$565.00") a número (2.4).
    3. Descarta avisos sin precio y los de más de CORTE_PRECIO BRL (2.4 y 2.5).
    4. Se queda sólo con las columnas que usan las features.

    Args:
        df: el CSV de Inside Airbnb tal como viene.

    Returns:
        DataFrame con COLUMNAS_CRUDAS más la columna `precio` (float, BRL).
    """
    df = df.drop(columns=COLS_FUGA, errors="ignore")
    df[TARGET] = df["price"].str.replace(r"[$,]", "", regex=True).astype(float)
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] <= CORTE_PRECIO]
    return df[[*COLUMNAS_CRUDAS, TARGET]].reset_index(drop=True)


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye las features de la sección 3.1 del TP.

    Todo lo que se calcula acá depende sólo de la propia fila (nunca de
    estadísticos de otras filas ni del target), así que puede hacerse antes del
    split sin fuga. Lo que sí se ajusta con datos (imputación, escalado,
    agrupación de categorías raras) vive dentro del pipeline, en `modelo.py`.

    Args:
        df: salida de `limpiar`.

    Returns:
        DataFrame con FEATURES más `precio`.
    """
    df = df.copy()

    # Antigüedad como anfitrión, en años con fracción.
    df["antiguedad_host"] = (
        df["hosts_time_as_host_years"].fillna(0)
        + df["hosts_time_as_host_months"].fillna(0) / 12
    )

    # Sin reseñas no hay puntaje: el faltante ES el dato. Bandera + relleno neutro.
    df["sin_resenas"] = (df["number_of_reviews"] == 0).astype(int)
    df[COLS_RESENAS] = df[COLS_RESENAS].fillna(0)

    # Atributos físicos no informados: bandera ANTES de que el imputer del
    # pipeline borre el rastro del faltante.
    for col, bandera in BANDERAS_FALTANTES.items():
        df[bandera] = df[col].isna().astype(int)

    # El baño compartido sale del texto libre de bathrooms_text.
    df["bano_compartido"] = (
        df["bathrooms_text"].str.contains("shared", case=False, na=False).astype(int)
    )

    # Amenities viene como una lista JSON en texto: una binaria por cada una.
    listas = df["amenities"].fillna("[]").apply(json.loads)
    for nombre, clave in AMENITIES.items():
        df[nombre] = listas.apply(
            lambda ls, k=clave: any(k in a.lower() for a in ls)
        ).astype(int)

    return df[[*FEATURES, TARGET]]
