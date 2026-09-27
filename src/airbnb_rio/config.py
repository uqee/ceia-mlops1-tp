"""Constantes del proyecto.

Todo lo que en el notebook de AMq1 era una constante suelta vive acá, para que
los DAGs y la API lean exactamente los mismos valores.
"""

# --- Reproducibilidad ---
SEMILLA = 42
TAMANO_TEST = 0.3

# --- Datos ---
# Snapshot fijo de Inside Airbnb (scrape del 24/06/2026). Se fija la fecha a
# propósito: Inside Airbnb re-scrapea periódicamente y entre marzo y junio los
# faltantes de `bedrooms` pasaron de 123 a 6.446. Con la URL fija, los números
# no cambian solos.
URL_RAW = (
    "https://data.insideairbnb.com/brazil/rj/rio-de-janeiro/"
    "2026-06-24/data/listings.csv.gz"
)

BUCKET = "data"
KEY_RAW = "raw/listings.csv.gz"
KEY_LIMPIO = "interim/listings_limpio.parquet"
KEY_FEATURES = "interim/listings_features.parquet"
KEY_TRAIN = "final/train.parquet"
KEY_TEST = "final/test.parquet"

# Precios por encima de este valor se descartan (sección 2.5 del TP de AMq1).
CORTE_PRECIO = 10_000

# Columnas que filtran el target (sección 2.2): nunca pueden entrar al modelo.
COLS_FUGA = [
    "price_quote_price_per_night",
    "price_quote_total_price",
    "price_quote_raw",
    "estimated_revenue_l365d",
    "price_quote_checkin_date",
    "price_quote_checkout_date",
]

# Columnas de reseñas que quedan nulas cuando el aviso no tiene reseñas.
COLS_RESENAS = [
    "reviews_per_month",
    "review_scores_rating",
    "review_scores_location",
    "review_scores_cleanliness",
    "review_scores_value",
]

# Bandera binaria de "no informado" para los atributos físicos (sección 3.1).
BANDERAS_FALTANTES = {
    "bedrooms": "sin_dormitorios",
    "bathrooms": "sin_banos",
    "beds": "sin_camas",
}

# Nombre de la feature binaria -> texto a buscar dentro de `amenities`.
AMENITIES = {
    "tiene_pileta": "pool",
    "tiene_aire": "air conditioning",
    "tiene_ascensor": "elevator",
    "tiene_gimnasio": "gym",
    "tiene_estacionamiento": "free parking",
    "tiene_lavarropas": "washer",
    "tiene_lavavajillas": "dishwasher",
    "tiene_vista_mar": "ocean view",
}

# --- Features ---
# Mismas features que el notebook. Los nombres con ñ se pasaron a ASCII
# (sin_resenas, sin_banos, bano_compartido) porque van a ser campos del JSON de la API.
FEAT_NUM = [
    "accommodates", "bedrooms", "beds", "bathrooms",
    "minimum_nights", "maximum_nights",
    "number_of_reviews", "reviews_per_month",
    "review_scores_rating", "review_scores_location",
    "review_scores_cleanliness", "review_scores_value",
    "availability_30", "availability_365",
    "estimated_occupancy_l365d", "calculated_host_listings_count",
    "antiguedad_host", "latitude", "longitude",
]

FEAT_BIN = [
    "sin_resenas", *BANDERAS_FALTANTES.values(), "bano_compartido",
    *AMENITIES.keys(),
]

FEAT_CAT = ["room_type", "property_type", "neighbourhood_cleansed"]

FEATURES = FEAT_NUM + FEAT_BIN + FEAT_CAT
TARGET = "precio"

# Categorías con menos avisos que esto se agrupan en una sola (sección 3.3).
UMBRAL_CATEGORIAS = 200

# --- Modelo ---
# Mejores hiperparámetros encontrados por Optuna en la sección 9.3 del TP de AMq1
# (100 trials, CV 3-fold). Con ellos el modelo dio MAE 250,5 BRL en test.
XGB_PARAMS = {
    "n_estimators": 719,
    "learning_rate": 0.0614611546741507,
    "max_depth": 8,
    "subsample": 0.6914652162026267,
    "colsample_bytree": 0.6281594438639964,
    "min_child_weight": 8,
    "reg_lambda": 0.13294694769765902,
    "reg_alpha": 2.254103823975803,
}

# Parámetros fijos (no tuneados) de XGBoost, iguales a XGB_FIJOS del notebook.
XGB_FIJOS = {
    "objective": "reg:absoluteerror",
    "random_state": SEMILLA,
    "n_jobs": -1,
    "tree_method": "hist",
}

# --- MLflow ---
EXPERIMENTO_MLFLOW = "airbnb_rio_price"
NOMBRE_MODELO = "airbnb_rio_price"
ALIAS_CHAMPION = "champion"
ALIAS_CHALLENGER = "challenger"
