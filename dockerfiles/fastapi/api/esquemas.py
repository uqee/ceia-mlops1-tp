"""Contratos JSON de la API (entrada y salida), con su documentación para /docs.

`AvisoEntrada` tiene un campo por cada columna cruda que usa
`airbnb_rio.features.construir_features`, con los mismos nombres que el CSV de
Inside Airbnb. La API no calcula features por su cuenta: se las pasa a ese mismo
código, el que usó el entrenamiento.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Aviso real de Copacabana del snapshot del 24/06/2026 (precio real: 383 BRL).
EJEMPLO_AVISO = {
    "latitude": -22.968,
    "longitude": -43.184,
    "neighbourhood_cleansed": "Copacabana",
    "room_type": "Entire home/apt",
    "property_type": "Entire rental unit",
    "accommodates": 2,
    "bedrooms": 1,
    "beds": 1,
    "bathrooms": 1,
    "bathrooms_text": "1 bath",
    "amenities": [
        "Wifi", "Hair dryer", "Refrigerator", "Iron", "Air conditioning", "Kitchen",
        "Self check-in", "TV", "Free parking on premises", "Smart lock",
    ],
    "minimum_nights": 1,
    "maximum_nights": 365,
    "availability_30": 8,
    "availability_365": 39,
    "estimated_occupancy_l365d": 102,
    "number_of_reviews": 17,
    "reviews_per_month": 2.71,
    "review_scores_rating": 4.94,
    "review_scores_location": 5.0,
    "review_scores_cleanliness": 5.0,
    "review_scores_value": 4.88,
    "calculated_host_listings_count": 1,
    "hosts_time_as_host_years": 0,
    "hosts_time_as_host_months": 6,
}

# Los campos opcionales pueden faltar en Inside Airbnb: el modelo los imputa
# (mediana de train) y marca el faltante con una bandera (sin_dormitorios, etc.).
Puntaje = Field(default=None, ge=0, le=5)


class AvisoEntrada(BaseModel):
    """Un aviso de Airbnb en Río de Janeiro, tal como lo publica Inside Airbnb."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EJEMPLO_AVISO]})

    # Ubicación
    latitude: float = Field(ge=-23.2, le=-22.6, description="Latitud (municipio de Río).")
    longitude: float = Field(ge=-43.9, le=-43.0, description="Longitud (municipio de Río).")
    neighbourhood_cleansed: str = Field(
        description="Barrio. Uno con pocos avisos o desconocido se trata como 'Otros'.",
    )

    # Tipo de aviso
    room_type: Literal["Entire home/apt", "Private room", "Shared room", "Hotel room"]
    property_type: str = Field(
        description="Tipo de propiedad, p. ej. 'Entire rental unit'. Los raros cuentan como 'Otros'.",
    )

    # Capacidad
    accommodates: int = Field(ge=1, description="Huéspedes que admite.")
    bedrooms: float | None = Field(default=None, ge=0, description="Dormitorios.")
    beds: float | None = Field(default=None, ge=0, description="Camas.")
    bathrooms: float | None = Field(default=None, ge=0, description="Baños.")
    bathrooms_text: str | None = Field(
        default=None, description="Texto libre de baños; si dice 'shared', el baño es compartido.",
    )
    amenities: list[str] = Field(
        default_factory=list,
        description="Comodidades. Se buscan pool, air conditioning, elevator, gym, "
                    "free parking, washer, dishwasher y ocean view.",
    )

    # Reglas y disponibilidad
    minimum_nights: int | None = Field(default=None, ge=1, description="Noches mínimas.")
    maximum_nights: int | None = Field(default=None, ge=1, description="Noches máximas.")
    availability_30: int = Field(ge=0, le=30, description="Noches libres en los próximos 30 días.")
    availability_365: int = Field(ge=0, le=365, description="Noches libres en el próximo año.")
    estimated_occupancy_l365d: int = Field(
        ge=0, le=365, description="Noches ocupadas estimadas en el último año.",
    )

    # Reseñas (sin reseñas, los puntajes van en null)
    number_of_reviews: int = Field(ge=0, description="Cantidad de reseñas.")
    reviews_per_month: float | None = Field(default=None, ge=0, description="Reseñas por mes.")
    review_scores_rating: float | None = Puntaje
    review_scores_location: float | None = Puntaje
    review_scores_cleanliness: float | None = Puntaje
    review_scores_value: float | None = Puntaje

    # Anfitrión
    calculated_host_listings_count: int = Field(ge=1, description="Avisos del anfitrión.")
    hosts_time_as_host_years: int | None = Field(
        default=None, ge=0, description="Años como anfitrión.",
    )
    hosts_time_as_host_months: int | None = Field(
        default=None, ge=0, le=11, description="Meses adicionales como anfitrión.",
    )


class PrediccionSalida(BaseModel):
    precio_brl: float = Field(description="Precio por noche estimado, en reales.")
    version_modelo: str = Field(description="Versión del modelo del registry que respondió.")


class ModeloInfo(BaseModel):
    nombre: str
    alias: str
    version: str
    test_mae: float | None = Field(description="MAE en test (BRL) medido por train_model.")
