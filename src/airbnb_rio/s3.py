"""Lectura y escritura en el data lake (MinIO, compatible con S3).

boto3 toma las credenciales y el endpoint de las variables de entorno que ya
define el docker-compose (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY y
AWS_ENDPOINT_URL_S3=http://s3:9000). Ojo: desde adentro de los contenedores el
host es `s3`, no `localhost`.
"""

import io

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from airbnb_rio.config import BUCKET


def _cliente():
    return boto3.client("s3")


def existe(key: str, bucket: str = BUCKET) -> bool:
    """True si el objeto ya está en el bucket.

    Sólo un 404 significa "no existe". Cualquier otro error (credenciales,
    permisos, MinIO caído) se propaga, para que la task falle con la causa real
    en vez de intentar descargar de nuevo.
    """
    try:
        _cliente().head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as error:
        if error.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def subir_archivo(ruta_local: str, key: str, bucket: str = BUCKET) -> str:
    """Sube un archivo local y devuelve su URI s3://."""
    _cliente().upload_file(ruta_local, bucket, key)
    return f"s3://{bucket}/{key}"


def leer_csv(key: str, bucket: str = BUCKET, **kwargs) -> pd.DataFrame:
    """Lee un CSV (comprimido o no) del bucket."""
    obj = _cliente().get_object(Bucket=bucket, Key=key)
    compresion = "gzip" if key.endswith(".gz") else None
    return pd.read_csv(io.BytesIO(obj["Body"].read()), compression=compresion, **kwargs)


def leer_parquet(key: str, bucket: str = BUCKET) -> pd.DataFrame:
    """Lee un parquet del bucket."""
    obj = _cliente().get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def escribir_parquet(df: pd.DataFrame, key: str, bucket: str = BUCKET) -> str:
    """Escribe un DataFrame como parquet y devuelve su URI s3://."""
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    _cliente().put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    return f"s3://{bucket}/{key}"
