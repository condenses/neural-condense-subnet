import minio
import io
import numpy as np


def upload_to_minio(
    minio_client: minio.Minio,
    bucket_name: str,
    object_name: str,
    data: tuple[tuple[np.ndarray, ...], ...],
):
    buffer = io.BytesIO()
    np.save(buffer, data)
    length = buffer.tell()
    buffer.seek(0)
    result = minio_client.put_object(bucket_name, object_name, buffer, length)
    return result
