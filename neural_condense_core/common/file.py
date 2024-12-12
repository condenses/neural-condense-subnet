import hf_transfer
import numpy as np
import io
import time
import httpx
import os
from rich.progress import track
from ..logger import logger
import asyncio
import sys

# Only clean tmp directory if running as validator
if (
    __name__ != "__main__"
    and os.path.basename(os.path.abspath(sys.argv[0])) == "validator.py"
):
    os.makedirs("tmp", exist_ok=True)
    # Remove all files in the tmp directory
    for file in track(os.listdir("tmp"), description="Cleaning tmp directory"):
        os.remove(os.path.join("tmp", file))


async def load_npy_from_url(url: str, max_size_mb: int = 1024):
    """
    Load a `.npy` file from a URL using the hf_transfer library for efficient downloading.

    Args:
        url (str): URL of the `.npy` file.
        max_size_mb (int): Maximum allowed file size in megabytes.

    Returns:
        tuple: A tuple containing the loaded data as a NumPy array and an error message (empty if no error occurred).
    """
    try:
        # Check file size using an HTTP HEAD request
        async with httpx.AsyncClient() as client:
            response = await client.head(url)
            if response.status_code != 200:
                return None, f"Failed to fetch file info: HTTP {response.status_code}"

            # Get content length in bytes
            content_length = int(response.headers.get("content-length", 0))
            max_size_bytes = max_size_mb * 1024 * 1024

            # Check if file exceeds the size limit
            if content_length > max_size_bytes:
                return (
                    None,
                    f"File too large: {content_length / (1024 * 1024):.1f}MB exceeds {max_size_mb}MB limit",
                )

        # Define parameters for hf_transfer
        def _download(url):
            filename = os.path.join("tmp", url.split("/")[-1])
            chunk_size = 1024 * 1024  # 1 MB chunks
            max_files = 16  # Number of parallel downloads
            parallel_failures = 2
            max_retries = 3

            start_time = time.time()
            hf_transfer.download(
                url=url,
                filename=filename,
                max_files=max_files,
                chunk_size=chunk_size,
                parallel_failures=parallel_failures,
                max_retries=max_retries,
                headers=None,  # Add headers if needed
            )
            end_time = time.time()
            logger.info(f"Time taken to download: {end_time - start_time:.2f} seconds")
            return filename, end_time - start_time

        filename, download_time = await asyncio.to_thread(_download, url)

        data = _load_and_cleanup(filename)
        return data, filename, download_time, ""
    except Exception as e:
        return None, "", 0, str(e)


def _load_and_cleanup(filename: str):
    """Helper function to handle blocking operations in a thread pool."""
    try:
        with open(filename, "rb") as f:
            buffer = io.BytesIO(f.read())
            data = np.load(buffer)
        return data.astype(np.float32)
    except Exception as e:
        logger.error(f"Error loading NPY file: {e}")
        return None
