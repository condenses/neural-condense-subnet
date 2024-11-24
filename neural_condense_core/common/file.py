import hf_transfer
import numpy as np
import io
import time
import httpx
import os
from tqdm import tqdm
from ..logger import logger

os.makedirs("tmp", exist_ok=True)


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
        filename = os.path.join("tmp", url.split("/")[-1])
        chunk_size = 1024 * 1024  # 1 MB chunks
        max_files = 128  # Number of parallel downloads
        parallel_failures = 2
        max_retries = 3

        # Initialize the tqdm progress bar
        with tqdm(
            total=content_length, unit="B", unit_scale=True, desc="Downloading"
        ) as pbar:

            def progress_callback(size):
                pbar.update(size)

            # Start measuring download time
            start_time = time.time()

            # Use hf_transfer to download the file
            hf_transfer.download(
                url=url,
                filename=filename,
                max_files=max_files,
                chunk_size=chunk_size,
                parallel_failures=parallel_failures,
                max_retries=max_retries,
                headers=None,  # Add headers if needed
                callback=progress_callback,
            )

            # Measure end time
            end_time = time.time()
            logger.info(f"Time taken to download: {end_time - start_time:.2f} seconds")

        # Load the NumPy array from the downloaded file
        with open(filename, "rb") as f:
            buffer = io.BytesIO(f.read())
            data = np.load(buffer)
        os.remove(filename)
        return data, ""

    except Exception as e:
        return None, str(e)
