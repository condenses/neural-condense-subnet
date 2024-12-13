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
import uuid
from concurrent.futures import ThreadPoolExecutor
from firerequests import FireRequests

fire_downloader = FireRequests()


def _clean_tmp_directory():
    """Clean the tmp directory if running as validator."""
    if (
        __name__ != "__main__"
        and os.path.basename(os.path.abspath(sys.argv[0])) == "validator.py"
    ):
        os.makedirs("tmp", exist_ok=True)
        for file in track(os.listdir("tmp"), description="Cleaning tmp directory"):
            os.remove(os.path.join("tmp", file))


def _check_file_size(response: httpx.Response, max_size_mb: int) -> tuple[bool, str]:
    """Check if file size is within limits."""
    content_length = int(response.headers.get("content-length", 0))
    max_size_bytes = max_size_mb * 1024 * 1024

    if content_length > max_size_bytes:
        return (
            False,
            f"File too large: {content_length / (1024 * 1024):.1f}MB exceeds {max_size_mb}MB limit",
        )
    return True, ""


def _generate_filename(url: str) -> str:
    """Generate a unique filename for downloaded file."""
    return os.path.join("tmp", str(uuid.uuid4()) + "_" + url.split("/")[-1])


async def _download(url: str) -> tuple[str, float, str]:
    """Download file using hf_transfer."""
    debug_start_time = time.time()
    try:
        filename = _generate_filename(url)
        start_time = time.time()

        await fire_downloader.download_file(
            url=url,
            filename=filename,
            max_files=10,  # Number of parallel downloads
            chunk_size=1024 * 1024,  # 1 MB chunks
            parallel_failures=3,
            max_retries=5,
            headers=None,
            show_progress=False,
        )

        download_time = time.time() - start_time
        logger.info(f"Time taken to download: {download_time:.2f} seconds")
        return filename, download_time, ""
    except Exception as e:
        return "", time.time() - debug_start_time, "Download failed: " + str(e)


def _load_and_cleanup(filename: str) -> tuple[np.ndarray | None, str]:
    """Load NPY file and convert to float32."""
    try:
        with open(filename, "rb") as f:
            buffer = io.BytesIO(f.read())
            data = np.load(buffer)
        return data.astype(np.float32), ""
    except Exception as e:
        logger.error(f"Error loading NPY file: {e}")
        return None, str(e)


async def load_npy_from_url(
    url: str, max_size_mb: int = 1024
) -> tuple[np.ndarray | None, str, float, str]:
    """
    Load a `.npy` file from a URL using the hf_transfer library for efficient downloading.

    Args:
        url (str): URL of the `.npy` file.
        max_size_mb (int): Maximum allowed file size in megabytes.

    Returns:
        tuple: (data, filename, download_time, error_message)
            - data: Loaded NumPy array or None if error
            - filename: Local filename where data was saved
            - download_time: Time taken to download in seconds
            - error_message: Empty string if successful, error description if failed
    """
    try:
        # Check file size using HTTP HEAD request
        async with httpx.AsyncClient() as client:
            response = await client.head(url)
            if response.status_code != 200:
                return (
                    None,
                    "",
                    0,
                    f"Failed to fetch file info: HTTP {response.status_code}",
                )

            size_ok, error = _check_file_size(response, max_size_mb)
            if not size_ok:
                return None, "", 0, error

        # Download and process file in thread pool asyncio
        filename, download_time, error = await _download(url)
        if error:
            return None, "", 0, error

        data, error = _load_and_cleanup(filename)
        if error:
            return None, "", 0, error

        return data, filename, download_time, ""

    except Exception as e:
        return None, "", 0, str(e)


# Clean tmp directory on module load if running as validator
_clean_tmp_directory()
