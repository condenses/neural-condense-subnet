import numpy as np
import io
import time
import aiohttp
import os
from rich.progress import track, Progress
from ..logger import logger
import asyncio
import sys
import uuid


def _clean_tmp_directory():
    """Clean the tmp directory if running as validator."""
    if (
        __name__ != "__main__"
        and os.path.basename(os.path.abspath(sys.argv[0])) == "validator.py"
    ):
        os.makedirs("tmp", exist_ok=True)
        for file in track(os.listdir("tmp"), description="Cleaning tmp directory"):
            os.remove(os.path.join("tmp", file))


def _check_file_size(content_length: int, max_size_mb: int) -> tuple[bool, str]:
    """Check if file size is within limits."""
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
    """Download file using aiohttp."""
    try:
        filename = _generate_filename(url)
        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return "", 0, f"Failed to download: HTTP {response.status}"

                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0
                with open(filename, "wb") as f:
                    async for chunk in response.content.iter_chunked(
                        1024 * 1024
                    ):  # 1MB chunks
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size:
                            percent = downloaded * 100 / total_size

        download_time = time.time() - start_time
        logger.info(
            f"Time taken to download: {download_time:.2f} seconds. File size: {total_size / (1024 * 1024):.1f}MB"
        )
        return filename, download_time, ""
    except Exception as e:
        return "", 0, str(e)


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
    Load a `.npy` file from a URL using aiohttp for efficient downloading.

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
        async with aiohttp.ClientSession() as session:
            async with session.head(url) as response:
                if response.status != 200:
                    return (
                        None,
                        "",
                        0,
                        f"Failed to fetch file info: HTTP {response.status}",
                    )

                content_length = int(response.headers.get("content-length", 0))
                size_ok, error = _check_file_size(content_length, max_size_mb)
                if not size_ok:
                    return None, "", 0, error

        # Download and process file
        filename, download_time, error = await _download(url)
        if error:
            return None, "", 0, error

        if not filename:
            return None, "", 0, "Download failed: Empty filename received"

        if not os.path.exists(filename):
            return None, "", 0, f"Downloaded file not found at {filename}"

        data, error = _load_and_cleanup(filename)
        if error:
            return None, "", 0, error

        return data, filename, download_time, ""

    except Exception as e:
        return None, "", 0, str(e)


# Clean tmp directory on module load if running as validator
_clean_tmp_directory()
