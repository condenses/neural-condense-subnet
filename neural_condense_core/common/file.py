import requests
import numpy as np


def download_file_from_url(url: str, max_size_mb: int = 10):
    try:
        # Stream the response to check size first
        response = requests.get(url, stream=True)

        # Check if the request was successful
        if response.status_code == 200:
            # Get content length in bytes
            content_length = int(response.headers.get("content-length", 0))

            # Convert max_size_mb to bytes
            max_size_bytes = max_size_mb * 1024 * 1024

            # Check if file is too large
            if content_length > max_size_bytes:
                return (
                    None,
                    f"File too large: {content_length/1024/1024:.1f}MB exceeds {max_size_mb}MB limit",
                )

            # Download and load if size is acceptable
            data = np.load(response.content)
            return data, ""

    except Exception as e:
        return None, str(e)
