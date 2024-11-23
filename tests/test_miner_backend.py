import requests
import argparse
from neural_condense_core.common.base64 import base64_to_ndarray

# Default values for the base URL and port
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8080
DEFAULT_API_PATH = "/condense"


def get_args():
    """
    Function to parse command-line arguments for test configuration.
    """
    parser = argparse.ArgumentParser(description="Test API Endpoints.")
    parser.add_argument(
        "--host", type=str, default=DEFAULT_HOST, help="API host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="API port (default: 8080)"
    )
    parser.add_argument(
        "--api-path",
        type=str,
        default=DEFAULT_API_PATH,
        help="API path (default: /condense)",
    )

    parser.add_argument(
        "--target-model",
        type=str,
        default="Condense-AI/Mistral-7B-Instruct-v0.1",
    )

    args, _ = parser.parse_known_args()  # Avoid conflict with pytest's arguments
    return args


# Get arguments from the command line
args = get_args()

# Construct the base URL using the provided arguments
BASE_URL = f"http://{args.host}:{args.port}{args.api_path}"


def get_api_url():
    """
    Function to provide the full API URL based on the host, port, and path.
    """
    return BASE_URL


def test_miner_api():
    """
    Test the prediction endpoint by sending a valid context and model request.
    """
    api_url = get_api_url()

    payload = {
        "context": "This is a long test context that needs to be compressed.",
        "target_model": args.target_model,
    }

    response = requests.post(api_url, json=payload)

    if response.status_code != 200:
        raise Exception(f"Expected status code 200 but got {response.status_code}")

    data = response.json()

    if "compressed_tokens_base64" not in data:
        raise Exception("Response should contain compressed_tokens_base64.")

    compressed_tokens = base64_to_ndarray(data["compressed_tokens_base64"])

    seq_len, hidden_size = compressed_tokens.shape

    print(f"Compressed tokens shape: {seq_len} x {hidden_size}")

    print("API test passed successfully!")
