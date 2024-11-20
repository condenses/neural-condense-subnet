import base64
import io
from flask import Flask, request, jsonify
import numpy as np
import time
from .soft_token_condenser_modeling import Condenser
from .logger import InferenceLogger

app = Flask(__name__)


# Helper function to convert NumPy array to base64
def ndarray_to_base64(array: np.ndarray) -> str:
    """Convert a NumPy array to a base64-encoded string."""
    buffer = io.BytesIO()
    np.save(buffer, array)
    buffer.seek(0)
    base64_str = base64.b64encode(buffer.read()).decode("utf-8")
    return base64_str


REPO_ID = "Condense-AI/Soft-Token-Condenser-Llama-3.2-1B"
condenser = Condenser.from_pretrained(REPO_ID)
condenser.eval()


# Define endpoint for compression
@app.route("/condense", methods=["POST"])
def compress_endpoint():
    t1 = time.time()
    data = request.get_json()
    context = data.get("context")
    target_model = data.get("target_model")

    if not context:
        return jsonify({"error": "Missing 'context' in request."}), 400

    try:
        # Compress context into condensed tokens
        compressed_tokens = condenser.compress(context)
        compressed_tokens = compressed_tokens.squeeze(0)
        compressed_tokens = compressed_tokens.cpu().numpy().astype(np.float32)
        compress_tokens_bs64 = ndarray_to_base64(compressed_tokens)

        # Log inference details
        InferenceLogger.log(
            "Predict",
            f"Compressed token length {len(compressed_tokens)} shape {compressed_tokens.shape}",
        )
        InferenceLogger.log("Inference time", time.time() - t1)

        return jsonify(
            {
                "compressed_tokens_b64": compress_tokens_bs64,
                "target_model": target_model,
            }
        )
    except Exception as e:
        InferenceLogger.log("Error", str(e))
        return (
            jsonify({"error": "Failed to process the request.", "details": str(e)}),
            500,
        )
