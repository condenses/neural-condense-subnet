from typing import Any, List
from pydantic import BaseModel
import numpy as np
import io
import base64


class MinerResponse(BaseModel):
    filename: str
    compressed_kv: Any = None

    def decode(self):
        self.compressed_kv = load_npy_from_filename(self.filename)


class GroundTruthRequest(BaseModel):
    context: str
    expected_completion: str
    activation_prompt: str
    model_name: str
    messages: List[dict]
    hidden_messages: List[dict]
    criterias: List[str]
    positive_chunks: List[str]
    negative_chunks: List[str]


class BatchedScoringRequest(BaseModel):
    miner_responses: List[MinerResponse]
    ground_truth_request: GroundTruthRequest


def load_npy_from_filename(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        buffer = io.BytesIO(f.read())
        return np.load(buffer).astype(np.float32)


def base64_to_ndarray(base64_str: str) -> np.ndarray:
    try:
        """Convert a base64-encoded string back to a NumPy array."""
        buffer = io.BytesIO(base64.b64decode(base64_str))
        buffer.seek(0)
        array = np.load(buffer)
        array = array.astype(np.float32)
    except Exception as e:
        print(e)
        return None
    return array


def ndarray_to_base64(array: np.ndarray) -> str:
    try:
        """Convert a NumPy array to a base64-encoded string."""
        buffer = io.BytesIO()
        np.save(buffer, array)
        buffer.seek(0)
        base64_str = base64.b64encode(buffer.read()).decode("utf-8")
    except Exception as e:
        print(e)
        return ""
    return base64_str
