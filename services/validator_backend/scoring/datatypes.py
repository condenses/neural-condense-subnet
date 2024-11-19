from typing import Any, List, Union, Dict
from pydantic import BaseModel
from .utils import base64_to_ndarray


class MinerResponse(BaseModel):
    compressed_tokens_b64: str
    compressed_tokens: Any = None

    def decode(self):
        self.compressed_tokens = base64_to_ndarray(self.compressed_tokens_b64)


class GroundTruthRequest(BaseModel):
    context: str
    expected_completion: str
    activation_prompt: str
    model_name: str
    criterias: List[str]


class BatchedScoringRequest(BaseModel):
    miner_responses: List[MinerResponse]
    ground_truth_request: GroundTruthRequest
