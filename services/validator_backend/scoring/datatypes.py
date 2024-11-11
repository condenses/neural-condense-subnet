from typing import Any, List
from pydantic import BaseModel


class ScoringRequest(BaseModel):
    compressed_tokens_b64: str
    compressed_tokens: Any = None


class GroundTruthRequest(BaseModel):
    context: str
    expected_completion: str
    activation_prompt: str
    model_name: str
    criterias: List[str]
    last_prompt: str = ""


class BatchedScoringRequest(BaseModel):
    miner_responses: List[ScoringRequest]
    ground_truth_request: GroundTruthRequest
