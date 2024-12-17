import re
from bittensor import Synapse
from typing import Any, List
import torch
from transformers import DynamicCache
from pydantic import BaseModel
from .common.file import load_npy_from_url
from .constants import TierConfig
import numpy as np
import io


class Metadata(Synapse):
    metadata: dict = {}


class TaskData(BaseModel):
    formatted_context: str = ""
    original_context: str = ""
    challenge_questions: List[str] = []
    challenge_answers: List[str] = []
    formatted_questions: List[str] = []
    negative_chunks: List[str] = []
    positive_chunks: List[str] = []


class UtilData(BaseModel):
    compressed_kv_b64: str = ""
    compressed_length: int = 0
    download_time: float = 0.0
    bonus_compress_size: float = 0.0
    bonus_time: float = 0.0
    local_filename: str = ""


class MinerResponse(BaseModel):
    filename: str = ""


class BatchedScoringRequest(BaseModel):
    miner_responses: List[MinerResponse] = []
    task_data: TaskData = TaskData()
    target_model: str = ""
    criterias: List[str] = []


class TextCompressProtocol(Synapse):
    context: str = ""
    target_model: str = ""
    compressed_kv_url: str = ""
    util_data: UtilData = UtilData()
    task_data: TaskData = TaskData()

    @property
    def accelerate_score(self) -> float:
        return (self.util_data.bonus_compress_size + self.bonus_time) / 2

    @property
    def bonus_time(self) -> float:
        return 1 - min(
            1,
            (self.dendrite.process_time + self.util_data.download_time) / self.timeout,
        )

    @property
    def miner_payload(self) -> dict:
        return {"context": self.context, "target_model": self.target_model}

    @property
    def miner_synapse(self, is_miner: bool = False):
        return TextCompressProtocol(
            **self.model_dump(include={"context", "target_model"})
        )

    @property
    def validator_payload(self) -> dict:
        return {
            "task_data": self.task_data.model_dump(),
            "util_data": self.util_data.model_dump(),
        }

    @staticmethod
    def get_scoring_payload(
        responses: List["TextCompressProtocol"],
        ground_truth_synapse: "TextCompressProtocol",
        target_model: str,
        criterias: List[str],
    ) -> BatchedScoringRequest:
        return BatchedScoringRequest(
            miner_responses=[
                {"filename": r.util_data.local_filename} for r in responses
            ],
            task_data=ground_truth_synapse.task_data,
            target_model=target_model,
            criterias=criterias,
        )

    @staticmethod
    async def verify(
        response: "TextCompressProtocol", tier_config: TierConfig
    ) -> tuple[bool, str]:
        if not re.match(r"^https?://.*\.npy$", response.compressed_kv_url):
            return False, "Compressed KV URL must use HTTP or HTTPS."

        compressed_kv, filename, download_time, error = await load_npy_from_url(
            response.compressed_kv_url
        )
        response.util_data.download_time = download_time
        response.util_data.local_filename = filename
        if compressed_kv is None:
            return (
                False,
                f"Failed to load url: {error}. {download_time} seconds. {filename}",
            )
        try:
            tensor = torch.from_numpy(compressed_kv)
            kv_cache = DynamicCache.from_legacy_cache(tensor)
        except Exception as e:
            return False, f"{error} -> {str(e)}"

        if not (
            tier_config.min_condensed_tokens
            <= kv_cache._seen_tokens
            <= tier_config.max_condensed_tokens
        ):
            return False, "Compressed tokens are not within the expected range."

        response.util_data.bonus_compress_size = 1 - (
            kv_cache._seen_tokens / tier_config.max_condensed_tokens
        )
        response.util_data.compressed_length = kv_cache._seen_tokens
        del kv_cache
        del compressed_kv
        return True, ""
