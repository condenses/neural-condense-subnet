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
from neural_condense_core.logger import logger


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
    compressed_context: str = ""


class BatchedScoringRequest(BaseModel):
    miner_responses: List[MinerResponse] = []
    task_data: TaskData = TaskData()
    target_model: str = ""
    criterias: List[str] = []


class TextCompressProtocol(Synapse):
    context: str = ""
    target_model: str = ""
    tier: str = ""
    compressed_kv_url: str = ""
    compressed_context: str = ""
    util_data: UtilData = UtilData()
    task_data: TaskData = TaskData()

    @property
    def accelerate_score(self) -> float:
        return self.util_data.bonus_compress_size

    @property
    def bonus_time(self) -> float:
        if self.tier == "universal":
            return 1 - min(1, self.dendrite.process_time / self.timeout)
        else:
            return 1 - min(
                1,
                (self.dendrite.process_time + self.util_data.download_time)
                / self.timeout,
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
            "tier": self.tier,
        }

    @staticmethod
    def get_scoring_payload(
        responses: List["TextCompressProtocol"],
        ground_truth_synapse: "TextCompressProtocol",
        target_model: str,
        criterias: List[str],
    ) -> BatchedScoringRequest:
        if ground_truth_synapse.tier != "universal":
            return BatchedScoringRequest(
                miner_responses=[
                    {"filename": r.util_data.local_filename} for r in responses
                ],
                task_data=ground_truth_synapse.task_data,
                target_model=target_model,
                criterias=criterias,
            )
        else:
            return BatchedScoringRequest(
                miner_responses=[
                    {"compressed_context": r.compressed_context} for r in responses
                ],
                task_data=ground_truth_synapse.task_data,
                target_model=target_model,
                criterias=criterias,
            )

    @staticmethod
    async def verify(
        response: "TextCompressProtocol",
        tier_config: TierConfig,
        tier: str,
        tokenizer=None,
        ground_truth_synapse: "TextCompressProtocol" = None,
    ) -> tuple[bool, str]:
        print(f"Verifying tier: {tier}")
        if tier == "universal":
            condensed_tokens = tokenizer.encode(response.compressed_context)
            original_tokens = tokenizer.encode(ground_truth_synapse.context)
            n_condensed_tokens = len(condensed_tokens)
            compress_rate = n_condensed_tokens / len(original_tokens)
            logger.info(f"Compress rate: {compress_rate}")
            if not (
                tier_config.min_compress_rate
                <= compress_rate
                <= tier_config.max_compress_rate
            ):
                return (
                    False,
                    f"Compressed tokens are not within the expected range. {compress_rate}. Valid range: {tier_config.min_compress_rate} to {tier_config.max_compress_rate}",
                )

            response.util_data.bonus_compress_size = 1 - compress_rate
            response.util_data.compressed_length = n_condensed_tokens
            return True, ""
        else:
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
