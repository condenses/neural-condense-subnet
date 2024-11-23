import re
from bittensor import Synapse
from typing import Any
import torch
from transformers import DynamicCache
import time
from .common.base64 import ndarray_to_base64
from .common.file import load_npy_from_url
from .constants import TierConfig


class Metadata(Synapse):
    metadata: dict = {}


class TextCompressProtocol(Synapse):
    context: str = ""
    compressed_kv_url: str = ""
    compressed_kv_b64: str = ""
    compressed_kv: Any = None
    compressed_length: int = 0
    expected_completion: str = ""
    activation_prompt: str = ""
    target_model: str = ""
    download_time: float = 0.0
    bonus_compress_size: float = 0.0

    @property
    def accelerate_score(self) -> float:
        return (self.bonus_compress_size + self.bonus_time) / 2

    @property
    def bonus_time(self) -> float:
        return min(0, (self.dendrite.process_time + self.download_time) / self.timeout)

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
            "context": self.context,
            "expected_completion": self.expected_completion,
            "activation_prompt": self.activation_prompt,
        }

    @staticmethod
    async def verify(
        response: "TextCompressProtocol", tier_config: TierConfig
    ) -> tuple[bool, str]:
        if not re.match(r"^https?://.*\.npy$", response.compressed_kv_url):
            return False, "Compressed KV URL must use HTTP or HTTPS."

        start_time = time.time()
        compressed_kv, error = await load_npy_from_url(response.compressed_kv_url)
        response.download_time = time.time() - start_time
        try:
            kv_cache = DynamicCache.from_legacy_cache(torch.from_numpy(compressed_kv))
        except Exception as e:
            return False, f"{error} -> {str(e)}"

        if not (
            tier_config.min_condensed_tokens
            <= kv_cache._seen_tokens
            <= tier_config.max_condensed_tokens
        ):
            return False, "Compressed tokens are not within the expected range."

        response.bonus_compress_size = 1 - (
            kv_cache._seen_tokens / tier_config.max_condensed_tokens
        )

        response.compressed_kv_b64 = ndarray_to_base64(compressed_kv)
        response.compressed_length = compressed_kv[0][0].shape[2]
        return True, ""
