import re
from bittensor import Synapse
from typing import Any, List
import torch
from transformers import DynamicCache
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
    messages: List[dict] = []
    hidden_messages: List[dict] = []
    activation_prompt: str = ""
    target_model: str = ""
    local_filename: str = ""
    download_time: float = 0.0
    bonus_compress_size: float = 0.0
    negative_chunks: List[str] = []
    positive_chunks: List[str] = []

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
            "messages": self.messages,
            "hidden_messages": self.hidden_messages,
            "positive_chunks": self.positive_chunks,
            "negative_chunks": self.negative_chunks,
        }

    @staticmethod
    async def verify(
        response: "TextCompressProtocol", tier_config: TierConfig
    ) -> tuple[bool, str]:
        if not re.match(r"^https?://.*\.npy$", response.compressed_kv_url):
            return False, "Compressed KV URL must use HTTP or HTTPS."

        compressed_kv, filename, download_time, error = await load_npy_from_url(
            response.compressed_kv_url
        )
        response.download_time = download_time
        response.local_filename = filename
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

        response.bonus_compress_size = 1 - (
            kv_cache._seen_tokens / tier_config.max_condensed_tokens
        )
        response.compressed_length = kv_cache._seen_tokens
        del kv_cache
        del compressed_kv
        return True, ""
