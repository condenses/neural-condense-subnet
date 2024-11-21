from bittensor import Synapse
from typing import Any
import torch
from transformers import DynamicCache
from .common import base64, file
from .constants import TierConfig


class Metadata(Synapse):
    metadata: dict = {}


class TextCompressProtocol(Synapse):
    context: str = ""
    compressed_kv_url: str = ""
    compressed_kv_b64: str = ""
    compressed_kv: Any = None
    expected_completion: str = ""
    activation_prompt: str = ""
    target_model: str = ""

    @property
    def miner_payload(self) -> dict:
        r"""
        Get the input for the miner.
        """
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
            "compressed_kv_url": self.compressed_kv_url,
            "expected_completion": self.expected_completion,
            "activation_prompt": self.activation_prompt,
        }

    @staticmethod
    def verify(
        response: "TextCompressProtocol", tier_config: TierConfig
    ) -> tuple[bool, str]:
        compressed_kv, error = file.download_file_from_url(response.compressed_kv_url)
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

        response.compressed_kv_b64 = base64.ndarray_to_base64(compressed_kv)
        return True, ""
