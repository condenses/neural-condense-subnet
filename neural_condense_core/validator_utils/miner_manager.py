import json
import os
import bittensor as bt
import numpy as np
from ..common import build_rate_limit
from ..protocol import Metadata
from ..constants import constants
import threading
import requests
import pydantic
import httpx
from fastapi import Request
import random 


class ServingCounter:
    """
    A counter for rate limiting requests to a miner from this validator.
    - rate_limit: int, the maximum number of requests allowed per epoch.
    - counter: int, the current number of requests made in the current epoch.
    """

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.counter = 0
        self.lock = threading.Lock()

    def increment(self) -> bool:
        """
        Increments the counter and returns True if the counter is less than or equal to the rate limit.
        """
        with self.lock:
            self.counter += 1
            return self.counter <= self.rate_limit


class MinerManager:
    r"""
    Manages the metadata and serving counter of miners.
    """

    def __init__(self, validator,wallet):
        self.validator = validator
        self.dendrite: bt.dendrite = validator.dendrite
        self.metagraph = validator.metagraph
        self.default_metadata_items = [
            ("tier", "unknown"),
        ]
        self.wallet = wallet
        self.message = "".join(random.choices("0123456789abcdef", k=16))
        self.metadata = self._init_metadata()
        bt.logging.info(f"Metadata: {self.metadata}")
        self.load_state()
        self.state_path = self.validator.config.full_path + "/state.json"
        self.sync()

    def _authenticate(self, request: Request):
        message = request.headers["message"]
        if message != self.message:
            raise Exception("Authentication failed.")


    def update_scores(self, scores: list[float], uids: list[int]):
        r"""
        Updates the scores of the miners.
        """
        for score, uid in zip(scores, uids):
            self.metadata[uid]["score"] = (
                constants.SCORE_MOVING_AVERAGE * score
                + (1 - constants.SCORE_MOVING_AVERAGE) * self.metadata[uid]["score"]
            )

    def get_normalized_scores(self, eps: float = 1e-6) -> np.ndarray:
        scores = np.zeros(len(self.metagraph.hotkeys))
        for uid, metadata in self.metadata.items():
            scores[uid] = metadata["score"]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + eps)
        return scores

    def load_state(self):
        try:
            if os.path.exists(self.state_path):
                state = json.load(open(self.config.full_path + "/state.json", "r"))
                self.metadata = state["metadata"]
                bt.logging.success("Loaded state.")
        except Exception as e:
            bt.logging.error(f"Failed to load state: {e}")

    def save_state(self):
        try:
            state = {"metadata": self.metadata}
            json.dump(state, open(self.state_path, "w"))
            bt.logging.success("Saved state.")
        except Exception as e:
            bt.logging.error(f"Failed to save state: {e}")

    def _init_metadata(self):
        r"""
        Initializes the metadata of the miners.
        """
        metadata = {
            uid: {
                "score": 0.0,
                "tier": "unknown",
            }
            for uid in self.metagraph.uids
        }
        return metadata

    def sync(self):
        r"""
        Synchronizes the metadata and serving counter of miners.
        """
        self.metadata = self._update_metadata()
        self.serving_counter: dict[int, ServingCounter] = self._create_serving_counter()

    def _update_metadata(self):
        r"""
        Updates the metadata of the miners by whitelisted synapse queries.
        It doesn't consume validator's serving counter.
        """
        synapse = Metadata()
        metadata = {}
        uids = [uid for uid in range(len(self.metagraph.hotkeys))]
        axons = [self.metagraph.axons[uid] for uid in uids]
        responses = self.dendrite.query(
            axons,
            synapse,
            deserialize=False,
            timeout=4,
        )
        bt.logging.info(f"Responses: {responses}")
        for uid, response in zip(uids, responses):
            metadata.setdefault(uid, {})
            current_tier = self.metadata.get(uid, {}).get("tier", "unknown")
            for key, default_value in self.default_metadata_items:
                if response:
                    metadata[uid][key] = response.metadata.get(key, default_value)
                else:
                    metadata[uid][key] = default_value

            if metadata[uid]["tier"] != current_tier:
                bt.logging.info(
                    f"Tier of uid {uid} changed from {current_tier} to {metadata[uid]['tier']}."
                )
                metadata[uid]["score"] = 0.0
            if "score" not in metadata[uid]:
                metadata[uid]["score"] = 0.0

        bt.logging.info(f"Metadata: {metadata}")
        bt.logging.success(f"Updated metadata for {len(uids)} uids.")
        return metadata

    def _create_serving_counter(self):
        r"""
        Creates a serving counter for each tier of miners.
        """
        rate_limit_per_tier = {
            tier: build_rate_limit(self.metagraph, self.validator.config, tier)[
                self.validator.uid
            ]
            for tier in constants.TIER_CONFIG.keys()
        }
        tier_group = {tier: {} for tier in constants.TIER_CONFIG.keys()}

        for uid, metadata in self.metadata.items():
            tier = metadata.get("tier", "unknown")
            if tier not in constants.TIER_CONFIG:
                continue
            if tier not in tier_group:
                tier_group[tier] = {}
            tier_group[tier][uid] = ServingCounter(rate_limit_per_tier[tier])

        return tier_group


    def store_miner_info(self, request: Request):

        try:
            signature = f"0x{self.dendrite.keypair.sign(self.message).hex()}"
            headers = {
                "Content-Type": "application/json",
                "message": self.message,
                "ss58_address": self.wallet.hotkey.ss58_address,
                "signature": signature,
            }
            self._authenticate(request)
            requests.post(
                url=self.validator.config.storage_url + "/store_miner_info",
                json={
                    "uid": self.validator.uid,
                    "info": self.metadata,
                },
                headers=headers,
            )
        except Exception as e:
            bt.logging.error(f"Failed to store miner info: {e}")
