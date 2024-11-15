import json
import os
import bittensor as bt
import numpy as np
import random
import httpx
import pandas as pd
from ..common import build_rate_limit
from ..protocol import Metadata
from ..constants import constants
from .elo import ELOSystem
import threading
from pydantic import BaseModel

class MetadataItem(BaseModel):
    tier: str = "unknown"
    elo_rating: float = constants.INITIAL_ELO_RATING

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

    def __init__(self, validator):
        self.validator = validator
        self.wallet = validator.wallet
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = validator.metagraph
        self.elo_system = ELOSystem()
        self.default_metadata_items = [
            ("tier", "unknown"),
        ]
        self.config = validator.config
        self.metadata = self._init_metadata()
        self.state_path = self.validator.config.full_path + "/state.json"
        self.message = "".join(random.choices("0123456789abcdef", k=16))
        self.load_state()
        self.sync()

    def update_ratings(self, scores: list[float], uids: list[int]):
        """
        Updates the ELO ratings of the miners.
        """
        # Get current ELO ratings for participating miners
        current_ratings = [self.metadata[uid].elo_rating for uid in uids]
        
        # Update ELO ratings based on performance scores
        new_ratings = self.elo_system.update_ratings(current_ratings, scores)
        
        # Update metadata with new ratings and scores
        for uid, new_rating in zip(uids, new_ratings):
            self.metadata[uid] = MetadataItem(tier=self.metadata[uid].tier, elo_rating=new_rating)

    def get_normalized_ratings(self) -> np.ndarray:
        """
        Get normalized ratings using ELO ratings within each tier.
        """
        weights = np.zeros(len(self.metagraph.hotkeys))
        
        for tier in constants.TIER_CONFIG.keys():
            # Get ELO ratings for miners in this tier
            tier_ratings = []
            tier_uids = []
            
            for uid, metadata in self.metadata.items():
                if metadata.tier == tier:
                    tier_ratings.append(metadata.elo_rating)
                    tier_uids.append(uid)
            
            if tier_ratings:
                # Normalize ELO ratings to weights, sum to 1
                normalized_ratings = self.elo_system.normalize_ratings(tier_ratings)
                
                # Apply tier incentive percentage
                tier_weights = np.array(normalized_ratings) * constants.TIER_CONFIG[tier].incentive_percentage
                
                # Assign weights to corresponding UIDs
                for uid, weight in zip(tier_uids, tier_weights):
                    weights[uid] = weight
        
        return weights

    def load_state(self):
        try:
            state = json.load(open(self.state_path, "r"))
            metadata_items = {int(k): MetadataItem(**v) for k, v in state["metadata"].items()}
            self.metadata = metadata_items
            self._log_metadata()
            bt.logging.success("Loaded state.")
        except Exception as e:
            bt.logging.error(f"Failed to load state: {e}")

    def save_state(self):
        try:
            metadata_dict = {k: v.dict() for k, v in self.metadata.items()}
            state = {"metadata": metadata_dict}
            json.dump(state, open(self.state_path, "w"))
            bt.logging.success("Saved state.")
        except Exception as e:
            bt.logging.error(f"Failed to save state: {e}")

    def _init_metadata(self):
        r"""
        Initializes the metadata of the miners.
        """
        metadata = {
            int(uid): MetadataItem()
            for uid in self.metagraph.uids
        }
        return metadata

    def sync(self):
        r"""
        Synchronizes the metadata and serving counter of miners.
        """
        self.metadata = self._update_metadata()
        self.serving_counter: dict[str, dict[int, ServingCounter]] = (
            self._create_serving_counter()
        )
        self._log_metadata()

    def _log_metadata(self):
        # Log metadata as pandas dataframe with uid, tier, and elo_rating
        metadata_dict = {uid: {"tier": m.tier, "elo_rating": m.elo_rating} for uid, m in self.metadata.items()}
        metadata_df = pd.DataFrame(metadata_dict).T
        metadata_df = metadata_df.reset_index()
        metadata_df.columns = ["uid", "tier", "elo_rating"]
        bt.logging.info("\n" + metadata_df.to_string(index=True))

    def _report(self):
        r"""
        Reports the metadata of the miners.
        """
        url = f"{self.config.validator.report_url}/api/report"
        signature = f"0x{self.dendrite.keypair.sign(self.message).hex()}"

        headers = {
            "Content-Type": "application/json",
            "message": self.message,
            "ss58_address": self.wallet.hotkey.ss58_address,
            "signature": signature,
        }

        metadata_dict = {k: v.dict() for k, v in self.metadata.items()}
        payload = {
            "metadata": metadata_dict,
        }

        with httpx.Client() as client:
            response = client.post(
                url,
                json=payload,
                headers=headers,
                timeout=32,
            )

        if response.status_code != 200:
            bt.logging.error(
                f"Failed to report metadata to the Validator Server. Response: {response.text}"
            )
        else:
            bt.logging.success("Reported metadata to the Validator Server.")

    def _update_metadata(self):
        r"""
        Updates the metadata of the miners by whitelisted synapse queries.
        It doesn't consume validator's serving counter.
        """
        synapse = Metadata()
        metadata = self.metadata.copy()
        uids = [uid for uid in range(len(self.metagraph.hotkeys))]
        axons = [self.metagraph.axons[uid] for uid in uids]

        responses = self.dendrite.query(
            axons,
            synapse,
            deserialize=False,
            timeout=16,
        )

        for uid, response in zip(uids, responses):
            # Keep track of the current tier
            current_tier = self.metadata[uid].tier if uid in self.metadata else "unknown"
            new_tier = current_tier

            # Update tier based on response
            if response and response.metadata.get("tier") is not None:
                new_tier = response.metadata["tier"]
            
            # Get current or initial ELO rating
            current_elo = self.metadata[uid].elo_rating if uid in self.metadata else constants.INITIAL_ELO_RATING
            
            # Reset ELO rating if tier changed
            if new_tier != current_tier:
                bt.logging.info(f"Tier of uid {uid} changed from {current_tier} to {new_tier}.")
                current_elo = constants.INITIAL_ELO_RATING

            metadata[uid] = MetadataItem(tier=new_tier, elo_rating=current_elo)

        # Update self.metadata with the newly computed metadata
        self.metadata = metadata
        bt.logging.success(f"Updated metadata for {len(uids)} uids.")
        return self.metadata

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
            tier = metadata.tier
            if tier not in constants.TIER_CONFIG:
                continue
            if tier not in tier_group:
                tier_group[tier] = {}
            tier_group[tier][uid] = ServingCounter(rate_limit_per_tier[tier])

        return tier_group
