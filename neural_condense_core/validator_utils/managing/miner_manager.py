import json
import bittensor as bt
import numpy as np
import httpx
import pandas as pd
import threading
import time
from pydantic import BaseModel
from .metric_converter import MetricConverter
from .elo import ELOSystem
from ...common import build_rate_limit
from ...protocol import Metadata
from ...constants import constants, TierConfig
from ...logger import logger


class MetadataItem(BaseModel):
    """
    Represents metadata for a miner including their tier and ELO rating.

    Attributes:
        tier (str): The tier level of the miner, defaults to "unknown"
        elo_rating (float): The ELO rating score of the miner, defaults to initial rating from constants
    """

    tier: str = "unknown"
    elo_rating: float = constants.INITIAL_ELO_RATING


class ServingCounter:
    """
    A counter for rate limiting requests to a miner from this validator.

    Attributes:
        rate_limit (int): The maximum number of requests allowed per epoch
        counter (int): The current number of requests made in the current epoch
        lock (threading.Lock): Thread lock for synchronizing counter access
    """

    def __init__(self, rate_limit: int):
        """
        Initialize the serving counter.

        Args:
            rate_limit (int): Maximum number of requests allowed per epoch
        """
        self.rate_limit = rate_limit
        self.counter = 0
        self.lock = threading.Lock()

    def increment(self) -> bool:
        """
        Increments the counter and checks if rate limit is exceeded.

        Returns:
            bool: True if counter is within rate limit, False otherwise
        """
        with self.lock:
            self.counter += 1
            return self.counter <= self.rate_limit


class MinerManager:
    """
    Manages metadata and serving counters for miners in the network.

    Attributes:
        validator: The validator instance this manager belongs to
        wallet: Bittensor wallet for the validator
        dendrite: Bittensor dendrite for network communication
        metagraph: Network metagraph containing miner information
        elo_system (ELOSystem): System for managing ELO ratings
        default_metadata_items (list): Default metadata fields
        config: Validator configuration
        metadata (dict): Miner metadata storage
        state_path (str): Path to save/load state
        message (str): Random message for signing
        metric_converter (MetricConverter): Converts metrics to scores
        serving_counter (dict): Rate limiting counters per tier
    """

    def __init__(self, validator):
        """
        Initialize the MinerManager.

        Args:
            validator: The validator instance this manager belongs to
        """
        self.config = validator.config
        self.uid = validator.uid
        self.wallet = validator.wallet
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = validator.metagraph
        self.elo_system = ELOSystem()
        self.default_metadata_items = [
            ("tier", "unknown"),
        ]
        self.config = validator.config
        self.metadata = self._init_metadata()
        self.state_path = self.config.full_path + "/state.json"
        self.metric_converter = MetricConverter()
        self.rate_limit_per_tier = self.get_rate_limit_per_tier()
        logger.info(f"Rate limit per tier: {self.rate_limit_per_tier}")
        self.load_state()
        self.sync()

    def update_ratings(
        self,
        metrics: dict[str, list[float]],
        k_factor: int,
        total_uids: list[int],
        tier_config: TierConfig,
    ):
        """
        Updates the ELO ratings of miners based on their performance.

        Args:
            metrics (dict[str, list[float]]): Performance metrics for each miner
            total_uids (list[int]): UIDs of all miners
            k_factor (int): ELO K-factor for rating adjustments
            tier_config (TierConfig): Tier configuration
        """
        # Get current ELO ratings for participating miners
        initial_ratings = [self.metadata[uid].elo_rating for uid in total_uids]
        performance_scores: dict[str, list[float]] = (
            self.metric_converter.convert_metrics_to_score(metrics, tier_config)
        )
        # Update ELO ratings based on performance scores
        metric_ratings = []
        for metric, scores in performance_scores.items():
            updated_ratings = self.elo_system.update_ratings(
                initial_ratings, scores, k_factor
            )
            metric_ratings.append(updated_ratings)

        final_ratings = np.mean(metric_ratings, axis=0)
        # Update metadata with new ratings and scores
        for uid, final_rating in zip(total_uids, final_ratings):
            self.metadata[uid] = MetadataItem(
                tier=self.metadata[uid].tier,
                elo_rating=max(constants.FLOOR_ELO_RATING, final_rating),
            )
        return final_ratings, initial_ratings

    def get_normalized_ratings(self, top_percentage: float = 1.0) -> np.ndarray:
        """
        Calculate normalized ratings for all miners based on their tier and ELO rating.

        Args:
            top_percentage (float): Percentage of miners to consider for normalization

        Returns:
            np.ndarray: Array of normalized ratings for all miners
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
            uids_ratings = list(zip(tier_uids, tier_ratings))
            if uids_ratings:
                # Give zeros to rating of miners not in top_percentage
                n_top_miners = max(1, int(len(tier_ratings) * top_percentage))
                top_miners = sorted(uids_ratings, key=lambda x: x[1], reverse=True)[
                    :n_top_miners
                ]
                top_uids, _ = zip(*top_miners)
                thresholded_ratings = tier_ratings.copy()
                for i in range(len(tier_ratings)):
                    if tier_uids[i] not in top_uids:
                        thresholded_ratings[i] = 0

                data = {
                    "uids": tier_uids,
                    "original_ratings": tier_ratings,
                    "thresholded_ratings": thresholded_ratings,
                }
                logger.info(
                    f"Thresholded Ratings for Tier {tier} (thresholded by {top_percentage}) :\n{pd.DataFrame(data).to_markdown()}"
                )
                # Normalize ELO ratings to weights, sum to 1
                normalized_ratings = self.elo_system.normalize_ratings(
                    thresholded_ratings
                )

                # Apply tier incentive percentage
                tier_weights = (
                    np.array(normalized_ratings)
                    * constants.TIER_CONFIG[tier].incentive_percentage
                )

                # Assign weights to corresponding UIDs
                for uid, weight in zip(tier_uids, tier_weights):
                    weights[uid] = weight

        return weights

    def load_state(self):
        """
        Load miner metadata state from disk.
        """
        try:
            state = json.load(open(self.state_path, "r"))
            metadata_items = {
                int(k): MetadataItem(**v) for k, v in state["metadata"].items()
            }
            self.metadata = metadata_items
            self._log_metadata()
            logger.info("Loaded state.")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

    def save_state(self):
        """
        Save current miner metadata state to disk.
        """
        try:
            metadata_dict = {k: v.dict() for k, v in self.metadata.items()}
            state = {"metadata": metadata_dict}
            json.dump(state, open(self.state_path, "w"))
            logger.info("Saved state.")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _init_metadata(self):
        """
        Initialize metadata for all miners in the network.

        Returns:
            dict: Initial metadata for all miners
        """
        metadata = {int(uid): MetadataItem() for uid in self.metagraph.uids}
        return metadata

    def sync(self):
        """
        Synchronize metadata and serving counters for all miners.
        """
        logger.info("Synchronizing metadata and serving counters.")
        self.rate_limit_per_tier = self.get_rate_limit_per_tier()
        logger.info(f"Rate limit per tier: {self.rate_limit_per_tier}")
        self.metadata = self._update_metadata()
        self.serving_counter: dict[str, dict[int, ServingCounter]] = (
            self._create_serving_counter()
        )
        self._log_metadata()

    def _log_metadata(self):
        """
        Log current miner metadata as a formatted pandas DataFrame.
        """
        # Log metadata as pandas dataframe with uid, tier, and elo_rating
        metadata_dict = {
            uid: {"tier": m.tier, "elo_rating": m.elo_rating}
            for uid, m in self.metadata.items()
        }
        metadata_df = pd.DataFrame(metadata_dict).T
        metadata_df = metadata_df.reset_index()
        metadata_df.columns = ["uid", "tier", "elo_rating"]
        logger.info("Metadata:\n" + metadata_df.to_markdown())

    async def report_metadata(self):
        """
        Report current miner metadata to the validator server.
        """
        metadata_dict = {k: v.dict() for k, v in self.metadata.items()}
        await self.report(metadata_dict, "api/report-metadata")

    async def report(self, payload: dict, endpoint: str):
        """
        Report current miner metadata to the validator server.
        """
        url = f"{self.config.validator.report_url}/{endpoint}"
        nonce = str(time.time_ns())
        signature = f"0x{self.dendrite.keypair.sign(nonce).hex()}"

        headers = {
            "Content-Type": "application/json",
            "message": nonce,
            "ss58_address": self.wallet.hotkey.ss58_address,
            "signature": signature,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload,
                headers=headers,
                timeout=32,
            )

        if response.status_code != 200:
            logger.error(
                f"Failed to report to the {endpoint}. Response: {response.text}"
            )
        else:
            logger.info(f"Reported to the {endpoint}.")

    def _update_metadata(self):
        """
        Update metadata for all miners by querying their status.
        Does not consume validator's serving counter.

        Returns:
            dict: Updated metadata for all miners
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
            current_tier = (
                self.metadata[uid].tier if uid in self.metadata else "unknown"
            )
            new_tier = current_tier

            # Update tier based on response
            if response and response.metadata.get("tier") is not None:
                new_tier = response.metadata["tier"]

            # Get current or initial ELO rating
            current_elo = (
                self.metadata[uid].elo_rating
                if uid in self.metadata
                else constants.INITIAL_ELO_RATING
            )

            # Reset ELO rating if tier changed
            if new_tier != current_tier:
                logger.info(
                    f"Tier of uid {uid} changed from {current_tier} to {new_tier}."
                )
                current_elo = constants.INITIAL_ELO_RATING

            metadata[uid] = MetadataItem(tier=new_tier, elo_rating=current_elo)

        # Update self.metadata with the newly computed metadata
        self.metadata = metadata
        logger.info(f"Updated metadata for {len(uids)} uids.")
        return self.metadata

    def get_rate_limit_per_tier(self):
        """
        Get rate limit per tier for the validator.
        """
        rate_limit_per_tier = {
            tier: build_rate_limit(self.metagraph, self.config, tier)[self.uid]
            for tier in constants.TIER_CONFIG.keys()
        }
        return rate_limit_per_tier

    def _create_serving_counter(self):
        """
        Create rate limiting counters for each tier of miners.

        Returns:
            dict: Serving counters organized by tier and UID
        """
        tier_group = {tier: {} for tier in constants.TIER_CONFIG.keys()}
        for uid, metadata in self.metadata.items():
            tier = metadata.tier
            if tier not in constants.TIER_CONFIG:
                continue
            if tier not in tier_group:
                tier_group[tier] = {}
            tier_group[tier][uid] = ServingCounter(self.rate_limit_per_tier[tier])
        return tier_group
