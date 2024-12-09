import json
import bittensor as bt
import numpy as np
import httpx
import pandas as pd
import threading
import time
import asyncio
from pydantic import BaseModel
from .metric_converter import MetricConverter
from .elo import ELOSystem
from ...common import build_rate_limit
from ...protocol import Metadata
from ...constants import constants, TierConfig
from ...logger import logger
import redis
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class MinerMetadata(Base):
    """SQLAlchemy model for miner metadata with Pydantic-like validation."""
    __tablename__ = 'miner_metadata'
    
    uid = Column(Integer, primary_key=True)
    tier = Column(String, default="unknown")
    elo_rating = Column(Float, default=constants.INITIAL_ELO_RATING)

    def __init__(self, uid, tier="unknown", elo_rating=constants.INITIAL_ELO_RATING):
        self.uid = uid
        self.tier = tier
        self.elo_rating = elo_rating

    def to_dict(self):
        """Convert to dictionary for easy serialization."""
        return {
            "uid": self.uid,
            "tier": self.tier,
            "elo_rating": self.elo_rating
        }


class ServingCounter:
    """
    A Redis-based counter for rate limiting requests to a miner from this validator.

    Attributes:
        rate_limit (int): The maximum number of requests allowed per epoch
        redis_client (redis.Redis): Redis client for distributed counting
        uid (int): Unique identifier for the miner
        tier (str): Tier of the miner
    """

    def __init__(self, rate_limit: int, uid: int, tier: str, redis_client: redis.Redis):
        """
        Initialize the serving counter.

        Args:
            rate_limit (int): Maximum number of requests allowed per epoch
            uid (int): Unique identifier for the miner
            tier (str): Tier of the miner
            redis_client (redis.Redis): Shared Redis client instance
        """
        self.rate_limit = rate_limit
        self.redis_client = redis_client
        self.key = constants.DATABASE_CONFIG.redis.serving_counter_key_format.format(
            tier=tier,
            uid=uid,
        )
        self.expire_time = constants.DATABASE_CONFIG.redis.expire_time


    def increment(self) -> bool:
        """
        Increments the counter and checks if rate limit is exceeded.
        Uses Redis INCR for atomic increment and EXPIRE for automatic cleanup.

        Returns:
            bool: True if counter is within rate limit, False otherwise
        """
        current_key = self.key
        # Atomic increment operation
        count = self.redis_client.incr(current_key)
        
        # Set expiration for 1 hour if this is the first increment
        if count == 1:
            self.redis_client.expire(current_key, self.expire_time)
        
        return count <= self.rate_limit


class MinerManager:
    """
    Manages metadata and serving counters for miners in the network.

    Attributes:
        wallet: Bittensor wallet for the validator
        dendrite: Bittensor dendrite for network communication
        metagraph: Network metagraph containing miner information
        elo_system (ELOSystem): System for managing ELO ratings
        default_metadata_items (list): Default metadata fields
        config: Validator configuration
        state_path (str): Path to save/load state
        message (str): Random message for signing
        metric_converter (MetricConverter): Converts metrics to scores
        serving_counter (dict): Rate limiting counters per tier
    """

    def __init__(self, uid, wallet, metagraph, config=None):
        """Initialize the MinerManager."""
        self.config = config
        self.uid = uid
        self.wallet = wallet
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = metagraph
        self.elo_system = ELOSystem()
        self.default_metadata_items = [
            ("tier", "unknown"),
        ]
        
        # Initialize SQLAlchemy with configured URL
        self.engine = create_engine(constants.DATABASE_CONFIG.sql.url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        self._init_metadata()
        self.metric_converter = MetricConverter()
        self.rate_limit_per_tier = self.get_rate_limit_per_tier()
        logger.info(f"Rate limit per tier: {self.rate_limit_per_tier}")
        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self.sync())
        
        # Create a single Redis client for all counters
        redis_config = constants.DATABASE_CONFIG.redis
        self.redis_client = redis.Redis(
            host=redis_config.host,
            port=redis_config.port,
            db=redis_config.db
        )

    def get_metadata(self, uids: list[int]) -> dict[int, MinerMetadata]:
        return {miner.uid: miner for miner in self.session.query(MinerMetadata).filter(MinerMetadata.uid.in_(uids)).all()}

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
        initial_ratings = []
        for uid in total_uids:
            miner = self.session.query(MinerMetadata).get(uid)
            initial_ratings.append(miner.elo_rating)

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
            miner = self.session.query(MinerMetadata).get(uid)
            if not miner:
                miner = MinerMetadata(uid=uid)
                self.session.add(miner)
            miner.elo_rating = max(constants.FLOOR_ELO_RATING, final_rating)
        self.session.commit()
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

            miners = self.session.query(MinerMetadata).filter_by(tier=tier).all()
            for miner in miners:
                tier_ratings.append(miner.elo_rating)
                tier_uids.append(miner.uid)
                
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

                thresholded_ratings = np.array(thresholded_ratings)

                # Adjust ratings to match expected mean and standard deviation
                nonzero_mask = thresholded_ratings > 0
                if np.any(nonzero_mask):
                    current_std = np.std(thresholded_ratings[nonzero_mask])
                    current_mean = np.mean(thresholded_ratings[nonzero_mask])

                    if current_std > 0:
                        # Clamp the standard deviation to a maximum value
                        max_allowed_std = constants.EXPECTED_MAX_STD_ELO_RATING
                        target_std = min(current_std, max_allowed_std)
                        scale_factor = target_std / current_std

                        # Center around mean and apply scaling
                        centered_ratings = (
                            thresholded_ratings[nonzero_mask] - current_mean
                        )
                        scaled_ratings = centered_ratings * scale_factor

                        # Apply sigmoid-like compression to reduce extreme values
                        compression_factor = 0.5
                        compressed_ratings = (
                            np.tanh(scaled_ratings * compression_factor) * target_std
                        )

                        # Shift back to target mean
                        thresholded_ratings[nonzero_mask] = (
                            compressed_ratings + constants.EXPECTED_MEAN_ELO_RATING
                        )

                        # Apply floor
                        thresholded_ratings[
                            thresholded_ratings < constants.FLOOR_ELO_RATING
                        ] = constants.FLOOR_ELO_RATING

                        logger.info(
                            "adjust_ratings",
                            tier=tier,
                            mean=current_mean,
                            std=current_std,
                            scale_factor=scale_factor,
                        )

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

    def _init_metadata(self):
        """
        Initialize metadata for all miners in the network.
        """
        for uid in self.metagraph.uids:
            miner = self.session.query(MinerMetadata).get(uid)
            if not miner:
                miner = MinerMetadata(uid=uid)
                self.session.add(miner)
        self.session.commit()

    async def sync(self):
        """
        Synchronize metadata and serving counters for all miners.
        """
        logger.info("Synchronizing metadata and serving counters.")
        self.rate_limit_per_tier = self.get_rate_limit_per_tier()
        logger.info(f"Rate limit per tier: {self.rate_limit_per_tier}")
        await self._update_metadata()
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
            miner.uid: {"tier": miner.tier, "elo_rating": miner.elo_rating}
            for miner in self.session.query(MinerMetadata).all()
        }
        metadata_df = pd.DataFrame(metadata_dict).T
        metadata_df = metadata_df.reset_index()
        metadata_df.columns = ["uid", "tier", "elo_rating"]
        logger.info("Metadata:\n" + metadata_df.to_markdown())

    async def report_metadata(self):
        """
        Report current miner metadata to the validator server.
        """
        metadata_dict = {
            miner.uid: {
                "tier": miner.tier,
                "elo_rating": miner.elo_rating
            }
            for miner in self.session.query(MinerMetadata).all()
        }
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

    async def _update_metadata(self):
        """
        Update metadata for all miners by querying their status.
        Does not consume validator's serving counter.
        """
        synapse = Metadata()
        uids = [uid for uid in range(len(self.metagraph.hotkeys))]
        axons = [self.metagraph.axons[uid] for uid in uids]

        responses = await self.dendrite.forward(
            axons,
            synapse,
            deserialize=False,
            timeout=16,
        )

        for uid, response in zip(uids, responses):
            miner = self.session.query(MinerMetadata).get(uid)
            if not miner:
                miner = MinerMetadata(uid=uid)
                self.session.add(miner)

            # Keep track of the current tier
            current_tier = miner.tier
            new_tier = current_tier

            # Update tier based on response
            if response and response.metadata.get("tier") is not None:
                new_tier = response.metadata["tier"]

            # Get current or initial ELO rating
            current_elo = miner.elo_rating

            # Reset ELO rating if tier changed
            if new_tier != current_tier:
                logger.info(
                    f"Tier of uid {uid} changed from {current_tier} to {new_tier}."
                )
                current_elo = constants.INITIAL_ELO_RATING

            miner.tier = new_tier
            miner.elo_rating = current_elo

        self.session.commit()
        logger.info(f"Updated metadata for {len(uids)} uids.")

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
        for miner in self.session.query(MinerMetadata).all():
            tier = miner.tier
            if tier not in constants.TIER_CONFIG:
                continue
            if tier not in tier_group:
                tier_group[tier] = {}
            tier_group[tier][miner.uid] = ServingCounter(
                self.rate_limit_per_tier[tier], 
                miner.uid, 
                tier,
                self.redis_client
            )
        return tier_group
