import bittensor as bt
import numpy as np
import httpx
import pandas as pd
import time
import asyncio
from .metric_converter import MetricConverter
from ...common import build_rate_limit
from ...protocol import Metadata
from ...constants import constants
from .utils import (
    apply_top_percentage_threshold,
    standardize_scores,
    normalize_and_weight_scores,
)
from ...logger import logger
import redis
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class MinerMetadata(Base):
    """
    SQLAlchemy model for storing miner metadata.

    Attributes:
        uid (int): Unique identifier for the miner
        tier (str): Miner's tier level (default: "unknown")
        score (float): Miner's performance score (default: 0.0)
    """

    __tablename__ = "miner_metadata"

    uid = Column(Integer, primary_key=True)
    tier = Column(String, default="unknown")
    score = Column(Float, default=0.0)

    def __init__(self, uid, tier="unknown", score=0.0):
        self.uid = uid
        self.tier = tier
        self.score = score

    def to_dict(self):
        """Convert metadata to dictionary format."""
        return {"uid": self.uid, "tier": self.tier, "score": self.score}


class ServingCounter:
    """
    Redis-based rate limiter for miner requests.

    Uses atomic Redis operations to track and limit request rates per miner.

    Attributes:
        rate_limit (int): Max requests allowed per epoch
        redis_client (redis.Redis): Redis connection for distributed counting
        key (str): Unique Redis key for this counter
        expire_time (int): TTL for counter keys in Redis
    """

    def __init__(
        self,
        rate_limit: int,
        uid: int,
        tier: str,
        redis_client: redis.Redis,
        postfix_key: str = "",
    ):
        self.rate_limit = rate_limit
        self.redis_client = redis_client
        self.key = constants.DATABASE_CONFIG.redis.serving_counter_key_format.format(
            tier=tier,
            uid=uid,
        ) + str(postfix_key)

    def increment(self) -> bool:
        """
        Increment request counter and check rate limit.

        Uses atomic Redis INCR operation and sets expiry on first increment.

        Reset the counter after EPOCH_LENGTH seconds.

        Returns:
            bool: True if under rate limit, False if exceeded
        """
        count = self.redis_client.incr(self.key)

        if count == 1:
            self.redis_client.expire(self.key, constants.EPOCH_LENGTH)

        if count <= self.rate_limit:
            return True

        logger.info(f"Rate limit exceeded for {self.key}")
        return False

    def get_current_count(self):
        return self.redis_client.get(self.key)

    def reset_counter(self):
        self.redis_client.set(self.key, 0)


class MinerManager:
    """
    Manages miner metadata, scoring and rate limiting.

    Handles:
    - Miner metadata storage and updates
    - Performance scoring and normalization
    - Request rate limiting per miner
    - Synchronization with validator network

    Attributes:
        wallet: Bittensor wallet
        dendrite: Network communication client
        metagraph: Network state/topology
        config: Validator configuration
        redis_client: Redis connection
        session: SQLAlchemy database session
        metric_converter: Converts raw metrics to scores
        rate_limit_per_tier: Request limits by tier
    """

    def __init__(self, uid, wallet, metagraph, config=None):
        self.is_main_process = bool(config)
        self.config = config
        self.uid = uid
        self.wallet = wallet
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = metagraph

        # Initialize Redis
        redis_config = constants.DATABASE_CONFIG.redis
        self.redis_client = redis.Redis(
            host=redis_config.host, port=redis_config.port, db=redis_config.db
        )

        # Initialize SQLAlchemy
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

    def get_metadata(self, uids: list[int] = []) -> dict[int, MinerMetadata]:
        """
        Get metadata for specified miner UIDs.

        Args:
            uids: List of miner UIDs to fetch. Empty list returns all miners.

        Returns:
            dict: Mapping of UID to miner metadata
        """
        query = self.session.query(MinerMetadata)
        if uids:
            query = query.filter(MinerMetadata.uid.in_(uids))
        return {miner.uid: miner for miner in query.all()}

    def update_scores(self, scores: list[float], total_uids: list[int]):
        """
        Update miner scores with exponential moving average.

        Args:
            scores: New performance scores
            total_uids: UIDs corresponding to scores

        Returns:
            tuple: Updated scores and previous scores
        """
        updated_scores = []
        previous_scores = []

        for uid, score in zip(total_uids, scores):
            miner = self.session.query(MinerMetadata).get(uid)
            previous_scores.append(miner.score)

            # EMA with 0.9 decay factor
            miner.score = miner.score * 0.9 + score * 0.1
            miner.score = max(0, miner.score)
            updated_scores.append(miner.score)

        self.session.commit()
        return updated_scores, previous_scores

    def get_normalized_ratings(self, top_percentage: float = 1.0) -> np.ndarray:
        """
        Calculate normalized ratings across all miners.

        Applies:
        1. Top percentage thresholding
        2. Score standardization
        3. Sigmoid compression
        4. Tier-based incentive weighting

        Args:
            top_percentage: Fraction of top miners to consider

        Returns:
            np.ndarray: Normalized ratings for all miners
        """
        weights = np.zeros(len(self.metagraph.hotkeys))

        for tier in constants.TIER_CONFIG:
            tier_weights = self._get_tier_weights(tier, top_percentage)
            for uid, weight in tier_weights.items():
                weights[uid] = weight

        return weights

    def _get_tier_weights(self, tier: str, top_percentage: float) -> dict[int, float]:
        """
        Calculate weights for miners in a specific tier.

        Args:
            tier: The tier to calculate weights for
            top_percentage: Fraction of top miners to consider

        Returns:
            dict: Mapping of UID to weight for miners in tier
        """
        # Get scores for miners in this tier
        miners = self.session.query(MinerMetadata).filter_by(tier=tier).all()
        tier_scores = [m.score for m in miners]
        tier_uids = [m.uid for m in miners]

        if not tier_scores:
            return {}

        scores = apply_top_percentage_threshold(tier_scores, tier_uids, top_percentage)
        scores = normalize_and_weight_scores(scores, tier)

        return dict(zip(tier_uids, scores))

    def _init_metadata(self):
        """Initialize metadata entries for all miners."""
        for uid in self.metagraph.uids:
            try:
                self.session.query(MinerMetadata).get(uid)
            except Exception as e:
                logger.info(f"Reinitialize uid {uid}, {e}")
                self.session.add(MinerMetadata(uid=uid))
        self.session.commit()

    async def sync(self):
        """Synchronize metadata and rate limiters."""
        logger.info("Synchronizing metadata and serving counters.")
        self.rate_limit_per_tier = self.get_rate_limit_per_tier()
        logger.info(f"Rate limit per tier: {self.rate_limit_per_tier}")

        self.serving_counter = self._create_serving_counter()

        if self.is_main_process:
            await self._update_metadata()
            self._log_metadata()

    def _log_metadata(self):
        """Log current metadata as formatted DataFrame."""
        metadata = {
            m.uid: {"tier": m.tier, "elo_rating": m.score * 100}
            for m in self.session.query(MinerMetadata).all()
        }
        df = pd.DataFrame(metadata).T.reset_index()
        df.columns = ["uid", "tier", "elo_rating"]
        logger.info("Metadata:\n" + df.to_markdown())

    async def report_metadata(self):
        """Report metadata to validator server."""
        metadata = {
            m.uid: {"tier": m.tier, "elo_rating": m.score * 100}
            for m in self.session.query(MinerMetadata).all()
        }
        await self.report(metadata, "api/report-metadata")

    async def report(self, payload: dict, endpoint: str):
        """
        Send signed report to validator server.

        Args:
            payload: Data to report
            endpoint: API endpoint path
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
            logger.error(f"Failed to report to {endpoint}. Response: {response.text}")
        else:
            logger.info(f"Reported to {endpoint}.")

    async def _update_metadata(self):
        """Update metadata by querying miner status."""
        synapse = Metadata()
        uids = list(range(len(self.metagraph.hotkeys)))
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

            current_tier = miner.tier
            new_tier = current_tier

            if response and response.metadata.get("tier") is not None:
                new_tier = response.metadata["tier"]

            if new_tier != current_tier:
                logger.info(
                    f"Tier of uid {uid} changed from {current_tier} to {new_tier}"
                )
                miner.score = 0

            miner.tier = new_tier

        self.session.commit()
        logger.info(f"Updated metadata for {len(uids)} uids")

    def get_rate_limit_per_tier(self):
        """Get request rate limits for each tier."""
        return {
            tier: build_rate_limit(self.metagraph, self.config, tier)[self.uid]
            for tier in constants.TIER_CONFIG
        }

    def _create_serving_counter(self):
        """
        Create rate limiters for each miner by tier.

        Returns:
            dict: Nested dict of tier -> uid -> counter
        """
        counters = {tier: {} for tier in constants.TIER_CONFIG}

        for miner in self.session.query(MinerMetadata).all():
            tier = miner.tier
            if tier not in constants.TIER_CONFIG:
                continue

            counter = ServingCounter(
                self.rate_limit_per_tier[tier], miner.uid, tier, self.redis_client
            )
            counters[tier][miner.uid] = counter

        return counters
