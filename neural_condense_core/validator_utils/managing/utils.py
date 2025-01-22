import numpy as np
from ...constants import constants
from ...logger import logger


def apply_top_percentage_threshold(
    scores: list[float], uids: list[int], top_percentage: float
) -> np.ndarray:
    """Apply threshold to keep only top percentage of scores."""
    n_top = max(1, int(len(scores) * top_percentage))
    top_miners = sorted(zip(uids, scores), key=lambda x: x[1], reverse=True)[:n_top]
    top_uids = {uid for uid, _ in top_miners}

    return np.array(
        [score if uid in top_uids else 0 for uid, score in zip(uids, scores)]
    )


def standardize_scores(scores: np.ndarray, tier: str) -> np.ndarray:
    """Standardize non-zero scores using mean and clamped standard deviation."""
    nonzero = scores > 0
    if not np.any(nonzero):
        return scores

    curr_std = np.std(scores[nonzero])
    curr_mean = np.mean(scores[nonzero])

    if curr_std > 0:
        target_std = min(curr_std, constants.EXPECTED_MAX_STD_SCORE)
        scale = target_std / curr_std

        centered = scores[nonzero] - curr_mean
        scaled = centered * scale
        compressed = np.tanh(scaled * 0.5) * target_std
        scores[nonzero] = compressed + constants.EXPECTED_MEAN_SCORE

        logger.info(
            "adjust_ratings",
            tier=tier,
            mean=curr_mean,
            std=curr_std,
            scale_factor=scale,
        )

    return scores


def normalize_and_weight_scores(scores: np.ndarray, tier: str) -> np.ndarray:
    """Normalize scores to sum to 1 and apply tier incentive weighting."""
    total = np.sum(scores)
    if total > 0:
        scores = scores / total

    # --Smoothing Update---
    from datetime import datetime, timezone

    current_datetime = datetime.now(timezone.utc)
    target_datetime = datetime(2025, 1, 24, 12, 0, 0, tzinfo=timezone.utc)

    if current_datetime < target_datetime:
        logger.info("Using early incentive scaling")
        if tier == "research":
            scale = 0.9
        elif tier == "universal":
            scale = 0.1
    else:
        logger.info("Using stable incentive scaling")
        scale = constants.TIER_CONFIG[tier].incentive_percentage

    return scores * scale
