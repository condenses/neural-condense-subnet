import numpy as np
import bittensor as bt
from ..constants import TierConfig


class MetricConverter:
    def __init__(self):
        self.converters = {
            "loss": self.loss_to_score,
            "accuracy": self.accuracy_to_score,
        }

    def convert_metrics_to_score(
        self, metrics: dict, tier_config: TierConfig
    ) -> dict[str, list[float]]:
        total_scores = {}
        accelerate_bonuses = self.get_accelerate_bonuses(metrics, tier_config)
        for metric, values in metrics.items():
            try:
                converter = self.converters[metric]
                scores = converter(values)
                scores = [s * (1 + a) for s, a in zip(scores, accelerate_bonuses)]
                total_scores[metric] = scores
            except KeyError:
                bt.logging.error(f"Unknown metric: {metric}")
        return total_scores

    def loss_to_score(self, losses: list[float]):
        pivot = max(losses)
        scores = pivot / np.array(losses)
        return scores.tolist()

    def accuracy_to_score(self, accuracies: list[float]):
        return accuracies

    def get_accelerate_bonuses(self, metrics: dict, tier_config: TierConfig):
        accelerate_metrics = metrics["accelerate_metrics"]
        return [s * tier_config.accelerate_reward_scalar for s in accelerate_metrics]
