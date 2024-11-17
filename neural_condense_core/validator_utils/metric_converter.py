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
                continue
        return total_scores

    def loss_to_score(self, losses: list[float]):
        for i in range(len(losses)):
            if losses[i] is None:
                losses[i] = 1000
        pivot = max(losses)
        scores = pivot / np.array(losses)
        return scores.tolist()

    def accuracy_to_score(self, accuracies: list[float]):
        for i in range(len(accuracies)):
            if accuracies[i] is None:
                accuracies[i] = 0
        return accuracies

    def get_accelerate_bonuses(self, metrics: dict, tier_config: TierConfig):
        accelerate_metrics = metrics["accelerate_metrics"]
        for i in range(len(accelerate_metrics)):
            if accelerate_metrics[i] is None:
                accelerate_metrics[i] = 0
        return [s * tier_config.accelerate_reward_scalar for s in accelerate_metrics]
