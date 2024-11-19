import numpy as np
from ...constants import TierConfig


class MetricConverter:
    def __init__(self):
        self.converters = {
            "perplexity": self.perplexity_to_score,
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

    def perplexity_to_score(self, perplexities: list[float]):
        for i in range(len(perplexities)):
            if perplexities[i] is None:
                perplexities[i] = 1000
        pivot = min(perplexities)
        scores = pivot / np.array(perplexities)
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
