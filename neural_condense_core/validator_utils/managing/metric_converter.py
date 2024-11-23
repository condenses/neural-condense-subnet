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
                scores = [
                    s * (1 + a) if a is not None else s
                    for s, a in zip(scores, accelerate_bonuses)
                ]
                total_scores[metric] = scores
            except KeyError:
                continue
        return total_scores

    def perplexity_to_score(self, perplexities: list[float]):
        valid_perplexities = [p for p in perplexities if p is not None]
        if not valid_perplexities:
            return perplexities
        pivot = min(valid_perplexities)
        scores = [pivot / p if p is not None else None for p in perplexities]
        return scores

    def accuracy_to_score(self, accuracies: list[float]):
        return accuracies

    def get_accelerate_bonuses(self, metrics: dict, tier_config: TierConfig):
        accelerate_metrics = metrics["accelerate_metrics"]
        return [
            s * tier_config.accelerate_reward_scalar if s is not None else None
            for s in accelerate_metrics
        ]
