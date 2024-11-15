import numpy as np
import bittensor as bt

class MetricConverter:
    def __init__(self):
        self.converters = {
            "loss": self.loss_to_score,
            "accuracy": self.accuracy_to_score,
        }

    def convert_metrics_to_score(self, metrics: dict) -> dict[str, list[float]]:
        total_scores = {}
        for metric, values in metrics.items():
            try:
                converter = self.converters[metric]
                scores = converter(values)
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
