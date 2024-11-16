from typing import List
import math
import numpy as np


class ELOSystem:
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A when facing player B."""
        return 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))

    def update_ratings(
        self, ratings: List[float], scores: List[float], k_factor: int
    ) -> List[float]:
        """
        Update ELO ratings for a batch of miners based on their performance scores.

        Args:
            ratings: Current ELO ratings for each miner
            scores: Performance scores from 0 to 1 for each miner

        Returns:
            List of updated ELO ratings
        """
        n = len(ratings)
        new_ratings = ratings.copy()

        # Compare each miner against every other miner
        for i in range(n):
            for j in range(i + 1, n):
                score_i = scores[i] + 1e-6
                score_j = scores[j] + 1e-6
                S_i = score_i / (score_i + score_j)
                S_j = score_j / (score_i + score_j)
                expected_i = self.expected_score(ratings[i], ratings[j])
                expected_j = self.expected_score(ratings[j], ratings[i])
                rating_change_i = k_factor * (S_i - expected_i)
                rating_change_j = k_factor * (S_j - expected_j)
                new_ratings[i] += rating_change_i
                new_ratings[j] += rating_change_j

        return new_ratings

    def normalize_ratings(
        self, ratings: List[float], min_val: float = 0, max_val: float = 1
    ) -> List[float]:
        """Normalize ratings to sum to 1 for weight setting using numpy."""
        if len(ratings) == 0:
            return []

        ratings_array = np.array(ratings)
        if np.all(ratings_array == ratings_array[0]):
            # If all ratings are equal, return equal weights that sum to 1
            return (np.ones(len(ratings)) / len(ratings)).tolist()

        # Normalize to sum to 1
        return (ratings_array / np.sum(ratings_array)).tolist()
