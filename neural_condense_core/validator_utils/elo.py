from typing import List, Tuple
import math
import numpy as np
class ELOSystem:
    def __init__(self, k_factor: float = 32, initial_rating: float = 1400):
        """
        Initialize the ELO rating system.
        
        Args:
            k_factor: The maximum rating change possible (default: 32)
            initial_rating: Starting rating for new miners (default: 1400)
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A when facing player B."""
        return 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))

    def update_ratings(self, ratings: List[float], scores: List[float]) -> List[float]:
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
                # Calculate score difference between miners
                score_diff = scores[i] - scores[j]
                
                # Convert to win/loss/draw
                if abs(score_diff) < 0.001:  # Draw threshold
                    actual_score = 0.5
                else:
                    actual_score = 1 if score_diff > 0 else 0

                # Calculate expected scores
                expected_i = self.expected_score(ratings[i], ratings[j])
                
                # Update ratings
                rating_change = self.k_factor * (actual_score - expected_i)
                new_ratings[i] += rating_change
                new_ratings[j] -= rating_change

        return new_ratings

    def normalize_ratings(self, ratings: List[float], min_val: float = 0, max_val: float = 1) -> List[float]:
        """Normalize ratings to sum to 1 for weight setting using numpy."""
        if len(ratings) == 0:
            return []
            
        ratings_array = np.array(ratings)
        if np.all(ratings_array == ratings_array[0]):
            # If all ratings are equal, return equal weights that sum to 1
            return (np.ones(len(ratings)) / len(ratings)).tolist()
        
        # Normalize to sum to 1
        return (ratings_array / np.sum(ratings_array)).tolist()
