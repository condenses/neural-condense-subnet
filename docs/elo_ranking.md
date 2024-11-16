## ELO Rating System

The validator uses an ELO rating system to track and evaluate miner performance over time. Similar to chess rankings, this system provides a dynamic way to rank miners based on their relative performance.

### How It Works

1. **Initial Rating**
   - All miners start with a base ELO rating (default: 1200)
   - Ratings have a floor value to prevent excessive penalties

2. **Performance Evaluation**
   - Miners are evaluated in batches against each other
   - Performance metrics are converted to normalized scores (0-1)
   - Each miner's score is compared against others in pairwise matchups

3. **Rating Updates**
   - Ratings are updated based on:
     - Expected performance (based on current ratings)
     - Actual performance (based on quality metrics)
     - K-factor (determines how quickly ratings can change)
   - The formula used is: `new_rating = old_rating + K * (actual_score - expected_score)`

4. **Tier System**
   - Miners are grouped into tiers (e.g., basic, standard, premium)
   - ELO ratings are maintained separately within each tier
   - When a miner changes tiers, their ELO rating resets to the initial value

5. **Weight Assignment**
   - Final weights for reward distribution are calculated by:
     - Normalizing ELO ratings within each tier
     - Applying tier-specific incentive percentages
     - Ensuring weights sum to 1 across all miners

### Example

```python
# Sample ELO calculation for two miners
miner_a_rating = 1200  # Current rating
miner_b_rating = 1300
k_factor = 32

# Calculate expected score for miner A
expected_score = 1 / (1 + 10**((miner_b_rating - miner_a_rating) / 400))

# If miner A performs better than expected (actual_score > expected_score)
actual_score = 0.7
rating_change = k_factor * (actual_score - expected_score)
new_rating = miner_a_rating + rating_change
```

### Benefits

- **Dynamic Rankings**: Ratings automatically adjust based on consistent performance
- **Fair Competition**: Miners compete primarily against others in their tier
- **Performance History**: Ratings provide a memory of past performance
- **Balanced Rewards**: Higher-rated miners receive proportionally higher weights
