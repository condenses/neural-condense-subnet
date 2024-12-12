from typing import List
from .context_loader import load_wikipedia_science_dataset
from .infinity_iterable_dataset import InfiniteDataset


def load_context_datasets() -> List[InfiniteDataset]:
    return [
        InfiniteDataset(load_wikipedia_science_dataset().shuffle(seed=42)),
    ]
