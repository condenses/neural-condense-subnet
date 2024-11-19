from typing import List
from .context_loader import (
    load_fineweb_context_dataset,
    load_fineweb_math_corpus_dataset,
)
from .instruction_loader import (
    load_orca_instruct_dataset,
    load_open_math_instruct_dataset,
)
from .infinity_iterable_dataset import InfiniteDataset


def load_instruct_datasets() -> List[InfiniteDataset]:
    return [
        InfiniteDataset(load_orca_instruct_dataset().shuffle(seed=42)),
        InfiniteDataset(load_open_math_instruct_dataset().shuffle(seed=42)),
    ]


def load_context_datasets() -> List[InfiniteDataset]:
    return [
        InfiniteDataset(load_fineweb_context_dataset().shuffle(seed=42)),
        InfiniteDataset(load_fineweb_math_corpus_dataset().shuffle(seed=42)),
    ]
