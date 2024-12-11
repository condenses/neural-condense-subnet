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

from datasets import load_dataset


def load_instruct_datasets() -> List[InfiniteDataset]:
    return [
        InfiniteDataset(load_orca_instruct_dataset().shuffle(seed=42)),
        InfiniteDataset(load_open_math_instruct_dataset().shuffle(seed=42)),
    ]


def load_wikipedia_science_dataset():
    ds = load_dataset(
        "Laz4rz/wikipedia_science_chunked_small_rag_512", streaming=True, split="train"
    )
    ds = ds.shuffle()
    ds = ds.filter(lambda x: len(x["text"]) > 512)
    ds = ds.map(lambda x: {"context": x["text"]})
    print("Loaded wikipedia science dataset")
    return ds


def load_context_datasets() -> List[InfiniteDataset]:
    return [
        # InfiniteDataset(load_fineweb_context_dataset().shuffle(seed=42)),
        # InfiniteDataset(load_fineweb_math_corpus_dataset().shuffle(seed=42)),
        InfiniteDataset(load_wikipedia_science_dataset().shuffle(seed=42)),
    ]
