from datasets import load_dataset, IterableDataset
from pydantic import BaseModel


class ContextItem(BaseModel):
    context: str


def load_fineweb_context_dataset() -> IterableDataset:
    ds = load_dataset("gair-prox/FineWeb-pro", split="train", streaming=True)
    ds = ds.filter(lambda x: 1024 <= len(x["text"]) <= 4096)
    ds = ds.map(lambda x: {"context": x["text"]})
    return ds


def load_fineweb_math_corpus_dataset() -> IterableDataset:
    ds = load_dataset(
        "OpenCoder-LLM/fineweb-math-corpus",
        split="train",
        streaming=True,
    )
    ds = ds.filter(lambda x: 256 <= len(x["text"]) <= 4096)
    ds = ds.map(lambda x: {"context": x["text"]})
    return ds
