from datasets import load_dataset


def load_wikipedia_science_dataset():
    ds = load_dataset(
        "Laz4rz/wikipedia_science_chunked_small_rag_512", streaming=True, split="train"
    )
    ds = ds.shuffle()
    ds = ds.filter(lambda x: len(x["text"]) > 512)
    ds = ds.map(lambda x: {"context": x["text"]})
    print("Loaded wikipedia science dataset")
    return ds
