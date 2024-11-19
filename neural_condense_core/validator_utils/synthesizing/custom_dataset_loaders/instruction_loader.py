from datasets import load_dataset, IterableDataset


def load_orca_instruct_dataset() -> IterableDataset:
    ds = load_dataset(
        "mlabonne/orca-agentinstruct-1M-v1-cleaned", split="train", streaming=True
    )
    ds = ds.map(
        lambda x: {
            "messages": x["messages"],
        }
    )
    return ds


def load_open_math_instruct_dataset() -> IterableDataset:
    ds = load_dataset("nvidia/OpenMathInstruct-2", split="train", streaming=True)
    ds = ds.map(
        lambda x: {
            "messages": [
                {"role": "user", "content": x["problem"]},
                {"role": "assistant", "content": x["generated_solution"]},
            ]
        }
    )
    return ds
