import bittensor as bt
import wandb
import pandas as pd


def log_wandb(logs: dict, uids: list[int], tier=""):
    try:
        for metric, values in logs.items():
            if metric == "perplexity":
                for uid, value in zip(uids, values):
                    if value is None:
                        continue
                    wandb.log({f"{tier}-{uid}/perplexity": abs(value)})
    except Exception as e:
        bt.logging.error(f"Error logging to wandb: {e}")


def log_as_dataframe(data: dict, name: str):
    for metric, values in data.items():
        for i in range(len(values)):
            if values[i] is None:
                values[i] = "N/A"
            if isinstance(values[i], float):
                values[i] = round(values[i], 2)
    df = pd.DataFrame(data)
    bt.logging.info(f"Logging dataframe {name}:\n{df}")
    return df
