import wandb
import pandas as pd
from ...logger import logger


def log_wandb(logs: dict, uids: list[int], tier=""):
    try:
        for metric, values in logs.items():
            if metric == "perplexity":
                for uid, value in zip(uids, values):
                    if value is None or not isinstance(value, float):
                        continue
                    wandb.log({f"{tier}-{uid}/perplexity": abs(value)})
    except Exception as e:
        logger.error(f"Error logging to wandb: {e}")


def log_as_dataframe(data: dict):
    for metric, values in data.items():
        for i in range(len(values)):
            if values[i] is None:
                values[i] = "N/A"
            if isinstance(values[i], float):
                values[i] = round(values[i], 2)
    df = pd.DataFrame(data)
    return df
