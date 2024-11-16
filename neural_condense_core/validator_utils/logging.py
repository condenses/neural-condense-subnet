import bittensor as bt
import wandb
import pandas as pd


def log_wandb(logs: dict, uids: list[int], tier=""):
    try:
        for metric, values in logs.items():
            if metric == "accuracy":
                pass
            if metric == "losses":
                for uid, value in zip(uids, values):
                    wandb.log({f"{tier}-{uid}/losses": abs(value)})
            if metric == "penalized_scores":
                for uid, value in zip(uids, values):
                    wandb.log({f"{tier}-{uid}/penalized_scores": value})
    except Exception as e:
        bt.logging.error(f"Error logging to wandb: {e}")


def log_as_dataframe(data: dict, name: str):
    df = pd.DataFrame(data)
    bt.logging.info(f"Logging dataframe {name}:\n{df}")
