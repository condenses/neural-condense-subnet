from pydantic import BaseModel
from typing import Callable, List
import os


class TierConfig(BaseModel):
    incentive_percentage: float
    requests_per_epoch: int
    timeout: int
    scoring_lambda: Callable[[dict], float]
    supporting_models: List[str]
    max_condensed_tokens: int
    min_condensed_tokens: int
    max_context_length_in_chars: int


class SyntheticTaskConfig(BaseModel):
    task: str
    criterias: List[str]
    rewarding_frequency: int
    weight: float


class Constants(BaseModel):
    TIER_CONFIG: dict[str, TierConfig] = {
        "research": TierConfig(
            incentive_percentage=1.0,
            requests_per_epoch=256,
            timeout=32,
            scoring_lambda=lambda x: x["normalized_score_in_batch"]
            - 0.1 * x["process_time/timeout"]
            + 0.1 * x["compress_rate_reward"],
            supporting_models=["Condense-AI/Mistral-7B-Instruct-v0.2"],
            max_condensed_tokens=1024,
            min_condensed_tokens=128,
            max_context_length_in_chars=10000,
        ),
        "inference_0": TierConfig(
            incentive_percentage=0.0,
            requests_per_epoch=1024,
            timeout=8,
            scoring_lambda=lambda x: max(
                0,
                x["normalized_score_in_batch"]
                - 0.1 * x["process_time/timeout"]
                + 0.1 * x["compress_rate_reward"],
            ),
            supporting_models=["Condense-AI/Mistral-7B-Instruct-v0.2"],
            max_condensed_tokens=1024,
            min_condensed_tokens=128,
            max_context_length_in_chars=15000,
        ),
        "inference_1": TierConfig(
            incentive_percentage=0.0,
            requests_per_epoch=1024,
            timeout=8,
            scoring_lambda=lambda x: max(
                0,
                x["normalized_score_in_batch"]
                - 0.1 * x["process_time/timeout"]
                + 0.1 * x["compress_rate_reward"],
            ),
            supporting_models=["Condense-AI/Mistral-7B-Instruct-v0.2"],
            max_condensed_tokens=2048,
            min_condensed_tokens=128,
            max_context_length_in_chars=20000,
        ),
    }

    SYNTHETIC_TASK_CONFIG: List[SyntheticTaskConfig] = [
        SyntheticTaskConfig(
            task="reconstruction",
            criterias=["loss"],
            rewarding_frequency=1,
            weight=0.8,
        ),
        SyntheticTaskConfig(
            task="question_answering",
            criterias=["accuracy"],
            rewarding_frequency=1,
            weight=0.2,
        ),
        SyntheticTaskConfig(
            task="continual_conversation",
            criterias=["reward_model"],
            rewarding_frequency=1,
            weight=0.0,
        ),
    ]

    # Default values
    EPOCH_LENGTH: int = 600
    SCORING_PER_MINER_PER_EPOCH: int = 1
    SUBNET_TEMPO: int = 360
    MIN_STAKE: int = int(os.environ.get("MIN_STAKE", 10000))
    RPE_PERCENTAGE_FOR_SYNTHETIC: float = 0.05
    BATCH_SIZE: int = 4
    SCORE_MOVING_AVERAGE: float = 0.05
    ORGANIC_CLIENT_URL: str = "https://ncs-client.condenses.ai"
    REPORT_URL: str = "https://report.condenses.ai"

    # Adjust values based on NETWORK environment variable
    def __init__(self, **data):
        super().__init__(**data)
        network = os.getenv("NETWORK")
        if network == "test":
            self.RPE_PERCENTAGE_FOR_SYNTHETIC = float(
                os.getenv("RPE_PERCENTAGE_FOR_SYNTHETIC", 0.5)
            )
            self.EPOCH_LENGTH = int(os.getenv("EPOCH_LENGTH", 600))
            self.MIN_STAKE = int(os.getenv("MIN_STAKE", 0))
            self.ORGANIC_CLIENT_URL = os.getenv(
                "ORGANIC_CLIENT_URL", "https://testnet-ncs-client.condenses.ai"
            )


constants = Constants()

if __name__ == "__main__":
    import rich

    for k, v in constants.model_dump().items():
        rich.print(f"- {k}: {v}")
