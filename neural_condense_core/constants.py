from pydantic import BaseModel
from typing import List
import os


class TierConfig(BaseModel):
    incentive_percentage: float
    requests_per_epoch: int
    timeout: int
    supporting_models: List[str]
    max_condensed_tokens: int
    min_condensed_tokens: int
    max_context_length_in_chars: int
    accelerate_reward_scalar: float


class SyntheticTaskConfig(BaseModel):
    task: str
    criterias: List[str]
    rewarding_frequency: int
    weight: float


class EloGroup(BaseModel):
    min_elo: int
    max_elo: int
    k_factor: int


class Constants(BaseModel):
    TIER_CONFIG: dict[str, TierConfig] = {
        "research": TierConfig(
            incentive_percentage=1.0,
            requests_per_epoch=256,
            timeout=32,
            accelerate_reward_scalar=0.1,
            supporting_models=["Condense-AI/Mistral-7B-Instruct-v0.2"],
            max_condensed_tokens=1024,
            min_condensed_tokens=128,
            max_context_length_in_chars=10000,
        ),
        "inference_0": TierConfig(
            incentive_percentage=0.0,
            requests_per_epoch=1024,
            timeout=8,
            accelerate_reward_scalar=0.1,
            supporting_models=["Condense-AI/Mistral-7B-Instruct-v0.2"],
            max_condensed_tokens=1024,
            min_condensed_tokens=128,
            max_context_length_in_chars=15000,
        ),
        "inference_1": TierConfig(
            incentive_percentage=0.0,
            requests_per_epoch=1024,
            timeout=8,
            accelerate_reward_scalar=0.1,
            supporting_models=["Condense-AI/Mistral-7B-Instruct-v0.2"],
            max_condensed_tokens=2048,
            min_condensed_tokens=128,
            max_context_length_in_chars=20000,
        ),
    }

    SYNTHETIC_TASK_CONFIG: List[SyntheticTaskConfig] = [
        SyntheticTaskConfig(
            task="causal_conversation",
            criterias=["perplexity"],
            rewarding_frequency=1,
            weight=1,
        ),
        SyntheticTaskConfig(
            task="question_answering",
            criterias=["perplexity"],
            rewarding_frequency=1,
            weight=1,
        ),
        SyntheticTaskConfig(
            task="reconstruct_conversation",
            criterias=["perplexity"],
            rewarding_frequency=1,
            weight=1,
        ),
        SyntheticTaskConfig(
            task="trivial_qa_conversation",
            criterias=["perplexity"],
            rewarding_frequency=1,
            weight=1,
        ),
    ]

    # Default values
    EPOCH_LENGTH: int = 600
    SCORING_PER_MINER_PER_EPOCH: int = 1
    SUBNET_TEMPO: int = 360
    MIN_STAKE: int = int(os.environ.get("MIN_STAKE", 10000))
    RPE_PERCENTAGE_FOR_SYNTHETIC: float = 0.05
    BATCH_SIZE: int = 4
    SET_WEIGHTS_TIMEOUT: int = 120
    ORGANIC_CLIENT_URL: str = "https://ncs-client.condenses.ai"
    REPORT_URL: str = "https://report.condenses.ai"
    INITIAL_ELO_RATING: float = 100.0
    FLOOR_ELO_RATING: float = 0.0
    ELO_GROUPS: dict[str, EloGroup] = {
        "beginner": EloGroup(min_elo=0, max_elo=800, k_factor=24),
        "intermediate": EloGroup(min_elo=800, max_elo=1600, k_factor=16),
        "advanced": EloGroup(min_elo=1600, max_elo=3000, k_factor=4),
    }
    ORGANIC_VERIFY_FREQUENCY: float = 0.1
    TOP_PERCENTAGE_FOR_ALLOCATING_WEIGHTS: float = 0.3

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
            self.REPORT_URL = os.getenv(
                "REPORT_URL", "https://testnet-report.condenses.ai"
            )


constants = Constants()

if __name__ == "__main__":
    import rich

    for k, v in constants.model_dump().items():
        rich.print(f"- {k}: {v}")
