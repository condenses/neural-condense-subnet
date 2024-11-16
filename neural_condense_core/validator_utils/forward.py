import neural_condense_core as ncc
import bittensor as bt
import requests
import random
import wandb
from ..protocol import TextCompressProtocol
from . import logging
from .synthetic_challenge import Challenger
from .miner_manager import MinerManager, ServingCounter, MetadataItem
from ..constants import SyntheticTaskConfig, TierConfig


def get_task_config() -> SyntheticTaskConfig:
    """
    Get a random task configuration based on weights.

    Returns:
        SyntheticTaskConfig: The selected task configuration
    """
    return random.choices(
        ncc.constants.SYNTHETIC_TASK_CONFIG,
        weights=[t.weight for t in ncc.constants.SYNTHETIC_TASK_CONFIG],
    )[0]


def prepare_synapse(
    challenger: Challenger,
    tokenizer,
    task_config: SyntheticTaskConfig,
    tier_config: TierConfig,
    model_name: str,
):
    """
    Prepare a synapse for validation.

    Args:
        tokenizer: The tokenizer to use
        task_config (SyntheticTaskConfig): Configuration for the task
        tier_config (TierConfig): Configuration for the tier
        model_name (str): Name of the model to use

    Returns:
        The prepared synapse object
    """
    synapse = challenger(
        tokenizer=tokenizer,
        task=task_config.task,
        max_context_length_in_chars=tier_config.max_context_length_in_chars,
    )
    synapse.target_model = model_name
    return synapse


def query_miners(
    dendrite: bt.dendrite,
    metagraph: bt.metagraph,
    uids: list[int],
    synapse,
    timeout: int,
):
    """
    Query a group of miners with a synapse.

    Args:
        dendrite: The dendrite connection
        uids (list[int]): List of miner UIDs to query
        synapse: The synapse to send
        timeout (int): Query timeout in seconds

    Returns:
        list: Responses from the miners
    """
    return dendrite.query(
        axons=[metagraph.axons[uid] for uid in uids],
        synapse=synapse,
        deserialize=False,
        timeout=timeout,
    )


def validate_responses(responses: list, uids: list[int], tier_config: TierConfig):
    """
    Validate responses from miners.

    Args:
        responses (list): List of miner responses
        uids (list[int]): List of miner UIDs
        tier_config (TierConfig): Configuration for the tier

    Returns:
        tuple: Lists of valid responses, valid UIDs, and invalid UIDs
    """
    valid_responses, valid_uids, invalid_uids = [], [], []
    for uid, response in zip(uids, responses):
        try:
            response.base64_to_ndarray()
            if (
                response
                and response.is_success
                and len(response.compressed_tokens.shape) == 2
                and tier_config.min_condensed_tokens
                <= len(response.compressed_tokens)
                <= tier_config.max_condensed_tokens
            ):
                valid_responses.append(response)
                valid_uids.append(uid)
            else:
                invalid_uids.append(uid)
        except Exception as e:
            bt.logging.error(f"Error: {e}")
            invalid_uids.append(uid)
    return valid_responses, valid_uids, invalid_uids


def process_and_score_responses(
    miner_manager: MinerManager,
    valid_responses: list[TextCompressProtocol],
    valid_uids: list[int],
    invalid_uids: list[int],
    ground_truth_synapse: TextCompressProtocol,
    model_name: str,
    task_config: SyntheticTaskConfig,
    tier_config: TierConfig,
    tier: str,
    k_factor: int,
    use_wandb: bool = False,
    config: bt.config = None,
    timeout: int = 120,
):
    """
    Process and score miner responses.

    Args:
        valid_responses (list): List of valid responses
        valid_uids (list[int]): List of valid miner UIDs
        invalid_uids (list[int]): List of invalid miner UIDs
        ground_truth_synapse: The ground truth synapse
        model_name (str): Name of the model used
        task_config (SyntheticTaskConfig): Task configuration
        tier_config (TierConfig): Tier configuration
        tier (str): The tier level
        k_factor (int): ELO rating K-factor
        timeout (int): Timeout for scoring backend
        use_wandb (bool): Whether to use wandb
    """
    metrics = get_scoring_metrics(
        valid_responses=valid_responses,
        ground_truth_synapse=ground_truth_synapse,
        model_name=model_name,
        task_config=task_config,
        timeout=timeout,
        config=config,
    )
    accelerate_metrics = get_accelerate_metrics(
        valid_responses=valid_responses,
        tier_config=tier_config,
    )
    metrics["accelerate_metrics"] = accelerate_metrics
    final_ratings, initial_ratings = miner_manager.update_ratings(
        metrics=metrics,
        valid_uids=valid_uids,
        k_factor=k_factor,
        invalid_uids=invalid_uids,
        tier_config=tier_config,
    )
    rating_changes = [
        f"{initial_ratings[i]} -> {final_ratings[i]}"
        for i in range(len(initial_ratings))
    ]

    metrics["rating_changes"] = rating_changes
    metrics["UIDs"] = valid_uids
    logging.log_as_dataframe(data=metrics, name="Batch Metrics")
    if use_wandb:
        logging.log_wandb(metrics, valid_uids, tier=tier)


def get_scoring_metrics(
    valid_responses: list,
    ground_truth_synapse,
    model_name: str,
    task_config: SyntheticTaskConfig,
    timeout: int = 120,
    config: bt.config = None,
):
    """
    Get scoring metrics for valid responses.
    """
    payload = {
        "miner_responses": [
            {"compressed_tokens_b64": r.compressed_tokens_b64} for r in valid_responses
        ],
        "ground_truth_request": ground_truth_synapse.deserialize()
        | {"model_name": model_name, "criterias": task_config.criterias},
    }

    scoring_response = requests.post(
        f"http://{config.validator.score_backend.host}:{config.validator.score_backend.port}/scoring",
        json=payload,
        timeout=timeout,
    ).json()

    metrics = scoring_response["metrics"]
    return metrics


def get_accelerate_metrics(
    valid_responses: list, tier_config: TierConfig
) -> list[float]:
    """
    Calculate additional rewards for miners based on compression and processing time.

    Args:
        valid_responses (list): List of valid responses
        tier_config (TierConfig): Tier configuration

    Returns:
        list[float]: List of additional rewards
    """
    compress_rate_rewards = [
        1 - len(r.compressed_tokens) / tier_config.max_condensed_tokens
        for r in valid_responses
    ]
    process_time_rewards = [
        1 - r.dendrite.process_time / tier_config.timeout for r in valid_responses
    ]
    rewards = [(c + p) / 2 for c, p in zip(compress_rate_rewards, process_time_rewards)]
    return rewards


def get_k_factor(miner_manager: MinerManager, uids: list[int]) -> tuple[int, float]:
    """
    Get the ELO K-factor and optimization bounty based on mean ELO rating.

    Args:
        uids (list[int]): List of miner UIDs

    Returns:
        tuple[int, float]: K-factor and optimization bounty
    """
    mean_elo = sum(miner_manager.metadata[uid].elo_rating for uid in uids) / len(uids)

    if mean_elo < ncc.constants.ELO_GROUPS["beginner"].max_elo:
        elo_group = ncc.constants.ELO_GROUPS["beginner"]
    elif mean_elo < ncc.constants.ELO_GROUPS["intermediate"].max_elo:
        elo_group = ncc.constants.ELO_GROUPS["intermediate"]
    else:
        elo_group = ncc.constants.ELO_GROUPS["advanced"]
    return elo_group.k_factor


def initialize_wandb(dendrite: bt.dendrite, metagraph: bt.metagraph, uid: int):
    try:
        message = "incentivized-decentralzied-condensed-ai" + "-".join(
            random.choices("0123456789abcdef", k=16)
        )
        signature = f"0x{dendrite.keypair.sign(message).hex()}"
        wandb.init(
            project="Neural-Condense-Subnet",
            name=f"validator-{uid}",
            entity="toilaluan",
            job_type="validation",
            group="validator",
            resume="auto",
            config={
                "signature": signature,
                "uid": uid,
                "message": message,
                "ss58_address": metagraph.hotkeys[uid],
            },
        )
    except Exception as e:
        bt.logging.error(f"Starting wandb error: {e}")


def get_batched_uids(
    serving_counter: dict[int, ServingCounter], metadata: dict[int, MetadataItem]
) -> list[list[int]]:
    """
    Get batched UIDs for validation.

    Args:
        serving_counter (dict[int, ServingCounter]): Serving counter

    Returns:
        list[list[int]]: Batched UIDs
    """
    uids = list(serving_counter.keys())
    uids = sorted(uids, key=lambda uid: metadata[uid].elo_rating, reverse=True)
    group_size = max(2, len(uids) // 4)
    groups = [uids[i : i + group_size] for i in range(0, len(uids), group_size)]
    for group in groups:
        random.shuffle(group)
    uids = [uid for group in groups for uid in group]
    pre_batched_uids = [
        uids[i : i + ncc.constants.BATCH_SIZE]
        for i in range(0, len(uids), ncc.constants.BATCH_SIZE)
    ]
    return pre_batched_uids
