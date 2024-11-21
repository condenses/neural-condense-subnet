import neural_condense_core as ncc
import bittensor as bt
import requests
import random
import wandb
from ...protocol import TextCompressProtocol
from .logging import log_wandb
from ..synthesizing.challenge_generator import ChallengeGenerator
from ..managing.miner_manager import MinerManager, ServingCounter, MetadataItem
from ...constants import SyntheticTaskConfig, TierConfig


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
    challenge_generator: ChallengeGenerator,
    tokenizer,
    task_config: SyntheticTaskConfig,
    tier_config: TierConfig,
    model_name: str,
) -> TextCompressProtocol:
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
    try:
        synapse = challenge_generator.generate_challenge(
            tokenizer=tokenizer,
            task=task_config.task,
            max_context_length_in_chars=tier_config.max_context_length_in_chars,
        )
        synapse.target_model = model_name
    except Exception:
        return None
    return synapse


def query_miners(
    dendrite: bt.dendrite,
    metagraph: bt.metagraph,
    uids: list[int],
    synapse,
    timeout: int,
) -> list[TextCompressProtocol]:
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


def validate_responses(
    responses: list[TextCompressProtocol], uids: list[int], tier_config: TierConfig
) -> tuple[list[TextCompressProtocol], list[int], list[int], list[str]]:
    valid_responses, valid_uids, invalid_uids, invalid_reasons = [], [], [], []
    for uid, response in zip(uids, responses):
        try:
            is_valid, reason = TextCompressProtocol.verify(response, tier_config)
            if is_valid:
                valid_responses.append(response)
                valid_uids.append(uid)
            else:
                invalid_uids.append(uid)
                invalid_reasons.append(reason)
        except Exception as e:
            invalid_uids.append(uid)
            invalid_reasons.append(str(e))
    return valid_responses, valid_uids, invalid_uids, invalid_reasons


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
    invalid_reasons: list[str] = [],
    timeout: int = 120,
) -> dict[str, list]:
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
    metrics = update_metrics_of_invalid_miners(
        invalid_uids=invalid_uids,
        metrics=metrics,
    )
    total_uids = valid_uids + invalid_uids
    final_ratings, initial_ratings = miner_manager.update_ratings(
        metrics=metrics,
        total_uids=total_uids,
        k_factor=k_factor,
        tier_config=tier_config,
    )
    rating_changes = [
        f"{int(initial_ratings[i])} -> {int(final_ratings[i])}"
        for i in range(len(initial_ratings))
    ]
    reasons = [""] * len(valid_uids) + invalid_reasons
    metrics["rating_change"] = rating_changes
    metrics["uid"] = total_uids
    metrics["invalid_reasons"] = reasons
    if use_wandb:
        log_wandb(metrics, total_uids, tier=tier)
    return metrics


def update_metrics_of_invalid_miners(
    invalid_uids: list[int],
    metrics: dict,
):
    for metric_name, values in metrics.items():
        values.extend([None] * len(invalid_uids))
    return metrics


def get_scoring_metrics(
    valid_responses: list,
    ground_truth_synapse: TextCompressProtocol,
    model_name: str,
    task_config: SyntheticTaskConfig,
    timeout: int = 120,
    config: bt.config = None,
) -> dict[str, list]:
    payload = {
        "miner_responses": [
            {"compressed_kv_b64": r.compressed_kv_b64} for r in valid_responses
        ],
        "ground_truth_request": ground_truth_synapse.validator_payload
        | {"model_name": model_name, "criterias": task_config.criterias},
    }

    scoring_response = requests.post(
        f"http://{config.validator.score_backend.host}:{config.validator.score_backend.port}/get_metrics",
        json=payload,
        timeout=timeout,
    ).json()

    metrics = scoring_response["metrics"]
    return metrics


def get_accelerate_metrics(
    valid_responses: list, tier_config: TierConfig
) -> list[float]:
    compress_rate_rewards = [
        1 - len(r.compressed_tokens) / tier_config.max_condensed_tokens
        for r in valid_responses
    ]
    process_time_rewards = [
        1 - r.dendrite.process_time / tier_config.timeout for r in valid_responses
    ]
    rewards = [
        max(0, (c + p) / 2) for c, p in zip(compress_rate_rewards, process_time_rewards)
    ]
    return rewards


def get_k_factor(miner_manager: MinerManager, uids: list[int]) -> tuple[int, float]:
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
            resume="allow",
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
