import neural_condense_core as ncc
import bittensor as bt
import random
import httpx
import wandb
from ...protocol import TextCompressProtocol
from ...logger import logger
from ..synthesizing.challenge_generator import ChallengeGenerator
from ..managing.miner_manager import MinerManager, ServingCounter, MetadataItem
from ...constants import SyntheticTaskConfig, TierConfig
import asyncio
import os
import traceback


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


async def prepare_synapse(
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
        synapse = await challenge_generator.generate_challenge(
            tokenizer=tokenizer,
            task=task_config.task,
            max_context_length_in_chars=tier_config.max_context_length_in_chars,
        )
        synapse.target_model = model_name
    except Exception as e:
        logger.error(f"Error generating challenge: {e}")
        traceback.print_exc()
        return None
    return synapse


async def query_miners(
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
    return await dendrite.forward(
        axons=[metagraph.axons[uid] for uid in uids],
        synapse=synapse,
        deserialize=False,
        timeout=timeout,
    )


async def validate_responses(
    responses: list[TextCompressProtocol], uids: list[int], tier_config: TierConfig
) -> tuple[list[TextCompressProtocol], list[int], list[int], list[str]]:
    valid_responses, valid_uids, invalid_uids, invalid_reasons = [], [], [], []

    # Add recursion limit protection
    async def verify_single_response(response):
        try:
            # Add timeout to prevent hanging
            is_valid, reason = await asyncio.wait_for(
                TextCompressProtocol.verify(response, tier_config), timeout=256
            )
            return is_valid, reason
        except asyncio.TimeoutError:
            return False, "Verification timeout"
        except RecursionError:
            return False, "Recursion limit exceeded"
        except Exception as e:
            return False, str(e)

    results = []
    for response in responses:
        verify_result = await verify_single_response(response)
        results.append(verify_result)

    # Process results maintaining order
    for uid, (is_valid, reason), response in zip(uids, results, responses):
        if is_valid:
            valid_responses.append(response)
            valid_uids.append(uid)
        else:
            invalid_uids.append(uid)
            invalid_reasons.append(reason)

    return valid_responses, valid_uids, invalid_uids, invalid_reasons


async def process_and_score_responses(
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
    metrics = await get_scoring_metrics(
        valid_responses=valid_responses,
        invalid_uids=invalid_uids,
        ground_truth_synapse=ground_truth_synapse,
        model_name=model_name,
        task_config=task_config,
        timeout=timeout,
        config=config,
    )
    total_uids = valid_uids + invalid_uids

    # Use run_in_threadpool instead of run_in_executor
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
    return metrics, total_uids


def update_metrics_of_invalid_miners(
    invalid_uids: list[int],
    metrics: dict,
):
    for metric_name, values in metrics.items():
        values.extend([None] * len(invalid_uids))
    return metrics


async def get_scoring_metrics(
    valid_responses: list,
    invalid_uids: list[int],
    ground_truth_synapse: TextCompressProtocol,
    model_name: str,
    task_config: SyntheticTaskConfig,
    timeout: int = 240,
    config: bt.config = None,
) -> dict[str, list]:
    # Move the payload creation to an executor
    payload = {
        "miner_responses": [{"filename": r.local_filename} for r in valid_responses],
        "ground_truth_request": ground_truth_synapse.validator_payload
        | {"model_name": model_name, "criterias": task_config.criterias},
    }
    logger.info(f"Sending payload to scoring backend")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"http://{config.validator.score_backend.host}:{config.validator.score_backend.port}/get_metrics",
                json=payload,
                timeout=timeout,
            )
        except Exception as e:
            logger.error(f"Error sending payload to scoring backend: {e}")
        for r in valid_responses:
            try:
                os.remove(r.local_filename)
            except Exception as e:
                logger.error(f"Error removing local file {r.local_filename}: {e}")
        logger.info("Removed all local files")
        if response.status_code != 200:
            raise Exception(
                f"Scoring backend returned status code {response.status_code}"
            )
        scoring_response = response.json()

    metrics = scoring_response["metrics"]
    # Move the accelerate_metrics calculation to an executor as well
    metrics["accelerate_metrics"] = [r.accelerate_score for r in valid_responses]

    # If update_metrics_of_invalid_miners is CPU-intensive, move it to executor too
    metrics = update_metrics_of_invalid_miners(invalid_uids, metrics)
    return metrics


def get_k_factor(miner_manager: MinerManager, uids: list[int]) -> tuple[int, float]:
    metadatas = miner_manager.get_metadata(uids)
    mean_elo = sum(metadata.elo_rating for metadata in metadatas.values()) / len(metadatas)

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
        logger.error(f"Starting wandb error: {e}")


def get_batched_uids(
    serving_counter: dict[int, ServingCounter], metadata: dict[int, MetadataItem]
) -> list[list[int]]:
    uids = list(serving_counter.keys())
    uids = sorted(uids, key=lambda uid: metadata[uid].elo_rating, reverse=True)
    n_folds = random.choice([2, 3, 4])
    group_size = max(2, len(uids) // n_folds)
    groups = [uids[i : i + group_size] for i in range(0, len(uids), group_size)]
    for group in groups:
        random.shuffle(group)
    uids = [uid for group in groups for uid in group]
    pre_batched_uids = [
        uids[i : i + ncc.constants.BATCH_SIZE]
        for i in range(0, len(uids), ncc.constants.BATCH_SIZE)
    ]
    return pre_batched_uids
