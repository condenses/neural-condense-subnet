import neural_condense_core as ncc
import bittensor as bt
import random
import httpx
import wandb
from ...protocol import TextCompressProtocol
from ...logger import logger
from ..synthesizing.challenge_generator import ChallengeGenerator
from ..managing.miner_manager import MinerManager, ServingCounter, MinerMetadata
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
    tier: str,
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
            model_name=model_name,
            tier=tier,
            task=task_config.task,
            max_context_length_in_chars=tier_config.max_context_length_in_chars,
        )
        synapse.target_model = model_name
        synapse.tier = tier
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
    batched_uids = [
        uids[i : i + ncc.constants.BATCH_SIZE]
        for i in range(0, len(uids), ncc.constants.BATCH_SIZE)
    ]
    all_responses = []
    for batch_uids in batched_uids:
        responses = await dendrite.forward(
            axons=[metagraph.axons[uid] for uid in batch_uids],
            synapse=synapse,
            deserialize=False,
            timeout=timeout,
        )
        all_responses.extend(responses)
    return all_responses


async def validate_responses(
    responses: list[TextCompressProtocol],
    uids: list[int],
    tier_config: TierConfig,
    tier: str,
    tokenizer=None,
    ground_truth_synapse: TextCompressProtocol = None,
) -> tuple[list[TextCompressProtocol], list[int], list[int], list[str]]:
    valid_responses, valid_uids, invalid_uids, invalid_reasons = [], [], [], []

    # Add recursion limit protection
    async def verify_single_response(response):
        try:
            # Add timeout to prevent hanging
            is_valid, reason = await asyncio.wait_for(
                TextCompressProtocol.verify(
                    response,
                    tier_config,
                    tier,
                    tokenizer,
                    ground_truth_synapse,
                ),
                timeout=360,
            )
            return is_valid, reason
        except asyncio.TimeoutError:
            return False, "Verification timeout"
        except RecursionError:
            return False, "Recursion limit exceeded"
        except Exception as e:
            return False, f"Failed to verify: {str(e)}"

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
    config: bt.config = None,
    invalid_reasons: list[str] = [],
    timeout: int = 120,
    tier: str = "",
) -> dict[str, list]:
    if len(valid_responses) > 0:
        accuracies, accelerate_rewards = await get_accuracies(
            valid_responses=valid_responses,
            ground_truth_synapse=ground_truth_synapse,
            model_name=model_name,
            task_config=task_config,
            timeout=timeout,
            config=config,
            tier=tier,
        )
        scores = [
            (
                accu * (1 - tier_config.accelerate_reward_scalar)
                + accel * tier_config.accelerate_reward_scalar
            )
            * (accu > 0)
            for accu, accel in zip(accuracies, accelerate_rewards)
        ] + [0] * len(invalid_uids)
    else:
        scores = [0] * len(valid_uids) + [0] * len(invalid_uids)
        accuracies = []
        accelerate_rewards = []
    total_uids = valid_uids + invalid_uids
    updated_scores, previous_scores = miner_manager.update_scores(
        scores=scores,
        total_uids=total_uids,
    )
    score_changes = [
        f"{round(previous_scores[i], 3)} -> {round(updated_scores[i], 3)}"
        for i in range(len(previous_scores))
    ]
    logs = {
        "uid": total_uids,
        "accuracy": accuracies + [0] * len(invalid_uids),
        "accelerate_reward": accelerate_rewards + [0] * len(invalid_uids),
        "score_change": score_changes,
        "invalid_reasons": [""] * len(valid_uids) + invalid_reasons,
    }
    return logs, total_uids


def update_metrics_of_invalid_miners(
    invalid_uids: list[int],
    metrics: dict,
):
    for metric_name, values in metrics.items():
        values.extend([0] * len(invalid_uids))
    return metrics


async def get_accuracies(
    valid_responses: list,
    ground_truth_synapse: TextCompressProtocol,
    model_name: str,
    task_config: SyntheticTaskConfig,
    timeout: int = 240,
    config: bt.config = None,
    tier: str = "",
) -> tuple[list, list]:
    payload = TextCompressProtocol.get_scoring_payload(
        responses=valid_responses,
        ground_truth_synapse=ground_truth_synapse,
        target_model=model_name,
        criterias=task_config.criterias,
    ).model_dump()
    if tier == "universal":
        url = f"http://{config.validator.universal_score_backend.host}:{config.validator.universal_score_backend.port}/get_metrics"
    else:
        url = f"http://{config.validator.score_backend.host}:{config.validator.score_backend.port}/get_metrics"
    logger.info(f"Sending payload to scoring backend: {url}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url,
                json=payload,
                timeout=timeout,
            )
        except Exception as e:
            logger.error(f"Error sending payload to scoring backend: {e}")
        for r in valid_responses:
            try:
                if r.util_data.local_filename:
                    os.remove(r.util_data.local_filename)
            except Exception as e:
                logger.error(
                    f"Error removing local file {r.util_data.local_filename}: {e}"
                )
        logger.info("Removed all local files")
        if response.status_code != 200:
            raise Exception(
                f"Scoring backend returned status code {response.status_code}"
            )
        scoring_response = response.json()

    accuracies = scoring_response["metrics"]["accuracy"]
    accelerate_rewards = [r.accelerate_score for r in valid_responses]
    return accuracies, accelerate_rewards


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
