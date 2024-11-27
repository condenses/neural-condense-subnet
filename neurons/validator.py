from neural_condense_core import (
    base,
    validator_utils as vutils,
    constants,
    __spec_version__,
    logger,
)
import pandas as pd
import bittensor as bt
import random
from transformers import AutoTokenizer
import numpy as np
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time


class Validator(base.BaseValidator):
    """
    Validator class that handles validation of miner responses and manages rewards.

    Attributes:
        tier_config (dict): Configuration for different tiers of miners
        miner_manager (MinerManager): Manager for handling miner metadata and state
        challenger (Challenger): Generates validation challenges for miners
        organic_gate (OrganicGate): Optional gate for handling organic traffic
    """

    def __init__(self):
        """Initialize the validator with required components and configurations."""
        super().__init__()
        self.miner_manager = vutils.managing.MinerManager(self)
        self.challenge_generator = vutils.synthesizing.ChallengeGenerator(
            keypair=self.dendrite.keypair
        )

        if self.config.validator.gate_port:
            try:
                self.organic_gate = vutils.monetize.OrganicGate(
                    miner_manager=self.miner_manager,
                    wallet=self.wallet,
                    config=self.config,
                    metagraph=self.metagraph,
                )
                logger.info("Starting organic gate.")
            except Exception as e:
                logger.error(f"Starting organic gate error: {e}")

        if self.config.validator.use_wandb:
            vutils.loop.initialize_wandb(self.dendrite, self.metagraph, self.uid)

        # Add a thread pool executor
        self.loop = asyncio.get_event_loop()
        try:
            self.set_weights()
        except Exception as e:
            logger.error(f"First try to set weights failed: {e}")
            traceback.print_exc()

    async def start_epoch(self):
        """
        Main validation loop that processes miners across all tiers.
        Syncs miner state and runs validation in parallel threads.
        """
        logger.info("Running epoch.")
        await self.miner_manager.sync()
        tasks = [
            self.loop.create_task(self._forward_tier(tier))
            for tier in constants.TIER_CONFIG
        ]
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.TimeoutError as e:
            logger.warning(f"Epoch tasks timed out: {e}")
        except Exception as e:
            logger.error(f"Error running epoch tasks: {e}")
            traceback.print_exc()

        try:
            await self.miner_manager.report_metadata()
            self.miner_manager.save_state()
        except Exception as e:
            logger.error(f"Failed to report metadata & save-state: {e}")

    async def _forward_tier(self, tier: str):
        """
        Process validation for a specific tier of miners.

        Args:
            tier (str): The tier level to process
        """
        if constants.TIER_CONFIG[tier].incentive_percentage == 0:
            logger.info(f"Tier {tier} has no incentive percentage.")
            return

        model_name = random.choice(constants.TIER_CONFIG[tier].supporting_models)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        serving_counter = self.miner_manager.serving_counter.get(tier, {})

        if not serving_counter:
            logger.info(f"No miners in tier {tier}.")
            return
        rate_limit = self.miner_manager.rate_limit_per_tier[tier]
        n_sets = max(
            int(rate_limit * constants.RPE_PERCENTAGE_FOR_SYNTHETIC),
            1,
        )
        sleep_per_set = constants.EPOCH_LENGTH / n_sets
        futures = []

        for i in range(n_sets):
            logger.info(
                f"Processing set {i}/{n_sets} then sleeping for {sleep_per_set} seconds."
            )
            pre_batched_uids = vutils.loop.get_batched_uids(
                serving_counter, self.miner_manager.metadata
            )
            sleep_per_batch = sleep_per_set / len(pre_batched_uids)
            for batch_uids in pre_batched_uids:
                batched_uids = [
                    uid for uid in batch_uids if serving_counter[uid].increment()
                ][: constants.BATCH_SIZE]

                if len(batched_uids) < 2:
                    continue
                future = self.loop.create_task(
                    self._forward_batch(tier, model_name, batched_uids, tokenizer)
                )
                futures.append(future)
                await asyncio.sleep(sleep_per_batch)

        await asyncio.gather(*futures, return_exceptions=True)

    async def _forward_batch(
        self,
        tier: str,
        model_name: str,
        batched_uids: list[int],
        tokenizer: AutoTokenizer,
    ):
        """
        Process a batch of miners for validation.

        Args:
            tier (str): The tier level being processed
            model_name (str): Name of the model to use for validation
            batched_uids (list[int]): List of miner UIDs to validate
            tokenizer: The tokenizer for the selected model
        """
        try:
            dendrite = bt.dendrite(self.wallet)
            task_config = vutils.loop.get_task_config()
            try:
                ground_truth_synapse = await vutils.loop.prepare_synapse(
                    challenge_generator=self.challenge_generator,
                    tokenizer=tokenizer,
                    task_config=task_config,
                    tier_config=constants.TIER_CONFIG[tier],
                    model_name=model_name,
                )
            except Exception as e:
                logger.error(f"Error preparing synapse: {e}")
                traceback.print_exc()
                return
            if not ground_truth_synapse:
                return
            synapse = ground_truth_synapse.miner_synapse
            k_factor = vutils.loop.get_k_factor(self.miner_manager, batched_uids)
            responses = await vutils.loop.query_miners(
                dendrite=dendrite,
                metagraph=self.metagraph,
                uids=batched_uids,
                synapse=synapse,
                timeout=constants.TIER_CONFIG[tier].timeout,
            )

            if not responses:
                logger.warning(f"No responses from {batched_uids}.")
                return
            try:
                (
                    valid_responses,
                    valid_uids,
                    invalid_uids,
                    invalid_reasons,
                ) = await vutils.loop.validate_responses(
                    responses=responses,
                    uids=batched_uids,
                    tier_config=constants.TIER_CONFIG[tier],
                )
            except Exception as e:
                logger.error(f"Error validating responses: {e}")
                traceback.print_exc()
                return
            try:
                metrics, total_uids = await vutils.loop.process_and_score_responses(
                    miner_manager=self.miner_manager,
                    valid_responses=valid_responses,
                    valid_uids=valid_uids,
                    invalid_uids=invalid_uids,
                    ground_truth_synapse=ground_truth_synapse,
                    model_name=model_name,
                    task_config=task_config,
                    tier_config=constants.TIER_CONFIG[tier],
                    tier=tier,
                    k_factor=k_factor,
                    use_wandb=self.config.validator.use_wandb,
                    config=self.config,
                    invalid_reasons=invalid_reasons,
                )
            except Exception as e:
                logger.error(f"Error processing and scoring responses: {e}")
                return

            batch_information = (
                f"Batch Metrics - {tier} - {model_name} - {task_config.task}"
            )
            batch_report_df = vutils.loop.logging.log_as_dataframe(metrics)
            logger.info(
                f"Logging dataframe {batch_information}:\n{batch_report_df.to_markdown()}"
            )

            if self.config.validator.use_wandb:
                vutils.loop.logging.log_wandb(metrics, total_uids, tier=tier)

            await self.miner_manager.report(
                {
                    "comparision": batch_report_df.to_dict(),
                    "challenge": ground_truth_synapse.validator_payload,
                    "task": task_config.task,
                    "tier": tier,
                },
                "api/report-batch",
            )

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error: {e}")

    def set_weights(self):
        """Set weights for miners based on their performance."""
        self.current_block = self.subtensor.get_current_block()
        self.last_update = self.metagraph.last_update[self.uid]
        weights = self.miner_manager.get_normalized_ratings(
            top_percentage=constants.TOP_PERCENTAGE_FOR_ALLOCATING_WEIGHTS
        )
        if np.all(weights == 0):
            weights = np.ones(len(self.metagraph.uids))
            logger.info("All weights are zero, setting to ones.")
        weight_info = list(zip(self.metagraph.uids, weights))
        weight_info_df = pd.DataFrame(weight_info, columns=["uid", "weight"])
        logger.info(f"Weight info:\n{weight_info_df.to_markdown()}")
        if self.current_block > self.last_update + constants.SUBNET_TEMPO:
            logger.info("Actually trying to set weights.")

            # Use ThreadPoolExecutor to add timeout capability
            with ThreadPoolExecutor(max_workers=1) as executor:
                try:
                    # Submit the task to the executor with a timeout
                    future = executor.submit(
                        self.subtensor.set_weights,
                        netuid=self.config.netuid,
                        wallet=self.wallet,
                        uids=self.metagraph.uids,
                        weights=weights,
                        wait_for_inclusion=True,
                        version_key=__spec_version__,
                    )

                    # Wait for the result with a timeout
                    result = future.result(timeout=120)  # 2 minute timeout
                    logger.info(f"Set weights result: {result}")
                    self.resync_metagraph()

                except TimeoutError:
                    logger.error("Setting weights timed out after 2 minutes")
                except Exception as e:
                    logger.error(f"Failed to set weights: {e}")
                    traceback.print_exc()
        else:
            logger.info(
                f"Not setting weights because current block {self.current_block} is not greater than last update {self.last_update} + tempo {constants.SUBNET_TEMPO}"
            )


if __name__ == "__main__":
    validator = Validator()
    asyncio.run(validator.run())
