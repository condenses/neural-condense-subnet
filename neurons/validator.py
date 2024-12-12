from neural_condense_core import (
    base,
    validator_utils as vutils,
    constants,
    logger,
)
from neural_condense_core.protocol import TextCompressProtocol
import pandas as pd
import bittensor as bt
import random
from transformers import AutoTokenizer
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time


class Validator(base.BaseValidator):
    def __init__(self):
        super().__init__()
        self.miner_manager = vutils.managing.MinerManager(
            uid=self.uid,
            wallet=self.wallet,
            metagraph=self.metagraph,
            config=self.config,
        )
        self.challenge_generator = vutils.synthesizing.ChallengeGenerator(
            keypair=self.dendrite.keypair
        )

        if self.config.validator.use_wandb:
            vutils.loop.initialize_wandb(self.dendrite, self.metagraph, self.uid)

        self.set_weights_executor = ThreadPoolExecutor(max_workers=1)

    async def start_epoch(self):
        logger.info("Running epoch.")
        await self.miner_manager.sync()
        try:
            await self.miner_manager.report_metadata()
        except Exception as e:
            logger.error(f"Failed to report metadata & save-state: {e}")
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

    async def _forward_tier(self, tier: str):
        try:
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
            futures = []
        except Exception as e:
            logger.error(f"Error in _forward_tier: {e}")
            traceback.print_exc()
            return

        task_config = vutils.loop.get_task_config()
        model_name = random.choice(constants.TIER_CONFIG[tier].supporting_models)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        ground_truth_synapses = [
            await vutils.loop.prepare_synapse(
                challenge_generator=self.challenge_generator,
                tokenizer=tokenizer,
                task_config=task_config,
                tier_config=constants.TIER_CONFIG[tier],
                model_name=model_name,
            )
            for _ in range(n_sets)
        ]

        sleep_per_set = constants.EPOCH_LENGTH / n_sets / 2

        logger.info(f"Prepared {len(ground_truth_synapses)} ground truth synapses.")

        for i, ground_truth_synapse in enumerate(ground_truth_synapses):
            logger.info(
                f"Processing set {i}/{n_sets} then sleeping for {sleep_per_set} seconds."
            )
            total_uids = list(serving_counter.keys())
            random.shuffle(total_uids)
            batched_uids = [total_uids[i : i + 4] for i in range(0, len(total_uids), 4)]
            futures = []
            for uids in batched_uids:
                logger.info(
                    "Processing batch",
                    uids=uids,
                    sleep=sleep_per_set / len(batched_uids),
                )
                future = self.loop.create_task(
                    self._forward_batch(
                        tier,
                        model_name,
                        uids,
                        ground_truth_synapse,
                        task_config,
                    )
                )
                futures.append(future)
                await asyncio.sleep(sleep_per_set / len(batched_uids))
        await asyncio.gather(*futures, return_exceptions=True)

    async def _forward_batch(
        self,
        tier: str,
        model_name: str,
        batched_uids: list[int],
        ground_truth_synapse: TextCompressProtocol,
        task_config,
    ):
        try:
            dendrite = bt.dendrite(self.wallet)
            synapse = ground_truth_synapse.miner_synapse
            logger.info(f"Querying miners {batched_uids}.")
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
                logger.info(f"Validating responses for {batched_uids}.")
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
                logger.info("Processing and scoring responses.")
                start_time = time.time()
                logs, total_uids = await vutils.loop.process_and_score_responses(
                    miner_manager=self.miner_manager,
                    valid_responses=valid_responses,
                    valid_uids=valid_uids,
                    invalid_uids=invalid_uids,
                    ground_truth_synapse=ground_truth_synapse,
                    model_name=model_name,
                    task_config=task_config,
                    tier_config=constants.TIER_CONFIG[tier],
                    config=self.config,
                    invalid_reasons=invalid_reasons,
                    timeout=300,
                )
                end_time = time.time()
                logger.info(
                    f"Time taken to process and score responses: {end_time - start_time:.2f} seconds"
                )
            except Exception as e:
                logger.error(f"Error processing and scoring responses: {e}")
                return

            batch_information = (
                f"Batch Metrics - {tier} - {model_name} - {task_config.task}"
            )
            batch_report_df = vutils.loop.logging.log_as_dataframe(logs)
            logger.info(
                f"Logging dataframe {batch_information}:\n{batch_report_df.to_markdown()}"
            )

            if self.config.validator.use_wandb:
                vutils.loop.logging.log_wandb(logs, total_uids, tier=tier)

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
        try:
            self.current_block = self.subtensor.get_current_block()
        except OSError as e:
            logger.warning(f"Subtensor not available, reconnecting: {e}")
            self.subtensor = bt.subtensor(config=self.config)
            logger.info("Reconnected to subtensor.")
            self.current_block = self.subtensor.get_current_block()
        except Exception as e:
            logger.error(f"Error getting current block: {e}")
            traceback.print_exc()
            return
        self.last_update = self.metagraph.last_update[self.uid]
        weights = self.miner_manager.get_normalized_ratings(
            top_percentage=constants.TOP_PERCENTAGE_FOR_ALLOCATING_WEIGHTS
        )
        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        (
            uint_uids,
            uint_weights,
        ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        if self.current_block > self.last_update + constants.SUBNET_TEMPO:
            weight_info = list(zip(uint_uids, uint_weights))
            weight_info_df = pd.DataFrame(weight_info, columns=["uid", "weight"])
            logger.info(f"Weight info:\n{weight_info_df.to_markdown()}")
            logger.info("Actually trying to set weights.")
            try:
                future = self.set_weights_executor.submit(
                    self.subtensor.set_weights,
                    netuid=self.config.netuid,
                    wallet=self.wallet,
                    uids=uint_uids,
                    weights=uint_weights,
                )
                result = future.result(timeout=120)
            except Exception as e:
                logger.error(f"Failed to set weights: {e}")
                traceback.print_exc()

            logger.info(f"Set weights result: {result}")
        else:
            logger.info(
                f"Not setting weights because current block {self.current_block} is not greater than last update {self.last_update} + tempo {constants.SUBNET_TEMPO}"
            )


if __name__ == "__main__":
    with Validator() as validator:
        while True:
            logger.info("validator_status", object=validator)
            if not validator.thread_set_weights.is_alive():
                logger.info("Starting set weights thread.")
                validator.thread_set_weights.start()
            else:
                logger.info("Set weights thread already running.")
            time.sleep(60)
