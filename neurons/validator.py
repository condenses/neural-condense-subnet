import neural_condense_core as ncc
import bittensor as bt
import threading
import random
from transformers import AutoTokenizer
import numpy as np
import time
import traceback
from neural_condense_core.validator_utils import forward as forward_utils


class Validator(ncc.base.BaseValidator):
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
        self.miner_manager = ncc.validator_utils.MinerManager(self)
        self.challenger = ncc.validator_utils.Challenger()

        if self.config.validator.gate_port:
            try:
                self.organic_gate = ncc.validator_utils.OrganicGate(
                    miner_manager=self.miner_manager,
                    wallet=self.wallet,
                    config=self.config,
                    metagraph=self.metagraph,
                )
                bt.logging.info("Starting organic gate.")
            except Exception as e:
                bt.logging.error(f"Starting organic gate error: {e}")

        if self.config.validator.use_wandb:
            forward_utils.initialize_wandb(self.dendrite, self.metagraph, self.uid)
        
        weights = self.miner_manager.get_normalized_ratings()
        bt.logging.info(f"Weights: {weights}")
        bt.logging.info(f"Uids: {self.metagraph.uids}")

    def start_epoch(self):
        """
        Main validation loop that processes miners across all tiers.
        Syncs miner state and runs validation in parallel threads.
        """
        bt.logging.info("Running epoch.")
        self.miner_manager.sync()
        threads = [
            threading.Thread(target=self._forward_tier, args=(tier,))
            for tier in ncc.constants.TIER_CONFIG
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        try:
            self.miner_manager.report()
            self.miner_manager.save_state()
        except Exception as e:
            bt.logging.error(f"Failed to report metadata & save-state: {e}")

    def _forward_tier(self, tier: str):
        """
        Process validation for a specific tier of miners.

        Args:
            tier (str): The tier level to process
        """
        if ncc.constants.TIER_CONFIG[tier].incentive_percentage == 0:
            bt.logging.info(f"Tier {tier} has no incentive percentage.")
            return

        model_name = random.choice(ncc.constants.TIER_CONFIG[tier].supporting_models)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        serving_counter = self.miner_manager.serving_counter.get(tier, {})

        if not serving_counter:
            bt.logging.info(f"No miners in tier {tier}.")
            return

        n_sets = max(
            int(
                ncc.constants.TIER_CONFIG[tier].requests_per_epoch
                * ncc.constants.RPE_PERCENTAGE_FOR_SYNTHETIC
            ),
            1,
        )
        sleep_per_set = ncc.constants.EPOCH_LENGTH / n_sets
        query_threads = []

        for _ in range(n_sets):
            pre_batched_uids = forward_utils.get_batched_uids(
                serving_counter, self.miner_manager.metadata
            )
            sleep_per_batch = sleep_per_set / len(pre_batched_uids)
            for batch_uids in pre_batched_uids:
                batched_uids = [
                    uid for uid in batch_uids if serving_counter[uid].increment()
                ][: ncc.constants.BATCH_SIZE]

                if len(batched_uids) < 2:
                    continue

                thread = threading.Thread(
                    target=self._forward_batch,
                    args=(tier, model_name, batched_uids, tokenizer),
                )
                query_threads.append(thread)
                thread.start()
                time.sleep(sleep_per_batch)

        for thread in query_threads:
            thread.join()

    def _forward_batch(
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
            task_config = forward_utils.get_task_config()

            ground_truth_synapse = forward_utils.prepare_synapse(
                challenger=self.challenger,
                tokenizer=tokenizer,
                task_config=task_config,
                tier_config=ncc.constants.TIER_CONFIG[tier],
                model_name=model_name,
            )
            bt.logging.info(f"Prepared ground truth synapse for {batched_uids}.")
            synapse = ground_truth_synapse.model_copy()
            synapse.hide_ground_truth()
            k_factor = forward_utils.get_k_factor(self.miner_manager, batched_uids)
            responses = forward_utils.query_miners(
                dendrite=dendrite,
                metagraph=self.metagraph,
                uids=batched_uids,
                synapse=synapse,
                timeout=ncc.constants.TIER_CONFIG[tier].timeout,
            )
            bt.logging.info(f"Queried miners for {batched_uids}.")
            valid_responses, valid_uids, invalid_uids = (
                forward_utils.validate_responses(
                    responses=responses,
                    uids=batched_uids,
                    tier_config=ncc.constants.TIER_CONFIG[tier],
                )
            )
            bt.logging.info(f"Validated responses for {batched_uids}.")
            if not valid_responses:
                bt.logging.info(f"No valid responses for batch {batched_uids}.")
                return

            if random.random() < task_config.rewarding_frequency:
                forward_utils.process_and_score_responses(
                    miner_manager=self.miner_manager,
                    valid_responses=valid_responses,
                    valid_uids=valid_uids,
                    invalid_uids=invalid_uids,
                    ground_truth_synapse=ground_truth_synapse,
                    model_name=model_name,
                    task_config=task_config,
                    tier_config=ncc.constants.TIER_CONFIG[tier],
                    tier=tier,
                    k_factor=k_factor,
                    use_wandb=self.config.validator.use_wandb,
                    config=self.config,
                )
                bt.logging.info(f"Processed and scored responses for {batched_uids}.")
            else:
                bt.logging.info(f"Not rewarding batch {batched_uids}.")

        except Exception as e:
            traceback.print_exc()
            bt.logging.error(f"Error: {e}")

    def set_weights(self):
        """Set weights for miners based on their performance."""
        self.current_block = self.subtensor.get_current_block()
        self.last_update = self.metagraph.last_update[self.uid]
        weights = self.miner_manager.get_normalized_ratings()
        if np.all(weights == 0):
            weights = np.ones(len(self.metagraph.uids))
            bt.logging.info("All weights are zero, setting to ones.")
        bt.logging.info(f"Weights: {weights}")
        bt.logging.info(f"Uids: {self.metagraph.uids}")
        if self.current_block > self.last_update + ncc.constants.SUBNET_TEMPO:
            result = self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=self.metagraph.uids,
                weights=weights,
                wait_for_inclusion=True,
                version_key=ncc.__spec_version__,
            )
            bt.logging.info(f"Set weights result: {result}")
            self.resync_metagraph()


if __name__ == "__main__":
    validator = Validator()
    validator.run()
