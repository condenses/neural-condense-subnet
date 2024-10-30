import neural_condense_core as ncc
import bittensor as bt
import threading
import random
from transformers import AutoTokenizer
import requests
import numpy as np
import time


class Validator(ncc.BaseValidator):
    def __init__(self):
        super().__init__()
        self.tier_config = ncc.constants.TIER_CONFIG
        self.miner_manager = ncc.MinerManager(self)
        self.challenger = ncc.Challenger()
        if self.config.validator.gate_port:
            self.organic_gate = ncc.OrganicGate(
                miner_manager=self.miner_manager,
                wallet=self.wallet,
                config=self.config,
                metagraph=self.metagraph,
            )

    def forward(self):
        bt.logging.info("Running epoch.")
        self.miner_manager.sync()
        threads = []
        for tier in self.tier_config:
            thread = threading.Thread(target=self._forward_tier, args=(tier,))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

    def _forward_tier(self, tier):
        supporting_models = ncc.constants.TIER_CONFIG[tier].supporting_models
        model_name = random.choice(supporting_models)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        serving_counter: dict[int, ncc.ServingCounter] = (
            self.miner_manager.serving_counter.get(tier, {})
        )
        bandwidth = sum([serving_counter[uid].rate_limit for uid in serving_counter])
        bandwidth_to_synthetic = int(
            bandwidth * ncc.constants.RPE_PERCENTAGE_FOR_SYNTHETIC
        )
        n_batch = bandwidth_to_synthetic // ncc.constants.BATCH_SIZE
        if n_batch:
            sleep_per_batch = ncc.constants.EPOCH_LENGTH // n_batch
        else:
            sleep_per_batch = ncc.constants.EPOCH_LENGTH

        log = (
            f"Tier: {tier}\n"
            f"Bandwidth: {bandwidth}\n"
            f"Bandwidth to synthetic: {bandwidth_to_synthetic}\n"
            f"Number of batches: {n_batch}\n"
            f"Sleep per batch: {sleep_per_batch}\n"
        )
        bt.logging.info(log)

        query_threads = []
        for _ in range(n_batch):
            random.shuffle(serving_counter)
            batched_uids = []
            for uid in serving_counter:
                if serving_counter[uid].increment():
                    batched_uids.append(uid)
                    if len(batched_uids) == ncc.constants.BATCH_SIZE:
                        break
            if not batched_uids:
                continue

            thread = threading.Thread(
                target=self._forward_batch,
                args=(tier, model_name, batched_uids, tokenizer),
            )
            query_threads.append(thread)
            thread.start()
            bt.logging.info(f"Forwarding batch to {tier}: {batched_uids}")
            bt.logging.info(f"Sleeping for {sleep_per_batch} seconds.")
            time.sleep(sleep_per_batch)

    def _forward_batch(self, tier, model_name, batched_uids, tokenizer):
        r"""
        Forward a batch of requests to the miners.
        Args:
        - tier (str): The tier name.
        - batched_uids (List[int]): The uids of the miners.
        - tokenizer (AutoTokenizer): The tokenizer for the model

        1. Randomly select a task configuration.
        2. Get the synthetic synapse.
        3. Hide the ground truth from miners.
        4. Query the miners.
        5. Update the scores of the miners with probability rewarding_frequency.
        """
        task_config = random.choice(ncc.constants.SYNTHETIC_TASK_CONFIG)
        task_name = task_config["task"]
        rewarding_frequency = task_config["rewarding_frequency"]
        groud_truth_synapse = self.challenger(tokenizer, task_name)
        groud_truth_synapse.target_model = model_name
        synapse = groud_truth_synapse.model_copy()
        synapse.hide_ground_truth()
        dendrite = bt.dendrite(self.wallet)
        axons = [self.metagraph.axons[int(uid)] for uid in batched_uids]
        responses: list[ncc.TextCompressProtocol] = dendrite.query(
            axons=axons,
            synapse=synapse,
            deserialize=False,
            timeout=ncc.constants.TIER_CONFIG[tier].timeout,
        )
        valid_responses: list[ncc.TextCompressProtocol] = []
        valid_uids: list[int] = []
        for uid, response in zip(batched_uids, responses):
            if (
                not response
                or not response.is_success
                or (
                    len(response.compressed_tokens)
                    > ncc.constants.TIER_CONFIG[tier].max_condensed_tokens
                )
            ):
                self.miner_manager.update_scores([uid], [0])
            else:
                valid_responses.append(response)
                valid_uids.append(uid)
        if valid_responses and random.random() < rewarding_frequency:
            payload = {
                "miner_responses": [r.deserialize() for r in valid_responses],
                "ground_truth_request": groud_truth_synapse.deserialize(),
            }
            payload["ground_truth_request"]["model_name"] = model_name
            payload["ground_truth_request"]["criterias"] = task_config["criterias"]

            scoring_response = requests.post(
                ncc.constants.SCORING_ENDPOINT, json=payload, timeout=120
            )
            scoring_response = scoring_response.json()

            scores: list[float] = scoring_response["scores"]

            factors_list = [
                {
                    "normalized_score_in_batch": score,
                    "process_time/timeout": response.dendrite.process_time
                    / ncc.constants.TIER_CONFIG[tier].timeout,
                }
                for score, response in zip(scores, valid_responses)
            ]
            scores = [
                ncc.constants.TIER_CONFIG[tier].scoring_lambda(factors)
                for factors in factors_list
            ]

            self.miner_manager.update_scores(valid_uids, scores)

    def set_weights(self):
        r"""
        Just normalize the scores and set the weights.
        """
        self.current_block = self.subtensor.get_current_block()
        self.last_update = self.metagraph.last_update[self.uid]
        weights: np.ndarray = self.miner_manager.get_normalized_scores()
        if self.last_update > self.current_block + ncc.constants.SUBNET_TEMPO:
            bt.logging.info(f"Setting weights: {weights}")
            result = self.subtensor.set_weights(
                netuid=self.netuid,
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
