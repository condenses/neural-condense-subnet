import neural_condense_core as ncc
import bittensor as bt
import threading
import random
from transformers import AutoTokenizer
from transformers.utils.logging import disable_propagation, disable_default_handler
import numpy as np
import time
import requests

disable_default_handler()
disable_propagation()


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
            bt.logging.info("Starting organic gate.")

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
        self.miner_manager.store_miner_info()

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
            n_batch = bandwidth_to_synthetic

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
        for thread in query_threads:
            thread.join()

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
        try:
            dendrite = bt.dendrite(self.wallet)
            task_weights = [
                task_config.weight
                for task_config in ncc.constants.SYNTHETIC_TASK_CONFIG
            ]
            task_config = random.choices(
                ncc.constants.SYNTHETIC_TASK_CONFIG, weights=task_weights
            )[0]
            task_name = task_config.task
            this_tier_config = ncc.constants.TIER_CONFIG[tier]
            rewarding_frequency = task_config.rewarding_frequency
            groud_truth_synapse = self.challenger(
                tokenizer=tokenizer,
                task=task_name,
                max_context_length_in_chars=this_tier_config.max_context_length_in_chars,
            )
            groud_truth_synapse.target_model = model_name
            synapse = groud_truth_synapse.model_copy()
            synapse.hide_ground_truth()
            axons = [self.metagraph.axons[int(uid)] for uid in batched_uids]
            bt.logging.info(f"Querying {tier} with uids: {batched_uids}")
            responses: list[ncc.TextCompressProtocol] = dendrite.query(
                axons=axons,
                synapse=synapse,
                deserialize=False,
                timeout=this_tier_config.timeout,
            )
            valid_responses: list[ncc.TextCompressProtocol] = []
            valid_uids: list[int] = []
            for uid, response in zip(batched_uids, responses):
                if (
                    not response
                    or not response.is_success
                    or (
                        len(response.compressed_tokens)
                        >= this_tier_config.max_condensed_tokens
                    )
                ):
                    bt.logging.info(f"Invalid response from uid {uid}")
                    self.miner_manager.update_scores( [0],[uid])
                else:
                    valid_responses.append(response)
                    valid_uids.append(uid)
            if not valid_responses:
                bt.logging.info("No valid responses.")
            if valid_responses and random.random() < rewarding_frequency:
                bt.logging.info(
                    f"Updating scores of {len(valid_responses)} valid responses."
                )
                payload = {
                    "miner_responses": [
                        {
                            "compressed_tokens": response.compressed_tokens,
                        }
                        for response in valid_responses
                    ],
                    "ground_truth_request": groud_truth_synapse.deserialize(),
                }
                payload["ground_truth_request"]["model_name"] = model_name
                payload["ground_truth_request"]["criterias"] = task_config.criterias

                scoring_response = requests.post(
                    f"http://{self.config.validator.score_backend.host}:{self.config.validator.score_backend.port}/scoring",
                    json=payload,
                    timeout=120,
                )
                scoring_response = scoring_response.json()

                scores: list[float] = scoring_response["scores"]

                factors_list = [
                    {
                        "normalized_score_in_batch": score,
                        "process_time/timeout": response.dendrite.process_time
                        / this_tier_config.timeout,
                    }
                    for score, response in zip(scores, valid_responses)
                ]
                penalized_scores = [
                    this_tier_config.scoring_lambda(factors) for factors in factors_list
                ]
                bt.logging.info(
                    f"Scores: {scores}\nFactors: {factors_list}\nPenalized scores: {penalized_scores}"
                )
                process_times = [
                    synapse.dendrite.process_time if synapse.is_success else -1
                    for synapse in responses
                ]
                self.miner_manager.update_scores(penalized_scores,valid_uids )
                self.miner_manager.update_ptime(process_times,valid_uids)

        except Exception as e:
            bt.logging.error(f"Error: {e}")

    def set_weights(self):
        r"""
        Just normalize the scores and set the weights.
        """
        self.current_block = self.subtensor.get_current_block()
        self.last_update = self.metagraph.last_update[self.uid]
        weights: np.ndarray = self.miner_manager.get_normalized_scores()
        if np.all(weights == 0):
            bt.logging.info(
                "All weights are zero. Setting all weights to 1 to prevent error."
            )
            weights = np.ones(len(self.metagraph.uids))
        bt.logging.info(
            f"Current block: {self.current_block}, Last Update: {self.last_update}"
        )
        if self.current_block > self.last_update + ncc.constants.SUBNET_TEMPO:
            bt.logging.info(f"Setting weights: {weights}")
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
