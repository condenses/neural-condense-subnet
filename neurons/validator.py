import neural_condense_core as ncc
import bittensor as bt
import threading
import random
from transformers import AutoTokenizer
from transformers.utils.logging import disable_propagation, disable_default_handler
import numpy as np
import time
import requests
import wandb
from neural_condense_core.validator_utils import logging

disable_default_handler()
disable_propagation()


class Validator(ncc.base.BaseValidator):
    def __init__(self):
        super().__init__()
        self.tier_config = ncc.constants.TIER_CONFIG
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
            try:
                message = "incentivized-decentralzied-condensed-ai" + "-".join(
                    random.choices("0123456789abcdef", k=16)
                )
                signature = f"0x{self.dendrite.keypair.sign(message).hex()}"
                wandb.init(
                    project="Neural-Condense-Subnet",
                    name=f"validator-{self.uid}",
                    entity="toilaluan",
                    job_type="validation",
                    group="validator",
                    resume="auto",
                    config={
                        "signature": signature,
                        "uid": self.uid,
                        "message": message,
                        "ss58_address": self.metagraph.hotkeys[self.uid],
                    },
                )
            except Exception as e:
                bt.logging.error(f"Starting wandb error: {e}")

    def forward(self):
        bt.logging.info("Running epoch.")
        self.miner_manager.sync()
        threads = [
            threading.Thread(target=self._forward_tier, args=(tier,))
            for tier in self.tier_config
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        try:
            self.miner_manager._report()
            self.miner_manager.save_state()
        except Exception as e:
            bt.logging.error(f"Failed to report metadata & save-state: {e}")

    def _forward_tier(self, tier):
        if ncc.constants.TIER_CONFIG[tier].incentive_percentage == 0:
            bt.logging.info(f"Tier {tier} has no incentive percentage.")
            return

        model_name = random.choice(ncc.constants.TIER_CONFIG[tier].supporting_models)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        serving_counter = self.miner_manager.serving_counter.get(tier, {})

        if not serving_counter:
            bt.logging.info(f"No miners in tier {tier}.")
            return

        n_sets = int(
            ncc.constants.TIER_CONFIG[tier].requests_per_epoch
            * ncc.constants.RPE_PERCENTAGE_FOR_SYNTHETIC
        )
        sleep_per_set = ncc.constants.EPOCH_LENGTH / n_sets
        query_threads = []

        for _ in range(n_sets):
            uids = list(serving_counter.keys())
            uids.sort(key=lambda uid: self.miner_manager.metadata[uid].elo_rating)
            group_size = max(2, len(uids) // 4)
            groups = [uids[i : i + group_size] for i in range(0, len(uids), group_size)]
            for group in groups:
                random.shuffle(group)
            uids = [uid for group in groups for uid in group]

            pre_batched_uids = [
                uids[i : i + ncc.constants.BATCH_SIZE]
                for i in range(0, len(uids), ncc.constants.BATCH_SIZE)
            ]
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

    def _forward_batch(self, tier, model_name, batched_uids, tokenizer):
        try:
            dendrite = bt.dendrite(self.wallet)
            task_config = self._get_task_config()
            this_tier_config = ncc.constants.TIER_CONFIG[tier]

            groud_truth_synapse = self._prepare_synapse(
                tokenizer, task_config, this_tier_config, model_name
            )
            bt.logging.info(f"Prepared ground truth synapse for {batched_uids}.")
            synapse = groud_truth_synapse.model_copy()
            synapse.hide_ground_truth()

            responses = self._query_miners(
                dendrite, batched_uids, synapse, this_tier_config.timeout
            )
            bt.logging.info(f"Queried miners for {batched_uids}.")
            valid_responses, valid_uids, invalid_uids = self._validate_responses(
                responses, batched_uids, this_tier_config
            )
            bt.logging.info(f"Validated responses for {batched_uids}.")
            if not valid_responses:
                bt.logging.info(f"No valid responses for batch {batched_uids}.")
                return

            if random.random() < task_config.rewarding_frequency:
                self._process_and_score_responses(
                    valid_responses,
                    valid_uids,
                    invalid_uids,
                    groud_truth_synapse,
                    model_name,
                    task_config,
                    this_tier_config,
                    tier,
                )
                bt.logging.info(f"Processed and scored responses for {batched_uids}.")
            else:
                bt.logging.info(f"Not rewarding batch {batched_uids}.")

        except Exception as e:
            bt.logging.error(f"Error: {e}")

    def _get_task_config(self):
        return random.choices(
            ncc.constants.SYNTHETIC_TASK_CONFIG,
            weights=[t.weight for t in ncc.constants.SYNTHETIC_TASK_CONFIG],
        )[0]

    def _prepare_synapse(self, tokenizer, task_config, tier_config, model_name):
        synapse = self.challenger(
            tokenizer=tokenizer,
            task=task_config.task,
            max_context_length_in_chars=tier_config.max_context_length_in_chars,
        )
        synapse.target_model = model_name
        return synapse

    def _query_miners(self, dendrite, uids, synapse, timeout):
        return dendrite.query(
            axons=[self.metagraph.axons[uid] for uid in uids],
            synapse=synapse,
            deserialize=False,
            timeout=timeout,
        )

    def _validate_responses(self, responses, uids, tier_config):
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

    def _process_and_score_responses(
        self,
        valid_responses,
        valid_uids,
        invalid_uids,
        ground_truth_synapse,
        model_name,
        task_config,
        tier_config,
        tier,
    ):
        payload = {
            "miner_responses": [
                {"compressed_tokens_b64": r.compressed_tokens_b64}
                for r in valid_responses
            ],
            "ground_truth_request": ground_truth_synapse.deserialize()
            | {"model_name": model_name, "criterias": task_config.criterias},
        }

        scoring_response = requests.post(
            f"http://{self.config.validator.score_backend.host}:{self.config.validator.score_backend.port}/scoring",
            json=payload,
            timeout=120,
        ).json()

        scores = scoring_response["scores"]
        compress_rate_rewards = [
            1 - len(r.compressed_tokens) / tier_config.max_condensed_tokens
            for r in valid_responses
        ]

        factors_list = [
            {
                "normalized_score_in_batch": score,
                "process_time/timeout": response.dendrite.process_time
                / tier_config.timeout,
                "compress_rate_reward": compress_rate_reward,
            }
            for score, compress_rate_reward, response in zip(
                scores, compress_rate_rewards, valid_responses
            )
        ]

        penalized_scores = [
            min(1, max(0, tier_config.scoring_lambda(f))) for f in factors_list
        ]
        merged_scores = penalized_scores + [0] * len(invalid_uids)
        merged_uids = valid_uids + invalid_uids

        k_factor = self.get_k_factor(merged_uids)
        self.miner_manager.update_ratings(merged_scores, merged_uids, k_factor)

        if self.config.validator.use_wandb:
            logs = scoring_response["logs"] | {"penalized_scores": penalized_scores}
            logging.log_as_dataframe(data=logs, name="Batch Logs")
            logging.log_wandb(logs, valid_uids, tier=tier)

    def set_weights(self):
        self.current_block = self.subtensor.get_current_block()
        self.last_update = self.metagraph.last_update[self.uid]
        weights = self.miner_manager.get_normalized_ratings()

        if np.all(weights == 0):
            weights = np.ones(len(self.metagraph.uids))

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

    def get_k_factor(self, uids):
        mean_elo = sum(
            self.miner_manager.metadata[uid].elo_rating for uid in uids
        ) / len(uids)
        if mean_elo < ncc.constants.ELO_GROUPS["beginner"].max_elo:
            return ncc.constants.ELO_GROUPS["beginner"].k_factor
        elif mean_elo < ncc.constants.ELO_GROUPS["intermediate"].max_elo:
            return ncc.constants.ELO_GROUPS["intermediate"].k_factor
        return ncc.constants.ELO_GROUPS["advanced"].k_factor


if __name__ == "__main__":
    validator = Validator()
    validator.run()
