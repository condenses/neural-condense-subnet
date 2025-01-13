import neural_condense_core as ncc
import httpx
from typing import Tuple
import bittensor as bt
import time
import traceback
import redis


class Miner(ncc.base.BaseMiner):
    def __init__(self):
        super().__init__()
        self.blacklist_fns.append(self.blacklist_fn)
        self.forward_fns.append(self.forward_text_compress)
        self.redis_config = ncc.constants.DATABASE_CONFIG.redis
        self.redis = redis.Redis(
            host=self.redis_config.host,
            port=self.redis_config.port,
            db=self.redis_config.db,
            decode_responses=True,
        )
        self.setup_logging()
        self._initialize_rate_limits()

    def _initialize_rate_limits(self):
        r"""
        Initializes the rate limits for the miners.
        """
        self.rate_limits = {
            uid: ncc.validator_utils.managing.ServingCounter(
                rate_limit=rate_limit,
                uid=uid,
                tier=self.config.miner.tier,
                redis_client=self.redis,
                postfix_key=self.config.axon.port,
            )
            for uid, rate_limit in ncc.common.build_rate_limit(
                self.metagraph, self.config, tier=self.config.miner.tier
            ).items()
        }
        for k, v in self.rate_limits.items():
            v.reset_counter()
            bt.logging.info(
                f"Reset rate limit for {k}: {v.get_current_count()}/{v.rate_limit}"
            )

    def run(self):
        self.setup_axon()
        bt.logging.info("Starting main loop")
        step = 0
        while True:
            try:
                # Periodically update our knowledge of the network graph.
                if step % 60 == 0:
                    self.metagraph.sync()
                    self._initialize_rate_limits()
                    log = (
                        f"Block: {self.metagraph.block.item()} | "
                        f"Incentive: {self.metagraph.I[self.my_subnet_uid]} | "
                    )
                    bt.logging.info(log)
                step += 1
                time.sleep(10)

            except KeyboardInterrupt:
                self.axon.stop()
                bt.logging.success("Miner killed by keyboard interrupt.")
                break
            except Exception as e:
                bt.logging.error(f"Miner exception: {e}")
                bt.logging.error(traceback.format_exc())
                continue

    async def forward_text_compress(
        self, synapse: ncc.protocol.TextCompressProtocol
    ) -> ncc.protocol.TextCompressProtocol:
        r"""
        Forward function for the Text-Compress task.
        Args:
            synapse (TextCompressProtocol): The synapse containing the text to compress.
        Returns:
            synapse (TextCompressProtocol): The synapse containing the compressed tokens.
        """
        bt.logging.info(
            f"Forwarding text compress: {synapse.context[:100]}...{synapse.context[-100:]}"
        )
        bt.logging.info(f"Context length: {len(synapse.context)}")

        payload = synapse.miner_payload

        async with httpx.AsyncClient(timeout=synapse.timeout) as client:
            response = await client.post(
                f"http://{self.config.miner.backend_host}:{self.config.miner.backend_port}/condense",
                json=payload,
            )
            response = response.json()

            if self.config.miner.tier == "universal":
                synapse.compressed_context = response["compressed_context"]
                return synapse

            elif self.config.miner.tier == "research":
                compressed_kv_url = response["compressed_kv_url"]
                bt.logging.info(f"Compressed & uploaded to {compressed_kv_url}")
                return ncc.protocol.TextCompressProtocol(
                    compressed_kv_url=compressed_kv_url
                )

    def blacklist_fn(
        self, synapse: ncc.protocol.TextCompressProtocol
    ) -> Tuple[bool, str]:
        r"""
        Blacklist function for the Text-Compress task.
        Args:
            synapse (TextCompressProtocol): The synapse containing the text to compress.
        Returns:
            bool: Whether to blacklist the synapse.
            reason (str): The reason for blacklisting the synapse.
        """
        hotkey = synapse.dendrite.hotkey
        uid = self.metagraph.hotkeys.index(hotkey)
        stake = self.metagraph.S[uid]
        if stake < ncc.constants.MIN_STAKE:
            return True, "Stake too low."
        allowed = self.rate_limits[uid].increment()
        bt.logging.info(
            f"Rate limit: {uid} {self.rate_limits[uid].get_current_count()}/{self.rate_limits[uid].rate_limit}"
        )
        if not allowed:
            return True, "Rate limit exceeded."
        return False, ""


if __name__ == "__main__":
    miner = Miner()
    miner.run()
