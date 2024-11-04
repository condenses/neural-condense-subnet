import neural_condense_core as ncc
import httpx
from typing import Tuple
import bittensor as bt
import numpy as np
import time
import traceback


class Miner(ncc.BaseMiner):
    def __init__(self):
        super().__init__()
        self.blacklist_fns.append(self.blacklist_fn)
        self.forward_fns.append(self.forward_text_compress)
        self.setup_logging()
        self._initialize_rate_limits()

    def _initialize_rate_limits(self):
        r"""
        Initializes the rate limits for the miners.
        """
        self.rate_limits = {
            uid: ncc.ServingCounter(rate_limit)
            for uid, rate_limit in ncc.build_rate_limit(
                self.metagraph, self.config
            ).items()
        }

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
        self, synapse: ncc.TextCompressProtocol
    ) -> ncc.TextCompressProtocol:
        r"""
        Forward function for the Text-Compress task.
        Args:
            synapse (TextCompressProtocol): The synapse containing the text to compress.
        Returns:
            synapse (TextCompressProtocol): The synapse containing the compressed tokens.
        """
        bt.logging.info(f"Forwarding text compress: {synapse.context[:100]}...")

        payload = synapse.get_miner_payload()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://{self.config.miner.backend_host}:{self.config.miner.backend_port}/condense",
                json=payload,
                timeout=synapse.timeout,
            )
            response = response.json()
            compressed_tokens = response["compressed_tokens"]
            synapse.compressed_tokens = compressed_tokens
        bt.logging.info(f"Compressed shape: {np.array(compressed_tokens).shape}")
        return synapse

    def blacklist_fn(self, synapse: ncc.TextCompressProtocol) -> Tuple[bool, str]:
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
            f"Rate limit: {uid} {self.rate_limits[uid].counter}/{self.rate_limits[uid].rate_limit}"
        )
        if not allowed:
            return True, "Rate limit exceeded."
        return False, ""


if __name__ == "__main__":
    miner = Miner()
    miner.run()
