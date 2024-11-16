import os
import argparse
import traceback
import bittensor as bt
import threading
import time
from substrateinterface import SubstrateInterface
from .config import add_common_config, add_validator_config
from abc import abstractmethod, ABC
from ..constants import constants


class Validator(ABC):
    def __init__(self):
        self.config = self.get_config()
        print(self.config)
        self.setup_logging()
        self.setup_bittensor_objects()
        self.last_update = 0
        self.current_block = 0
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.node = SubstrateInterface(url=self.config.subtensor.chain_endpoint)

    def get_config(self):
        parser = argparse.ArgumentParser()
        parser = add_validator_config(parser)
        parser = add_common_config(parser)
        config = bt.config(parser)
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey_str,
                config.netuid,
                "validator",
            )
        )
        print(config)
        os.makedirs(config.full_path, exist_ok=True)
        return config

    def setup_logging(self):
        bt.logging.enable_default()
        bt.logging.enable_info()

        if self.config.logging.debug:
            bt.logging.enable_debug()
        if self.config.logging.trace:
            bt.logging.enable_trace()
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(
            f"Running validator for subnet: {self.config.netuid} on network: {self.config.subtensor.network} with config:"
        )
        bt.logging.info(self.config)

    def setup_bittensor_objects(self):
        bt.logging.info("Setting up Bittensor objects.")
        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")
        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")
        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"\nYour validator: {self.wallet} is not registered to chain connection: {self.subtensor} \nRun 'btcli register' and try again."
            )
            exit()
        else:
            self.my_subnet_uid = self.metagraph.hotkeys.index(
                self.wallet.hotkey.ss58_address
            )
            bt.logging.info(f"Running validator on uid: {self.my_subnet_uid}")

    def setup_axon(self):
        self.axon = bt.axon(wallet=self.wallet, config=self.config)
        bt.logging.info(
            f"Serving axon on network: {self.config.subtensor.network} with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        bt.logging.info(f"Axon: {self.axon}")
        bt.logging.info(f"Starting axon server on port: {self.config.axon.port}")
        self.axon.start()

    def node_query(self, module, method, params):
        try:
            result = self.node.query(module, method, params).value

        except Exception:
            # reinitilize node
            self.node = SubstrateInterface(url=self.config.subtensor.chain_endpoint)
            result = self.node.query(module, method, params).value

        return result

    def synthetic_loop_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    @abstractmethod
    def start_epoch(self):
        pass

    def run(self):
        self.setup_axon()
        bt.logging.info("Starting validator loop.")
        while True:
            start_epoch = time.time()

            try:
                self.start_epoch()
            except Exception as e:
                bt.logging.error(f"Forward error: {e}")
                traceback.print_exc()

            end_epoch = time.time()
            elapsed = end_epoch - start_epoch
            time_to_sleep = max(0, constants.EPOCH_LENGTH - elapsed)
            bt.logging.info(f"Epoch finished. Sleeping for {time_to_sleep} seconds.")
            time.sleep(time_to_sleep)

            try:
                set_weights_thread = threading.Thread(target=self.set_weights)
                set_weights_thread.start()
                set_weights_thread.join(timeout=constants.SET_WEIGHTS_TIMEOUT)

                if set_weights_thread.is_alive():
                    bt.logging.warning(
                        "Set weights timeout reached, continuing to next epoch"
                    )
            except Exception as e:
                bt.logging.error(f"Set weights error: {e}")
                traceback.print_exc()

            try:
                self.resync_metagraph()
            except Exception as e:
                bt.logging.error(f"Resync metagraph error: {e}")
                traceback.print_exc()

            except KeyboardInterrupt:
                bt.logging.success("Keyboard interrupt detected. Exiting validator.")
                exit()

    @abstractmethod
    def set_weights(self):
        pass

    def resync_metagraph(self):
        self.metagraph.sync()
