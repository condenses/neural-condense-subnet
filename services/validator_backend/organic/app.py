import neural_condense_core.validator_utils as vutils
from neural_condense_core.base.config import add_common_config, add_validator_config
import argparse
import bittensor as bt

def setup_config():
    parser = argparse.ArgumentParser()
    parser = add_common_config(parser)
    parser = add_validator_config(parser)
    config = bt.config(parser)
    bt.logging.info(f"Config: {config}")
    return config
    
def setup_bittensor_objects(config: bt.config):
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)
    return wallet, metagraph

def setup_miner_manager(config: bt.config, wallet, metagraph: bt.metagraph):
    miner_manager = vutils.managing.MinerManager(uid=config.netuid, wallet=wallet, metagraph=metagraph, config=None)
    return miner_manager

def setup_organic_gate(config: bt.config, miner_manager: vutils.managing.MinerManager):
    organic_gate = vutils.monetize.OrganicGate(miner_manager=miner_manager)
    return organic_gate

def main():
    config = setup_config()
    wallet, metagraph = setup_bittensor_objects(config)
    miner_manager = setup_miner_manager(config, wallet, metagraph)
    organic_gate = setup_organic_gate(config, miner_manager)
    organic_gate.start_server()

if __name__ == "__main__":
    main()