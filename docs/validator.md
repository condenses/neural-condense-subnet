<div align="center">

# ⚡ Validator Documentation

</div>

## Minimum Requirements
- GPU with at least 80GB of VRAM (A100, H100, etc.) to run LLMs and Reward Model
- 512GB of SSD storage
- CUDA, NVIDIA Driver installed
- Internet connection with at least 4Gbps
- PM2 install (see [Guide to install PM2](./pm2.md))

## What does a Validator do?

- Synthetic request & evaluate miner's performance by using prepared tasks: autoencoder, question-answering, conservation, etc.
- Forward Organic API if you want to sell your bandwidth to the end-users.

## Steps to setup a Miner

1. Clone the repository
```bash
git clone https://github.com/condenses/neural-condense-subnet
cd neural-condense-subnet
```

2. Install the dependencies
```bash
pip install -e .
```

3. Config your wallet, backend host, and port. Below just an example:

**Parameters**
- `--netuid` - The network UID of the subnet.
- `--subtensor.network` - The Subtensor network to connect to. `finney` for the mainnet. `test` for the testnet.
- `--wallet.name` - The name of the wallet to use.
- `--wallet.hotkey` - The hotkey of the wallet to use.
- `--axon.port` - The port to be posted to metagraph.
- `--validator.scoring_backend.host` - The host of the validator backend for scoring.
- `--validator.scoring_backend.port` - The port of the validator backend for scoring.
- `--validator.gate_port` - The port to open for the validator to forward the request from end-users to the miner. It should be an open port in your firewall. It's optional

**Important**: `axon_port` and `gate_port` must be opened in your firewall.

```bash
val_wallet="my_wallet"
val_hotkey="my_hotkey"
val_backend_host="localhost"
val_backend_port=8080
val_axon_port=12345
val_gate_port=12346
val_netuid=52
val_subtensor_network="finney"
```

4. Run the validator backend.
```bash
pm2 start python --name condense_validator_backend \
-- -m uvicorn services.validator_backend.scoring.app:app \
--port $val_backend_port \
--host 0.0.0.0
```

5. Run the validator script
```bash
pm2 start python --name condense_validator \
-- -m neurons.validator \
--netuid $val_netuid \
--subtensor.network $val_subtensor_network \
--wallet.name $val_wallet \
--wallet.hotkey $val_hotkey \
--axon.port $val_axon_port \
--validator.gate_port $val_gate_port \
--validator.scoring_backend.host $val_backend_host \
--validator.scoring_backend.port $val_backend_port
```

If you want to update the parameters, you can use the following command:
```bash
pm2 restart condense_validator --update-env
```