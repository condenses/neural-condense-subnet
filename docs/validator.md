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

## Steps to setup a Validator

1. Clone the repository
```bash
git clone https://github.com/condenses/neural-condense-subnet
cd neural-condense-subnet
```

2. Install the dependencies
```bash
pip install uv
uv sync --prerelease=allow
. .venv/bin/activate
. scripts/install_redis.sh
```
To test if Redis is working correctly, run `redis-cli ping` and it should return `PONG`.

**Optional**
- Login to Weights & Biases to use the logging feature
```bash
wandb login
```

3. Config your wallet, backend host, and port. Below just an example:

**Parameters**
- `--netuid` - The network UID of the subnet.
- `--subtensor.network` - The Subtensor network to connect to. `finney` for the mainnet. `test` for the testnet.
- `--wallet.name` - The name of the wallet to use.
- `--wallet.hotkey` - The hotkey of the wallet to use.
- `--axon.port` - The port to be posted to metagraph.
- `--validator.score_backend.host` - The host of the validator backend for scoring.
- `--validator.score_backend.port` - The port of the validator backend for scoring.
- `--validator.gate_port` - The port to open for the validator to forward the request from end-users to the miner. It should be an open port in your firewall. It's optional
- `--validator.use_wandb` - Use Weights & Biases for logging. It's optional.

**Important**: `axon_port` and `gate_port` must be opened in your firewall.

**Define bash variable in your terminal**
```bash
val_wallet="my_wallet"
val_hotkey="my_hotkey"
val_backend_host="localhost"
val_backend_port=8080
val_axon_port=12345
val_gate_port=12346
val_netuid=47
val_subtensor_network="finney"
```

4. Run the validator backend.
```bash
pm2 start python --name condense_validator_backend \
-- -m gunicorn services.validator_backend.scoring.app:app \
--workers 1 \
--bind $val_backend_host:$val_backend_port \
--timeout 0
```

5. Run the validator script
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
pm2 start python --name condense_validator \
-- -m neurons.validator \
--netuid $val_netuid \
--subtensor.network $val_subtensor_network \
--wallet.name $val_wallet \
--wallet.hotkey $val_hotkey \
--axon.port $val_axon_port \
--validator.score_backend.host $val_backend_host \
--validator.score_backend.port $val_backend_port \
--validator.use_wandb
```

6. Run the auto update script, it will check for updates every 30 minutes
```bash
pm2 start auto_update.sh --name "auto_updater"
```

7. Run Organic Server for using Organic API
```bash
pm2 start python --name condense_organic \
-- -m services.validator_backend.organic.app:app \
--netuid $val_netuid \
--subtensor.network $val_subtensor_network \
--wallet.name $val_wallet \
--wallet.hotkey $val_hotkey \
--axon.port $val_axon_port \
--validator.gate_port $val_gate_port \
```
