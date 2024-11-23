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

2. Install Poetry and dependencies
```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# Install Redis
. scripts/install_redis.sh
```
To test if Redis is working correctly, run `redis-cli ping` and it should return `PONG`.

**Optional**
- Login to Weights & Biases to use the logging feature
```bash
poetry run wandb login
```

3. Config your wallet, backend host, and port. Below just an example:

[rest of the documentation remains the same until step 4]

4. Run the validator backend.
```bash
pm2 start poetry --name condense_validator_backend \
-- run python -m gunicorn services.validator_backend.scoring.app:app \
--workers 1 \
--bind $val_backend_host:$val_backend_port \
--timeout 0
```

5. Run the validator script
```bash
pm2 start poetry --name condense_validator \
-- run python -m neurons.validator \
--netuid $val_netuid \
--subtensor.network $val_subtensor_network \
--wallet.name $val_wallet \
--wallet.hotkey $val_hotkey \
--axon.port $val_axon_port \
--validator.gate_port $val_gate_port \
--validator.score_backend.host $val_backend_host \
--validator.score_backend.port $val_backend_port \
--validator.use_wandb
```

6. Run the auto update script, it will check for updates every 30 minutes
```bash
pm2 start auto_update.sh --name "auto_updater"
```