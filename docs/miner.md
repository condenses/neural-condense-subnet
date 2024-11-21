<div align="center">

# ⚡ Miner Documentation

</div>

## Minimum Requirements for Baseline
- GPU with at least 24GB of VRAM (RTX 4090, A6000, A100, H100, etc.) to run Baseline Model
- CUDA, NVIDIA Driver installed
- PM2 install (see [Guide to install PM2](./pm2.md))
- `minio` Public Storage (see [Guide to install MINIO](./minio.md))

## What does a Miner do?

A miner is a node that is responsible for condensing a long text into much shorter as condensed tokens. These condensed tokens are then used to feed to Large Language Models like Llama, Gemma, Mistral, etc.

## How does a Miner work?

We (subnet owner) provide some baselines for miners. But miners have to research their own algorithms to be more competitive. We also have a mission to push the latest SOTA algorithms to the miners as soon as possible.

So basically, there are somethings that a miner has to do:

1. Select a TIER: we have 3 tiers: research, inference_0, inference_1. Each tier is tailored for different API need, example `inference_0` for long text and `inference_1` for very long text. You can see the details in the miner's config file: `neural_condense_core/constants.py` or at the [README.md](../README.md) doc.

2. Implement your own algorithm or pick one of our baseline algorithms. You can find the baseline algorithms in the `services/miner_backend/serving` folder.
The schema of backend api is very simple: `Validator` sends you a dictionary with the `context: str` and you have to return a `list[list[floats]]` `(seq_len x hidden_size)` which is the condensed tokens.

3. After having a competitive backend, you need to measure it to meet speed and load defined in the tier. **Our baselines are required to use GPU**.

4. Register your slot and start mining.

## Steps to setup a Miner

1. Clone the repository
```bash
git clone https://github.com/condenses/neural-condense-subnet
cd neural-condense-subnet
```

2. Install the dependencies
```bash
pip install -e . --ignore-installed
pip install "numpy<2"
. services/miner_backend/serving/download_checkpoint.sh
```

3. Config your wallet, backend, etc... Below just an example:

**Parameters**
- `--miner.tier` - The selected tier should be suitable with your backend.
- `--netuid` - The network UID of the subnet.
- `--subtensor.network` - The Subtensor network to connect to. `finney` for the mainnet. `test` for the testnet.
- `--wallet.name` - The name of the wallet to use.
- `--wallet.hotkey` - The hotkey of the wallet to use.
- `--axon.port` - The port to be posted to metagraph.
- `--miner.backend.host` - The host of the miner backend for condensing.
- `--miner.backend.port` - The port of the miner backend for condensing.

**Important**: `axon_port` must be opened in your firewall.

**Define bash variable in your terminal**
```bash
miner_tier="inference_0"
miner_wallet="my_wallet"
miner_hotkey="my_hotkey"
miner_backend_host="localhost"
miner_backend_port=8080
miner_axon_port=12345
miner_netuid=47
miner_subtensor_network="finney"
```

4. Run the miner backend. Example of using our baseline KVPress as a backend (export the minio credentials):
```bash
export MINIO_ACCESS_KEY="your_minio_access_key"
export MINIO_SECRET_KEY="your_minio_secret_key"
export MINIO_BUCKET="your_minio_bucket"
export MINIO_SERVER="your_minio_server"
pm2 start python --name condense_miner_backend \
-- -m gunicorn services.miner_backend.kvpress.app:app \
--timeout 120 \
--bind 0.0.0.0:$miner_backend_port
```

5. Run the mining script
```bash
pm2 start python --name condense_miner \
-- -m neurons.miner \
--netuid $miner_netuid \
--subtensor.network $miner_subtensor_network \
--wallet.name $miner_wallet \
--wallet.hotkey $miner_hotkey \
--miner.tier $miner_tier \
--miner.backend_host $miner_backend_host \
--miner.backend_port $miner_backend_port \
--axon.port $miner_axon_port
```