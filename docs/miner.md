<div align="center">

# ⚡ Miner Documentation

</div>

## Minimum Requirements for Baseline
- GPU with at least 24GB of VRAM (RTX 4090, A6000, A100, H100, etc.) to run Baseline Model
- CUDA, NVIDIA Driver installed
- PM2 install (see [Guide to install PM2](./pm2.md))
- Setup a cloud storage for uploading miner outputs. Here are some recommended options:
    - `Huggingface Hub` (free but has some limitations)
    - `AWS S3`
    - `minio` (open-source version of AWS S3) (see [Guide to install MINIO](./minio.md))
    - `Google Cloud Storage`
    - `Azure Blob Storage`

## What does a Miner do?

A miner is a node that is responsible for condensing a long text into much shorter as condensed tokens & activations. These condensed tokens & activations are then used to feed to Large Language Models like Llama, Gemma, Mistral, etc.

## How does a Miner work?

We (subnet owner) provide some baselines for miners. But miners have to research their own algorithms to be more competitive. We also have a mission to push the latest SOTA algorithms to the miners as soon as possible.

So basically, there are somethings that a miner has to do:

1. Select a TIER: we have 2 tiers: research, universal. You can see the details in the miner's config file: `neural_condense_core/constants.py` or at the [README.md](../README.md) doc.

2. Implement your own algorithm or pick one of our baseline algorithms. You can find the baseline algorithms in the `services/miner_backend/` folder.
The schema of backend api is very simple: `Validator` sends you a dictionary with the `context: str` and `target_model: str`.
- For `research` tier:
Miner runs their own backend that results in KV Cache of the target LLM model. Then miner uploads the KV Cache to the `minio` storage and returns the `minio` path to the `Validator`.
  - `past_key_values: Tuple[Tuple[torch.FloatTensor]]` is the format of the KV Cache. It would be loaded into the LLM using `torch.DynamicCache.from_legacy_cache(past_key_values)`.
- For `universal` tier:
Miner runs their own backend that results in compressed text representation and returns the compressed text to the `Validator` as an attribute of `SynapseResponse`.


3. After having a competitive backend, you need to measure it to meet speed and load defined in the tier. **Our baselines are required to use GPU**.

4. Register your slot and start mining.

## Steps to setup a Miner

### 1. Clone the repository
```bash
git clone https://github.com/condenses/neural-condense-subnet
cd neural-condense-subnet
```

### 2. Install the dependencies
```bash
pip install uv
uv sync --prerelease=allow
. .venv/bin/activate
. scripts/install_redis.sh
```

### 3. Config your wallet, backend, etc... Below just an example:

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
miner_tier="research" # or "universal"
miner_wallet="my_wallet"
miner_hotkey="my_hotkey"
miner_backend_host="localhost"
miner_backend_port=8080
miner_axon_port=12345
miner_netuid=47
miner_subtensor_network="finney"
```

### 4. Run the miner backend. <br>

#### 4.a. Research tier: <br>
You have to collect the `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET`, `MINIO_SERVER` from the minio setup (see [minio.md](./minio.md)).

There are three compression algorithms available:
- `kvpress`: Basic KV-cache compression
- `soft_token`: Soft token compression (requires additional model)
- `activation_beacon`: Activation beacon compression

```bash
export MINIO_ACCESS_KEY="your_minio_access_key"
export MINIO_SECRET_KEY="your_minio_secret_key"
export MINIO_BUCKET="condense"
export MINIO_SERVER="your_minio_server"

# Choose one of the following commands based on your preferred algorithm:

# For KVPress compression:
pm2 start python --name condense_miner_backend \
-- -m gunicorn "services.miner_backend.app:create_app('kvpress')" \
--timeout 120 \
--bind 0.0.0.0:$miner_backend_port

# For Soft Token compression:
pm2 start python --name condense_miner_backend \
-- -m gunicorn "services.miner_backend.app:create_app('soft_token')" \
--timeout 120 \
--bind 0.0.0.0:$miner_backend_port

# For Activation Beacon compression:
pm2 start python --name condense_miner_backend \
-- -m gunicorn "services.miner_backend.app:create_app('activation_beacon')" \
--timeout 120 \
--bind 0.0.0.0:$miner_backend_port
```

**Note**: 
- If using `soft_token` algorithm, you can train your own model using our prepared trainer at [Condense-Trainer](https://github.com/condenses/condense-trainer).
- Each algorithm has different GPU memory requirements:
  - `kvpress`: ~24GB VRAM
  - `soft_token`: ~24GB VRAM + additional memory for condenser model
  - `activation_beacon`: ~24GB VRAM

You can also run the backend directly without PM2 for testing:
```bash
python -m gunicorn "services.miner_backend.app:create_app('kvpress')" --bind 0.0.0.0:8080
```
#### 4.b. Universal tier:
You can check the default `llmlingua-2` model in the `services.miner_backend.universal_app` folder, and develop your own model to further improve the performance.
```bash
pm2 start python --name miner_universal_backend \
	-- -m gunicorn "services.miner_backend.universal_app:create_app('llmlingua-2')" \
	--timeout 120 \
	--bind 0.0.0.0:8080
```


### 5. Run the mining script
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