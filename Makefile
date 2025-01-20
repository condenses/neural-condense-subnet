# define miner variables
miner_tier="universal"
miner_wallet="tnm"
miner_hotkey="0"
miner_backend_host="localhost"
miner_backend_port=8080
miner_axon_port=12345
miner_netuid=245 # testuid
miner_subtensor_network="test"

# run command
run_miner_universal_backend:
	pm2 start python --name miner_universal_backend \
	-- -m gunicorn "services.miner_backend.universal_app:create_app('llmlingua-2')" \
	--timeout 120 \
	--bind 0.0.0.0:$(miner_backend_port)

run_miner_research_backend:
	pm2 start python --name miner_research_backend \
	-- -m gunicorn "services.miner_backend.app:create_app('llmlingua-2')" \
	--timeout 120 \
	--bind 0.0.0.0:$(miner_backend_port)

run_miner:
	pm2 start python --name miner \
	-- -m neurons.miner \
	--netuid $(miner_netuid) \
	--subtensor.network $(miner_subtensor_network) \
	--wallet.name $(miner_wallet) \
	--wallet.hotkey $(miner_hotkey) \
	--miner.tier $(miner_tier) \
	--miner.backend_host $(miner_backend_host) \
	--miner.backend_port $(miner_backend_port) \
	--axon.port $(miner_axon_port)

# define validator variables
val_wallet="tnv"
val_hotkey="0"
val_backend_host="localhost"
val_backend_port=8089
val_universal_backend_host="localhost"
val_universal_backend_port=8090
val_axon_port=12346
val_gate_port=12347
val_netuid=245 # testuid
val_subtensor_network="test"

run_val_research_backend:
	pm2 start python --name val_research_backend \
	-- -m gunicorn services.validator_backend.scoring.app:app \
	--workers 1 \
	--bind $(val_backend_host):$(val_backend_port) \
	--timeout 0

run_val_universal_backend:
	pm2 start python --name val_universal_backend \
	-- -m gunicorn services.validator_backend.universal_scoring.app:app \
	--workers 1 -k uvicorn.workers.UvicornWorker \
	--bind $(val_universal_backend_host):$(val_universal_backend_port) \
	--timeout 0

run_validator:
	export HF_HUB_ENABLE_HF_TRANSFER=1
	pm2 start python --name validator \
	-- -m neurons.validator \
	--netuid $(val_netuid) \
	--subtensor.network $(val_subtensor_network) \
	--wallet.name $(val_wallet) \
	--wallet.hotkey $(val_hotkey) \
	--axon.port $(val_axon_port) \
	--validator.score_backend.host $(val_backend_host) \
	--validator.score_backend.port $(val_backend_port) \
	--validator.universal_score_backend.host $(val_universal_backend_host) \
	--validator.universal_score_backend.port $(val_universal_backend_port)

all:
	make run_miner_universal_backend
	make run_miner
	make run_val_universal_backend
	make run_validator

# export TOGETHER_API_KEY=e5473bbf93db6340c70c816053ab307b372febe7717ce01c26d0829c8878f7a2 & \
# python -m gunicorn services.validator_backend.universal_scoring.app:app \
# 	--workers 1 -k uvicorn.workers.UvicornWorker \
# 	--bind localhost:8090 \
# 	--timeout 0