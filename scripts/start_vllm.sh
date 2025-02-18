uv venv .vllm-venv
. .vllm-venv/bin/activate
uv pip install vllm
pm2 start --name vllm "vllm serve unsloth/Llama-3.1-8B-Instruct"
. .venv/bin/activate