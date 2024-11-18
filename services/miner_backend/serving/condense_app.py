import torch
import base64
import io
import huggingface_hub
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import numpy as np
import rich
import time

app = Flask(__name__)


class InferenceLogger:
    """
    Logger class for inference processes. This logs key-value pairs of information
    during the execution of the backend inference, using the rich library for better readability.
    """

    @staticmethod
    def log(key, value):
        rich.print(f"Inference Backend -- {key}: {value}")


class Condenser(nn.Module):
    """
    A neural module for condensing large text contexts into smaller dense representations
    """

    def __init__(self, num_condense_tokens):
        super().__init__()
        self.dtype = torch.bfloat16
        self.condense_model = AutoModelForCausalLM.from_pretrained(
            "Condense-AI/Condenser-Llama-3.2-1B-20241117-173040", torch_dtype=self.dtype
        ).to("cuda")
        self.condense_tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B"
        )
        self.condense_tokenizer.pad_token = self.condense_tokenizer.eos_token
        self.num_condense_tokens = num_condense_tokens
        self.hidden_size = self.condense_model.config.hidden_size
        self.n_last_hidden_states = 2
        self.norm = nn.LayerNorm(self.hidden_size * n_last_hidden_states).to(
            dtype=self.dtype, device="cuda"
        )
        self.linear = nn.Linear(self.hidden_size * n_last_hidden_states, 4096).to(
            dtype=self.dtype, device="cuda"
        )
        self.pre_condensed_tokens = nn.Parameter(
            torch.randn(
                1,
                num_condense_tokens,
                self.hidden_size,
                dtype=self.dtype,
                device="cuda",
            )
        )

    def load_state_dict(self, state_dict):
        self.pre_condensed_tokens.data = state_dict["pre_condensed_tokens"].to(
            dtype=self.dtype, device="cuda"
        )
        self.linear.load_state_dict(
            {
                k: v.to(dtype=self.dtype, device="cuda")
                for k, v in state_dict["linear_state_dict"].items()
            }
        )
        self.norm.load_state_dict(
            {
                k: v.to(dtype=self.dtype, device="cuda")
                for k, v in state_dict["norm_state_dict"].items()
            }
        )

    @torch.no_grad()
    def compress(self, context: str) -> torch.Tensor:
        output = self.condense_tokenizer(
            context,
            return_tensors="pt",
            add_special_tokens=False,
            padding="max_length",
            max_length=4096,
            truncation=True,
            return_attention_mask=True,
        )
        context_ids = output.input_ids.to(device="cuda")
        attention_mask = output.attention_mask.to(device="cuda")

        # Processing embedding and condensation
        context_embeds = self.condense_model.get_input_embeddings()(context_ids)
        inputs_embeds_condense = torch.cat(
            [context_embeds, self.pre_condensed_tokens], dim=1
        )
        expanded_attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    attention_mask.shape[0],
                    self.num_condense_tokens,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=1,
        )

        # Generate condensed tokens
        output = self.condense_model(
            inputs_embeds=inputs_embeds_condense,
            output_hidden_states=True,
            attention_mask=expanded_attention_mask,
        )
        hidden_states = torch.cat(
            output.hidden_states[-self.n_last_hidden_states :], dim=-1
        )[:, -self.num_condense_tokens :, :]

        return self.linear(self.norm(hidden_states)).to(torch.float32)


def ndarray_to_base64(array: np.ndarray) -> str:
    """Convert a NumPy array to a base64-encoded string."""
    buffer = io.BytesIO()
    np.save(buffer, array)
    buffer.seek(0)
    base64_str = base64.b64encode(buffer.read()).decode("utf-8")
    return base64_str


# Load state dict and get model configuration
file_path = huggingface_hub.hf_hub_download(
    repo_id="Condense-AI/Condenser-Llama-3.2-1B-20241117-173040",
    filename="checkpoints/modules.pt",
    local_dir="./",
)
state_dict = torch.load(file_path)
num_condense_tokens = state_dict["modules"]["pre_condensed_tokens"].shape[1]
n_last_hidden_states = 2

# Initialize Condenser
condenser = Condenser(num_condense_tokens)
condenser.load_state_dict(state_dict["modules"])
condenser.eval()


@app.route("/condense", methods=["POST"])
def compress_endpoint():
    """
    Endpoint for prediction requests. Receives JSON data with 'context' and 'target_model'.
    """
    t1 = time.time()
    data = request.get_json()
    context = data.get("context")
    target_model = data.get("target_model")
    if not context:
        return jsonify({"error": "Missing 'context' in request."}), 400

    compressed_tokens = condenser.compress(context)
    compressed_tokens = compressed_tokens.squeeze(0)
    compressed_tokens = compressed_tokens.cpu().numpy()
    compress_tokens_bs64 = ndarray_to_base64(compressed_tokens)

    InferenceLogger.log(
        "Predict",
        f"Compress token length {len(compressed_tokens)} shape{compressed_tokens.shape}",
    )
    InferenceLogger.log("Inference time", time.time() - t1)

    return jsonify(
        {"compressed_tokens_b64": compress_tokens_bs64, "target_model": target_model}
    )
